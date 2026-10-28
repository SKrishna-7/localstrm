"""Comprehensive unit and integration tests for VRAM memory budgeting and KV-cache sizing.

Verifies:
1. Exact integer byte calculations per KV block across data types (FP16, BF16, FP32, INT8).
2. Parsing and conversion of user VRAM budgets across binary and decimal units.
3. Component estimation: model weights, runtime overhead, workspace, safety margin.
4. Preflight rejection on insufficient budget or exceeding hardware capacity.
5. Mutual exclusivity and conflict detection across CacheConfig and MiniVLLMEngine.
6. Execution prevention: preflight rejection happens BEFORE loading model weights.
7. Engine metadata exposure for budget resolution.
8. Error cleanup and recovery under memory pressure.
"""

from __future__ import annotations

import types
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config import CacheConfig
from src.engine.llm_engine import MiniVLLMEngine
from src.memory.preflight import (
    HardwareCapacity,
    ModelMetadata,
    VRAMBudgetEstimate,
    calculate_kv_block_bytes,
    estimate_vram_budget,
    format_binary_bytes,
    parse_vram_budget_bytes,
)


class MockTokenizer:
    """Lightweight deterministic tokenizer for testing."""

    def __init__(self, vocab: dict[str, int] | None = None, eos_token_id: int = 2) -> None:
        self.vocab = vocab or {"hello": 10, "world": 11, "antigravity": 12, "test": 13}
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        words = text.strip().split()
        return [self.vocab.get(w, ord(w[0]) if w else 1) for w in words] or [1]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(t) for t in token_ids)


class LayerCache:
    def __init__(self, num_tokens: int = 1, num_kv_heads: int = 2, head_dim: int = 64) -> None:
        self.keys = [torch.zeros((num_tokens, num_kv_heads, head_dim))]
        self.values = [torch.zeros((num_tokens, num_kv_heads, head_dim))]


class DynamicCacheMock:
    def __init__(self, num_layers: int = 24, num_tokens: int = 1, num_kv_heads: int = 2, head_dim: int = 64) -> None:
        self.layers = [LayerCache(num_tokens, num_kv_heads, head_dim) for _ in range(num_layers)]


def _create_mock_hf_model(
    num_layers: int = 24,
    num_heads: int = 16,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    hidden_size: int = 1024,
    vocab_size: int = 1000,
    device: str = "cpu",
) -> MagicMock:
    """Build a mock model with the specified architectural config and embedding weights."""
    model = MagicMock()
    model.config = types.SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    model.device = torch.device(device)
    model.to.return_value = model
    model.eval.return_value = model

    def forward_impl(input_tensor: Any, past_key_values: Any = None, use_cache: bool = True, **kwargs: Any) -> Any:
        num_tokens = input_tensor.shape[1]
        logits = torch.zeros((1, num_tokens, vocab_size))
        logits[0, -1, 2] = 10.0  # eos
        cache = DynamicCacheMock(
            num_layers=num_layers,
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        return types.SimpleNamespace(logits=logits, past_key_values=cache)

    model.side_effect = forward_impl
    return model


# ---------------------------------------------------------------------------
# 1. Byte Calculation Tests
# ---------------------------------------------------------------------------


def test_calculate_kv_block_bytes_qwen25_05b_fp16():
    """Verify exact 192 KiB (196,608 bytes) for Qwen2.5-0.5B under FP16.

    Formula: 24 layers * 2 (K+V) * 16 block_size * 2 kv_heads * 64 head_dim * 2 bytes = 196,608 bytes.
    """
    bytes_per_block = calculate_kv_block_bytes(
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
        kv_cache_dtype="fp16",
    )
    assert bytes_per_block == 196_608
    assert bytes_per_block == 192 * 1024


def test_calculate_kv_block_bytes_across_dtypes():
    """Verify calculations for BF16, FP32, INT8, and unsupported types."""
    # BF16: 2 bytes per element -> 192 KiB
    bf16_bytes = calculate_kv_block_bytes(24, 2, 64, 16, "bf16")
    assert bf16_bytes == 196_608

    # FP32: 4 bytes per element -> 384 KiB
    fp32_bytes = calculate_kv_block_bytes(24, 2, 64, 16, "fp32")
    assert fp32_bytes == 393_216

    # INT8: 1 byte per element + float16 per-token-head scale tensors
    # Tensor bytes: 24 * 2 * 16 * 2 * 64 * 1 = 98,304 bytes
    # Scale bytes:  24 * 2 * 16 * 2 * 1  * 2 = 3,072 bytes
    # Total INT8: 101,376 bytes = 99 KiB
    int8_bytes = calculate_kv_block_bytes(24, 2, 64, 16, "int8")
    assert int8_bytes == 101_376
    assert int8_bytes == 99 * 1024

    # Unsupported dtype
    with pytest.raises(ValueError, match="Unsupported kv_cache_dtype"):
        calculate_kv_block_bytes(24, 2, 64, 16, "int4")


# ---------------------------------------------------------------------------
# 2. Budget Parsing and Binary Units Tests
# ---------------------------------------------------------------------------


def test_parse_vram_budget_bytes_valid():
    """Verify parsing of binary, decimal, integer, and float budget specifications."""
    assert parse_vram_budget_bytes("2GiB") == 2 * (1024**3)
    assert parse_vram_budget_bytes("2048MiB") == 2048 * (1024**2)
    assert parse_vram_budget_bytes("2GB") == 2 * (1000**3)
    assert parse_vram_budget_bytes(2.0) == 2 * (1024**3)
    assert parse_vram_budget_bytes(2) == 2 * (1024**3)
    assert parse_vram_budget_bytes(2147483648) == 2147483648


def test_parse_vram_budget_bytes_invalid():
    """Verify rejection of invalid, non-positive, or unparseable formats."""
    with pytest.raises(ValueError, match="vram_budget must be positive"):
        parse_vram_budget_bytes(-10)
    with pytest.raises(ValueError, match="vram_budget must be positive"):
        parse_vram_budget_bytes(0)
    with pytest.raises(ValueError, match="Invalid vram_budget"):
        parse_vram_budget_bytes("")
    with pytest.raises(ValueError, match="Could not parse numeric part of vram_budget"):
        parse_vram_budget_bytes("two_gib")
    with pytest.raises(ValueError, match="Unrecognized vram_budget format"):
        parse_vram_budget_bytes("gibberish")


def test_format_binary_bytes():
    """Verify human-readable binary formatting."""
    assert format_binary_bytes(1024) == "1.00 KiB"
    assert format_binary_bytes(196_608) == "192.00 KiB"
    assert format_binary_bytes(2 * (1024**3)) == "2.00 GiB"


# ---------------------------------------------------------------------------
# 3. Budget Estimator Logic & Rounding Tests
# ---------------------------------------------------------------------------


def test_estimate_vram_budget_success_and_integer_floor():
    """Verify budget estimation breakdown and integer floor division for total_blocks."""
    meta = ModelMetadata(
        model_id="test-model",
        is_local=True,
        weight_bytes=940 * (1024**2),  # ~940 MiB
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
    )
    hw = HardwareCapacity(
        device_type="cuda",
        device_name="NVIDIA Test GPU",
        total_memory=6 * (1024**3),
        available_memory=5 * (1024**3),
        disk_free=50 * (1024**3),
        cache_dir="/tmp",
    )

    # Total non-KV requirement:
    # 940 MiB (model) + 450 MiB (CUDA context overhead) + 64 MiB (chunked workspace) + 128 MiB (safety)
    # = 1582 MiB = 1,658,847,232 bytes
    # Budget: 2 GiB = 2,147,483,648 bytes
    # Available for KV: 2,147,483,648 - 1,658,847,232 = 488,636,416 bytes
    # Bytes per block: 196,608 bytes (192 KiB)
    # Total blocks: 488,636,416 // 196,608 = 2485 blocks
    est = estimate_vram_budget(
        model_id="test-model",
        vram_budget="2GiB",
        device="cuda",
        chunked_prefill=True,
        metadata=meta,
        hardware=hw,
    )

    assert est.is_valid
    assert est.bytes_per_block == 196_608
    assert est.total_blocks == 2485
    assert est.max_context_tokens == 2485 * 16
    assert est.available_for_kv_bytes >= est.total_blocks * est.bytes_per_block
    assert "2485 KV blocks" in est.summary()


def test_estimate_vram_budget_insufficient_budget():
    """Verify rejection when budget cannot cover non-KV requirements + 1 block."""
    meta = ModelMetadata(
        model_id="test-model",
        is_local=True,
        weight_bytes=1000 * (1024**2),  # 1000 MiB
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
    )
    hw = HardwareCapacity(
        device_type="cuda",
        device_name="Test GPU",
        total_memory=6 * (1024**3),
        available_memory=5 * (1024**3),
        disk_free=50 * (1024**3),
        cache_dir="/tmp",
    )

    # 1100 MiB budget < 1000 MiB (weights) + 120 MiB (overhead) + 64 MiB (workspace) + 128 MiB (safety) = 1312 MiB
    with pytest.raises(ValueError, match="Insufficient VRAM budget"):
        estimate_vram_budget(
            model_id="test-model",
            vram_budget="1100MiB",
            device="cuda",
            chunked_prefill=True,
            metadata=meta,
            hardware=hw,
        )


def test_estimate_vram_budget_exceeds_device_capacity():
    """Verify rejection when requested budget exceeds available device memory."""
    meta = ModelMetadata(
        model_id="test-model",
        is_local=True,
        weight_bytes=500 * (1024**2),
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
    )
    hw = HardwareCapacity(
        device_type="cuda",
        device_name="Test GPU",
        total_memory=4 * (1024**3),
        available_memory=3 * (1024**3),  # Only 3 GiB available
        disk_free=50 * (1024**3),
        cache_dir="/tmp",
    )

    with pytest.raises(ValueError, match="exceeds available device memory"):
        estimate_vram_budget(
            model_id="test-model",
            vram_budget="4GiB",  # 4 GiB > 3 GiB
            device="cuda",
            metadata=meta,
            hardware=hw,
        )


# ---------------------------------------------------------------------------
# 4. Configuration Conflict Validation Tests
# ---------------------------------------------------------------------------


def test_cache_config_mutual_exclusivity():
    """Verify that specifying both total_blocks and vram_budget_bytes raises ValueError."""
    with pytest.raises(ValueError, match="Conflicting KV-cache configuration"):
        CacheConfig(total_blocks=128, vram_budget_bytes=2 * (1024**3))

    # Single specification is allowed
    cfg_blocks = CacheConfig(total_blocks=128)
    assert cfg_blocks.total_blocks == 128
    assert cfg_blocks.vram_budget_bytes is None

    cfg_budget = CacheConfig(vram_budget_bytes=2 * (1024**3))
    assert cfg_budget.vram_budget_bytes == 2 * (1024**3)
    assert cfg_budget.total_blocks is None

    # Default fallback when neither is provided
    cfg_default = CacheConfig()
    assert cfg_default.total_blocks == 64
    assert cfg_default.vram_budget_bytes is None


def test_minivllm_engine_constructor_conflict():
    """Verify MiniVLLMEngine rejects conflicting total_blocks and vram_budget arguments."""
    with pytest.raises(ValueError, match="Conflicting KV-cache configuration"):
        MiniVLLMEngine(
            model_name="dummy",
            total_blocks=128,
            vram_budget="2GiB",
            device="cpu",
        )


# ---------------------------------------------------------------------------
# 5. Preflight Timing: Rejection BEFORE Weight Loading
# ---------------------------------------------------------------------------


def test_preflight_rejection_prevents_model_loading():
    """Ensure that an invalid or insufficient VRAM budget halts before loading weights."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.side_effect = RuntimeError("Weight loading should NOT be reached!")

        meta = ModelMetadata(
            model_id="mock-model",
            is_local=True,
            weight_bytes=2 * (1024**3),  # 2 GiB weights
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
        )

        with patch("src.memory.preflight.get_model_metadata", return_value=meta):
            with pytest.raises(ValueError, match="Insufficient VRAM budget"):
                MiniVLLMEngine(
                    model_name="mock-model",
                    vram_budget="500MiB",  # 500 MiB << 2 GiB
                    device="cpu",
                )

        mock_from_pretrained.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Engine Budget Metadata Exposure Tests
# ---------------------------------------------------------------------------


def test_minivllm_engine_metadata_exposes_budget():
    """Verify that MiniVLLMEngine exposes resolved total_blocks, budget bytes, and summary."""
    mock_model = _create_mock_hf_model()
    mock_tok = MockTokenizer()

    # Budget estimate stub
    budget_est = VRAMBudgetEstimate(
        vram_budget_bytes=2 * (1024**3),
        bytes_per_block=196_608,
        model_resident_bytes=940 * (1024**2),
        runtime_overhead_bytes=0,
        execution_workspace_bytes=64 * (1024**2),
        safety_margin_bytes=128 * (1024**2),
        non_kv_bytes=1132 * (1024**2),
        available_for_kv_bytes=896 * (1024**2),
        total_blocks=256,
        max_context_tokens=256 * 16,
        hardware_available_bytes=4 * (1024**3),
        device_type="cpu",
        is_valid=True,
    )

    with patch("src.memory.preflight.estimate_vram_budget", return_value=budget_est):
        engine = MiniVLLMEngine(
            model=mock_model,
            tokenizer=mock_tok,
            vram_budget="2GiB",
            device="cpu",
        )

    assert engine.total_blocks == 256
    assert engine.config.cache.vram_budget_bytes == 2 * (1024**3)
    assert engine.budget_resolution is not None
    summary = engine.budget_resolution.summary()
    assert "256 KV blocks" in summary
    assert "2.00 GiB" in summary


# ---------------------------------------------------------------------------
# 7. Memory Pressure & Reliable Recovery Tests
# ---------------------------------------------------------------------------


def test_engine_oversized_prompt_failure_and_subsequent_recovery():
    """Verify that an oversized request cleanly fails with MemoryError, releases blocks,

    and leaves the engine in a healthy state for subsequent valid requests.
    """
    mock_model = _create_mock_hf_model()
    # Configure 4 total blocks = 64 tokens capacity (at 16 tokens/block)
    mock_tok = MockTokenizer()

    engine = MiniVLLMEngine(
        model=mock_model,
        tokenizer=mock_tok,
        total_blocks=4,
        block_size=16,
        device="cpu",
        enable_prefix_caching=False,
    )
    initial_free = engine.num_free_blocks
    assert initial_free == 4

    # Request requiring 5 blocks: 70 prompt tokens + 10 max_tokens = 80 tokens -> 5 blocks > 4 total
    long_prompt = "hello " * 70
    engine.add_request(prompt=long_prompt, max_tokens=10)

    with pytest.raises(MemoryError, match="requires 5 KV blocks, but the pool has only 4"):
        engine.step()

    # Verify that blocks were released and queue is cleaned up
    assert engine.num_free_blocks == initial_free
    assert len(engine.waiting_queue) == 0
    assert len(engine.running_batch) == 0

    # Submit a normal request that fits (1 token + 2 max_tokens = 3 tokens -> 1 block)
    engine.add_request(prompt="hello", max_tokens=2)
    assert len(engine.waiting_queue) == 1

    # Should schedule and run without issue to completion (generating EOS)
    engine.step()
    assert len(engine.completed_requests) == 1
    assert engine.completed_requests[0].num_generated == 1
    assert engine.num_free_blocks == initial_free
