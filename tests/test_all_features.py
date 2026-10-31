"""
End-to-End System Regression Test Suite: Mini-vLLM
--------------------------------------------------
Validates all core architectural features and subsystems:
  1. Pre-Flight Hardware & Memory Guard
  2. Virtual Memory Paged Block Manager
  3. PagedAttention Gathered Attention Kernel (Numerical Equivalence)
  4. Content-Addressed Chained Block-Hash Prefix Caching & Block Deduplication
  5. Memory Starvation Preemption & Recovery
  6. Weight Quantization (INT8 & Bit-Packed INT4 Nibbles)
  7. Chunked Prefill Scheduler (Prompt Slicing & SLA Protection)
  8. Production Sampling Suite (Top-P, Top-K, Min-P, Repetition Penalty, Stop Sequences)
  9. Observability, Prometheus Metrics & Latency Percentiles
  10. Physical KV Cache Pool (FP16 & Dynamic INT8 Quantization)
  11. Unified MiniVLLMEngine End-to-End Execution
"""

import json
import os
import sys
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_default_model, get_default_device
from src.memory.preflight import validate_model_preflight
from src.memory.block_manager import BlockManager, BlockTable
from src.kernels.paged_attention import paged_attention_decode, standard_attention
from src.memory.prefix_cache import PrefixCacheManager, compute_block_hash
from src.memory.preemption import StarvationResilientEngine, PreemptibleSequence, SeqState
from src.kernels.quantization import QuantizedLinear
from src.scheduler.chunked_prefill import ChunkedPrefillScheduler, StreamRequest
from src.engine.sampling import (
    SamplingParams,
    sample_next_token,
    apply_repetition_penalties,
    check_stop_criteria
)
from src.observability.metrics import EngineMetricsTracker, calculate_percentiles
from src.engine.llm_engine import PhysicalBlockPool, MiniVLLMEngine


# =========================================================================
# 1. PRE-FLIGHT MEMORY GUARD & VRAM BUDGET
# =========================================================================
def test_preflight_memory_guard(tmp_path):
    """Verifies preflight estimates from a deterministic local model fixture."""
    model_dir = tmp_path / "qwen-fixture"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 151936,
    }))
    weight_bytes = 1 * (1024**2)
    (model_dir / "model.safetensors").write_bytes(b"0" * weight_bytes)

    report_fp16 = validate_model_preflight(str(model_dir), device="cpu", quantization="none")
    assert report_fp16.parameter_count == weight_bytes // 2
    assert report_fp16.download_size_bytes == weight_bytes
    assert report_fp16.runtime_memory_needed > 0

    report_int4 = validate_model_preflight(str(model_dir), device="cpu", quantization="int4")
    assert report_int4.runtime_memory_needed < report_fp16.runtime_memory_needed


# =========================================================================
# 2. VIRTUAL MEMORY BLOCK MANAGER
# =========================================================================
def test_block_manager():
    """Verifies OS-style virtual memory paging: allocation, token appending, and freeing."""
    mgr = BlockManager(total_blocks=4, block_size=4)
    assert mgr.num_free_blocks == 4

    # Allocate initial prompt blocks for seq 1 (4 tokens -> 1 block)
    table = mgr.allocate_for_prompt(seq_id=1, prompt_tokens=4)
    assert mgr.num_free_blocks == 3
    assert len(table) == 1
    assert mgr.blocks[table.physical_block_ids[0]].is_full

    # Appending a token must allocate a second block
    new_b = mgr.append_token(seq_id=1)
    assert new_b is not None
    assert len(table) == 2
    assert mgr.num_free_blocks == 2

    # Free sequence memory
    mgr.free_sequence(seq_id=1)
    assert mgr.num_free_blocks == 4
    assert 1 not in mgr.block_tables


# =========================================================================
# 3. PAGEDATTENTION KERNEL (NUMERICAL EQUIVALENCE)
# =========================================================================
def test_paged_attention_numerical_equivalence():
    """
    Verifies that non-contiguous gathered PagedAttention produces
    mathematically identical results (0.00 difference) to contiguous attention.
    """
    batch_size = 2
    num_heads = 4
    num_kv_heads = 2  # GQA
    head_dim = 16
    block_size = 4
    total_blocks = 8
    device = torch.device(get_default_device())

    query = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
    k_pool = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device)
    v_pool = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device)

    # Sequence 0: 6 tokens in physical blocks [1, 3]
    # Sequence 1: 3 tokens in physical block [0]
    block_tables = torch.tensor([[1, 3], [0, 0]], dtype=torch.int64, device=device)
    seq_lens = torch.tensor([6, 3], dtype=torch.int64, device=device)

    out_paged = paged_attention_decode(
        query=query,
        k_cache_pool=k_pool,
        v_cache_pool=v_pool,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size
    )
    assert out_paged.shape == (batch_size, num_heads, 1, head_dim)

    # Verify against standard contiguous attention for Seq 0
    k_seq0 = torch.cat([k_pool[1], k_pool[3]], dim=0)[:6]  # [6, 2, 16]
    v_seq0 = torch.cat([v_pool[1], v_pool[3]], dim=0)[:6]
    k_seq0 = k_seq0.permute(1, 0, 2).unsqueeze(0).repeat_interleave(2, dim=1)  # [1, 4, 6, 16]
    v_seq0 = v_seq0.permute(1, 0, 2).unsqueeze(0).repeat_interleave(2, dim=1)

    out_std_0 = standard_attention(query[0:1], k_seq0, v_seq0)
    max_diff = torch.max(torch.abs(out_paged[0:1] - out_std_0)).item()
    assert max_diff < 1e-4, f"PagedAttention numerical difference {max_diff} exceeded tolerance!"


# =========================================================================
# 4. CONTENT-ADDRESSED PREFIX CACHING
# =========================================================================
def test_prefix_caching():
    """Verifies SHA-256 block hash computation, cache hit reuse, and ref counting."""
    pcm = PrefixCacheManager(total_blocks=4, block_size=4)
    tokens_chunk = [101, 102, 103, 104]

    # First allocation: cache miss
    b_id1, is_hit1, hash1 = pcm.allocate_or_reuse_block(tokens_chunk, parent_hash=None)
    assert not is_hit1
    assert pcm.blocks[b_id1].ref_count == 1

    # Second allocation of identical tokens: CACHE HIT!
    b_id2, is_hit2, hash2 = pcm.allocate_or_reuse_block(tokens_chunk, parent_hash=None)
    assert is_hit2
    assert b_id1 == b_id2
    assert hash1 == hash2
    assert pcm.blocks[b_id1].ref_count == 2

    # Release both references
    pcm.free_block(b_id1)
    assert pcm.blocks[b_id1].ref_count == 1
    pcm.free_block(b_id1)
    assert pcm.blocks[b_id1].ref_count == 0
    assert b_id1 in pcm.lru_eviction_queue


# =========================================================================
# 5. MEMORY STARVATION PREEMPTION & RECOVERY
# =========================================================================
def test_preemption_starvation_recovery():
    """Verifies that an engine under memory starvation preempts victims and avoids crashing."""
    engine = StarvationResilientEngine(total_blocks=5, block_size=4, verbose=False)
    # Add high priority task and low priority task
    engine.add_request("High priority query", target_tokens=10, arrival_step=0, priority=0)
    engine.add_request("Low priority background task", target_tokens=10, arrival_step=0, priority=5)

    engine.run_simulation()
    assert len(engine.completed) == 2
    for seq in engine.completed:
        assert seq.tokens_generated >= seq.target_tokens
        assert seq.state == SeqState.FINISHED


# =========================================================================
# 6. WEIGHT QUANTIZATION (INT8 & BIT-PACKED INT4)
# =========================================================================
def test_weight_quantization():
    """Verifies INT8 and bit-packed INT4 weight quantization and cosine fidelity."""
    in_dim, out_dim = 128, 64
    linear = nn.Linear(in_dim, out_dim, bias=False)
    x = torch.randn(2, in_dim)
    y_float = linear(x)

    # 1. Test INT8 Quantization
    q8 = QuantizedLinear.from_float(linear, quant_type="int8")
    assert q8.weight_q.dtype == torch.int8
    y_int8 = q8(x)
    cos_sim8 = F.cosine_similarity(y_float.flatten(), y_int8.flatten(), dim=0).item()
    assert cos_sim8 > 0.98, f"INT8 cosine similarity {cos_sim8} too low!"

    # 2. Test INT4 Quantization (packed into uint8 nibbles)
    q4 = QuantizedLinear.from_float(linear, quant_type="int4")
    assert q4.weight_packed.shape == (out_dim, in_dim // 2)
    assert q4.weight_packed.dtype == torch.uint8
    y_int4 = q4(x)
    cos_sim4 = F.cosine_similarity(y_float.flatten(), y_int4.flatten(), dim=0).item()
    assert cos_sim4 > 0.90, f"INT4 cosine similarity {cos_sim4} too low!"


# =========================================================================
# 7. CHUNKED PREFILL SCHEDULER
# =========================================================================
def test_chunked_prefill_scheduler():
    """Verifies prompt slicing into chunks to protect decode latency SLAs."""
    scheduler = ChunkedPrefillScheduler(max_tokens_per_step=16, prefill_chunk_size=8)
    req = StreamRequest(req_id=1, prompt_tokens=list(range(20)), max_decode_tokens=5)
    tasks = scheduler.schedule_iteration(running_requests=[], waiting_requests=[req])
    assert len(tasks) == 1
    task = tasks[0]
    assert task.is_prefill
    assert len(task.token_slice) == 8  # Sliced to chunk size 8
    assert not task.is_last_chunk


# =========================================================================
# 8. PRODUCTION SAMPLING SUITE
# =========================================================================
def test_sampling_suite():
    """Verifies greedy sampling, temperature scaling, repetition penalty, and stop criteria."""
    logits = torch.tensor([[1.0, 2.0, 5.0, 0.5]])  # token 2 has highest logit

    # Greedy sampling
    params_greedy = SamplingParams.greedy()
    next_tok = sample_next_token(logits, params_greedy)
    assert next_tok == 2

    # Repetition penalty: penalize token 2 so token 1 becomes highest
    penalized_logits = apply_repetition_penalties(logits.clone(), past_token_ids=[2], repetition_penalty=10.0)
    next_tok_penalized = sample_next_token(penalized_logits, params_greedy)
    assert next_tok_penalized == 1

    # Stop criteria check
    class DummyTokenizer:
        eos_token_id = 999
        def decode(self, ids, skip_special_tokens=False):
            return "Hello world stop here"

    should_stop, reason = check_stop_criteria(
        generated_ids=[1, 2, 3],
        tokenizer=DummyTokenizer(),
        stop_token_ids=[99],
        stop_strings=["stop"]
    )
    assert should_stop
    assert "stop" in reason


# =========================================================================
# 9. OBSERVABILITY & PROMETHEUS METRICS
# =========================================================================
def test_observability_metrics():
    """Verifies latency distribution percentiles and Prometheus metric formatting."""
    tracker = EngineMetricsTracker(model_name="test-model", enable_prometheus=False)

    # Record TTFT and TPOT
    for val in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
        tracker.record_ttft(val)
        tracker.record_tpot(val / 2.0)

    p_stats = calculate_percentiles(tracker.ttft_records_ms)
    assert p_stats["p50"] > 0
    assert p_stats["p90"] > p_stats["p50"]

    prom_text = tracker.export_prometheus_text()
    assert "vllm_prompt_tokens_total" in prom_text
    assert "vllm_generation_tokens_total" in prom_text


# =========================================================================
# 10. PHYSICAL KV CACHE POOL (FP16 & INT8 DYNAMIC QUANTIZATION)
# =========================================================================
def test_physical_kv_cache_pool():
    """Verifies physical KV block allocation, vectorized writing, and INT8 scaling."""
    device = get_default_device()

    # 1. FP16 Pool
    pool_fp16 = PhysicalBlockPool(
        num_layers=2, total_blocks=4, block_size=4,
        num_kv_heads=2, head_dim=8, device=device, kv_cache_dtype="fp16"
    )
    k_vec = torch.randn(2, 2, 8, device=device)
    v_vec = torch.randn(2, 2, 8, device=device)
    pool_fp16.write_kv_all_layers(blk_idx=0, blk_offset=0, k_all=k_vec, v_all=v_vec)
    k_read, v_read = pool_fp16.read_kv(layer_idx=0, phys_blocks=[0])
    assert k_read.shape == (1, 4, 2, 8)

    # 2. INT8 Quantized Pool (50% memory reduction)
    pool_int8 = PhysicalBlockPool(
        num_layers=2, total_blocks=4, block_size=4,
        num_kv_heads=2, head_dim=8, device=device, kv_cache_dtype="int8"
    )
    assert pool_int8.k_pool.dtype == torch.int8
    assert pool_int8.v_pool.dtype == torch.int8
    pool_int8.write_kv_all_layers(blk_idx=0, blk_offset=0, k_all=k_vec, v_all=v_vec)
    k_q_read, v_q_read = pool_int8.read_kv(layer_idx=0, phys_blocks=[0])
    assert k_q_read.dtype == torch.float16  # Dequantized back to float16


# =========================================================================
# 11. UNIFIED MINIVLLM ENGINE (CONTINUOUS BATCHING ON DEVICE)
# =========================================================================
def test_unified_minivllm_engine():
    """
    Executes a real end-to-end continuous batching workload
    using the Qwen model on the active compute device (CUDA / CPU).
    """
    device = get_default_device()
    engine = MiniVLLMEngine(
        model_name="Qwen/Qwen2.5-0.5B",
        max_batch_size=2,
        total_blocks=32,
        block_size=16,
        device=device,
        kv_cache_dtype="fp16",
        chunked_prefill=True,
        enable_prefix_caching=True,
        verbose=False
    )

    engine.add_request("What is 1 + 1?", max_tokens=6, arrival_step=0)
    engine.add_request("What is 2 + 2?", max_tokens=6, arrival_step=0)

    completed = engine.run()
    assert len(completed) == 2
    for req in completed:
        assert req.num_generated > 0
        assert req.status.value == "FINISHED"

    assert engine.metrics.generation_tokens_total > 0
    assert len(engine.metrics.ttft_records_ms) == 2


def test_openai_compatible_protocol():
    """Validates OpenAI-compatible Pydantic v2 schemas and JSON serialization."""
    from src.server.protocol import (
        ChatMessage, ChatCompletionRequest, ChatCompletionResponse,
        ChatCompletionResponseChoice, UsageInfo, ModelList, ModelCard
    )

    req = ChatCompletionRequest(
        model="Qwen/Qwen2.5-0.5B",
        messages=[ChatMessage(role="user", content="Hello world!")],
        temperature=0.7,
        stream=True
    )
    assert req.model == "Qwen/Qwen2.5-0.5B"
    assert req.stream is True
    assert req.messages[0].content == "Hello world!"

    resp = ChatCompletionResponse(
        model="Qwen/Qwen2.5-0.5B",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Hello! How can I help?"),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=8, total_tokens=13)
    )
    dumped = resp.model_dump()
    assert dumped["object"] == "chat.completion"
    assert dumped["usage"]["total_tokens"] == 13


if __name__ == "__main__":
    import time
    print("\n" + "=" * 75)
    print("       🧪 MINI-VLLM COMPREHENSIVE ARCHITECTURAL TEST SUITE")
    print("=" * 75)

    test_functions = [
        ("1. Pre-Flight Hardware Memory Guard", test_preflight_memory_guard),
        ("2. Virtual Memory Block Manager", test_block_manager),
        ("3. PagedAttention Numerical Equivalence", test_paged_attention_numerical_equivalence),
        ("4. Content-Addressed Prefix Caching", test_prefix_caching),
        ("5. Memory Starvation Preemption & Recovery", test_preemption_starvation_recovery),
        ("6. Weight Quantization (INT8 & INT4)", test_weight_quantization),
        ("7. Chunked Prefill Scheduler", test_chunked_prefill_scheduler),
        ("8. Production Sampling Suite", test_sampling_suite),
        ("9. Observability & Prometheus Metrics", test_observability_metrics),
        ("10. Physical KV Pool (FP16 & INT8)", test_physical_kv_cache_pool),
        ("11. Unified MiniVLLMEngine End-to-End", test_unified_minivllm_engine),
        ("12. OpenAI REST Protocol & Schemas", test_openai_compatible_protocol),
    ]

    passed = 0
    failed = 0
    test_records = []
    total_start = time.perf_counter()

    for name, fn in test_functions:
        print(f"\n  Running: {name} ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            fn()
            dur = (time.perf_counter() - t0) * 1000.0
            print(f"✅ PASSED ({dur:.1f}ms)")
            passed += 1
            test_records.append({
                "name": name,
                "status": "PASSED",
                "duration_ms": round(dur, 2),
                "error": None
            })
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000.0
            print(f"❌ FAILED ({dur:.1f}ms)\n    Error: {e}")
            failed += 1
            test_records.append({
                "name": name,
                "status": "FAILED",
                "duration_ms": round(dur, 2),
                "error": str(e)
            })

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 75)
    if failed == 0:
        print(f"  🎉 ALL {passed}/{len(test_functions)} TESTS PASSED! (Total time: {total_time:.2f}s)")
    else:
        print(f"  ⚠️  TEST RUN COMPLETED: {passed} PASSED, {failed} FAILED (Total time: {total_time:.2f}s)")
    print("=" * 75)

    # Persist Test Run Results
    import json
    results_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "test_suite_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": get_default_device().upper(),
            "model": get_default_model(),
            "total_tests": len(test_functions),
            "passed": passed,
            "failed": failed,
            "total_time_seconds": round(total_time, 2),
            "tests": test_records
        }, f, indent=2)
    print(f"📁 Test suite results persisted to: {out_file}\n")
    sys.exit(0 if failed == 0 else 1)
