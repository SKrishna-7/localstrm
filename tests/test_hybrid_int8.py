"""
Phase 8.13: Comprehensive Focused Test Suite for Opt-In Hybrid INT8 Quantization
================================================================================
Validates:
  1. Default behavior preservation: 'int8' remains 100% dynamic dequantization (QuantizedLinear);
     'none' remains unquantized float.
  2. Opt-in 'hybrid_int8' routing: attention projections -> CachedQuantizedLinear (FP16 cached),
     eligible MLP projections -> Int8GEMMLinear (cuBLASLt INT8 GEMM).
  3. Shape alignment & safe fallback: non-multiple-of-8 MLP projections safely fall back
     to QuantizedLinear with observable logging.
  4. Numerical fidelity: Cosine similarity >= 0.99 against FP16 reference across 2D/3D shapes,
     including batch sizes and sequence lengths where M % 32 != 0 (validating row padding).
  5. Bias handling: Correct epilogue addition in float32 rescaled space.
  6. Kill-switch: MINIVLLM_DISABLE_INT8_GEMM=1 forces dynamic dequantization fallback.
  7. CPU / Non-CUDA safety: Int8GEMMLinear automatically falls back to dynamic dequantization
     when executed on CPU without crashing.
  8. CLI and EngineConfig propagation: Verifies CLI subcommands parse --quant hybrid_int8
     and propagate into EngineConfig.
  9. Preflight memory budget: validate_model_preflight and estimate_vram_budget compute
     effective 0.55 weight multiplier for hybrid_int8.
 10. CPU-staged model loader: load_and_quantize_model stages synthetic model on CPU and
     successfully quantizes into hybrid_int8 structures.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kernels.quantization import (
    QuantizedLinear,
    CachedQuantizedLinear,
    Int8GEMMLinear,
    can_use_int8_gemm,
    apply_optional_quantization,
    load_and_quantize_model,
)
from src.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from src.memory.preflight import validate_model_preflight, estimate_vram_budget, ModelMetadata
from cli import build_parser


# ==============================================================================
# Deterministic Synthetic Transformer Models
# ==============================================================================

class SyntheticAttention(nn.Module):
    def __init__(self, hidden_size: int = 64, with_bias: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=with_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=with_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=with_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5), dim=-1)
        return self.o_proj(torch.matmul(scores, v))


class SyntheticMLP(nn.Module):
    def __init__(self, hidden_size: int = 64, intermediate_size: int = 128, with_bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=with_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=with_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=with_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SyntheticDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 64, intermediate_size: int = 128, attn_bias: bool = True, mlp_bias: bool = False):
        super().__init__()
        self.self_attn = SyntheticAttention(hidden_size, with_bias=attn_bias)
        self.mlp = SyntheticMLP(hidden_size, intermediate_size, with_bias=mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.self_attn(x) + self.mlp(x)


class SyntheticTransformer(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, attn_bias: bool = True, mlp_bias: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            SyntheticDecoderLayer(hidden_size, hidden_size * 2, attn_bias=attn_bias, mlp_bias=mlp_bias)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SyntheticCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, vocab_size: int = 256, attn_bias: bool = True, mlp_bias: bool = False):
        super().__init__()
        self.model = SyntheticTransformer(hidden_size, num_layers, attn_bias=attn_bias, mlp_bias=mlp_bias)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(x))


def create_synthetic_model(seed: int = 42, hidden_size: int = 64, num_layers: int = 2) -> SyntheticCausalLM:
    torch.manual_seed(seed)
    model = SyntheticCausalLM(hidden_size=hidden_size, num_layers=num_layers)
    model.eval()
    return model


# ==============================================================================
# Unit & Integration Tests
# ==============================================================================

class TestHybridInt8(unittest.TestCase):
    """Test suite for Phase 8.13 opt-in hybrid INT8 quantization."""

    def setUp(self):
        torch.manual_seed(42)
        # Clean environment before each test
        os.environ.pop("MINIVLLM_DISABLE_INT8_GEMM", None)

    def tearDown(self):
        os.environ.pop("MINIVLLM_DISABLE_INT8_GEMM", None)

    # --------------------------------------------------------------------------
    # 1. Default Behavior Preservation
    # --------------------------------------------------------------------------
    def test_default_int8_mode_preserved(self):
        """Verify that default 'int8' mode still routes 100% of layers to QuantizedLinear."""
        model = create_synthetic_model()
        orig_bytes, quant_bytes = apply_optional_quantization(model, quant_type="int8")

        for layer in model.model.layers:
            # Attention projections must all be QuantizedLinear
            self.assertIsInstance(layer.self_attn.q_proj, QuantizedLinear)
            self.assertIsInstance(layer.self_attn.k_proj, QuantizedLinear)
            self.assertIsInstance(layer.self_attn.v_proj, QuantizedLinear)
            self.assertIsInstance(layer.self_attn.o_proj, QuantizedLinear)
            # MLP projections must all be QuantizedLinear
            self.assertIsInstance(layer.mlp.gate_proj, QuantizedLinear)
            self.assertIsInstance(layer.mlp.up_proj, QuantizedLinear)
            self.assertIsInstance(layer.mlp.down_proj, QuantizedLinear)

            self.assertEqual(layer.self_attn.q_proj.quant_type, "int8")
            self.assertEqual(layer.mlp.gate_proj.quant_type, "int8")

    def test_default_none_mode_preserved(self):
        """Verify that 'none' mode leaves nn.Linear untouched."""
        model = create_synthetic_model()
        orig_bytes, quant_bytes = apply_optional_quantization(model, quant_type="none")
        self.assertEqual(orig_bytes, quant_bytes)
        for layer in model.model.layers:
            self.assertIsInstance(layer.self_attn.q_proj, nn.Linear)
            self.assertIsInstance(layer.mlp.gate_proj, nn.Linear)

    # --------------------------------------------------------------------------
    # 2. Opt-In Hybrid INT8 Routing
    # --------------------------------------------------------------------------
    def test_hybrid_int8_routing(self):
        """Verify hybrid_int8 routes attention to CachedQuantizedLinear and MLP to Int8GEMMLinear."""
        model = create_synthetic_model()
        orig_bytes, quant_bytes = apply_optional_quantization(model, quant_type="hybrid_int8")

        for i, layer in enumerate(model.model.layers):
            # Attention projections -> CachedQuantizedLinear (FP16 cached)
            self.assertIsInstance(layer.self_attn.q_proj, CachedQuantizedLinear, f"L{i} q_proj")
            self.assertIsInstance(layer.self_attn.k_proj, CachedQuantizedLinear, f"L{i} k_proj")
            self.assertIsInstance(layer.self_attn.v_proj, CachedQuantizedLinear, f"L{i} v_proj")
            self.assertIsInstance(layer.self_attn.o_proj, CachedQuantizedLinear, f"L{i} o_proj")

            # MLP projections -> Int8GEMMLinear (INT8 weights)
            self.assertIsInstance(layer.mlp.gate_proj, Int8GEMMLinear, f"L{i} gate_proj")
            self.assertIsInstance(layer.mlp.up_proj, Int8GEMMLinear, f"L{i} up_proj")
            self.assertIsInstance(layer.mlp.down_proj, Int8GEMMLinear, f"L{i} down_proj")

        # Memory verification: hybrid_int8 must be smaller than original uncompressed
        self.assertLess(quant_bytes, orig_bytes)
        # Ratio should be approximately ~50-60% of original
        ratio = quant_bytes / orig_bytes
        self.assertGreater(ratio, 0.40)
        self.assertLess(ratio, 0.70)

    # --------------------------------------------------------------------------
    # 3. Shape Alignment & Fallback for Non-Multiple-of-8 MLP
    # --------------------------------------------------------------------------
    def test_hybrid_int8_unaligned_mlp_fallback(self):
        """Verify MLP projections not divisible by 8 safely fall back to QuantizedLinear."""
        # Create a layer with hidden_size=63 (not divisible by 8)
        class OddMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(63, 127)  # neither 63 nor 127 is divisible by 8
                self.up_proj = nn.Linear(63, 127)
                self.down_proj = nn.Linear(127, 63)

        class OddLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = SyntheticAttention(hidden_size=64)  # aligned
                self.mlp = OddMLP()

        class OddModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([OddLayer()])

        odd_model = OddModel()
        apply_optional_quantization(odd_model, quant_type="hybrid_int8")

        # Attention is aligned -> CachedQuantizedLinear
        self.assertIsInstance(odd_model.layers[0].self_attn.q_proj, CachedQuantizedLinear)
        # MLP is unaligned -> falls back to QuantizedLinear
        self.assertIsInstance(odd_model.layers[0].mlp.gate_proj, QuantizedLinear)
        self.assertEqual(odd_model.layers[0].mlp.gate_proj.quant_type, "int8")

    # --------------------------------------------------------------------------
    # 4. Numerical Accuracy and Rescaling
    # --------------------------------------------------------------------------
    def test_cached_quantized_linear_numerical_accuracy(self):
        """Verify CachedQuantizedLinear closely matches standard float Linear."""
        lin = nn.Linear(64, 128, bias=True)
        cached_layer = CachedQuantizedLinear.from_float(lin, dtype=torch.float32)

        x = torch.randn(4, 16, 64)
        ref_out = lin(x)
        cached_out = cached_layer(x)

        # Per-channel INT8 quantization has cosine similarity > 0.99 with float32
        cos_sim = F.cosine_similarity(ref_out.flatten(), cached_out.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.99)

    def test_int8_gemm_fallback_on_cpu(self):
        """Verify Int8GEMMLinear safely falls back to dynamic dequantization on CPU."""
        lin = nn.Linear(64, 128, bias=True)
        int8_layer = Int8GEMMLinear.from_float(lin, dtype=torch.float32)

        x = torch.randn(2, 8, 64)
        out = int8_layer(x)  # CPU input triggers dynamic dequantization fallback
        ref_out = lin(x)

        self.assertEqual(out.shape, (2, 8, 128))
        cos_sim = F.cosine_similarity(ref_out.flatten(), out.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.99)

    @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "_int_mm"), "Requires CUDA and torch._int_mm")
    def test_int8_gemm_cuda_execution_and_padding(self):
        """Verify native CUDA INT8 GEMM with row-padding on shapes where M % 32 != 0."""
        device = torch.device("cuda")
        # Ensure compute capability >= 8.0
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("Requires CUDA SM 8.0+")

        in_f, out_f = 64, 128
        lin = nn.Linear(in_f, out_f, bias=True, device=device, dtype=torch.float16)
        int8_layer = Int8GEMMLinear.from_float(lin, dtype=torch.float16).to(device)

        # Test various batch and sequence lengths:
        # 1 token (M=1 % 32 = 1) -> padded to 32
        # 4 tokens (M=4 % 32 = 4) -> padded to 32
        # 37 tokens (M=37 % 32 = 5) -> padded to 64
        # 64 tokens (M=64 % 32 = 0) -> no padding
        for shape in [(1, 1, in_f), (4, 1, in_f), (1, 37, in_f), (2, 32, in_f)]:
            x = torch.randn(shape, device=device, dtype=torch.float16)
            ref_out = lin(x)
            int8_out = int8_layer(x)

            self.assertEqual(int8_out.shape, (*shape[:-1], out_f))
            self.assertEqual(int8_out.dtype, torch.float16)

            cos_sim = F.cosine_similarity(ref_out.float().flatten(), int8_out.float().flatten(), dim=0).item()
            self.assertGreater(cos_sim, 0.99, f"Shape {shape} failed cosine similarity: {cos_sim}")

    # --------------------------------------------------------------------------
    # 5. Kill-Switch Validation
    # --------------------------------------------------------------------------
    def test_kill_switch_forces_fallback(self):
        """Verify MINIVLLM_DISABLE_INT8_GEMM=1 triggers fallback path."""
        lin = nn.Linear(64, 128, bias=True)
        int8_layer = Int8GEMMLinear.from_float(lin, dtype=torch.float32)

        os.environ["MINIVLLM_DISABLE_INT8_GEMM"] = "1"
        self.assertFalse(can_use_int8_gemm()[0])

        x = torch.randn(1, 1, 64)
        out = int8_layer(x)
        self.assertEqual(out.shape, (1, 1, 128))
        self.assertTrue(int8_layer._warned_fallback)

    # --------------------------------------------------------------------------
    # 6. CLI Argument Parsing and Config Propagation
    # --------------------------------------------------------------------------
    def test_cli_accepts_hybrid_int8(self):
        """Verify cli.py argument parser accepts --quant hybrid_int8 across all relevant subcommands."""
        parser = build_parser()

        # chat
        args_chat = parser.parse_args(["chat", "--quant", "hybrid_int8"])
        self.assertEqual(args_chat.quant, "hybrid_int8")

        # engine
        args_engine = parser.parse_args(["engine", "--quant", "hybrid_int8"])
        self.assertEqual(args_engine.quant, "hybrid_int8")

        # quantize
        args_quant = parser.parse_args(["quantize", "--quant", "hybrid_int8"])
        self.assertEqual(args_quant.quant, "hybrid_int8")

        # check
        args_check = parser.parse_args(["check", "--quant", "hybrid_int8"])
        self.assertEqual(args_check.quant, "hybrid_int8")

        # serve
        args_serve = parser.parse_args(["serve", "--quant", "hybrid_int8"])
        self.assertEqual(args_serve.quant, "hybrid_int8")

    def test_engine_config_propagation(self):
        """Verify EngineConfig.from_args cleanly propagates hybrid_int8."""
        parser = build_parser()
        args = parser.parse_args(["engine", "--quant", "hybrid_int8", "--batch-size", "2"])
        config = EngineConfig.from_args(args)
        self.assertEqual(config.quantization, "hybrid_int8")

    # --------------------------------------------------------------------------
    # 7. Memory Preflight Calculations
    # --------------------------------------------------------------------------
    def test_preflight_hybrid_multiplier(self):
        """Verify preflight validates hybrid_int8 with 0.55 multiplier."""
        meta = ModelMetadata(
            model_id="test-model",
            is_local=True,
            weight_bytes=1000 * (1024**2),  # 1000 MiB
            num_layers=12,
            num_kv_heads=2,
            head_dim=64,
        )

        with patch("src.memory.preflight.get_model_metadata", return_value=meta):
            # int8 -> 0.50 multiplier -> ~500 MiB weights
            report_int8 = validate_model_preflight("test-model", device="cpu", quantization="int8")
            # hybrid_int8 -> 0.55 multiplier -> ~550 MiB weights
            report_hybrid = validate_model_preflight("test-model", device="cpu", quantization="hybrid_int8")

            # Check that hybrid_int8 needed memory is slightly higher than int8 but much less than none
            report_none = validate_model_preflight("test-model", device="cpu", quantization="none")
            self.assertGreater(report_hybrid.runtime_memory_needed, report_int8.runtime_memory_needed)
            self.assertLess(report_hybrid.runtime_memory_needed, report_none.runtime_memory_needed)

    def test_estimate_vram_budget_hybrid_int8(self):
        """Verify estimate_vram_budget handles hybrid_int8 properly."""
        meta = ModelMetadata(
            model_id="test-3b",
            is_local=True,
            weight_bytes=5600 * (1024**2),  # ~5.6 GB FP16 uncompressed
            num_layers=36,
            num_heads=16,
            num_kv_heads=2,
            head_dim=128,
        )

        # Test budget estimate for hybrid_int8 under 5.0 GiB budget
        estimate = estimate_vram_budget(
            model_id="test-3b",
            vram_budget="5.0GiB",
            device="cuda",
            quantization="hybrid_int8",
            metadata=meta,
            raise_on_error=False,
        )
        self.assertIsNotNone(estimate)
        # Resident weight should be ~55% of 5600 MiB = ~3080 MiB
        expected_resident = int(5600 * (1024**2) * 0.55)
        self.assertEqual(estimate.model_resident_bytes, expected_resident)

    # --------------------------------------------------------------------------
    # 8. CPU-Staged Loading
    # --------------------------------------------------------------------------
    def test_load_and_quantize_synthetic_hybrid(self):
        """Verify load_and_quantize_model successfully loads a synthetic model in hybrid_int8 mode."""
        model = create_synthetic_model()
        device = torch.device("cpu")

        quant_model, orig_b, quant_b = load_and_quantize_model(
            model,
            device=device,
            quantization="hybrid_int8"
        )
        self.assertIsInstance(quant_model.model.layers[0].self_attn.q_proj, CachedQuantizedLinear)
        self.assertIsInstance(quant_model.model.layers[0].mlp.gate_proj, Int8GEMMLinear)
        self.assertLess(quant_b, orig_b)


if __name__ == "__main__":
    unittest.main()
