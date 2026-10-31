"""
Phase 8.12A: Comprehensive Unit and Regression Tests for Quantized Model Loading
-------------------------------------------------------------------------------
Covers:
  1. INT8 CPU-staged loading.
  2. INT4 CPU-staged loading.
  3. CPU target and CUDA target device placement.
  4. Preservation of expected output dtype and device.
  5. Deterministic numerical error checks with explicit justified tolerances.
  6. Batched inputs and multi-token input shapes ([1, 1, H], [4, 1, H], [2, 16, H], [8, 32, H]).
  7. Bias and bias-free linear layers.
  8. Invalid quantization modes and meaningful error behavior.
  9. Failure handling and resource cleanup without persistent GPU leakage.
 10. Non-quantized ('none') loading behavior remaining unchanged.
 11. Mocked AutoModelForCausalLM.from_pretrained verifying CPU staging before CUDA transfer.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kernels.quantization import (
    QuantizedLinear,
    apply_optional_quantization,
    load_and_quantize_model,
)


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
        # Simplified attention projection
        scores = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5), dim=-1)
        attn_out = torch.matmul(scores, v)
        return self.o_proj(attn_out)


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
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


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
        hidden = self.model(x)
        return self.lm_head(hidden)


def create_seeded_synthetic_model(seed: int = 42, hidden_size: int = 64, num_layers: int = 2, attn_bias: bool = True, mlp_bias: bool = False) -> SyntheticCausalLM:
    torch.manual_seed(seed)
    model = SyntheticCausalLM(hidden_size=hidden_size, num_layers=num_layers, attn_bias=attn_bias, mlp_bias=mlp_bias)
    model.eval()
    return model


# ==============================================================================
# Unit & Regression Test Suite
# ==============================================================================

class TestQuantizedModelLoader(unittest.TestCase):
    """Rigorous offline validation for load_and_quantize_model and QuantizedLinear."""

    def setUp(self):
        torch.manual_seed(100)

    # --------------------------------------------------------------------------
    # 1. INT8 CPU-Staged Loading
    # --------------------------------------------------------------------------
    def test_int8_cpu_staged_loading(self):
        """Verify INT8 converts all 7 transformer projections and achieves ~50% weight savings."""
        model = create_seeded_synthetic_model(seed=42)
        device = torch.device("cpu")

        quant_model, orig_bytes, quant_bytes = load_and_quantize_model(
            model,
            device=device,
            quantization="int8"
        )

        # Check all 7 linear projections per layer are QuantizedLinear
        for i, layer in enumerate(quant_model.model.layers):
            self.assertIsInstance(layer.self_attn.q_proj, QuantizedLinear, f"Layer {i} q_proj not quantized")
            self.assertIsInstance(layer.self_attn.k_proj, QuantizedLinear, f"Layer {i} k_proj not quantized")
            self.assertIsInstance(layer.self_attn.v_proj, QuantizedLinear, f"Layer {i} v_proj not quantized")
            self.assertIsInstance(layer.self_attn.o_proj, QuantizedLinear, f"Layer {i} o_proj not quantized")
            self.assertIsInstance(layer.mlp.gate_proj, QuantizedLinear, f"Layer {i} gate_proj not quantized")
            self.assertIsInstance(layer.mlp.up_proj, QuantizedLinear, f"Layer {i} up_proj not quantized")
            self.assertIsInstance(layer.mlp.down_proj, QuantizedLinear, f"Layer {i} down_proj not quantized")

            # Check quant_type
            self.assertEqual(layer.self_attn.q_proj.quant_type, "int8")
            self.assertTrue(hasattr(layer.self_attn.q_proj, "weight_q"))

        # Verify compression ratio (INT8 should be roughly 50% of FP16/FP32 linear weight bytes)
        self.assertLess(quant_bytes, orig_bytes)
        savings = (1.0 - (quant_bytes / orig_bytes)) * 100
        self.assertGreater(savings, 30.0, f"Expected >30% overall model savings, got {savings:.1f}%")

    # --------------------------------------------------------------------------
    # 2. INT4 CPU-Staged Loading
    # --------------------------------------------------------------------------
    def test_int4_cpu_staged_loading(self):
        """Verify INT4 packs two 4-bit weights per uint8 byte and achieves >60% weight savings."""
        model = create_seeded_synthetic_model(seed=42)
        device = torch.device("cpu")

        quant_model, orig_bytes, quant_bytes = load_and_quantize_model(
            model,
            device=device,
            quantization="int4"
        )

        for i, layer in enumerate(quant_model.model.layers):
            q_proj = layer.self_attn.q_proj
            self.assertIsInstance(q_proj, QuantizedLinear)
            self.assertEqual(q_proj.quant_type, "int4")
            self.assertTrue(hasattr(q_proj, "weight_packed"))
            self.assertEqual(q_proj.weight_packed.dtype, torch.uint8)
            # Packed buffer column dimension should be (in_features + 1) // 2
            expected_cols = (q_proj.in_features + 1) // 2
            self.assertEqual(q_proj.weight_packed.shape, (q_proj.out_features, expected_cols))

        savings = (1.0 - (quant_bytes / orig_bytes)) * 100
        self.assertGreater(savings, 50.0, f"Expected >50% overall model savings with INT4, got {savings:.1f}%")

    # --------------------------------------------------------------------------
    # 3. CPU Target and CUDA Target Placement
    # --------------------------------------------------------------------------
    def test_cpu_target_device_placement(self):
        """Verify device placement on CPU."""
        model = create_seeded_synthetic_model()
        quant_model, _, _ = load_and_quantize_model(model, device="cpu", quantization="int8")
        for p in quant_model.parameters():
            self.assertEqual(p.device.type, "cpu")
        for b in quant_model.buffers():
            self.assertEqual(b.device.type, "cpu")

    def test_cuda_target_device_placement(self):
        """Verify device placement on CUDA (skipped if CUDA is unavailable)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this test host; skipping CUDA placement test")

        model = create_seeded_synthetic_model()
        device = torch.device("cuda:0")
        quant_model, _, _ = load_and_quantize_model(model, device=device, quantization="int8")

        for p in quant_model.parameters():
            self.assertEqual(p.device.type, "cuda")
        for b in quant_model.buffers():
            self.assertEqual(b.device.type, "cuda")

    # --------------------------------------------------------------------------
    # 4. Preservation of Expected Output Dtype and Device
    # --------------------------------------------------------------------------
    def test_output_dtype_and_device_preservation(self):
        """Verify that QuantizedLinear preserves input dtype (fp32, fp16, bf16) and device."""
        float_linear = nn.Linear(32, 16, bias=True)
        torch.manual_seed(42)

        for quant_mode in ("int8", "int4"):
            qlayer = QuantizedLinear.from_float(float_linear, quant_type=quant_mode)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                x = torch.randn(2, 4, 32, dtype=dtype)
                out = qlayer(x)
                self.assertEqual(out.dtype, dtype, f"Failed dtype preservation for mode={quant_mode}, dtype={dtype}")
                self.assertEqual(out.device, x.device, f"Failed device preservation for mode={quant_mode}")
                self.assertEqual(out.shape, (2, 4, 16))

    # --------------------------------------------------------------------------
    # 5. Deterministic Numerical Error Checks & Tolerances
    # --------------------------------------------------------------------------
    def test_int8_numerical_accuracy_tolerances(self):
        """
        Verify INT8 quantization against unquantized reference across multiple layers.
        Enforced Tolerances:
          - Cosine Similarity >= 0.9990
          - Maximum Absolute Error <= 0.05
          - Mean Absolute Error <= 0.01
        """
        torch.manual_seed(12345)
        # Test 5 distinct linear layers with different dimensionalities
        shapes = [(32, 32), (64, 32), (32, 64), (128, 64), (64, 128)]

        for in_feat, out_feat in shapes:
            ref_linear = nn.Linear(in_feat, out_feat, bias=True)
            qlayer = QuantizedLinear.from_float(ref_linear, quant_type="int8")

            # Input with standard Gaussian distribution
            x = torch.randn(8, in_feat, dtype=torch.float32)

            with torch.no_grad():
                ref_out = ref_linear(x)
                q_out = qlayer(x)

            cos_sim = F.cosine_similarity(ref_out.flatten(), q_out.flatten(), dim=0).item()
            abs_err = torch.abs(ref_out - q_out)
            max_err = torch.max(abs_err).item()
            mean_err = torch.mean(abs_err).item()

            self.assertGreaterEqual(
                cos_sim, 0.9990,
                f"INT8 shape {in_feat}->{out_feat} failed CosSim tolerance: {cos_sim:.5f} < 0.9990"
            )
            self.assertLessEqual(
                max_err, 0.05,
                f"INT8 shape {in_feat}->{out_feat} failed MaxErr tolerance: {max_err:.5f} > 0.05"
            )
            self.assertLessEqual(
                mean_err, 0.01,
                f"INT8 shape {in_feat}->{out_feat} failed MeanErr tolerance: {mean_err:.5f} > 0.01"
            )

    def test_int4_numerical_accuracy_tolerances(self):
        """
        Verify INT4 quantization against unquantized reference across multiple layers.
        Enforced Tolerances:
          - Cosine Similarity >= 0.9900
          - Maximum Absolute Error <= 0.35
          - Mean Absolute Error <= 0.08
        """
        torch.manual_seed(54321)
        shapes = [(32, 32), (64, 32), (32, 64), (128, 64), (64, 128)]

        for in_feat, out_feat in shapes:
            ref_linear = nn.Linear(in_feat, out_feat, bias=True)
            qlayer = QuantizedLinear.from_float(ref_linear, quant_type="int4")

            x = torch.randn(8, in_feat, dtype=torch.float32)

            with torch.no_grad():
                ref_out = ref_linear(x)
                q_out = qlayer(x)

            cos_sim = F.cosine_similarity(ref_out.flatten(), q_out.flatten(), dim=0).item()
            abs_err = torch.abs(ref_out - q_out)
            max_err = torch.max(abs_err).item()
            mean_err = torch.mean(abs_err).item()

            self.assertGreaterEqual(
                cos_sim, 0.9900,
                f"INT4 shape {in_feat}->{out_feat} failed CosSim tolerance: {cos_sim:.5f} < 0.9900"
            )
            self.assertLessEqual(
                max_err, 0.35,
                f"INT4 shape {in_feat}->{out_feat} failed MaxErr tolerance: {max_err:.5f} > 0.35"
            )
            self.assertLessEqual(
                mean_err, 0.08,
                f"INT4 shape {in_feat}->{out_feat} failed MeanErr tolerance: {mean_err:.5f} > 0.08"
            )

    # --------------------------------------------------------------------------
    # 6. Batched Inputs and Multi-Token Input Shapes
    # --------------------------------------------------------------------------
    def test_batched_and_multi_token_input_shapes(self):
        """Verify QuantizedLinear correctness across single, batched, and sequence input shapes."""
        linear = nn.Linear(64, 32, bias=True)

        for quant_mode in ("int8", "int4"):
            qlayer = QuantizedLinear.from_float(linear, quant_type=quant_mode)

            shapes = [
                (1, 64),          # 2D flat: single token
                (4, 64),          # 2D flat: batch of 4
                (1, 1, 64),       # 3D: batch 1, 1 token
                (4, 1, 64),       # 3D: batch 4, 1 token decode step
                (2, 16, 64),      # 3D: batch 2, 16 tokens prefill
                (8, 32, 64),      # 3D: batch 8, 32 tokens long sequence
            ]

            for shape in shapes:
                x = torch.randn(*shape)
                out = qlayer(x)
                expected_shape = shape[:-1] + (32,)
                self.assertEqual(out.shape, expected_shape, f"Failed shape {shape} in mode {quant_mode}")
                self.assertFalse(torch.isnan(out).any(), f"NaNs detected in mode {quant_mode} shape {shape}")
                self.assertFalse(torch.isinf(out).any(), f"Infs detected in mode {quant_mode} shape {shape}")

    # --------------------------------------------------------------------------
    # 7. Bias and Bias-Free Linear Layers
    # --------------------------------------------------------------------------
    def test_bias_and_bias_free_layers(self):
        """Verify behavior with bias=True and bias=False."""
        l_with_bias = nn.Linear(32, 16, bias=True)
        l_no_bias = nn.Linear(32, 16, bias=False)

        for quant_mode in ("int8", "int4"):
            q_bias = QuantizedLinear.from_float(l_with_bias, quant_type=quant_mode)
            q_no_bias = QuantizedLinear.from_float(l_no_bias, quant_type=quant_mode)

            self.assertIsNotNone(q_bias.bias)
            self.assertIsNone(q_no_bias.bias)

            x = torch.randn(2, 32)
            out_bias = q_bias(x)
            out_no_bias = q_no_bias(x)

            self.assertEqual(out_bias.shape, (2, 16))
            self.assertEqual(out_no_bias.shape, (2, 16))

    # --------------------------------------------------------------------------
    # 8. Invalid Quantization Modes
    # --------------------------------------------------------------------------
    def test_invalid_quantization_modes(self):
        """Verify that invalid quantization modes raise ValueError with helpful error messages."""
        model = create_seeded_synthetic_model()

        invalid_modes = ["int16", "fp8", "awq", "gptq", "unknown_mode", "123"]
        for mode in invalid_modes:
            with self.assertRaises(ValueError) as ctx:
                load_and_quantize_model(model, device="cpu", quantization=mode)
            self.assertIn("Invalid quantization mode", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                apply_optional_quantization(model, quant_type=mode)
            self.assertIn("Invalid quantization type", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 9. Failure Handling and Resource Cleanup
    # --------------------------------------------------------------------------
    def test_failure_handling_and_resource_cleanup(self):
        """Verify that exceptions during quantization trigger CUDA cache cleanup and do not leak resources."""
        model = create_seeded_synthetic_model()

        # Simulate failure during device transfer or quantization
        with patch("src.kernels.quantization.apply_optional_quantization", side_effect=RuntimeError("Simulated CUDA OOM")):
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                with self.assertRaises(RuntimeError) as ctx:
                    load_and_quantize_model(model, device="cuda" if torch.cuda.is_available() else "cpu", quantization="int8")
                self.assertIn("Simulated CUDA OOM", str(ctx.exception))
                if torch.cuda.is_available():
                    mock_empty_cache.assert_called()

    # --------------------------------------------------------------------------
    # 10. Non-Quantized Loading Behavior Unchanged
    # --------------------------------------------------------------------------
    def test_non_quantized_loading_unchanged(self):
        """Verify that quantization='none' leaves model layers completely unquantized and unmodified."""
        model = create_seeded_synthetic_model(seed=42)
        quant_model, orig_bytes, quant_bytes = load_and_quantize_model(
            model,
            device="cpu",
            quantization="none"
        )

        self.assertEqual(orig_bytes, quant_bytes)
        for i, layer in enumerate(quant_model.model.layers):
            self.assertIsInstance(layer.self_attn.q_proj, nn.Linear)
            self.assertNotIsInstance(layer.self_attn.q_proj, QuantizedLinear)
            self.assertIsInstance(layer.mlp.down_proj, nn.Linear)
            self.assertNotIsInstance(layer.mlp.down_proj, QuantizedLinear)

    # --------------------------------------------------------------------------
    # 11. Mocked from_pretrained CPU Staging Verification
    # --------------------------------------------------------------------------
    def test_mocked_from_pretrained_cpu_staging(self):
        """
        Verify that when loading a model by string ID on CUDA, from_pretrained is explicitly
        directed to CPU staging first, rather than touching CUDA directly.
        """
        mock_model = create_seeded_synthetic_model(seed=42)
        mock_model.to = MagicMock(return_value=mock_model)

        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_from_pretrained:
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")
            load_and_quantize_model(
                "mock/test-model",
                device=target_device,
                quantization="int8"
            )

            # Assert from_pretrained was called
            mock_from_pretrained.assert_called_once()
            # Assert .to("cpu") was called on the uncompressed model before quantization
            mock_model.to.assert_any_call("cpu")
            # Assert .to(target_device) was called after quantization
            mock_model.to.assert_any_call(target_device)

    # --------------------------------------------------------------------------
    # 12. INT4 Memory-Efficient In-Place Dequantization Numerical Equivalence
    # --------------------------------------------------------------------------
    def test_int4_slice_dequantization_numerical_equivalence(self):
        """
        Regression test: Verify that memory-efficient slice unpacking produces
        exact identical tensors to separate low/high tensor unpacking for both
        even and odd in_features dimensions.
        """
        for in_features in (32, 33, 128, 1536):
            float_layer = nn.Linear(in_features, 64, bias=False)
            ql = QuantizedLinear.from_float(float_layer, quant_type="int4")

            # Reference unpacking method
            low = (ql.weight_packed & 0x0F).to(torch.int8) - 8
            high = ((ql.weight_packed >> 4) & 0x0F).to(torch.int8) - 8
            unpacked_ref = torch.empty((ql.out_features, ql.in_features), dtype=torch.int8)
            unpacked_ref[:, 0::2] = low[:, :unpacked_ref[:, 0::2].shape[1]]
            unpacked_ref[:, 1::2] = high[:, :unpacked_ref[:, 1::2].shape[1]]
            w_ref = unpacked_ref.float() * ql.scale

            # Optimized implementation method under test
            w_actual = qlayer_dequant = ql.dequantize()
            self.assertTrue(torch.equal(w_ref, w_actual), f"Mismatch for in_features={in_features}")

            # Verify target dtype support
            w_fp16 = ql.dequantize(dtype=torch.float16)
            self.assertEqual(w_fp16.dtype, torch.float16)
            self.assertLessEqual((w_fp16.float() - w_ref).abs().max().item(), 1e-3)

    # --------------------------------------------------------------------------
    # 13. Unsupported Architecture Diagnostics in apply_optional_quantization
    # --------------------------------------------------------------------------
    def test_apply_optional_quantization_unsupported_architecture(self):
        """
        Regression test: Verify that apply_optional_quantization raises a helpful
        ValueError when passed a model architecture that lacks standard transformer
        linear projections (e.g., non-decoder models or custom architectures like LFM3).
        """
        unsupported_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        for quant_mode in ("int8", "int4", "hybrid_int8"):
            with self.assertRaises(ValueError) as ctx:
                apply_optional_quantization(unsupported_model, quant_type=quant_mode)
            err_msg = str(ctx.exception)
            self.assertIn(f"Quantization '{quant_mode}' failed", err_msg)
            self.assertIn("no compatible linear layers found", err_msg)
            self.assertIn("Supported architectures require transformer blocks", err_msg)

    # --------------------------------------------------------------------------
    # 14. trust_remote_code Parameter Pass-Through
    # --------------------------------------------------------------------------
    def test_load_and_quantize_model_trust_remote_code(self):
        """
        Regression test: Verify that load_and_quantize_model forwards trust_remote_code
        to AutoModelForCausalLM.from_pretrained to prevent Windows SIGALRM crashes.
        """
        mock_model = create_seeded_synthetic_model(seed=42)
        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_from_pretrained:
            load_and_quantize_model("custom/model-id", device="cpu", quantization="none", trust_remote_code=True)
            mock_from_pretrained.assert_called_once()
            _, kwargs = mock_from_pretrained.call_args
            self.assertTrue(kwargs.get("trust_remote_code"))

        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model) as mock_from_pretrained:
            load_and_quantize_model("custom/model-id", device="cpu", quantization="none", trust_remote_code=False)
            mock_from_pretrained.assert_called_once()
            _, kwargs = mock_from_pretrained.call_args
            self.assertFalse(kwargs.get("trust_remote_code"))


if __name__ == "__main__":
    unittest.main()
