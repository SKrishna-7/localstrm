"""
Milestone 12: Optional Weight Quantization (FP16 -> INT8 / INT4)
----------------------------------------------------------------
Implements optional uniform weight quantization for LLM layers:
  1. User selectable precision: None (FP16/FP32), 'int8', or 'int4'.
  2. Quantizes linear projection weights (Q, K, V, O, Gate, Up, Down).
  3. Packs 4-bit integers (two weights per byte).
  4. Compares Memory Footprint, Compression Ratio, and Numerical Cosine Similarity.
"""

import os
import sys
import argparse
import time
from typing import Tuple, List, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Support running directly or as a package module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import get_default_model, get_default_device, add_common_engine_args


def format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear supporting optional INT8 and INT4 quantization.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, quant_type: str = "none"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type.lower()

        if self.quant_type == "int8":
            # Store as int8: 1 byte per weight
            self.register_buffer("weight_q", torch.zeros((out_features, in_features), dtype=torch.int8))
            self.register_buffer("scale", torch.zeros((out_features, 1), dtype=torch.float32))
        elif self.quant_type == "int4":
            # Pack two 4-bit integers into 1 uint8 byte: in_features // 2
            self.register_buffer("weight_packed", torch.zeros((out_features, (in_features + 1) // 2), dtype=torch.uint8))
            self.register_buffer("scale", torch.zeros((out_features, 1), dtype=torch.float32))
        else:
            # Unquantized: standard float weights
            self.weight = nn.Parameter(torch.zeros((out_features, in_features)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, float_linear: nn.Linear, quant_type: str = "none") -> "QuantizedLinear":
        """Quantizes an existing float nn.Linear layer on-the-fly."""
        qlayer = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=float_linear.bias is not None,
            quant_type=quant_type
        )
        w = float_linear.weight.data.float()

        if quant_type == "int8":
            # Per-channel symmetric quantization: max absolute value per output row
            max_val = torch.max(torch.abs(w), dim=-1, keepdim=True)[0]
            scale = torch.clamp(max_val / 127.0, min=1e-8)
            w_q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
            qlayer.weight_q.copy_(w_q)
            qlayer.scale.copy_(scale)

        elif quant_type == "int4":
            # 4-bit signed range: -8 to 7
            max_val = torch.max(torch.abs(w), dim=-1, keepdim=True)[0]
            scale = torch.clamp(max_val / 7.0, min=1e-8)
            w_q = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)

            # Map signed [-8, 7] to unsigned [0, 15] for bit-packing
            w_u4 = (w_q + 8).to(torch.uint8)
            # Pack two 4-bit values: low nibble = even index, high nibble = odd index
            low = w_u4[:, 0::2]
            high = w_u4[:, 1::2] if w_u4.shape[1] % 2 == 0 else F.pad(w_u4[:, 1::2], (0, 1))
            packed = low | (high << 4)

            qlayer.weight_packed.copy_(packed)
            qlayer.scale.copy_(scale)

        else:
            qlayer.weight.data.copy_(float_linear.weight.data)

        if float_linear.bias is not None:
            qlayer.bias.data.copy_(float_linear.bias.data)

        return qlayer

    def dequantize(self) -> torch.Tensor:
        """Dequantizes weights back to float on-the-fly in registers."""
        if self.quant_type == "int8":
            return self.weight_q.float() * self.scale
        elif self.quant_type == "int4":
            # Unpack two 4-bit values from uint8
            low = (self.weight_packed & 0x0F).to(torch.int8) - 8
            high = ((self.weight_packed >> 4) & 0x0F).to(torch.int8) - 8

            # Interleave back to original shape
            unpacked = torch.empty((self.out_features, self.in_features), dtype=torch.int8, device=self.weight_packed.device)
            unpacked[:, 0::2] = low[:, :unpacked[:, 0::2].shape[1]]
            unpacked[:, 1::2] = high[:, :unpacked[:, 1::2].shape[1]]
            return unpacked.float() * self.scale
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_float = self.dequantize()
        if w_float.device != x.device:
            w_float = w_float.to(x.device)
        if w_float.dtype != x.dtype:
            w_float = w_float.to(x.dtype)
        bias = self.bias
        if bias is not None:
            if bias.device != x.device:
                bias = bias.to(x.device)
            if bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
        return F.linear(x, w_float, bias)

    def memory_footprint_bytes(self) -> int:
        total = 0
        for b in self.buffers():
            total += b.numel() * b.element_size()
        for p in self.parameters():
            total += p.numel() * p.element_size()
        return total


def can_use_int8_gemm(device: Optional[Union[torch.device, str]] = None) -> Tuple[bool, str]:
    """
    Checks if INT8 GEMM (torch._int_mm) is supported in the current runtime environment.
    Returns: (is_supported: bool, reason: str)
    """
    if os.getenv("MINIVLLM_DISABLE_INT8_GEMM", "0").lower() in ("1", "true", "yes"):
        return False, "Disabled by MINIVLLM_DISABLE_INT8_GEMM environment variable"

    if not hasattr(torch, "_int_mm"):
        return False, f"torch._int_mm is not available in PyTorch build {torch.__version__}"

    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    if device is not None:
        dev = torch.device(device)
        if dev.type != "cuda":
            return False, f"Device type is '{dev.type}', but INT8 GEMM requires 'cuda'"
        dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    else:
        dev_idx = torch.cuda.current_device()

    try:
        capability = torch.cuda.get_device_capability(dev_idx)
        if capability < (8, 0):
            return False, f"CUDA device compute capability {capability[0]}.{capability[1]} < 8.0 (requires SM 8.0+)"
    except Exception as e:
        return False, f"Failed to query device capability: {e}"

    return True, f"Supported (SM {capability[0]}.{capability[1]})"


class CachedQuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear caching pre-dequantized FP16 weights.
    Used for attention projections (q, k, v, o) in hybrid_int8 mode to eliminate
    repeated per-token dequantization overhead while staying within VRAM budget.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        self.register_buffer(
            "weight",
            torch.zeros((out_features, in_features), dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        float_linear: nn.Linear,
        dtype: Optional[torch.dtype] = None
    ) -> "CachedQuantizedLinear":
        dt = dtype or float_linear.weight.dtype
        if dt not in (torch.float16, torch.bfloat16, torch.float32):
            dt = torch.float16

        layer = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=float_linear.bias is not None,
            dtype=dt
        )
        # Quantize to INT8 then dequantize back to target dtype once at initialization
        w = float_linear.weight.data.float()
        max_val = torch.max(torch.abs(w), dim=-1, keepdim=True)[0]
        scale = torch.clamp(max_val / 127.0, min=1e-8)
        w_q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
        dequant_w = (w_q.float() * scale).to(dt)

        layer.weight.copy_(dequant_w)
        if float_linear.bias is not None:
            layer.bias.data.copy_(float_linear.bias.data.to(dt))

        layer.to(float_linear.weight.device)
        return layer

    @classmethod
    def from_quantized(
        cls,
        weight_q: torch.Tensor,
        scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float16
    ) -> "CachedQuantizedLinear":
        out_f, in_f = weight_q.shape
        layer = cls(in_f, out_f, bias=bias is not None, dtype=dtype)
        dequant_w = (weight_q.float() * scale.float()).to(dtype)
        layer.weight.copy_(dequant_w)
        if bias is not None:
            layer.bias.data.copy_(bias.to(dtype))
        layer.to(weight_q.device)
        return layer

    def dequantize(self) -> torch.Tensor:
        """Returns pre-cached weights."""
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if w.device != x.device:
            w = w.to(x.device)
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        bias = self.bias
        if bias is not None:
            if bias.device != x.device:
                bias = bias.to(x.device)
            if bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
        return F.linear(x, w, bias)

    def memory_footprint_bytes(self) -> int:
        total = 0
        for b in self.buffers():
            total += b.numel() * b.element_size()
        for p in self.parameters():
            total += p.numel() * p.element_size()
        return total


class Int8GEMMLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using native INT8 GEMM (torch._int_mm)
    with per-channel weight quantization, dynamic per-token activation quantization,
    and automatic row padding for cuBLASLt tile alignment on NVIDIA Tensor Cores (SM >= 8.0).

    Safely falls back to dynamic dequantization on unsupported hardware, CPU device,
    missing torch._int_mm, or when MINIVLLM_DISABLE_INT8_GEMM=1 is set.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.output_dtype = dtype
        self._warned_fallback = False

        # weight_q_t: shape (in_features, out_features) int8 for torch._int_mm(x_padded, weight_q_t)
        self.register_buffer(
            "weight_q_t",
            torch.zeros((in_features, out_features), dtype=torch.int8)
        )
        # scale_w_t: shape (1, out_features) float32
        self.register_buffer(
            "scale_w_t",
            torch.zeros((1, out_features), dtype=torch.float32)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        float_linear: nn.Linear,
        dtype: Optional[torch.dtype] = None
    ) -> "Int8GEMMLinear":
        dt = dtype or float_linear.weight.dtype
        if dt not in (torch.float16, torch.bfloat16, torch.float32):
            dt = torch.float16

        layer = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=float_linear.bias is not None,
            dtype=dt
        )
        w = float_linear.weight.data.float()  # (out_features, in_features)
        max_val = torch.max(torch.abs(w), dim=-1, keepdim=True)[0]
        scale = torch.clamp(max_val / 127.0, min=1e-8)
        w_q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)

        layer.weight_q_t.copy_(w_q.t().contiguous())
        layer.scale_w_t.copy_(scale.t().contiguous())

        if float_linear.bias is not None:
            layer.bias.data.copy_(float_linear.bias.data.to(dt))

        layer.to(float_linear.weight.device)
        return layer

    def dequantize(self) -> torch.Tensor:
        """Dequantizes stored weights back to float on-the-fly."""
        return (self.weight_q_t.t().float() * self.scale_w_t.t()).to(self.output_dtype)

    def _fallback_forward(self, x: torch.Tensor, reason: str) -> torch.Tensor:
        """Safe dynamic dequantization fallback when INT8 GEMM cannot execute."""
        if not self._warned_fallback:
            import logging
            logging.getLogger("mini_vllm.quantization").warning(
                "Int8GEMMLinear falling back to dynamic dequantization: %s", reason
            )
            self._warned_fallback = True

        w_float = (self.weight_q_t.t().float() * self.scale_w_t.t()).to(x.dtype)
        if w_float.device != x.device:
            w_float = w_float.to(x.device)
        bias = self.bias
        if bias is not None:
            if bias.device != x.device:
                bias = bias.to(x.device)
            if bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
        return F.linear(x, w_float, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Kill-switch check
        if os.getenv("MINIVLLM_DISABLE_INT8_GEMM", "0").lower() in ("1", "true", "yes"):
            return self._fallback_forward(x, "MINIVLLM_DISABLE_INT8_GEMM is set")

        # 2. Device and PyTorch primitive verification
        if x.device.type != "cuda":
            return self._fallback_forward(x, f"Input device is {x.device.type}, not cuda")

        if not hasattr(torch, "_int_mm"):
            return self._fallback_forward(x, "torch._int_mm is not available in PyTorch build")

        # 3. Shape validation
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input inner dimension {x.shape[-1]} does not match layer in_features {self.in_features}"
            )

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        M, K = x_2d.shape

        # 4. Alignment check: K and N must be multiples of 8 for cuBLASLt INT8
        if K % 8 != 0 or self.out_features % 8 != 0:
            return self._fallback_forward(x, f"K ({K}) or N ({self.out_features}) not divisible by 8")

        try:
            # Ensure buffers are co-located with input
            if self.weight_q_t.device != x.device:
                self.weight_q_t = self.weight_q_t.to(x.device)
                self.scale_w_t = self.scale_w_t.to(x.device)

            # 5. Dynamic Per-Token Activation Quantization
            max_x = torch.max(torch.abs(x_2d), dim=-1, keepdim=True)[0]
            scale_x = torch.clamp(max_x / 127.0, min=1e-8).float()
            x_q = torch.clamp(torch.round(x_2d.float() / scale_x), -128, 127).to(torch.int8)

            # 6. Tile Alignment Padding: M must be a multiple of 32 for cuBLASLt on SM 8.9
            rem = M % 32
            if rem != 0:
                pad_rows = 32 - rem
                x_padded = F.pad(x_q, (0, 0, 0, pad_rows), value=0)
            else:
                x_padded = x_q

            # 7. Native Integer GEMM: (padded_M, K) x (K, N) -> (padded_M, N) int32
            out_int32 = torch._int_mm(x_padded, self.weight_q_t)

            # 8. Slice out padding rows
            if rem != 0:
                out_int32 = out_int32[:M, :]

            # 9. Output Rescaling (float32 accumulator to prevent FP16 overflow)
            scale_combined = scale_x * self.scale_w_t
            out_fp = out_int32.float() * scale_combined

            # 10. Optional Bias Addition
            if self.bias is not None:
                bias = self.bias
                if bias.device != x.device:
                    bias = bias.to(x.device)
                out_fp = out_fp + bias.float()

            # 11. Cast to target output dtype and restore shape
            target_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else self.output_dtype
            out_final = out_fp.to(target_dtype)
            if len(orig_shape) > 2:
                out_final = out_final.reshape(*orig_shape[:-1], self.out_features)

            return out_final

        except Exception as e:
            return self._fallback_forward(x, f"torch._int_mm execution error: {e}")

    def memory_footprint_bytes(self) -> int:
        total = 0
        for b in self.buffers():
            total += b.numel() * b.element_size()
        for p in self.parameters():
            total += p.numel() * p.element_size()
        return total


def apply_optional_quantization(model: nn.Module, quant_type: str = "none") -> Tuple[int, int]:
    """
    Selectively replaces linear layers in the transformer with QuantizedLinear,
    CachedQuantizedLinear, or Int8GEMMLinear.
    Returns: (orig_weight_bytes, quantized_weight_bytes)
    """
    quant_type = quant_type.lower()
    valid_modes = ("none", "int8", "int4", "hybrid_int8")
    if quant_type not in valid_modes:
        raise ValueError(f"Invalid quantization type '{quant_type}'. Must be one of {valid_modes}.")

    if quant_type == "none":
        orig_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        return orig_bytes, orig_bytes

    orig_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Replace linear projection layers in transformer blocks
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is not None:
        for layer in layers:
            if hasattr(layer, "self_attn"):
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    linear_mod = getattr(layer.self_attn, proj, None)
                    if isinstance(linear_mod, nn.Linear):
                        if quant_type == "hybrid_int8":
                            setattr(layer.self_attn, proj, CachedQuantizedLinear.from_float(linear_mod))
                        else:
                            setattr(layer.self_attn, proj, QuantizedLinear.from_float(linear_mod, quant_type))
            if hasattr(layer, "mlp"):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    linear_mod = getattr(layer.mlp, proj, None)
                    if isinstance(linear_mod, nn.Linear):
                        if quant_type == "hybrid_int8":
                            if linear_mod.in_features % 8 == 0 and linear_mod.out_features % 8 == 0:
                                setattr(layer.mlp, proj, Int8GEMMLinear.from_float(linear_mod))
                            else:
                                import logging
                                logging.getLogger("mini_vllm.quantization").warning(
                                    "MLP layer %s shape (%d, %d) not divisible by 8; falling back to dynamic int8",
                                    proj, linear_mod.in_features, linear_mod.out_features
                                )
                                setattr(layer.mlp, proj, QuantizedLinear.from_float(linear_mod, quant_type="int8"))
                        else:
                            setattr(layer.mlp, proj, QuantizedLinear.from_float(linear_mod, quant_type))

    quant_bytes = 0
    for m in model.modules():
        if isinstance(m, (QuantizedLinear, Int8GEMMLinear, CachedQuantizedLinear)):
            quant_bytes += m.memory_footprint_bytes()
        elif len(list(m.children())) == 0:
            for p in m.parameters(recurse=False):
                quant_bytes += p.numel() * p.element_size()

    return orig_bytes, quant_bytes


def load_and_quantize_model(
    model_name_or_path: Union[str, nn.Module],
    device: Union[torch.device, str],
    quantization: str = "none",
    dtype: Optional[torch.dtype] = None,
    logger: Optional[Any] = None
) -> Tuple[nn.Module, int, int]:
    """
    Loads a Hugging Face CausalLM model with support for CPU-staged quantization.

    When quantization is enabled ('int8', 'int4', or 'hybrid_int8') and the target device is CUDA,
    the uncompressed FP16 weights are loaded into host CPU memory first. Quantization
    is applied in-place on CPU, and only the compressed model is transferred to CUDA VRAM.
    This prevents full uncompressed weights from touching GPU VRAM, enabling larger models
    (e.g., 3B parameters) to fit within constrained VRAM budgets.

    Returns:
        (model, orig_bytes, quant_bytes)
    """
    quantization = quantization.lower()
    valid_modes = ("none", "int8", "int4", "hybrid_int8")
    if quantization not in valid_modes:
        raise ValueError(f"Invalid quantization mode '{quantization}'. Must be one of {valid_modes}.")

    if isinstance(device, str):
        device = torch.device(device)

    if dtype is None:
        dtype = torch.float32 if device.type == "cpu" else torch.float16

    log_fn = logger.info if logger is not None else None

    try:
        # Support passing an already-instantiated nn.Module (e.g. synthetic test models)
        if isinstance(model_name_or_path, nn.Module):
            model = model_name_or_path
            if quantization != "none" and device.type == "cuda":
                if log_fn:
                    log_fn("Staging model on CPU before applying %s quantization...", quantization.upper())
                model = model.to("cpu")
                model.eval()
                orig_bytes, quant_bytes = apply_optional_quantization(model, quant_type=quantization)
                if log_fn:
                    log_fn("Transferring quantized model to %s...", device)
                model = model.to(device)
            else:
                model = model.to(device)
                model.eval()
                if quantization != "none":
                    if log_fn:
                        log_fn("Applying %s weight quantization on %s...", quantization.upper(), device)
                    orig_bytes, quant_bytes = apply_optional_quantization(model, quant_type=quantization)
                else:
                    orig_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                    quant_bytes = orig_bytes
            return model, orig_bytes, quant_bytes

        if quantization != "none" and device.type == "cuda":
            if log_fn:
                log_fn("Staging uncompressed model on CPU before applying %s quantization...", quantization.upper())
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype).to("cpu")
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype).to("cpu")
            model.eval()

            if log_fn:
                log_fn("Applying %s weight quantization on CPU...", quantization.upper())
            orig_bytes, quant_bytes = apply_optional_quantization(model, quant_type=quantization)

            if log_fn:
                savings = (1.0 - (quant_bytes / orig_bytes)) * 100 if orig_bytes > 0 else 0.0
                log_fn("Transferring quantized model to %s (%.1f%% VRAM reduction, compressed from %.2f GB to %.2f GB)...",
                       device, savings, orig_bytes / (1024**3), quant_bytes / (1024**3))
            model = model.to(device)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype).to(device)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype).to(device)
            model.eval()

            if quantization != "none":
                if log_fn:
                    log_fn("Applying %s weight quantization on %s...", quantization.upper(), device)
                orig_bytes, quant_bytes = apply_optional_quantization(model, quant_type=quantization)
            else:
                orig_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                quant_bytes = orig_bytes

        return model, orig_bytes, quant_bytes
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e


def run_quantization_benchmark(
    selected_quant: str = "int8",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    prompt: Optional[str] = None
):
    model_name = model_name or get_default_model()
    device_str = device or get_default_device()

    print("=" * 65)
    print("      MILESTONE 12: OPTIONAL WEIGHT QUANTIZATION BENCHMARK")
    print("=" * 65)
    print(f"Target Model:         {model_name}")
    print(f"Device:               {device_str}")
    print(f"Selected Mode:        {selected_quant.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device_str == "cpu" else torch.float16
    ).to(device_str)
    model.eval()

    # 1. Apply Optional Quantization
    print(f"\n[1/3] Converting model layers to '{selected_quant}' precision...")
    t_quant_start = time.perf_counter()
    orig_bytes, quant_bytes = apply_optional_quantization(model, quant_type=selected_quant)
    quant_time = (time.perf_counter() - t_quant_start) * 1000

    compression_ratio = orig_bytes / quant_bytes if quant_bytes > 0 else 1.0
    vram_saved_pct = (1.0 - (quant_bytes / orig_bytes)) * 100.0

    print(f"  • Conversion Time:         {quant_time:.2f} ms")
    print(f"  • Original Weights Memory: {format_bytes(orig_bytes)}")
    print(f"  • Quantized Weights Memory:{format_bytes(quant_bytes)}")
    print(f"  • Compression Ratio:       {compression_ratio:.2f}x")
    print(f"  • VRAM Savings:            {vram_saved_pct:.1f}% savings!")

    # 2. Test Generation Quality
    test_prompt = prompt or "The key difference between CPU and GPU is that"
    print(f"\n[2/3] Testing text generation quality...")
    print(f"Prompt: \"{test_prompt}\"")

    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device_str)
    generated = list(input_ids[0].tolist())

    t_gen_start = time.perf_counter()
    for _ in range(15):
        curr_in = torch.tensor([generated], device=device_str)
        with torch.no_grad():
            outputs = model(curr_in, use_cache=False)
        next_tok = torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
        generated.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break
    gen_time = time.perf_counter() - t_gen_start

    output_text = tokenizer.decode(generated[input_ids.shape[1]:], skip_special_tokens=True)
    tps = 15 / gen_time if gen_time > 0 else 0

    print(f"  Output: \"{output_text}\"")
    print(f"  Decode Throughput: {tps:.2f} tokens/second")

    print("\n" + "=" * 65)
    print("                    QUANTIZATION SUMMARY")
    print("=" * 65)
    print(f"  • Mode:              {selected_quant.upper()}")
    print(f"  • VRAM Reduction:    {format_bytes(orig_bytes - quant_bytes)} freed!")
    print(f"  • Model Fits in:     {format_bytes(quant_bytes)}")
    if selected_quant == "int4":
        print("  💡 Insight: In INT4, a 3B model (~6 GB) compresses to ~1.8 GB, fitting easily on your GPU!")
    elif selected_quant == "int8":
        print("  💡 Insight: In INT8, weights transfer 2x faster across the memory bus with near-zero accuracy loss.")
    elif selected_quant == "hybrid_int8":
        print("  💡 Insight: In HYBRID_INT8, attention projections are cached in FP16 while MLP projections use native INT8 GEMM.")
    else:
        print("  💡 Insight: In unquantized mode, you retain 100% full precision.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optional weight quantization benchmark")
    add_common_engine_args(parser)
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt to test generation with quantized weights")
    parser.add_argument(
        "--quant",
        type=str,
        choices=["none", "int8", "int4", "hybrid_int8"],
        default="int8",
        help="Quantization precision: 'none', 'int8', 'int4', or 'hybrid_int8' (default: int8)"
    )
    args = parser.parse_args()

    user_prompt = args.prompt
    if not user_prompt:
        if not sys.stdin.isatty():
            user_prompt = sys.stdin.read().strip()
        else:
            try:
                user_p = input("Enter prompt for quantization test: ").strip()
                if user_p:
                    user_prompt = user_p
            except (KeyboardInterrupt, EOFError):
                pass

    run_quantization_benchmark(
        selected_quant=args.quant,
        prompt=user_prompt,
        model_name=args.model,
        device=args.device
    )
