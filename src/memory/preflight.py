"""
Pre-Flight Hardware & Memory Guard
----------------------------------
Prevents downloading or loading models that exceed the host system's GPU VRAM,
system RAM, or local disk space.

Inspects lightweight HuggingFace Hub metadata (or local directory) in milliseconds
BEFORE initiating multi-gigabyte weight downloads, protecting users from:
  1. Wasted network bandwidth and lengthy downloads.
  2. Out of disk space crashes (OSError: No space left on device).
  3. Inevitable CUDA Out of Memory (OOM) crashes.
"""

import os
import sys
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:
    HfApi = None
    hf_hub_download = None
    snapshot_download = None

try:
    from huggingface_hub.constants import HF_HUB_CACHE
except ImportError:
    HF_HUB_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


# Module-level single source of truth constants for hardware and memory budgeting
SAFETY_MARGIN_FRACTION: float = 0.05  # 5% safety margin on GPU VRAM
DISK_SAFETY_BUFFER_FRACTION: float = 0.10  # 10% safety buffer on disk download capacity
CUDA_CONTEXT_BYTES: int = int(450 * (1024**2))  # 450 MiB CUDA runtime, driver tables & cuBLAS handles
CPU_OVERHEAD_BYTES: int = int(400 * (1024**2))  # 400 MiB CPU runtime activation overhead
CHUNKED_WORKSPACE_BYTES: int = int(64 * (1024**2))  # 64 MiB execution workspace for chunked prefill
MONOLITHIC_WORKSPACE_BYTES: int = int(450 * (1024**2))  # 450 MiB execution workspace for monolithic prefill
DEFAULT_SAFETY_MARGIN_BYTES: int = int(128 * (1024**2))  # 128 MiB static VRAM safety buffer



def format_bytes(num_bytes: int) -> str:
    """Formats raw bytes into human-readable B, KB, MB, or GB."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def format_binary_bytes(num_bytes: int) -> str:
    """Formats raw bytes into exact binary B, KiB, MiB, GiB, or TiB."""
    num = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TiB"


def parse_vram_budget_bytes(budget: Union[int, float, str]) -> int:
    """Parses a user-specified VRAM budget into exact integer bytes."""
    if isinstance(budget, (int, float)):
        if budget <= 0:
            raise ValueError(f"vram_budget must be positive, got {budget}")
        # If float or small integer <= 128, interpret as GiB (e.g. 2 or 2.0 -> 2 GiB)
        if isinstance(budget, float) and budget <= 128.0:
            return int(budget * (1024**3))
        if isinstance(budget, int) and budget <= 128:
            return int(budget * (1024**3))
        return int(budget)

    if not isinstance(budget, str) or not budget.strip():
        raise ValueError(f"Invalid vram_budget: {budget!r}")

    s = budget.strip().lower()
    multipliers = [
        ("gib", 1024**3),
        ("gi", 1024**3),
        ("g", 1024**3),
        ("gb", 1000**3),
        ("mib", 1024**2),
        ("mi", 1024**2),
        ("m", 1024**2),
        ("mb", 1000**2),
        ("kib", 1024),
        ("ki", 1024),
        ("k", 1024),
        ("kb", 1000),
        ("b", 1),
    ]
    for unit, mult in multipliers:
        if s.endswith(unit):
            num_part = s[: -len(unit)].strip()
            try:
                val = float(num_part)
                if val <= 0:
                    raise ValueError
                return int(val * mult)
            except ValueError:
                raise ValueError(f"Could not parse numeric part of vram_budget: {budget!r}")

    try:
        val = float(s)
        if val <= 0:
            raise ValueError
        if val <= 128.0:
            return int(val * (1024**3))
        return int(val)
    except ValueError:
        raise ValueError(f"Unrecognized vram_budget format: {budget!r}")


def calculate_kv_block_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int = 16,
    kv_cache_dtype: str = "fp16",
) -> int:
    """Calculates exact integer bytes required for one physical KV cache block across all layers."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    dtype_lower = kv_cache_dtype.lower()
    if dtype_lower in ("fp16", "float16", "bf16", "bfloat16"):
        bytes_per_elem = 2
        return int(num_layers * 2 * block_size * num_kv_heads * head_dim * bytes_per_elem)
    elif dtype_lower == "int8":
        # In Mini-vLLM PhysicalBlockPool:
        # k_pool: int8 (1 byte), v_pool: int8 (1 byte)
        # k_scales: float16 (2 bytes), shape (num_layers, total_blocks, block_size, num_kv_heads, 1)
        # v_scales: float16 (2 bytes), shape (num_layers, total_blocks, block_size, num_kv_heads, 1)
        tensor_bytes = num_layers * 2 * block_size * num_kv_heads * head_dim * 1
        scale_bytes = num_layers * 2 * block_size * num_kv_heads * 1 * 2
        return int(tensor_bytes + scale_bytes)
    elif dtype_lower in ("fp32", "float32"):
        bytes_per_elem = 4
        return int(num_layers * 2 * block_size * num_kv_heads * head_dim * bytes_per_elem)
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype!r}; supported dtypes: fp16, bf16, int8, fp32")


@dataclass(frozen=True)
class VRAMBudgetEstimate:
    vram_budget_bytes: int
    bytes_per_block: int
    model_resident_bytes: int
    runtime_overhead_bytes: int
    execution_workspace_bytes: int
    safety_margin_bytes: int
    non_kv_bytes: int
    available_for_kv_bytes: int
    total_blocks: int
    max_context_tokens: int
    hardware_available_bytes: int
    device_type: str
    is_valid: bool
    error_message: Optional[str] = None

    def summary(self) -> str:
        if not self.is_valid:
            return f"VRAMBudgetEstimate(INVALID: {self.error_message})"
        return (
            f"VRAM Budget: {format_binary_bytes(self.vram_budget_bytes)} -> "
            f"{self.total_blocks} KV blocks ({self.max_context_tokens} max tokens) | "
            f"Resident Model: {format_binary_bytes(self.model_resident_bytes)}, "
            f"KV Pool: {format_binary_bytes(self.total_blocks * self.bytes_per_block)}, "
            f"Workspace: {format_binary_bytes(self.execution_workspace_bytes)}, "
            f"Overhead: {format_binary_bytes(self.runtime_overhead_bytes)}, "
            f"Safety Margin: {format_binary_bytes(self.safety_margin_bytes)}"
        )


@dataclass
class HardwareCapacity:
    device_type: str
    device_name: str
    total_memory: int
    available_memory: int
    disk_free: int
    cache_dir: str


@dataclass
class ModelMetadata:
    model_id: str
    is_local: bool
    parameter_count: Optional[int] = None
    weight_bytes: int = 0
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None


@dataclass
class PreflightReport:
    model_id: str
    device: str
    quantization: str
    parameter_count: Optional[int]
    download_size_bytes: int
    runtime_memory_needed: int
    hardware_available_memory: int
    disk_available_bytes: int
    fits_memory: bool
    fits_disk: bool
    can_run: bool
    recommendations: List[str] = field(default_factory=list)


def _get_hf_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Resolves the Hugging Face hub cache directory.

    Honors explicit cache_dir, HF_HUB_CACHE, and HF_HOME environment variables,
    falling back to huggingface_hub.constants.HF_HUB_CACHE.
    """
    if cache_dir is not None:
        return cache_dir
    if "HF_HUB_CACHE" in os.environ:
        return os.path.expandvars(os.path.expanduser(os.environ["HF_HUB_CACHE"]))
    if "HF_HOME" in os.environ:
        return os.path.join(os.environ["HF_HOME"], "hub")
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as _HUB_CACHE
        return str(_HUB_CACHE)
    except ImportError:
        return os.path.expanduser("~/.cache/huggingface/hub")


def filter_weight_files(file_entries: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Filters candidate weight files to prevent double-counting when multiple serialization formats exist.

    If any .safetensors files are present in file_entries, only the .safetensors files are returned.
    Otherwise, returns any .bin or .pt files.

    Note: This covers the common Hugging Face repository layout where models are provided
    in both SafeTensors and legacy PyTorch (.bin/.pt) formats. It does not handle exotic
    multi-format configurations.
    """
    safetensors = [entry for entry in file_entries if entry[0].endswith(".safetensors")]
    if safetensors:
        return safetensors
    return [entry for entry in file_entries if entry[0].endswith((".bin", ".pt"))]


def _inspect_local_model_directory(directory_path: str) -> ModelMetadata:
    """Extracts lightweight metadata from a local directory or cached snapshot."""
    raw_files: List[Tuple[str, int]] = []
    for root, _, files in os.walk(directory_path):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".pt")):
                fp = os.path.join(root, f)
                try:
                    raw_files.append((f, os.path.getsize(fp)))
                except OSError:
                    pass

    selected_files = filter_weight_files(raw_files)
    weight_bytes = sum(sz for _, sz in selected_files)

    cfg_file = os.path.join(directory_path, "config.json")
    meta = ModelMetadata(model_id=directory_path, is_local=True, weight_bytes=weight_bytes)
    if os.path.exists(cfg_file):
        try:
            import json
            with open(cfg_file, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)
            cfg_dict = raw_cfg.get("text_config", raw_cfg) if isinstance(raw_cfg, dict) else {}
            meta.hidden_size = cfg_dict.get("hidden_size")
            meta.num_layers = cfg_dict.get("num_hidden_layers")
            meta.num_heads = cfg_dict.get("num_attention_heads")
            meta.num_kv_heads = cfg_dict.get("num_key_value_heads", meta.num_heads)
            meta.head_dim = cfg_dict.get("head_dim", (meta.hidden_size // meta.num_heads) if meta.hidden_size and meta.num_heads else 64)
            meta.vocab_size = cfg_dict.get("vocab_size")
        except Exception:
            pass

    if meta.weight_bytes > 0 and meta.parameter_count is None:
        meta.parameter_count = int(meta.weight_bytes // 2)
    return meta


def get_hardware_capacity(device_str: str = "cuda") -> HardwareCapacity:
    """Inspects host GPU, CPU, and disk storage capacity."""
    # 1. Disk capacity for HuggingFace cache
    cache_dir = _get_hf_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    try:
        disk_usage = shutil.disk_usage(cache_dir)
        disk_free = disk_usage.free
    except Exception:
        disk_free = 100 * (1024**3)  # fallback 100 GB

    # 2. Compute device capacity
    if device_str == "cuda" and torch.cuda.is_available():
        device_idx = 0
        device_name = torch.cuda.get_device_name(device_idx)
        total_mem = torch.cuda.get_device_properties(device_idx).total_memory
        try:
            free_mem, _ = torch.cuda.mem_get_info(device_idx)
        except Exception:
            free_mem = total_mem
        return HardwareCapacity(
            device_type="cuda",
            device_name=device_name,
            total_memory=total_mem,
            available_memory=free_mem,
            disk_free=disk_free,
            cache_dir=cache_dir
        )
    else:
        # System RAM
        if psutil:
            vmem = psutil.virtual_memory()
            total_ram = vmem.total
            avail_ram = vmem.available
        else:
            total_ram = 16 * (1024**3)
            avail_ram = 8 * (1024**3)
        return HardwareCapacity(
            device_type="cpu",
            device_name="Host System RAM",
            total_memory=total_ram,
            available_memory=avail_ram,
            disk_free=disk_free,
            cache_dir=cache_dir
        )


def get_model_metadata(model_id: str) -> ModelMetadata:
    """
    Fetches lightweight metadata for a local directory or remote HuggingFace model
    WITHOUT downloading model weight tensors.
    """
    # 1. Local filesystem path check
    if os.path.exists(model_id) and os.path.isdir(model_id):
        return _inspect_local_model_directory(model_id)

    # 2. Remote HuggingFace Hub metadata check
    meta = ModelMetadata(model_id=model_id, is_local=False)
    raw_cfg = None
    if HfApi:
        try:
            api = HfApi()
            info = api.model_info(model_id, files_metadata=True)

            # Calculate total download size from file siblings, avoiding double-counting
            raw_siblings = [
                (s.rfilename, s.size)
                for s in (info.siblings or [])
                if s.size and s.rfilename.endswith((".safetensors", ".bin", ".pt"))
            ]
            selected_siblings = filter_weight_files(raw_siblings)
            if selected_siblings:
                meta.weight_bytes = sum(sz for _, sz in selected_siblings)

            # Check safetensors parameter metadata
            if getattr(info, "safetensors", None) and getattr(info.safetensors, "total", None):
                meta.parameter_count = info.safetensors.total
        except Exception:
            # When offline (e.g. HF_HUB_OFFLINE=1), model_info() fails.
            # Before giving up, attempt to resolve a local cached snapshot:
            try:
                from huggingface_hub import snapshot_download as _snap_dl
                local_snapshot_dir = _snap_dl(model_id, cache_dir=_get_hf_cache_dir(), local_files_only=True)
                if local_snapshot_dir and os.path.isdir(local_snapshot_dir):
                    local_meta = _inspect_local_model_directory(local_snapshot_dir)
                    local_meta.model_id = model_id
                    local_meta.is_local = False
                    return local_meta
            except Exception:
                pass

            # Also check directly in resolved cache_dir/models--.../snapshots
            repo_folder_name = "models--" + model_id.replace("/", "--")
            cached_snapshots = os.path.join(_get_hf_cache_dir(), repo_folder_name, "snapshots")
            if os.path.exists(cached_snapshots):
                revs = [d for d in os.listdir(cached_snapshots) if os.path.isdir(os.path.join(cached_snapshots, d))]
                if revs:
                    latest_snap = os.path.join(cached_snapshots, revs[-1])
                    local_meta = _inspect_local_model_directory(latest_snap)
                    local_meta.model_id = model_id
                    local_meta.is_local = False
                    return local_meta

    # 3. Download and inspect lightweight config.json (~3 KB) if architectural info is needed
    if hf_hub_download and (meta.num_layers is None or meta.hidden_size is None):
        try:
            import json
            cfg_path = hf_hub_download(model_id, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw_cfg = json.load(f)

            # If multimodal/MoE with nested text_config, unwrap it
            cfg_dict = raw_cfg.get("text_config", raw_cfg) if isinstance(raw_cfg, dict) else {}
            meta.hidden_size = cfg_dict.get("hidden_size")
            meta.num_layers = cfg_dict.get("num_hidden_layers")
            meta.num_heads = cfg_dict.get("num_attention_heads")
            meta.num_kv_heads = cfg_dict.get("num_key_value_heads", meta.num_heads)
            meta.head_dim = cfg_dict.get("head_dim", (meta.hidden_size // meta.num_heads) if meta.hidden_size and meta.num_heads else 64)
            meta.vocab_size = cfg_dict.get("vocab_size")
        except Exception:
            pass

    # 4. Fallback to AutoConfig if not loaded via raw JSON
    if meta.hidden_size is None or meta.num_layers is None:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id)
            meta.hidden_size = getattr(cfg, "hidden_size", None)
            meta.num_layers = getattr(cfg, "num_hidden_layers", None)
            meta.num_heads = getattr(cfg, "num_attention_heads", None)
            meta.num_kv_heads = getattr(cfg, "num_key_value_heads", meta.num_heads)
            meta.head_dim = getattr(cfg, "head_dim", (meta.hidden_size // meta.num_heads) if meta.hidden_size and meta.num_heads else 64)
            meta.vocab_size = getattr(cfg, "vocab_size", None)
        except Exception:
            pass

    # 5. Deduce parameter count / weight bytes if one is missing
    if meta.weight_bytes > 0 and meta.parameter_count is None:
        meta.parameter_count = int(meta.weight_bytes // 2)
    elif meta.parameter_count and meta.weight_bytes == 0:
        meta.weight_bytes = int(meta.parameter_count * 2)

    return meta


def validate_model_preflight(
    model_id: str,
    device: str = "cuda",
    quantization: str = "none",
    batch_size: int = 1,
    max_context: int = 2048,
    force: bool = False
) -> PreflightReport:
    """
    Validates whether a model can be safely downloaded and executed on the host system.
    """
    hw = get_hardware_capacity(device)
    meta = get_model_metadata(model_id)

    # 1. Scale weights by quantization factor
    quant_multiplier = 1.0
    if quantization == "int8":
        quant_multiplier = 0.50
    elif quantization == "int4":
        quant_multiplier = 0.25
    elif quantization == "hybrid_int8":
        quant_multiplier = 0.55

    effective_weight_bytes = int(meta.weight_bytes * quant_multiplier)

    # 2. Check for missing architectural metadata before computing KV cache
    missing_arch_fields = []
    if meta.num_layers is None or meta.num_layers <= 0:
        missing_arch_fields.append("num_hidden_layers")
    if meta.num_kv_heads is None or meta.num_kv_heads <= 0:
        missing_arch_fields.append("num_key_value_heads")
    if meta.head_dim is None or meta.head_dim <= 0:
        missing_arch_fields.append("head_dim")

    if missing_arch_fields:
        recs = [
            f"Incomplete model architectural configuration: missing required fields {missing_arch_fields}. "
            "Cannot reliably estimate KV-cache memory requirements before execution."
        ]
        return PreflightReport(
            model_id=model_id,
            device=hw.device_type,
            quantization=quantization,
            parameter_count=meta.parameter_count,
            download_size_bytes=meta.weight_bytes,
            runtime_memory_needed=0,
            hardware_available_memory=hw.available_memory,
            disk_available_bytes=hw.disk_free,
            fits_memory=False,
            fits_disk=False,
            can_run=force,
            recommendations=recs
        )

    num_layers = meta.num_layers
    num_kv_heads = meta.num_kv_heads
    head_dim = meta.head_dim
    bytes_per_elem = 2  # FP16
    kv_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
    total_kv_cache = kv_cache_per_token * max_context * batch_size

    # 3. Runtime Activation Buffers & Context Overhead
    runtime_overhead = CUDA_CONTEXT_BYTES if hw.device_type == "cuda" else CPU_OVERHEAD_BYTES

    total_memory_needed = effective_weight_bytes + total_kv_cache + runtime_overhead

    # 4. Memory & Disk Fit Evaluation
    if meta.weight_bytes == 0 and meta.parameter_count is None:
        fits_memory = False
        fits_disk = False
        can_run = force
        recs = [
            "Could not retrieve model metadata or weight sizes from repository. "
            "Verify that the model ID exists, is spelled correctly, and is publicly accessible (or authenticate with HF_TOKEN)."
        ]
        return PreflightReport(
            model_id=model_id,
            device=hw.device_type,
            quantization=quantization,
            parameter_count=None,
            download_size_bytes=0,
            runtime_memory_needed=0,
            hardware_available_memory=hw.available_memory,
            disk_available_bytes=hw.disk_free,
            fits_memory=False,
            fits_disk=False,
            can_run=can_run,
            recommendations=recs
        )

    # Require at least a 5% safety margin on GPU VRAM
    available_with_margin = int(hw.available_memory * (1.0 - SAFETY_MARGIN_FRACTION))
    fits_memory = total_memory_needed <= available_with_margin

    # Check disk download space (needs uncompressed weights + 10% buffer if remote and not already cached)
    is_cached = False
    repo_folder_name = "models--" + model_id.replace("/", "--")
    cached_repo_path = os.path.join(hw.cache_dir, repo_folder_name)
    if os.path.exists(cached_repo_path):
        snapshots_dir = os.path.join(cached_repo_path, "snapshots")
        if os.path.exists(snapshots_dir) and os.listdir(snapshots_dir):
            is_cached = True

    download_needed = 0 if (meta.is_local or is_cached) else meta.weight_bytes
    fits_disk = (download_needed * (1.0 + DISK_SAFETY_BUFFER_FRACTION)) <= hw.disk_free

    # 5. Formulate tailored recommendations
    recs = []
    if fits_memory and quantization != "none" and hw.device_type == "cuda" and psutil:
        avail_ram = psutil.virtual_memory().available
        if meta.weight_bytes > avail_ram * 0.9:
            fits_memory = False
            recs.append(f"Insufficient host CPU RAM to stage uncompressed model before quantization: need ~{format_bytes(meta.weight_bytes)}, but host has only {format_bytes(avail_ram)} available RAM.")

    can_run = (fits_memory and fits_disk) or force
    if not fits_memory:
        # Check if INT4 or INT8 would make it fit
        int4_needed = int(meta.weight_bytes * 0.25) + total_kv_cache + runtime_overhead
        int8_needed = int(meta.weight_bytes * 0.50) + total_kv_cache + runtime_overhead

        if quantization == "none" and int4_needed <= available_with_margin:
            recs.append(f"Compress weights with 4-bit quantization: pass '--quant int4' (reduces footprint to ~{format_bytes(int4_needed)}).")
        elif quantization == "none" and int8_needed <= available_with_margin:
            recs.append(f"Compress weights with 8-bit quantization: pass '--quant int8' (reduces footprint to ~{format_bytes(int8_needed)}).")

        # Check if system CPU RAM is large enough
        if hw.device_type == "cuda" and psutil:
            avail_ram = psutil.virtual_memory().available
            if total_memory_needed <= avail_ram * 0.9:
                recs.append(f"Offload execution to system RAM: pass '--device cpu' (Host has {format_bytes(avail_ram)} available RAM).")

        # Suggest smaller models that natively fit
        recs.append("Consider a lighter architecture that fits natively: e.g. 'Qwen/Qwen2.5-0.5B' (~0.92 GB), 'Qwen/Qwen2.5-1.5B' (~3.1 GB), or 'meta-llama/Llama-3.2-1B' (~2.5 GB).")

    if not fits_disk:
        recs.append(f"Free up disk space on '{hw.cache_dir}'. Need ~{format_bytes(download_needed)}, but only {format_bytes(hw.disk_free)} is free.")

    return PreflightReport(
        model_id=model_id,
        device=hw.device_type,
        quantization=quantization,
        parameter_count=meta.parameter_count,
        download_size_bytes=meta.weight_bytes,
        runtime_memory_needed=total_memory_needed,
        hardware_available_memory=hw.available_memory,
        disk_available_bytes=hw.disk_free,
        fits_memory=fits_memory,
        fits_disk=fits_disk,
        can_run=can_run,
        recommendations=recs
    )


def estimate_vram_budget(
    model_id: str,
    vram_budget: Union[int, float, str],
    device: str = "cuda",
    quantization: str = "none",
    block_size: int = 16,
    kv_cache_dtype: str = "fp16",
    chunked_prefill: bool = True,
    runtime_overhead_bytes: Optional[int] = None,
    execution_workspace_bytes: Optional[int] = None,
    safety_margin_bytes: Optional[int] = None,
    metadata: Optional[ModelMetadata] = None,
    hardware: Optional[HardwareCapacity] = None,
    raise_on_error: bool = True,
) -> VRAMBudgetEstimate:
    """Estimates static KV cache capacity from a user VRAM budget before loading weights.

    Uses integer byte arithmetic and explicit binary units.
    Fails safely if required architectural metadata is unavailable or budget is insufficient.
    """
    budget_bytes = parse_vram_budget_bytes(vram_budget)
    hw = hardware if hardware is not None else get_hardware_capacity(device)
    hw_avail = hw.available_memory

    meta = metadata if metadata is not None else get_model_metadata(model_id)

    # 1. Validate architectural metadata
    missing_fields = []
    if meta.num_layers is None or meta.num_layers <= 0:
        missing_fields.append("num_layers")
    if meta.num_kv_heads is None or meta.num_kv_heads <= 0:
        missing_fields.append("num_kv_heads")
    if meta.head_dim is None or meta.head_dim <= 0:
        missing_fields.append("head_dim")

    if missing_fields:
        msg = (
            f"Incomplete architectural metadata for model '{model_id}': missing {missing_fields}. "
            "Cannot reliably calculate bytes per KV block."
        )
        if raise_on_error:
            raise ValueError(msg)
        return VRAMBudgetEstimate(
            vram_budget_bytes=budget_bytes,
            bytes_per_block=0,
            model_resident_bytes=0,
            runtime_overhead_bytes=0,
            execution_workspace_bytes=0,
            safety_margin_bytes=0,
            non_kv_bytes=0,
            available_for_kv_bytes=0,
            total_blocks=0,
            max_context_tokens=0,
            hardware_available_bytes=hw_avail,
            device_type=hw.device_type,
            is_valid=False,
            error_message=msg,
        )

    # 2. Derive bytes per block
    bytes_per_block = calculate_kv_block_bytes(
        num_layers=meta.num_layers,
        num_kv_heads=meta.num_kv_heads,
        head_dim=meta.head_dim,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
    )

    # 3. Model residency calculation
    quant_multiplier = 1.0
    if quantization == "int8":
        quant_multiplier = 0.50
    elif quantization == "int4":
        quant_multiplier = 0.25
    elif quantization == "hybrid_int8":
        quant_multiplier = 0.55

    if meta.weight_bytes > 0:
        model_resident_bytes = int(meta.weight_bytes * quant_multiplier)
    elif meta.parameter_count is not None and meta.parameter_count > 0:
        model_resident_bytes = int(meta.parameter_count * 2 * quant_multiplier)
    else:
        msg = f"Cannot estimate weight residency for '{model_id}': 0 weight bytes and unknown parameter count."
        if raise_on_error:
            raise ValueError(msg)
        return VRAMBudgetEstimate(
            vram_budget_bytes=budget_bytes,
            bytes_per_block=bytes_per_block,
            model_resident_bytes=0,
            runtime_overhead_bytes=0,
            execution_workspace_bytes=0,
            safety_margin_bytes=0,
            non_kv_bytes=0,
            available_for_kv_bytes=0,
            total_blocks=0,
            max_context_tokens=0,
            hardware_available_bytes=hw_avail,
            device_type=hw.device_type,
            is_valid=False,
            error_message=msg,
        )

    # 4. Runtime overhead, workspace, and safety margin
    if runtime_overhead_bytes is None:
        # CUDA context (~450 MiB on Linux/RTX 4050) covers driver tables, JIT cache, cublas handles
        runtime_overhead_bytes = CUDA_CONTEXT_BYTES if hw.device_type == "cuda" else 0
    if execution_workspace_bytes is None:
        execution_workspace_bytes = CHUNKED_WORKSPACE_BYTES if chunked_prefill else MONOLITHIC_WORKSPACE_BYTES
    if safety_margin_bytes is None:
        safety_margin_bytes = DEFAULT_SAFETY_MARGIN_BYTES

    non_kv_bytes = (
        model_resident_bytes
        + runtime_overhead_bytes
        + execution_workspace_bytes
        + safety_margin_bytes
    )

    # 5. Check hardware limit constraint
    if budget_bytes > hw_avail:
        msg = (
            f"Requested VRAM budget ({format_binary_bytes(budget_bytes)}) exceeds available device memory "
            f"({format_binary_bytes(hw_avail)}) on {hw.device_name}."
        )
        if raise_on_error:
            raise ValueError(msg)
        return VRAMBudgetEstimate(
            vram_budget_bytes=budget_bytes,
            bytes_per_block=bytes_per_block,
            model_resident_bytes=model_resident_bytes,
            runtime_overhead_bytes=runtime_overhead_bytes,
            execution_workspace_bytes=execution_workspace_bytes,
            safety_margin_bytes=safety_margin_bytes,
            non_kv_bytes=non_kv_bytes,
            available_for_kv_bytes=0,
            total_blocks=0,
            max_context_tokens=0,
            hardware_available_bytes=hw_avail,
            device_type=hw.device_type,
            is_valid=False,
            error_message=msg,
        )

    # 6. Check budget sufficiency
    available_for_kv_bytes = budget_bytes - non_kv_bytes
    if available_for_kv_bytes < bytes_per_block:
        msg = (
            f"Insufficient VRAM budget: non-KV requirement of {format_binary_bytes(non_kv_bytes)} "
            f"(model: {format_binary_bytes(model_resident_bytes)}, workspace: {format_binary_bytes(execution_workspace_bytes)}, "
            f"overhead: {format_binary_bytes(runtime_overhead_bytes)}, safety: {format_binary_bytes(safety_margin_bytes)}) "
            f"leaves {format_binary_bytes(max(0, available_for_kv_bytes))} for KV pool, which is less than "
            f"the required {format_binary_bytes(bytes_per_block)} for 1 block."
        )
        if raise_on_error:
            raise ValueError(msg)
        return VRAMBudgetEstimate(
            vram_budget_bytes=budget_bytes,
            bytes_per_block=bytes_per_block,
            model_resident_bytes=model_resident_bytes,
            runtime_overhead_bytes=runtime_overhead_bytes,
            execution_workspace_bytes=execution_workspace_bytes,
            safety_margin_bytes=safety_margin_bytes,
            non_kv_bytes=non_kv_bytes,
            available_for_kv_bytes=available_for_kv_bytes,
            total_blocks=0,
            max_context_tokens=0,
            hardware_available_bytes=hw_avail,
            device_type=hw.device_type,
            is_valid=False,
            error_message=msg,
        )

    total_blocks = available_for_kv_bytes // bytes_per_block
    max_context_tokens = total_blocks * block_size

    return VRAMBudgetEstimate(
        vram_budget_bytes=budget_bytes,
        bytes_per_block=bytes_per_block,
        model_resident_bytes=model_resident_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        execution_workspace_bytes=execution_workspace_bytes,
        safety_margin_bytes=safety_margin_bytes,
        non_kv_bytes=non_kv_bytes,
        available_for_kv_bytes=available_for_kv_bytes,
        total_blocks=total_blocks,
        max_context_tokens=max_context_tokens,
        hardware_available_bytes=hw_avail,
        device_type=hw.device_type,
        is_valid=True,
        error_message=None,
    )


def enforce_preflight_check(
    model_id: str,
    device: str = "cuda",
    quantization: str = "none",
    force: bool = False
) -> PreflightReport:
    """
    Enforces the pre-flight check. If the model will NOT fit in the system,
    prints a clear diagnostic alert and halts BEFORE downloading weights.
    """
    report = validate_model_preflight(model_id, device=device, quantization=quantization, force=force)

    # ANSI Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    if report.fits_memory and report.fits_disk:
        # Fits comfortably
        param_str = f" ({report.parameter_count / 1e9:.1f}B params)" if report.parameter_count else ""
        print(f"{GREEN}✓ [Pre-Flight Check Passed]{RESET} Model '{model_id}'{param_str} "
              f"requires ~{format_bytes(report.runtime_memory_needed)} | "
              f"Available: {format_bytes(report.hardware_available_memory)} ({report.device.upper()})")
        return report

    if force:
        print(f"{YELLOW}⚠️  [Pre-Flight Warning]{RESET} Model '{model_id}' exceeds estimated capacity, "
              f"but proceeding due to '--force' flag.")
        return report

    # Failed check: Halt before downloading!
    print(f"\n{RED}{BOLD}╔═══════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{RED}{BOLD}║         🚨 PRE-FLIGHT CHECK FAILED: INSUFFICIENT SYSTEM CAPACITY          ║{RESET}")
    print(f"{RED}{BOLD}╚═══════════════════════════════════════════════════════════════════════════╝{RESET}\n")

    param_str = f"{report.parameter_count / 1e9:.2f} Billion" if report.parameter_count else "Unknown"
    print(f"{BOLD}Model ID:{RESET}                 {model_id}")
    print(f"{BOLD}Parameter Count:{RESET}          {param_str}")
    print(f"{BOLD}Download Weight Size:{RESET}     {format_bytes(report.download_size_bytes)}")
    print(f"{BOLD}Estimated Memory Needed:{RESET}  {RED}{format_bytes(report.runtime_memory_needed)}{RESET} "
          f"(Weights + KV Cache + CUDA Overhead)")
    print(f"{BOLD}Available {report.device.upper()} Memory:{RESET}    {GREEN}{format_bytes(report.hardware_available_memory)}{RESET}")

    deficit = report.runtime_memory_needed - report.hardware_available_memory
    if deficit > 0:
        print(f"{BOLD}Memory Shortage:{RESET}          {RED}{format_bytes(deficit)} deficit{RESET}")

    if not report.fits_disk:
        disk_deficit = report.download_size_bytes - report.disk_available_bytes
        print(f"{BOLD}Disk Shortage:{RESET}            {RED}{format_bytes(disk_deficit)} deficit on cache drive{RESET}")

    print(f"\n{CYAN}{BOLD}💡 Recommended Solutions:{RESET}")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

    print(f"\n{YELLOW}To override this safety guard and force download anyway, pass '--force'.{RESET}\n")
    sys.exit(1)


@dataclass
class InstalledModelInfo:
    repo_id: str
    size_bytes: int
    size_str: str
    status: str  # "READY" (weights present) or "METADATA ONLY"
    num_files: int
    model_type: str
    snapshot_path: Optional[str] = None
    has_weights: bool = False
    last_modified_str: str = "N/A"
    weight_files: List[str] = field(default_factory=list)


def list_installed_models(
    cache_dir: Optional[str] = None,
    include_metadata_only: bool = True
) -> List[InstalledModelInfo]:
    """
    Scans local Hugging Face cache and local model directories to report installed
    models, their size on disk, architecture, and whether full weights are downloaded.
    """
    cache_dir = _get_hf_cache_dir(cache_dir)

    results: List[InstalledModelInfo] = []
    scanned_repos = set()

    # 1. Primary scan using huggingface_hub if available
    try:
        import huggingface_hub
        if hasattr(huggingface_hub, "scan_cache_dir"):
            cache_info = huggingface_hub.scan_cache_dir(cache_dir if os.path.exists(cache_dir) else None)
            for repo in cache_info.repos:
                if getattr(repo, "repo_type", None) != "model":
                    continue
                scanned_repos.add(repo.repo_id)
                has_weights = False
                weight_files = []
                snap_path = None
                arch = "Unknown"

                for rev in getattr(repo, "revisions", []):
                    if snap_path is None and getattr(rev, "snapshot_path", None):
                        snap_path = str(rev.snapshot_path)
                    for f in getattr(rev, "files", []):
                        fname = getattr(f, "file_name", "")
                        if fname.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt")):
                            has_weights = True
                            if fname not in weight_files:
                                weight_files.append(fname)
                    cfg_path = os.path.join(str(rev.snapshot_path), "config.json") if getattr(rev, "snapshot_path", None) else ""
                    if os.path.exists(cfg_path):
                        try:
                            import json
                            with open(cfg_path, "r", encoding="utf-8") as fp:
                                cfg = json.load(fp)
                            arch = cfg.get("model_type", cfg.get("architectures", ["Unknown"])[0] if isinstance(cfg.get("architectures"), list) else "Unknown")
                        except Exception:
                            pass

                status = "READY" if has_weights else "METADATA ONLY"
                if include_metadata_only or has_weights:
                    results.append(InstalledModelInfo(
                        repo_id=repo.repo_id,
                        size_bytes=getattr(repo, "size_on_disk", 0),
                        size_str=getattr(repo, "size_on_disk_str", format_binary_bytes(getattr(repo, "size_on_disk", 0))),
                        status=status,
                        num_files=getattr(repo, "nb_files", len(weight_files)),
                        model_type=arch,
                        snapshot_path=snap_path,
                        has_weights=has_weights,
                        last_modified_str=getattr(repo, "last_modified_str", "N/A"),
                        weight_files=weight_files
                    ))
    except Exception:
        pass

    # 2. Filesystem fallback scan if huggingface_hub scan didn't find repos or wasn't available
    if os.path.exists(cache_dir):
        try:
            for entry in sorted(os.listdir(cache_dir)):
                if entry.startswith("models--"):
                    repo_id = entry[len("models--"):].replace("--", "/")
                    if repo_id in scanned_repos:
                        continue
                    scanned_repos.add(repo_id)
                    repo_path = os.path.join(cache_dir, entry)
                    snapshots_dir = os.path.join(repo_path, "snapshots")
                    has_weights = False
                    weight_files = []
                    snap_path = None
                    arch = "Unknown"
                    total_size = 0

                    if os.path.exists(snapshots_dir):
                        snaps = sorted(os.listdir(snapshots_dir))
                        if snaps:
                            snap_path = os.path.join(snapshots_dir, snaps[-1])
                            cfg_file = os.path.join(snap_path, "config.json")
                            if os.path.exists(cfg_file):
                                try:
                                    import json
                                    with open(cfg_file, "r", encoding="utf-8") as fp:
                                        cfg = json.load(fp)
                                    arch = cfg.get("model_type", cfg.get("architectures", ["Unknown"])[0] if isinstance(cfg.get("architectures"), list) else "Unknown")
                                except Exception:
                                    pass

                    for root, _, files in os.walk(repo_path):
                        for f in files:
                            fp = os.path.join(root, f)
                            try:
                                total_size += os.path.getsize(fp)
                            except OSError:
                                pass
                            if f.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt")):
                                has_weights = True
                                if f not in weight_files:
                                    weight_files.append(f)

                    status = "READY" if has_weights else "METADATA ONLY"
                    if include_metadata_only or has_weights:
                        results.append(InstalledModelInfo(
                            repo_id=repo_id,
                            size_bytes=total_size,
                            size_str=format_binary_bytes(total_size),
                            status=status,
                            num_files=len(weight_files) if weight_files else 1,
                            model_type=arch,
                            snapshot_path=snap_path,
                            has_weights=has_weights,
                            last_modified_str="Local Cache",
                            weight_files=weight_files
                        ))
        except Exception:
            pass

    # 3. Check local ./models directory if present in working directory
    local_models_dir = os.path.abspath("models")
    if os.path.exists(local_models_dir) and os.path.isdir(local_models_dir):
        try:
            for entry in sorted(os.listdir(local_models_dir)):
                model_path = os.path.join(local_models_dir, entry)
                if os.path.isdir(model_path) and entry not in scanned_repos:
                    has_weights = False
                    weight_files = []
                    total_size = 0
                    arch = "Unknown"
                    for root, _, files in os.walk(model_path):
                        for f in files:
                            fp = os.path.join(root, f)
                            try:
                                total_size += os.path.getsize(fp)
                            except OSError:
                                pass
                            if f.endswith((".safetensors", ".bin", ".pt")):
                                has_weights = True
                                if f not in weight_files:
                                    weight_files.append(f)
                            if f == "config.json":
                                try:
                                    import json
                                    with open(fp, "r", encoding="utf-8") as fp_in:
                                        cfg = json.load(fp_in)
                                    arch = cfg.get("model_type", "Unknown")
                                except Exception:
                                    pass
                    if has_weights:
                        scanned_repos.add(entry)
                        results.append(InstalledModelInfo(
                            repo_id=f"local:{entry}",
                            size_bytes=total_size,
                            size_str=format_binary_bytes(total_size),
                            status="READY",
                            num_files=len(weight_files),
                            model_type=arch,
                            snapshot_path=model_path,
                            has_weights=True,
                            last_modified_str="Local Directory",
                            weight_files=weight_files
                        ))
        except Exception:
            pass

    # Sort: READY models first, then alphabetical
    return sorted(results, key=lambda x: (not x.has_weights, x.repo_id.lower()))


def print_installed_models_report(
    models: List[InstalledModelInfo],
    cache_dir: Optional[str] = None,
    verbose: bool = False
) -> None:
    """Formats and prints the list of installed and cached models."""
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    cache_dir = _get_hf_cache_dir(cache_dir)

    print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║                 📦 LOCALLY INSTALLED & CACHED MODELS REPORT               ║{RESET}")
    print(f"{CYAN}{BOLD}╚═══════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{BOLD}Cache Location:{RESET} {cache_dir}")

    if not models:
        print(f"\n{YELLOW}No locally installed or cached models found in '{cache_dir}'.{RESET}")
        print("Tip: Run an inference command to download a model, for example:")
        print("  python cli.py engine --model Qwen/Qwen2.5-0.5B\n")
        return

    ready_models = [m for m in models if m.has_weights]
    meta_models = [m for m in models if not m.has_weights]

    print(f"\n{BOLD}{'MODEL ID':<48} {'STATUS':<15} {'DISK SIZE':<11} {'ARCH':<14} {'WEIGHTS'}{RESET}")
    print("─" * 96)

    for m in models:
        plain_status = "READY" if m.has_weights else "METADATA ONLY"
        padded_status = f"{plain_status:<15}"
        if m.has_weights:
            status_str = f"{GREEN}{BOLD}{padded_status}{RESET}"
            weight_summary = f"{len(m.weight_files)} file{'s' if len(m.weight_files) != 1 else ''}"
            if len(m.weight_files) == 1:
                weight_summary = m.weight_files[0]
        else:
            status_str = f"{DIM}{padded_status}{RESET}"
            weight_summary = f"{DIM}(no weights){RESET}"

        repo_display = m.repo_id if len(m.repo_id) <= 46 else m.repo_id[:43] + "..."
        arch_display = m.model_type if len(m.model_type) <= 12 else m.model_type[:10] + ".."
        size_display = m.size_str

        print(f"{repo_display:<48} {status_str} {size_display:<11} {arch_display:<14} {weight_summary}")

        if verbose and m.snapshot_path:
            print(f"  {DIM}└─ Snapshot: {m.snapshot_path}{RESET}")
            if m.weight_files and len(m.weight_files) > 1:
                print(f"  {DIM}   Files: {', '.join(m.weight_files[:4])}{' ...' if len(m.weight_files) > 4 else ''}{RESET}")

    print("─" * 96)
    total_size = sum(m.size_bytes for m in models)
    print(f"{BOLD}Summary:{RESET} {len(models)} cached repos found: "
          f"{GREEN}{len(ready_models)} READY{RESET} (weights downloaded), "
          f"{YELLOW}{len(meta_models)} METADATA ONLY{RESET} | "
          f"Total Disk Usage: ~{format_bytes(total_size)}")
    print(f"{DIM}Tip: Run `python cli.py check --model <model-id>` to verify hardware fit before loading.{RESET}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-flight LLM Memory & Capacity Checker")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model ID or path to verify")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--quant", type=str, default="none", choices=["none", "int8", "int4", "hybrid_int8"])
    parser.add_argument("--force", action="store_true", help="Bypass check")
    parser.add_argument("-l", "--list-models", action="store_true", help="List locally installed and cached models")
    args = parser.parse_args()

    if args.list_models:
        models = list_installed_models()
        print_installed_models_report(models)
    else:
        enforce_preflight_check(args.model, device=args.device, quantization=args.quant, force=args.force)
