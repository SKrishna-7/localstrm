"""
Benchmark Instrumentation & Profiling Subsystem
-----------------------------------------------
Captures raw, granular, per-request metrics across the complete request lifecycle:
  - Arrival, scheduling, first-token, and finish timestamps
  - TTFT, TPOT, Queue Wait, Decode Duration, and E2E latency
  - Input & output token accounting
  - Peak total GPU VRAM and analytical KV cache memory footprint
  - Execution bubble metrics (wasted idle GPU slots)
"""

import time
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import torch


@dataclass
class RequestMetricRecord:
    """Raw, granular per-request measurement record."""
    request_id: str
    engine: str                 # 'baseline_fp16', 'minivllm_fp16', 'minivllm_int8'
    workload_type: str          # 'fixed' or 'mixed'
    concurrency: int
    repetition: int
    batch_size: int

    arrival_time: float         # time.perf_counter()
    schedule_time: float        # time.perf_counter() when admitted to batch
    first_token_time: float     # time.perf_counter() when first decode token is emitted
    finish_time: float          # time.perf_counter() when generation completes

    prompt_tokens: int
    output_tokens: int
    target_tokens: int

    queue_wait_ms: float        # (schedule_time - arrival_time) * 1000
    engine_ttft_ms: float       # (first_token_time - schedule_time) * 1000 (pure prefill execution)
    client_ttft_ms: float       # (first_token_time - arrival_time) * 1000 (queue wait + prefill)
    ttft_ms: float              # client_ttft_ms (standard client-observed metric)
    decode_duration_ms: float   # (finish_time - first_token_time) * 1000
    e2e_latency_ms: float       # (finish_time - arrival_time) * 1000
    tpot_ms: float              # decode_duration_ms / max(1, output_tokens - 1)

    status: str                 # 'SUCCESS', 'ERROR', 'OOM', 'TIMEOUT'
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sync_gpu(device: torch.device):
    """
    Synchronizes CUDA device strictly at phase boundaries (prefill start/end, decode end)
    to guarantee accurate GPU timestamps without distorting normal kernel pipelining.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_vram(device: torch.device):
    """Resets peak memory tracking stats on CUDA device."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_vram_mb(device: torch.device) -> float:
    """Returns peak allocated memory on CUDA device in Megabytes."""
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    return 0.0


def calculate_analytical_kv_cache_mb(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    total_tokens: int,
    kv_cache_dtype: str = "fp16"
) -> float:
    """
    Analytically calculates dedicated KV cache memory independent of model weights
    and PyTorch runtime overhead:
      Memory = 2 (K & V) * num_layers * total_tokens * num_kv_heads * head_dim * bytes_per_elem
    """
    bytes_per_elem = 1.0 if kv_cache_dtype.lower() == "int8" else 2.0
    elements = 2 * num_layers * total_tokens * num_kv_heads * head_dim
    bytes_total = elements * bytes_per_elem
    if kv_cache_dtype.lower() == "int8":
        # Add FP16 dynamic scale factors: 2 * num_layers * total_tokens * num_kv_heads * 2 bytes
        scale_bytes = 2 * num_layers * total_tokens * num_kv_heads * 2.0
        bytes_total += scale_bytes
    return bytes_total / (1024.0 ** 2)


def calculate_bubble_percentage(
    actual_tokens_per_seq: List[int],
    max_tokens_in_batch: int
) -> float:
    """
    Calculates execution bubble percentage (wasted idle slots) caused by
    request length disparity in locked static batches:
      wasted_slots = sum(max_tokens - actual_tokens_i)
      useful_slots = sum(actual_tokens_i)
      bubble_pct = wasted_slots / (wasted_slots + useful_slots) * 100%
    """
    if not actual_tokens_per_seq:
        return 0.0
    useful_slots = sum(actual_tokens_per_seq)
    wasted_slots = sum(max_tokens_in_batch - t for t in actual_tokens_per_seq)
    total_slots = useful_slots + wasted_slots
    if total_slots == 0:
        return 0.0
    return (wasted_slots / total_slots) * 100.0
