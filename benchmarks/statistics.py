"""
Statistical Calculation and Distribution Aggregator
---------------------------------------------------
Implements verified statistical distributions and latency percentiles:
  - Method: Linear interpolation (matches standard numpy method='linear')
  - Generates: Mean, Std, Median (P50), P90, P95, P99, Min, Max, and Sample Count
  - Sample size validation: explicitly flags p99 as exploratory when N < 1000
"""

from typing import List, Dict, Any, Optional
import math
import numpy as np


def compute_distribution(
    values: List[float],
    metric_name: str = "",
    min_samples_for_p99: int = 1000
) -> Dict[str, Any]:
    """
    Computes rigorous statistical distributions for latency measurements.

    Percentile Convention:
      Uses linear interpolation between data points:
      i = (p / 100) * (N - 1)
      val = sorted[floor(i)] + (i - floor(i)) * (sorted[ceil(i)] - sorted[floor(i)])
      This matches numpy.percentile(..., method='linear').
    """
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p99_confidence": "insufficient_samples"
        }

    arr = np.array(values, dtype=np.float64)
    n = len(arr)

    p50, p90, p95, p99 = np.percentile(arr, [50, 90, 95, 99], method="linear")

    confidence = "statistically_sound" if n >= min_samples_for_p99 else f"exploratory (N={n} < {min_samples_for_p99})"

    return {
        "count": int(n),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "p99_confidence": confidence
    }


def aggregate_run_metrics(
    records: List[Any],
    wall_time_s: float,
    peak_vram_mb: float,
    kv_cache_dedicated_mb: float,
    bubble_percentage: float
) -> Dict[str, Any]:
    """
    Aggregates per-request raw records into an executive performance summary.
    """
    successful = [r for r in records if r.status == "SUCCESS"]
    failed = [r for r in records if r.status != "SUCCESS"]

    total_requests = len(records)
    total_completed = len(successful)
    total_out_tokens = sum(r.output_tokens for r in successful)
    total_in_tokens = sum(r.prompt_tokens for r in successful)

    req_per_s = total_completed / wall_time_s if wall_time_s > 0 else 0.0
    out_tok_per_s = total_out_tokens / wall_time_s if wall_time_s > 0 else 0.0
    in_tok_per_s = total_in_tokens / wall_time_s if wall_time_s > 0 else 0.0

    client_ttft_vals = [getattr(r, "client_ttft_ms", r.ttft_ms) for r in successful]
    engine_ttft_vals = [getattr(r, "engine_ttft_ms", r.ttft_ms) for r in successful]
    tpot_vals = [r.tpot_ms for r in successful]
    e2e_vals = [r.e2e_latency_ms for r in successful]
    wait_vals = [r.queue_wait_ms for r in successful]

    return {
        "total_requests": total_requests,
        "completed_requests": total_completed,
        "failed_requests": len(failed),
        "error_rate_pct": (len(failed) / total_requests * 100.0) if total_requests > 0 else 0.0,
        "wall_time_s": float(wall_time_s),
        "throughput": {
            "completed_req_per_s": float(req_per_s),
            "output_tokens_per_s": float(out_tok_per_s),
            "input_tokens_per_s": float(in_tok_per_s),
            "total_tokens_generated": total_out_tokens,
            "total_prompt_tokens": total_in_tokens
        },
        "ttft_ms": compute_distribution(client_ttft_vals, "ClientTTFT"),
        "client_ttft_ms": compute_distribution(client_ttft_vals, "ClientTTFT"),
        "engine_ttft_ms": compute_distribution(engine_ttft_vals, "EngineTTFT"),
        "tpot_ms": compute_distribution(tpot_vals, "TPOT"),
        "e2e_latency_ms": compute_distribution(e2e_vals, "E2E"),
        "queue_wait_ms": compute_distribution(wait_vals, "QueueWait"),
        "vram": {
            "peak_process_mb": float(peak_vram_mb),
            "dedicated_kv_cache_mb": float(kv_cache_dedicated_mb)
        },
        "bubble_percentage": float(bubble_percentage)
    }
