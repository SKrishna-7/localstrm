#!/usr/bin/env python3
"""
Rigorous LLM Serving Performance Benchmark Runner
-------------------------------------------------
Comprehensive, reproducible benchmark suite comparing:
  - Static Baseline (HuggingFace PyTorch left-padded batching)
  - Mini-vLLM with FP16 KV Cache (PagedAttention + Continuous Batching)
  - Mini-vLLM with INT8 KV Cache (50% physical pool memory reduction)

Supports:
  - Concurrency levels: 1, 2, 4, 8, 16 (and optional 32)
  - Two distinct workloads: Fixed-Length (Workload A) and Mixed-Length (Workload B)
  - Profiles: Smoke profile (fast validation) and Full profile (complete matrix)
  - Granular per-request lifecycle logging (CSV and JSONL)
  - Analytical and peak VRAM measurement
  - Latency distributions (P50, P90, P95, P99)
  - Presentation-grade PNG plots and executive Markdown reports
"""

import os
import sys
import time
import argparse
import json
import csv
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from src.config import get_default_model, get_default_device, add_common_engine_args
from src.memory.preflight import enforce_preflight_check
from benchmarks.workload import generate_fixed_workload, generate_mixed_workload, BenchmarkRequest
from benchmarks.instrumentation import RequestMetricRecord
from benchmarks.statistics import aggregate_run_metrics
from benchmarks.engine_adapters import BaselineAdapter, MiniVLLMAdapter
from benchmarks.plotting import generate_benchmark_plots
from benchmarks.report_generator import generate_markdown_report


def run_benchmark_suite(
    model_name: str,
    device_str: str,
    profile: str = "smoke",
    concurrencies: Optional[List[int]] = None,
    engines: Optional[List[str]] = None,
    workloads: Optional[List[str]] = None,
    num_requests_per_concurrency: Optional[int] = None,
    repetitions: int = 1,
    output_dir: Optional[str] = None,
    seed: int = 42
):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results", f"benchmark_{profile}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Resolve Profile Configurations
    if profile == "smoke":
        active_concurrencies = concurrencies or [1, 2, 4]
        active_engines = engines or ["baseline", "minivllm-fp16", "minivllm-int8"]
        active_workloads = workloads or ["mixed"]
        req_count = num_requests_per_concurrency or 4
        reps = 1
    else:  # full profile
        active_concurrencies = concurrencies or [1, 2, 4, 8, 16]
        active_engines = engines or ["baseline", "minivllm-fp16", "minivllm-int8"]
        active_workloads = workloads or ["fixed", "mixed"]
        req_count = num_requests_per_concurrency or 16
        reps = repetitions

    print("\n" + "=" * 80)
    print("       🚀 MINI-VLLM RIGOROUS PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)
    print(f"  Profile:        {profile.upper()}")
    print(f"  Model:          {model_name}")
    print(f"  Compute Device: {device_str.upper()}")
    print(f"  Concurrencies:  {active_concurrencies}")
    print(f"  Engines:        {active_engines}")
    print(f"  Workloads:      {active_workloads}")
    print(f"  Requests/Batch: {req_count} requests per concurrency level")
    print(f"  Repetitions:    {reps}")
    print(f"  Output Dir:     {output_dir}")
    print("=" * 80 + "\n")

    # Pre-flight hardware guard check
    enforce_preflight_check(model_name, device=device_str, force=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_raw_records: List[RequestMetricRecord] = []
    summary_results: List[Dict[str, Any]] = []

    # Initialize Adapters
    baseline_adapter = BaselineAdapter(model_name, device_str)
    minivllm_adapter = MiniVLLMAdapter(model_name, device_str)

    # Total test step estimation
    total_runs = len(active_workloads) * len(active_concurrencies) * len(active_engines) * reps
    current_run_idx = 0

    print(f"Estimated Workload: {total_runs} benchmark runs total.\n")

    for wl_type in active_workloads:
        for conc in active_concurrencies:
            # Check safety: for conc >= 32 on 6GB GPU
            if conc > 16 and device_str == "cuda":
                print(f"⚠️ [Safety Notice] Evaluating concurrency {conc} with high VRAM guard...")

            for rep in range(1, reps + 1):
                # Generate identical workload requests for all engines in this cell
                if wl_type == "fixed":
                    requests = generate_fixed_workload(tokenizer, num_requests=req_count, seed=seed + rep)
                else:
                    requests = generate_mixed_workload(tokenizer, num_requests=req_count, seed=seed + rep)

                for eng_key in active_engines:
                    current_run_idx += 1
                    status_line = f"[{current_run_idx}/{total_runs}] Workload: {wl_type.upper():<5} | Conc: {conc:2d} | Rep: {rep}/{reps} | Engine: {eng_key:<14}"
                    print(f"  {status_line} ... ", end="", flush=True)

                    t0 = time.perf_counter()
                    try:
                        if eng_key == "baseline":
                            # Run baseline with FP16
                            records, wall_time, peak_vram, kv_mb, bubble = baseline_adapter.run_workload(
                                requests=requests,
                                concurrency=conc,
                                repetition=rep,
                                kv_cache_dtype="fp16"
                            )
                            engine_label = "baseline_fp16"
                            baseline_adapter.cleanup()

                        elif eng_key == "minivllm-fp16":
                            records, wall_time, peak_vram, kv_mb, bubble = minivllm_adapter.run_workload(
                                requests=requests,
                                concurrency=conc,
                                repetition=rep,
                                kv_cache_dtype="fp16"
                            )
                            engine_label = "minivllm_fp16"

                        elif eng_key == "minivllm-int8":
                            records, wall_time, peak_vram, kv_mb, bubble = minivllm_adapter.run_workload(
                                requests=requests,
                                concurrency=conc,
                                repetition=rep,
                                kv_cache_dtype="int8"
                            )
                            engine_label = "minivllm_int8"

                        elif eng_key == "baseline-int8":
                            # Explicitly handle unsupported request
                            print("SKIPPED (UNSUPPORTED: Baseline does not support INT8 KV cache)")
                            continue
                        else:
                            print(f"UNKNOWN ENGINE: {eng_key}")
                            continue

                        all_raw_records.extend(records)
                        run_summary = aggregate_run_metrics(
                            records=records,
                            wall_time_s=wall_time,
                            peak_vram_mb=peak_vram,
                            kv_cache_dedicated_mb=kv_mb,
                            bubble_percentage=bubble
                        )
                        run_summary["engine"] = engine_label
                        run_summary["workload_type"] = wl_type
                        run_summary["concurrency"] = conc
                        run_summary["repetition"] = rep

                        summary_results.append(run_summary)
                        dur = time.perf_counter() - t0
                        tps = run_summary["throughput"]["output_tokens_per_s"]
                        ttft = run_summary["ttft_ms"]["p50"]
                        tpot = run_summary["tpot_ms"]["p50"]
                        print(f"DONE ({dur:4.1f}s) -> {tps:5.1f} tok/s | TTFT P50: {ttft:4.1f}ms | TPOT P50: {tpot:4.1f}ms")

                    except Exception as e:
                        dur = time.perf_counter() - t0
                        print(f"FAILED ({dur:4.1f}s): {e}")

    # Clean up baseline model from GPU
    baseline_adapter.cleanup()

    # =========================================================================
    # EXPORT RAW RESULTS (JSONL & CSV)
    # =========================================================================
    raw_jsonl_path = os.path.join(output_dir, "raw_requests.jsonl")
    with open(raw_jsonl_path, "w") as f:
        for r in all_raw_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    raw_csv_path = os.path.join(output_dir, "raw_requests.csv")
    if all_raw_records:
        keys = list(all_raw_records[0].to_dict().keys())
        with open(raw_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_raw_records:
                writer.writerow(r.to_dict())

    # =========================================================================
    # EXPORT AGGREGATED JSON SUMMARY
    # =========================================================================
    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "profile": profile,
            "model": model_name,
            "device": device_str,
            "concurrencies": active_concurrencies,
            "summary_data": summary_results
        }, f, indent=2)

    # =========================================================================
    # GENERATE PLOTS
    # =========================================================================
    print("\n[Reporting] Generating comparative performance plots...")
    plot_paths = generate_benchmark_plots(summary_results, plots_dir)

    # =========================================================================
    # GENERATE MARKDOWN REPORT
    # =========================================================================
    print("[Reporting] Compiling executive Markdown report...")
    report_md_path = os.path.join(output_dir, "BENCHMARK_REPORT.md")
    generate_markdown_report(
        summary_data=summary_results,
        output_path=report_md_path,
        raw_csv_path=raw_csv_path,
        raw_jsonl_path=raw_jsonl_path,
        summary_json_path=summary_json_path,
        plot_paths=plot_paths,
        model_name=model_name,
        device_str=device_str,
        profile_mode=profile
    )

    print("\n" + "=" * 80)
    print("  🎉 BENCHMARK RUN COMPLETE!")
    print(f"  • Executive Markdown Report: {report_md_path}")
    print(f"  • Raw Per-Request CSV:       {raw_csv_path}")
    print(f"  • Aggregated JSON Summary:   {summary_json_path}")
    print(f"  • Visual Plots Directory:    {plots_dir}")
    print("=" * 80 + "\n")

    return output_dir, report_md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigorous Performance Benchmark Suite for Mini-vLLM")
    add_common_engine_args(parser)
    parser.add_argument("--profile", type=str, choices=["smoke", "full"], default="smoke",
                        help="Execution profile: 'smoke' (fast validation, ~15s) or 'full' (complete matrix)")
    parser.add_argument("--concurrencies", type=str, default=None,
                        help="Comma-separated concurrency levels to test (e.g. '1,2,4,8,16')")
    parser.add_argument("--engines", type=str, default=None,
                        help="Comma-separated engine keys: 'baseline,minivllm-fp16,minivllm-int8'")
    parser.add_argument("--workloads", type=str, default=None,
                        help="Comma-separated workloads: 'fixed,mixed'")
    parser.add_argument("--num-requests", type=int, default=None,
                        help="Number of requests per concurrency run")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="Number of independent repetitions per test cell (default: 1 for smoke, 3 for full)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable workload generation")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for artifacts")

    args = parser.parse_args()

    concurrency_list = [int(c.strip()) for c in args.concurrencies.split(",")] if args.concurrencies else None
    engine_list = [e.strip() for e in args.engines.split(",")] if args.engines else None
    workload_list = [w.strip() for w in args.workloads.split(",")] if args.workloads else None

    run_benchmark_suite(
        model_name=args.model,
        device_str=args.device,
        profile=args.profile,
        concurrencies=concurrency_list,
        engines=engine_list,
        workloads=workload_list,
        num_requests_per_concurrency=args.num_requests,
        repetitions=args.repetitions,
        output_dir=args.output_dir,
        seed=args.seed
    )
