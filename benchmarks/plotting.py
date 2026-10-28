"""
Performance Benchmark Plotting Subsystem
-----------------------------------------
Generates presentation-grade performance charts:
  1. Throughput vs Concurrency (Tokens/Second)
  2. TTFT (P50 and P95) vs Concurrency (ms)
  3. TPOT (P50 and P95) vs Concurrency (ms)
  4. VRAM Footprint vs Concurrency (MB)
"""

import os
from typing import Dict, List, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt


def generate_benchmark_plots(
    summary_data: List[Dict[str, Any]],
    output_dir: str
) -> Dict[str, str]:
    """
    Renders 4 publication-quality comparative benchmark plots and saves them as PNGs.
    Returns: mapping of plot name to file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = {}

    # Separate records by engine and workload
    # We plot the primary workload (or mixed workload if both present)
    engines = ["baseline_fp16", "minivllm_fp16", "minivllm_int8"]
    engine_labels = {
        "baseline_fp16": "Baseline Static (FP16)",
        "minivllm_fp16": "Mini-vLLM (FP16 KV)",
        "minivllm_int8": "Mini-vLLM (INT8 KV)"
    }
    colors = {
        "baseline_fp16": "#e74c3c",   # Red
        "minivllm_fp16": "#2ecc71",   # Green
        "minivllm_int8": "#3498db"    # Blue
    }
    markers = {
        "baseline_fp16": "o",
        "minivllm_fp16": "s",
        "minivllm_int8": "^"
    }

    # Set universal aesthetic
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#bdc3c7"
    plt.rcParams["axes.linewidth"] = 0.8

    # =========================================================================
    # PLOT 1: Throughput (Tokens/Second) vs Concurrency
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for eng in engines:
        eng_data = [d for d in summary_data if d["engine"] == eng]
        if not eng_data:
            continue
        eng_data = sorted(eng_data, key=lambda x: x["concurrency"])
        concs = [d["concurrency"] for d in eng_data]
        tps = [d["throughput"]["output_tokens_per_s"] for d in eng_data]
        ax.plot(
            concs, tps,
            label=engine_labels.get(eng, eng),
            color=colors.get(eng, "#7f8c8d"),
            marker=markers.get(eng, "o"),
            linewidth=2.2,
            markersize=7
        )

    ax.set_title("Inference Throughput vs Concurrency", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=11, labelpad=8)
    ax.set_ylabel("Output Throughput (Tokens / Second)", fontsize=11, labelpad=8)
    ax.set_xticks(sorted(list(set(d["concurrency"] for d in summary_data))))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()
    p1_path = os.path.join(output_dir, "throughput_vs_concurrency.png")
    fig.savefig(p1_path)
    plt.close(fig)
    generated_plots["throughput"] = p1_path

    # =========================================================================
    # PLOT 2: TTFT (Time to First Token: P50 & P95) vs Concurrency
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for eng in engines:
        eng_data = [d for d in summary_data if d["engine"] == eng]
        if not eng_data:
            continue
        eng_data = sorted(eng_data, key=lambda x: x["concurrency"])
        concs = [d["concurrency"] for d in eng_data]
        p50 = [d["ttft_ms"]["p50"] for d in eng_data]
        p95 = [d["ttft_ms"]["p95"] for d in eng_data]

        col = colors.get(eng, "#7f8c8d")
        ax.plot(concs, p50, label=f"{engine_labels.get(eng, eng)} (P50)", color=col,
                marker=markers.get(eng, "o"), linewidth=2.0, markersize=6)
        ax.plot(concs, p95, label=f"{engine_labels.get(eng, eng)} (P95)", color=col,
                linestyle="--", marker=markers.get(eng, "o"), linewidth=1.5, markersize=5, alpha=0.75)

    ax.set_title("Time to First Token (TTFT) vs Concurrency", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=11, labelpad=8)
    ax.set_ylabel("TTFT Latency (Milliseconds)", fontsize=11, labelpad=8)
    ax.set_xticks(sorted(list(set(d["concurrency"] for d in summary_data))))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#ffffff", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    p2_path = os.path.join(output_dir, "ttft_vs_concurrency.png")
    fig.savefig(p2_path)
    plt.close(fig)
    generated_plots["ttft"] = p2_path

    # =========================================================================
    # PLOT 3: TPOT (Time Per Output Token: P50 & P95) vs Concurrency
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for eng in engines:
        eng_data = [d for d in summary_data if d["engine"] == eng]
        if not eng_data:
            continue
        eng_data = sorted(eng_data, key=lambda x: x["concurrency"])
        concs = [d["concurrency"] for d in eng_data]
        p50 = [d["tpot_ms"]["p50"] for d in eng_data]
        p95 = [d["tpot_ms"]["p95"] for d in eng_data]

        col = colors.get(eng, "#7f8c8d")
        ax.plot(concs, p50, label=f"{engine_labels.get(eng, eng)} (P50)", color=col,
                marker=markers.get(eng, "o"), linewidth=2.0, markersize=6)
        ax.plot(concs, p95, label=f"{engine_labels.get(eng, eng)} (P95)", color=col,
                linestyle="--", marker=markers.get(eng, "o"), linewidth=1.5, markersize=5, alpha=0.75)

    ax.set_title("Time per Output Token (TPOT) vs Concurrency", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=11, labelpad=8)
    ax.set_ylabel("TPOT Latency (Milliseconds / Token)", fontsize=11, labelpad=8)
    ax.set_xticks(sorted(list(set(d["concurrency"] for d in summary_data))))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#ffffff", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    p3_path = os.path.join(output_dir, "tpot_vs_concurrency.png")
    fig.savefig(p3_path)
    plt.close(fig)
    generated_plots["tpot"] = p3_path

    # =========================================================================
    # PLOT 4: Peak VRAM vs Concurrency
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for eng in engines:
        eng_data = [d for d in summary_data if d["engine"] == eng]
        if not eng_data:
            continue
        eng_data = sorted(eng_data, key=lambda x: x["concurrency"])
        concs = [d["concurrency"] for d in eng_data]
        vrams = [d["vram"]["peak_process_mb"] for d in eng_data]
        ax.plot(
            concs, vrams,
            label=engine_labels.get(eng, eng),
            color=colors.get(eng, "#7f8c8d"),
            marker=markers.get(eng, "o"),
            linewidth=2.2,
            markersize=7
        )

    ax.set_title("Total Peak GPU VRAM Footprint vs Concurrency", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=11, labelpad=8)
    ax.set_ylabel("Peak VRAM (Megabytes)", fontsize=11, labelpad=8)
    ax.set_xticks(sorted(list(set(d["concurrency"] for d in summary_data))))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()
    p4_path = os.path.join(output_dir, "vram_vs_concurrency.png")
    fig.savefig(p4_path)
    plt.close(fig)
    generated_plots["vram"] = p4_path

    return generated_plots
