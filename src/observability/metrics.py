"""
Production LLM Observability & Prometheus Telemetry Subsystem
--------------------------------------------------------------
Provides enterprise-grade telemetry, metrics tracking, and real-time
performance profiling for the Mini-vLLM inference engine:

1. Real-time Gauges:
   - vllm_gpu_cache_usage_factor (KV cache physical pool saturation)
   - vllm_num_requests_running (Active concurrent batch size)
   - vllm_num_requests_waiting (Pending scheduler queue depth)
   - vllm_prefix_cache_hit_rate (Radix deduplication efficiency)

2. Latency Distributions (p50, p90, p95, p99):
   - TTFT (Time to First Token / Prefill latency)
   - TPOT (Time Per Output Token / Inter-token decode jitter)
   - E2E Request Latency

3. Counters & Throughput:
   - vllm_prompt_tokens_total
   - vllm_generation_tokens_total
   - vllm_preemptions_total
   - Real-time Token Generation Throughput (tok/s)

4. Exporters:
   - Prometheus Plaintext Exporter (/metrics standard HTTP endpoint)
   - Terminal Telemetry Dashboard
"""

import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import threading

try:
    from prometheus_client import (
        Gauge, Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


def calculate_percentiles(values: List[float], percentiles: Tuple[float, ...] = (50, 90, 95, 99)) -> Dict[str, float]:
    """Calculates percentile values from a list of raw float measurements."""
    if not values:
        return {f"p{int(p)}": 0.0 for p in percentiles}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    results = {}
    for p in percentiles:
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            val = sorted_vals[int(k)]
        else:
            val = sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)
        results[f"p{int(p)}"] = val
    return results


class EngineMetricsTracker:
    """
    Centralized metrics tracking engine.
    Maintains thread-safe counters, latency histograms, and Prometheus gauges.
    """
    def __init__(self, model_name: str = "default", enable_prometheus: bool = True):
        self.model_name = model_name
        self._lock = threading.Lock()

        # Raw data stores for latency distributions
        self.ttft_records_ms: List[float] = []
        self.tpot_records_ms: List[float] = []
        self.e2e_records_ms: List[float] = []

        # Real-time state
        self.num_requests_running: int = 0
        self.num_requests_waiting: int = 0
        self.gpu_cache_usage_factor: float = 0.0
        self.prefix_cache_hit_rate: float = 0.0

        # Cumulative counters
        self.prompt_tokens_total: int = 0
        self.generation_tokens_total: int = 0
        self.requests_completed_total: int = 0
        self.preemptions_total: int = 0

        # Benchmark timers
        self.start_time: float = time.perf_counter()
        self.total_decode_time_seconds: float = 0.0

        # Prometheus registry and metrics
        self.registry = None
        self.prom_gauges: Dict[str, Any] = {}
        self.prom_counters: Dict[str, Any] = {}
        self.prom_histograms: Dict[str, Any] = {}

        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()

    def _init_prometheus_metrics(self):
        """Initializes Prometheus metric objects with clean labels."""
        self.registry = CollectorRegistry(auto_describe=True)

        self.prom_gauges["running"] = Gauge(
            "vllm_num_requests_running",
            "Number of requests currently being decoded in iteration batch",
            ["model"],
            registry=self.registry
        )
        self.prom_gauges["waiting"] = Gauge(
            "vllm_num_requests_waiting",
            "Number of requests currently queued in the waiting queue",
            ["model"],
            registry=self.registry
        )
        self.prom_gauges["gpu_cache"] = Gauge(
            "vllm_gpu_cache_usage_factor",
            "Fraction of physical KV cache blocks allocated in VRAM pool (0.0 to 1.0)",
            ["model"],
            registry=self.registry
        )
        self.prom_gauges["prefix_hits"] = Gauge(
            "vllm_prefix_cache_hit_rate",
            "Ratio of prefix tokens reused via Radix block deduplication",
            ["model"],
            registry=self.registry
        )

        self.prom_counters["prompt_tokens"] = Counter(
            "vllm_prompt_tokens_total",
            "Total number of prompt tokens processed across all requests",
            ["model"],
            registry=self.registry
        )
        self.prom_counters["gen_tokens"] = Counter(
            "vllm_generation_tokens_total",
            "Total number of generation/decode tokens emitted across all requests",
            ["model"],
            registry=self.registry
        )
        self.prom_counters["completed"] = Counter(
            "vllm_requests_completed_total",
            "Total number of successfully finished inference requests",
            ["model"],
            registry=self.registry
        )
        self.prom_counters["preemptions"] = Counter(
            "vllm_preemptions_total",
            "Total number of sequence preemptions due to memory starvation",
            ["model"],
            registry=self.registry
        )

        self.prom_histograms["ttft"] = Histogram(
            "vllm_time_to_first_token_seconds",
            "Time to First Token (TTFT / Prefill Latency) in seconds",
            ["model"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        self.prom_histograms["tpot"] = Histogram(
            "vllm_time_per_output_token_seconds",
            "Time Per Output Token (TPOT / Decode step latency) in seconds",
            ["model"],
            buckets=[0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5],
            registry=self.registry
        )
        self.prom_histograms["e2e"] = Histogram(
            "vllm_e2e_request_latency_seconds",
            "End-to-end request turnaround duration in seconds",
            ["model"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

    # -------------------------------------------------------------
    # Live Recording Methods (Invoked by LLM Engine)
    # -------------------------------------------------------------
    def record_queue_state(self, running: int, waiting: int):
        with self._lock:
            self.num_requests_running = running
            self.num_requests_waiting = waiting
            if self.registry:
                self.prom_gauges["running"].labels(model=self.model_name).set(running)
                self.prom_gauges["waiting"].labels(model=self.model_name).set(waiting)

    def record_cache_usage(self, allocated_blocks: int, total_blocks: int):
        with self._lock:
            factor = (allocated_blocks / total_blocks) if total_blocks > 0 else 0.0
            self.gpu_cache_usage_factor = factor
            if self.registry:
                self.prom_gauges["gpu_cache"].labels(model=self.model_name).set(factor)

    def record_prefix_cache_hit_rate(self, hit_tokens: int, total_prompt_tokens: int):
        with self._lock:
            rate = (hit_tokens / total_prompt_tokens) if total_prompt_tokens > 0 else 0.0
            self.prefix_cache_hit_rate = rate
            if self.registry:
                self.prom_gauges["prefix_hits"].labels(model=self.model_name).set(rate)

    def record_ttft(self, latency_ms: float):
        with self._lock:
            self.ttft_records_ms.append(latency_ms)
            if self.registry:
                self.prom_histograms["ttft"].labels(model=self.model_name).observe(latency_ms / 1000.0)

    def record_tpot(self, latency_ms: float):
        with self._lock:
            self.tpot_records_ms.append(latency_ms)
            if self.registry:
                self.prom_histograms["tpot"].labels(model=self.model_name).observe(latency_ms / 1000.0)

    def record_request_completed(
        self,
        prompt_tokens: int,
        generated_tokens: int,
        e2e_latency_ms: float
    ):
        with self._lock:
            self.prompt_tokens_total += prompt_tokens
            self.generation_tokens_total += generated_tokens
            self.requests_completed_total += 1
            self.e2e_records_ms.append(e2e_latency_ms)

            if self.registry:
                self.prom_counters["prompt_tokens"].labels(model=self.model_name).inc(prompt_tokens)
                self.prom_counters["gen_tokens"].labels(model=self.model_name).inc(generated_tokens)
                self.prom_counters["completed"].labels(model=self.model_name).inc(1)
                self.prom_histograms["e2e"].labels(model=self.model_name).observe(e2e_latency_ms / 1000.0)

    def record_preemption(self):
        with self._lock:
            self.preemptions_total += 1
            if self.registry:
                self.prom_counters["preemptions"].labels(model=self.model_name).inc(1)

    # -------------------------------------------------------------
    # Exporters & Dashboards
    # -------------------------------------------------------------
    def export_prometheus_text(self) -> str:
        """Exports metrics in official Prometheus plaintext exposition format."""
        if self.registry and PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode("utf-8")

        # Fallback manual exporter if prometheus_client is not installed
        lines = [
            f"# HELP vllm_num_requests_running Number of requests running in continuous batch.",
            f"# TYPE vllm_num_requests_running gauge",
            f'vllm_num_requests_running{{model="{self.model_name}"}} {self.num_requests_running}',
            f"# HELP vllm_num_requests_waiting Number of requests waiting in queue.",
            f"# TYPE vllm_num_requests_waiting gauge",
            f'vllm_num_requests_waiting{{model="{self.model_name}"}} {self.num_requests_waiting}',
            f"# HELP vllm_gpu_cache_usage_factor Fraction of KV cache blocks allocated.",
            f"# TYPE vllm_gpu_cache_usage_factor gauge",
            f'vllm_gpu_cache_usage_factor{{model="{self.model_name}"}} {self.gpu_cache_usage_factor:.4f}',
            f"# HELP vllm_prompt_tokens_total Total prompt tokens processed.",
            f"# TYPE vllm_prompt_tokens_total counter",
            f'vllm_prompt_tokens_total{{model="{self.model_name}"}} {self.prompt_tokens_total}',
            f"# HELP vllm_generation_tokens_total Total generation tokens produced.",
            f"# TYPE vllm_generation_tokens_total counter",
            f'vllm_generation_tokens_total{{model="{self.model_name}"}} {self.generation_tokens_total}',
            f"# HELP vllm_requests_completed_total Total requests successfully finished.",
            f"# TYPE vllm_requests_completed_total counter",
            f'vllm_requests_completed_total{{model="{self.model_name}"}} {self.requests_completed_total}',
            f"# HELP vllm_preemptions_total Total requests preempted.",
            f"# TYPE vllm_preemptions_total counter",
            f'vllm_preemptions_total{{model="{self.model_name}"}} {self.preemptions_total}',
        ]
        return "\n".join(lines) + "\n"

    def start_http_server(self, port: int = 9090):
        """Starts background Prometheus /metrics scraping HTTP daemon on the specified port."""
        if PROMETHEUS_AVAILABLE and self.registry:
            start_http_server(port=port, registry=self.registry)
            return True
        return False

    def render_ascii_dashboard(self, elapsed_seconds: Optional[float] = None) -> str:
        """Renders an ASCII telemetry dashboard summarizing engine health and latencies."""
        duration = elapsed_seconds or max(0.001, time.perf_counter() - self.start_time)
        throughput = self.generation_tokens_total / duration if duration > 0 else 0.0

        ttft_stats = calculate_percentiles(self.ttft_records_ms)
        tpot_stats = calculate_percentiles(self.tpot_records_ms)
        e2e_stats = calculate_percentiles(self.e2e_records_ms)

        cache_pct = self.gpu_cache_usage_factor * 100.0
        prefix_pct = self.prefix_cache_hit_rate * 100.0

        # Memory progress bar: [████████░░░░░░░░] 50.0%
        bar_len = 24
        filled = int((cache_pct / 100.0) * bar_len)
        bar_str = "█" * filled + "░" * (bar_len - filled)

        dashboard = f"""
┌────────────────────────────────────────────────────────────────────────┐
│                   📊 MINI-VLLM ENGINE TELEMETRY                        │
├────────────────────────────────────────────────────────────────────────┤
│  Model: {self.model_name:<30}  Throughput: {throughput:6.1f} tok/s       │
│  KV Cache Usage: [{bar_str}] {cache_pct:5.1f}%       │
│  Prefix Cache Hit Ratio:    {prefix_pct:5.1f}%       Queue: {self.num_requests_running} active / {self.num_requests_waiting} waiting │
├────────────────────────────────────────────────────────────────────────┤
│  TOKEN COUNTERS                                                        │
│  • Total Prompt Tokens:     {self.prompt_tokens_total:<10d} Completed Requests: {self.requests_completed_total:<6d}│
│  • Total Generated Tokens:  {self.generation_tokens_total:<10d} Preemptions Total:   {self.preemptions_total:<6d}│
├────────────────────────────────────────────────────────────────────────┤
│  LATENCY PERCENTILES (ms)       p50       p90       p95       p99      │
│  • TTFT (Time to First Token): {ttft_stats['p50']:7.1f}   {ttft_stats['p90']:7.1f}   {ttft_stats['p95']:7.1f}   {ttft_stats['p99']:7.1f}   │
│  • TPOT (Decode per Token):    {tpot_stats['p50']:7.1f}   {tpot_stats['p90']:7.1f}   {tpot_stats['p95']:7.1f}   {tpot_stats['p99']:7.1f}   │
│  • E2E Request Duration:       {e2e_stats['p50']:7.1f}   {e2e_stats['p90']:7.1f}   {e2e_stats['p95']:7.1f}   {e2e_stats['p99']:7.1f}   │
└────────────────────────────────────────────────────────────────────────┘
"""
        return dashboard.strip()
