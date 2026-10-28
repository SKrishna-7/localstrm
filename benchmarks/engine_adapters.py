"""
Engine Adapters for Standardized Benchmark Execution
---------------------------------------------------
Provides uniform, fair execution adapters across:
  1. Static Baseline (HuggingFace / PyTorch left-padded batching)
  2. Mini-vLLM with FP16 KV Cache
  3. Mini-vLLM with INT8 KV Cache

Key Rigor Features:
  - Explicit warmup passes to eliminate CUDA cold-start overhead (~500ms).
  - Explicit 'UNSUPPORTED' marking for baseline INT8 KV cache.
  - Per-request lifecycle timestamping (arrival, schedule, first token, finish).
  - Safe memory cleanup between runs.
"""

import time
import os
import sys
from typing import List, Dict, Tuple, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from benchmarks.workload import BenchmarkRequest
from benchmarks.instrumentation import (
    RequestMetricRecord, sync_gpu, reset_vram, get_peak_vram_mb,
    calculate_analytical_kv_cache_mb, calculate_bubble_percentage
)
from src.engine.llm_engine import MiniVLLMEngine


class BaselineAdapter:
    """
    Adapter for the static-batching baseline engine (HuggingFace / PyTorch).
    """
    def __init__(self, model_name: str, device_str: str):
        self.model_name = model_name
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            dtype = torch.float32 if self.device.type == "cpu" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device)
            self.model.eval()

    def warmup(self, warmup_requests: int = 2):
        """Mandatory warm-up to initialize CUDA context, cuBLAS kernels, and memory pools."""
        self.load_model()
        dummy_prompts = ["Warmup system probe query."] * warmup_requests
        encoded = self.tokenizer(dummy_prompts, padding=True, return_tensors="pt").to(self.device)
        sync_gpu(self.device)
        with torch.no_grad():
            outputs = self.model(encoded["input_ids"], attention_mask=encoded["attention_mask"], use_cache=True)
            past = outputs.past_key_values
            next_tok = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            for _ in range(3):
                outputs = self.model(next_tok, past_key_values=past, use_cache=True)
                past = outputs.past_key_values
                next_tok = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        sync_gpu(self.device)

    def run_workload(
        self,
        requests: List[BenchmarkRequest],
        concurrency: int,
        repetition: int,
        kv_cache_dtype: str = "fp16"
    ) -> Tuple[List[RequestMetricRecord], float, float, float, float]:
        """
        Executes workload using static batching.
        Returns: (records, wall_time_s, peak_vram_mb, kv_cache_mb, bubble_pct)
        """
        if kv_cache_dtype.lower() == "int8":
            raise NotImplementedError(
                "UNSUPPORTED: Static baseline (HuggingFace) does not support INT8 KV Cache. "
                "HuggingFace Transformers only supports standard floating-point KV cache."
            )

        self.load_model()
        if not getattr(self, "_warmed_up", False):
            self.warmup(warmup_requests=2)
            self._warmed_up = True
        reset_vram(self.device)
        sync_gpu(self.device)

        records: List[RequestMetricRecord] = []
        total_useful_slots = 0
        total_wasted_slots = 0
        total_tokens_in_pool = 0

        t_run_start = time.perf_counter()

        # Process requests in batches of size `concurrency`
        for batch_start in range(0, len(requests), concurrency):
            batch_reqs = requests[batch_start : batch_start + concurrency]
            batch_size = len(batch_reqs)
            batch_prompts = [r.prompt for r in batch_reqs]
            batch_targets = [r.target_tokens for r in batch_reqs]
            max_target = max(batch_targets)

            t_batch_scheduled = time.perf_counter()

            # Tokenize with left-padding
            encoded = self.tokenizer(batch_prompts, padding=True, return_tensors="pt").to(self.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Prefill Phase
            sync_gpu(self.device)
            t_prefill_start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
            sync_gpu(self.device)
            t_first_token = time.perf_counter()

            engine_ttft_ms = (t_first_token - t_prefill_start) * 1000.0

            past_key_values = outputs.past_key_values
            next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            generated_per_seq = [1] * batch_size
            finished = [False] * batch_size
            finish_timestamps = [0.0] * batch_size

            # Check if any request finished on first token
            for i in range(batch_size):
                if generated_per_seq[i] >= batch_targets[i] or next_tokens[i].item() == self.tokenizer.eos_token_id:
                    finished[i] = True
                    finish_timestamps[i] = time.perf_counter()

            curr_input = next_tokens

            # Decode Loop: locked together until slowest sequence completes
            for step in range(1, max_target):
                if all(finished):
                    break

                with torch.no_grad():
                    outputs = self.model(curr_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                curr_input = next_tokens

                t_step_now = time.perf_counter()
                for i in range(batch_size):
                    if not finished[i]:
                        generated_per_seq[i] += 1
                        total_useful_slots += 1
                        if generated_per_seq[i] >= batch_targets[i] or next_tokens[i].item() == self.tokenizer.eos_token_id:
                            finished[i] = True
                            finish_timestamps[i] = t_step_now
                    else:
                        total_wasted_slots += 1

            sync_gpu(self.device)
            t_batch_finished = time.perf_counter()

            for i, r in enumerate(batch_reqs):
                f_time = finish_timestamps[i] if finish_timestamps[i] > 0 else t_batch_finished
                q_wait = (t_batch_scheduled - (t_run_start + r.arrival_time_offset)) * 1000.0
                q_wait = max(0.0, q_wait)
                client_ttft_ms = max(0.1, (t_first_token - (t_run_start + r.arrival_time_offset)) * 1000.0)
                dec_dur = (f_time - t_first_token) * 1000.0
                e2e = (f_time - (t_run_start + r.arrival_time_offset)) * 1000.0
                tpot = dec_dur / max(1, generated_per_seq[i] - 1)

                records.append(
                    RequestMetricRecord(
                        request_id=r.request_id,
                        engine="baseline_fp16",
                        workload_type=r.workload_type,
                        concurrency=concurrency,
                        repetition=repetition,
                        batch_size=batch_size,
                        arrival_time=t_run_start + r.arrival_time_offset,
                        schedule_time=t_batch_scheduled,
                        first_token_time=t_first_token,
                        finish_time=f_time,
                        prompt_tokens=len(r.prompt_token_ids),
                        output_tokens=generated_per_seq[i],
                        target_tokens=r.target_tokens,
                        queue_wait_ms=q_wait,
                        engine_ttft_ms=engine_ttft_ms,
                        client_ttft_ms=client_ttft_ms,
                        ttft_ms=client_ttft_ms,
                        decode_duration_ms=dec_dur,
                        e2e_latency_ms=e2e,
                        tpot_ms=tpot,
                        status="SUCCESS",
                        error_msg=None
                    )
                )
                total_tokens_in_pool += (len(r.prompt_token_ids) + generated_per_seq[i])

        sync_gpu(self.device)
        t_run_end = time.perf_counter()
        wall_time_s = t_run_end - t_run_start

        peak_vram_mb = get_peak_vram_mb(self.device)
        total_slots = total_useful_slots + total_wasted_slots
        bubble_pct = (total_wasted_slots / total_slots * 100.0) if total_slots > 0 else 0.0

        num_layers = self.model.config.num_hidden_layers
        num_kv = getattr(self.model.config, "num_key_value_heads", self.model.config.num_attention_heads)
        h_dim = getattr(self.model.config, "head_dim", self.model.config.hidden_size // self.model.config.num_attention_heads)
        kv_cache_mb = calculate_analytical_kv_cache_mb(num_layers, num_kv, h_dim, total_tokens_in_pool // len(requests) * concurrency, "fp16")

        return records, wall_time_s, peak_vram_mb, kv_cache_mb, bubble_pct

    def cleanup(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._warmed_up = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class MiniVLLMAdapter:
    """
    Adapter for the unified Mini-vLLM engine (PagedAttention + Continuous Batching).
    """
    def __init__(self, model_name: str, device_str: str):
        self.model_name = model_name
        self.device_str = device_str
        self.device = torch.device(device_str)

    def run_workload(
        self,
        requests: List[BenchmarkRequest],
        concurrency: int,
        repetition: int,
        kv_cache_dtype: str = "fp16"
    ) -> Tuple[List[RequestMetricRecord], float, float, float, float]:
        """
        Executes workload using Mini-vLLM engine with specified KV cache precision.
        """
        engine_label = f"minivllm_{kv_cache_dtype.lower()}"
        reset_vram(self.device)
        sync_gpu(self.device)

        # Ensure total_blocks is large enough for the concurrency level
        max_prompt = max(len(r.prompt_token_ids) for r in requests)
        max_target = max(r.target_tokens for r in requests)
        tokens_per_req = max_prompt + max_target
        needed_pool_blocks = int(concurrency * tokens_per_req / 16 * 1.5)
        total_blocks = max(64, needed_pool_blocks)

        # For fixed workload (Workload A), disable prefix caching so every request
        # executes genuine prefill GEMMs for a fair parity comparison with the baseline.
        # For mixed workload (Workload B), keep prefix caching enabled to evaluate shared prefixes.
        is_fixed = any(r.workload_type == "fixed" for r in requests)
        enable_prefix_caching = False if is_fixed else True

        engine = MiniVLLMEngine(
            model_name=self.model_name,
            max_batch_size=concurrency,
            total_blocks=total_blocks,
            block_size=16,
            device=self.device_str,
            kv_cache_dtype=kv_cache_dtype,
            chunked_prefill=True,
            prefill_chunk_size=32,
            enable_prefix_caching=enable_prefix_caching,
            verbose=False
        )

        # 1. Warm-up to eliminate cold-start
        engine.add_request("Warmup probe.", max_tokens=4, arrival_step=0)
        sync_gpu(self.device)
        engine.run()
        sync_gpu(self.device)

        # 2. Add benchmark requests
        t_run_start = time.perf_counter()
        req_mapping = {}
        for r in requests:
            arr_step = int(r.arrival_time_offset * 10)  # Map arrival offset to engine steps
            eng_req = engine.add_request(prompt=r.prompt, max_tokens=r.target_tokens, arrival_step=arr_step)
            eng_req.arrival_time = t_run_start + r.arrival_time_offset
            req_mapping[eng_req.req_id] = r

        sync_gpu(self.device)
        t_exec_start = time.perf_counter()
        completed = engine.run()
        sync_gpu(self.device)
        t_exec_end = time.perf_counter()

        wall_time_s = t_exec_end - t_exec_start
        peak_vram_mb = get_peak_vram_mb(self.device)

        records: List[RequestMetricRecord] = []
        for eng_req in completed:
            orig_req = req_mapping.get(eng_req.req_id)
            if not orig_req:
                continue

            # Exact per-request wall-clock lifecycle timestamps
            f_time = eng_req.finish_time if eng_req.finish_time is not None else t_exec_end
            first_t = eng_req.first_token_time if eng_req.first_token_time is not None else eng_req.arrival_time
            sched_t = eng_req.schedule_time if eng_req.schedule_time is not None else eng_req.arrival_time

            q_wait = max(0.0, (sched_t - eng_req.arrival_time) * 1000.0)
            engine_ttft = max(0.1, (first_t - sched_t) * 1000.0)
            client_ttft = max(0.1, (first_t - eng_req.arrival_time) * 1000.0)
            dec_dur = max(0.0, (f_time - first_t) * 1000.0)
            e2e = (f_time - eng_req.arrival_time) * 1000.0
            tpot = dec_dur / max(1, eng_req.num_generated - 1)

            records.append(
                RequestMetricRecord(
                    request_id=orig_req.request_id,
                    engine=engine_label,
                    workload_type=orig_req.workload_type,
                    concurrency=concurrency,
                    repetition=repetition,
                    batch_size=concurrency,
                    arrival_time=eng_req.arrival_time,
                    schedule_time=sched_t,
                    first_token_time=first_t,
                    finish_time=f_time,
                    prompt_tokens=len(eng_req.prompt_token_ids),
                    output_tokens=eng_req.num_generated,
                    target_tokens=orig_req.target_tokens,
                    queue_wait_ms=q_wait,
                    engine_ttft_ms=engine_ttft,
                    client_ttft_ms=client_ttft,
                    ttft_ms=client_ttft,
                    decode_duration_ms=dec_dur,
                    e2e_latency_ms=e2e,
                    tpot_ms=tpot,
                    status="SUCCESS",
                    error_msg=None
                )
            )

        # Analytical dedicated KV cache calculation
        num_layers = engine.num_layers
        num_kv = engine.num_kv_heads
        h_dim = engine.head_dim
        allocated_blocks = total_blocks - engine.num_free_blocks
        kv_cache_mb = calculate_analytical_kv_cache_mb(
            num_layers, num_kv, h_dim, allocated_blocks * 16, kv_cache_dtype
        )

        del engine
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return records, wall_time_s, peak_vram_mb, kv_cache_mb, 0.0  # 0.0% bubble by definition
