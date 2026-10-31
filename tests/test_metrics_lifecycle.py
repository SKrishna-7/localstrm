"""
Tests for Metrics Lifecycle Consistency and Queue/KV-Cache Gauge Synchronization.
----------------------------------------------------------------------------------
Phase 8.5.2 Regression Suite verifying:
  A. Single-request idle recovery (queues, metrics, Prometheus gauges return to 0).
  B. Concurrent-batch recovery (multi-request batch tracks actively and recovers to 0).
  C. Admission rejection (impossible capacity request rejected, gauges reflect 0).
  D. Decode-time allocation failure (isolated failure evicts failing sequence, updates gauges, preserves peer).
  E. Request cancellation (explicit abort on waiting and running requests updates metrics).
  F. Catastrophic engine cleanup (model execution error aborts batch, metrics return to idle 0).
  G. API-facing verification (FastAPI /health and /metrics report 0 idle state post-generation, post-rejection, and post-cancellation).
"""

from __future__ import annotations

import time
from typing import Any
import pytest
import torch
from fastapi.testclient import TestClient

from src.engine.llm_engine import (
    MiniVLLMEngine,
    RequestStatus,
    SamplingParams,
)
from src.server.app import create_app
from src.server.serving_adapter import AsyncServingAdapter
from tests.test_memory_constrained import make_test_engine, MockTokenizer, create_test_model


# =========================================================================
# Test A: Single-request idle recovery
# =========================================================================
def test_single_request_idle_recovery():
    """Verify that a single completed request clears queues and restores metrics to 0."""
    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=4)
    model_name = engine.metrics.model_name
    initial_free = engine.num_free_blocks

    # Pre-execution idle checks
    assert len(engine.running_batch) == 0
    assert len(engine.waiting_queue) == 0
    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor == 0.0

    req = engine.add_request("hello world", max_tokens=2, sampling_params=SamplingParams.greedy())
    assert engine.metrics.num_requests_waiting == 1
    assert engine.metrics.num_requests_running == 0

    # Step: admits and decodes
    while engine.step():
        assert engine.metrics.num_requests_running >= 1
        assert engine.metrics.num_requests_waiting == 0
        assert engine.metrics.gpu_cache_usage_factor > 0.0

    # Terminal state checks
    assert req.status is RequestStatus.FINISHED
    assert len(engine.running_batch) == 0
    assert len(engine.waiting_queue) == 0
    assert engine.num_free_blocks == initial_free

    # Metric and Prometheus synchronization checks
    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor == 0.0

    prom_text = engine.metrics.export_prometheus_text()
    assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in prom_text
    assert f'vllm_num_requests_waiting{{model="{model_name}"}} 0.0' in prom_text
    assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in prom_text


# =========================================================================
# Test B: Concurrent-batch recovery
# =========================================================================
def test_concurrent_batch_recovery():
    """Verify that a multi-request concurrent batch tracks running count and drains to 0."""
    engine = make_test_engine(total_blocks=16, block_size=4, max_batch_size=4)
    model_name = engine.metrics.model_name
    initial_free = engine.num_free_blocks

    r1 = engine.add_request("hello", max_tokens=6, sampling_params=SamplingParams.greedy())
    r2 = engine.add_request("world", max_tokens=7, sampling_params=SamplingParams.greedy())
    r3 = engine.add_request("antigravity", max_tokens=8, sampling_params=SamplingParams.greedy())

    assert engine.metrics.num_requests_waiting == 3
    assert engine.metrics.num_requests_running == 0

    # First step: admits all 3 (none finish in step 0 since max_tokens >= 6)
    engine.step()
    assert len(engine.running_batch) == 3
    assert engine.metrics.num_requests_running == 3
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor > 0.0

    # Run loop to drain
    while engine.step():
        assert 0 <= engine.metrics.num_requests_running <= 3
        assert engine.metrics.num_requests_running == len(engine.running_batch)

    assert r1.status is RequestStatus.FINISHED
    assert r2.status is RequestStatus.FINISHED
    assert r3.status is RequestStatus.FINISHED

    assert len(engine.running_batch) == 0
    assert len(engine.waiting_queue) == 0
    assert engine.num_free_blocks == initial_free

    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor == 0.0

    prom_text = engine.metrics.export_prometheus_text()
    assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in prom_text
    assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in prom_text


# =========================================================================
# Test C: Admission rejection
# =========================================================================
def test_admission_rejection_metrics():
    """Verify that impossible-capacity admission failure does not leave phantom gauges."""
    engine = make_test_engine(total_blocks=2, block_size=4, max_batch_size=2)
    model_name = engine.metrics.model_name
    initial_free = engine.num_free_blocks

    # 12 tokens prompt + 4 tokens max = 16 tokens -> 4 blocks > total_blocks (2)
    impossible_prompt = "hello " * 12
    req = engine.add_request(impossible_prompt, max_tokens=4)
    assert engine.metrics.num_requests_waiting == 1
    assert engine.metrics.num_requests_running == 0

    with pytest.raises(MemoryError, match="requires 4 KV blocks, but the pool has only 2"):
        engine.step()

    # Rejection verification
    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "kv_cache_full"
    assert len(engine.waiting_queue) == 0
    assert len(engine.running_batch) == 0
    assert engine.num_free_blocks == initial_free

    # Gauge verification: must return to 0 running, 0 waiting, 0.0 cache
    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor == 0.0

    prom_text = engine.metrics.export_prometheus_text()
    assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in prom_text
    assert f'vllm_num_requests_waiting{{model="{model_name}"}} 0.0' in prom_text
    assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in prom_text


# =========================================================================
# Test D: Decode-time allocation failure
# =========================================================================
def test_decode_allocation_failure_metrics():
    """Verify decode block allocation failure evicts failed request and updates metrics."""
    engine = make_test_engine(total_blocks=4, block_size=4, max_batch_size=2)
    model_name = engine.metrics.model_name
    initial_free = engine.num_free_blocks

    req_healthy = engine.add_request("hello", max_tokens=5, sampling_params=SamplingParams.greedy())
    req_greedy = engine.add_request("hello world", max_tokens=5, sampling_params=SamplingParams.greedy())

    # Step 0: Admits both
    engine.step()
    assert len(engine.running_batch) == 2
    assert engine.metrics.num_requests_running == 2

    # Inject failure on next allocation
    original_allocate = engine._allocate_block

    def constrained_allocate() -> int:
        raise MemoryError("Out of KV Cache physical blocks!")

    engine._allocate_block = constrained_allocate

    # Step 1: req_greedy needs a second block and fails; req_healthy does not
    engine.step()

    assert req_greedy.status is RequestStatus.FAILED
    assert req_greedy.stop_reason == "kv_cache_full"
    assert req_greedy not in engine.running_batch
    assert req_healthy in engine.running_batch

    # Metrics must now reflect exactly 1 running request and cache for 1 request
    assert engine.metrics.num_requests_running == 1
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor == (len(req_healthy.physical_block_ids) / engine.total_blocks)

    # Restore allocator and finish req_healthy
    engine._allocate_block = original_allocate
    while engine.step():
        pass

    assert req_healthy.status is RequestStatus.FINISHED
    assert len(engine.running_batch) == 0
    assert engine.num_free_blocks == initial_free

    # All metrics must be back to idle 0
    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor == 0.0

    prom_text = engine.metrics.export_prometheus_text()
    assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in prom_text
    assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in prom_text


# =========================================================================
# Test E: Request cancellation
# =========================================================================
def test_request_cancellation_metrics():
    """Verify abort_request updates queue, running, and cache metrics for waiting and running requests."""
    engine = make_test_engine(total_blocks=4, block_size=4, max_batch_size=1)
    model_name = engine.metrics.model_name
    initial_free = engine.num_free_blocks

    req1 = engine.add_request("first running request", max_tokens=5)
    req2 = engine.add_request("second waiting request", max_tokens=5)

    assert engine.metrics.num_requests_waiting == 2
    assert engine.metrics.num_requests_running == 0

    # Step: admits req1 (batch size 1), req2 remains waiting
    engine.step()
    assert req1 in engine.running_batch
    assert req2 in engine.waiting_queue
    assert engine.metrics.num_requests_running == 1
    assert engine.metrics.num_requests_waiting == 1
    assert engine.metrics.gpu_cache_usage_factor > 0.0

    # 1. Abort waiting request
    success = engine.abort_request(req2.req_id)
    assert success is True
    assert req2.status is RequestStatus.FAILED
    assert req2 not in engine.waiting_queue
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.num_requests_running == 1

    # 2. Abort running request
    success = engine.abort_request(req1.req_id)
    assert success is True
    assert req1.status is RequestStatus.FAILED
    assert req1 not in engine.running_batch
    assert engine.num_free_blocks == initial_free

    # Metrics must immediately reflect 0 running, 0 waiting, 0.0 cache
    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor == 0.0

    prom_text = engine.metrics.export_prometheus_text()
    assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in prom_text
    assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in prom_text


# =========================================================================
# Test F: Catastrophic engine cleanup
# =========================================================================
def test_catastrophic_engine_cleanup_metrics():
    """Verify that model execution failure aborts all running requests and clears gauges."""
    engine = make_test_engine(total_blocks=4, block_size=4, max_batch_size=2)
    model_name = engine.metrics.model_name
    initial_free = engine.num_free_blocks

    req = engine.add_request("hello world", max_tokens=4)

    # Inject model forward failure
    def raising_forward(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated forward pass crash")

    engine.model.side_effect = raising_forward

    with pytest.raises(RuntimeError, match="simulated forward pass crash"):
        engine.step()

    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "error"
    assert len(engine.running_batch) == 0
    assert len(engine.waiting_queue) == 0
    assert engine.num_free_blocks == initial_free

    # Gauges must return to idle 0
    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.num_requests_waiting == 0
    assert engine.metrics.gpu_cache_usage_factor == 0.0

    prom_text = engine.metrics.export_prometheus_text()
    assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in prom_text
    assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in prom_text


# =========================================================================
# Test G: API-facing verification
# =========================================================================
def test_api_health_and_metrics_idle_recovery():
    """Verify /health and /metrics report 0 running requests and 0.0 cache through FastAPI."""
    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=2)
    model_name = engine.metrics.model_name
    adapter = AsyncServingAdapter(engine)
    app = create_app(adapter)

    try:
        with TestClient(app) as client:
            # 1. Initial health and metrics
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["running_requests"] == 0

            resp_m = client.get("/metrics")
            assert resp_m.status_code == 200
            assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in resp_m.text
            assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in resp_m.text

            # 2. Complete non-streaming chat request
            resp_chat = client.post(
                "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hello world"}],
                    "max_tokens": 3
                }
            )
            assert resp_chat.status_code == 200

            # Wait briefly for adapter channel cleanup
            time.sleep(0.05)

            # Post-completion: /health and /metrics must report 0 running requests!
            resp_post = client.get("/health")
            assert resp_post.status_code == 200
            assert resp_post.json()["running_requests"] == 0

            resp_m_post = client.get("/metrics")
            assert resp_m_post.status_code == 200
            assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in resp_m_post.text
            assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in resp_m_post.text

            # 3. Capacity rejection via API
            # Prompt requiring more than 8 blocks (8 * 4 = 32 tokens)
            long_prompt = "hello " * 40
            resp_rej = client.post(
                "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": long_prompt}],
                    "max_tokens": 10
                }
            )
            assert resp_rej.status_code == 503
            assert resp_rej.json()["error"]["code"] == "kv_cache_full"

            time.sleep(0.05)

            resp_post_rej = client.get("/health")
            assert resp_post_rej.status_code == 200
            assert resp_post_rej.json()["running_requests"] == 0

            # 4. Client disconnect cancellation
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "stream cancel test"}],
                    "max_tokens": 10,
                    "stream": True
                }
            ) as stream_resp:
                assert stream_resp.status_code == 200
                # Consume single chunk then close
                for line in stream_resp.iter_lines():
                    if line:
                        break

            # Allow worker thread to process abort command
            time.sleep(0.1)

            resp_post_cancel = client.get("/health")
            assert resp_post_cancel.status_code == 200
            assert resp_post_cancel.json()["running_requests"] == 0

            resp_m_cancel = client.get("/metrics")
            assert resp_m_cancel.status_code == 200
            assert f'vllm_num_requests_running{{model="{model_name}"}} 0.0' in resp_m_cancel.text
            assert f'vllm_gpu_cache_usage_factor{{model="{model_name}"}} 0.0' in resp_m_cancel.text

    finally:
        adapter.shutdown_sync()


# =========================================================================
# Test H: Exact Token Accounting without Text Re-encoding Skew
# =========================================================================
def test_token_accounting_exact_counts_without_reencoding():
    """
    Verify completion_tokens is derived from actual generated token IDs rather than
    re-encoding full_text.
    """
    class MergeTokenizer:
        """Tokenizer where decoding and re-encoding changes token count (e.g. whitespace merging)."""
        def __init__(self):
            self.eos_token_id = 999
            self.chat_template = None

        def encode(self, text: str) -> list[int]:
            if not text:
                return []
            # Splits on spaces: "a b" -> 2 tokens, but "  " merges into 1 token!
            return [10] * len(text.split())

        def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
            # Emits spaces between tokens
            return " " * len(token_ids)

    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=2)
    engine.tokenizer = MergeTokenizer()
    adapter = AsyncServingAdapter(engine)

    try:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            full_text, prompt_len, gen_tokens, finish_reason = loop.run_until_complete(
                adapter.generate_complete("hello world", max_tokens=3)
            )
            # Generated 3 tokens. Re-encoding full_text ("   ") with MergeTokenizer gives 0 or 1 tokens!
            # Exact token count must be 3, proving no re-encoding was performed!
            assert gen_tokens == 3, f"Expected exactly 3 generated tokens, got {gen_tokens}"
            assert prompt_len == 2, f"Expected 2 prompt tokens for 'hello world', got {prompt_len}"
            assert finish_reason in ("stop", "length")
        finally:
            loop.close()
    finally:
        adapter.shutdown_sync()


# =========================================================================
# Test I: Streaming vs Non-Streaming Token Parity and Stream Options Usage
# =========================================================================
def test_streaming_usage_and_parity_with_non_streaming():
    """Verify stream_options.include_usage emits exact token usage matching non-streaming."""
    import json
    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=2)
    model_name = engine.metrics.model_name
    adapter = AsyncServingAdapter(engine)
    app = create_app(adapter)

    try:
        with TestClient(app) as client:
            prompt_content = "hello test parity"
            max_toks = 4

            # 1. Non-streaming call
            resp_non_stream = client.post(
                "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt_content}],
                    "max_tokens": max_toks,
                    "stream": False
                }
            )
            assert resp_non_stream.status_code == 200
            non_stream_usage = resp_non_stream.json()["usage"]
            assert non_stream_usage["prompt_tokens"] > 0
            assert non_stream_usage["completion_tokens"] == max_toks
            assert non_stream_usage["total_tokens"] == non_stream_usage["prompt_tokens"] + non_stream_usage["completion_tokens"]

            # 2. Streaming call with stream_options.include_usage = True
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt_content}],
                    "max_tokens": max_toks,
                    "stream": True,
                    "stream_options": {"include_usage": True}
                }
            ) as stream_resp:
                assert stream_resp.status_code == 200
                lines = [line.strip() for line in stream_resp.iter_lines() if line.strip()]

            # Must end with usage chunk then [DONE]
            assert lines[-1] == "data: [DONE]"
            usage_line = lines[-2]
            assert usage_line.startswith("data: ")
            usage_chunk = json.loads(usage_line[6:])

            assert usage_chunk["choices"] == []
            assert "usage" in usage_chunk
            streaming_usage = usage_chunk["usage"]

            # 3. Assert exact parity between streaming and non-streaming accounting
            assert streaming_usage["prompt_tokens"] == non_stream_usage["prompt_tokens"]
            assert streaming_usage["completion_tokens"] == non_stream_usage["completion_tokens"]
            assert streaming_usage["total_tokens"] == non_stream_usage["total_tokens"]
    finally:
        adapter.shutdown_sync()


# =========================================================================
# Test J: Failure Metrics Classification and Non-Counted Completed Requests
# =========================================================================
def test_failure_metrics_classification_and_isolation():
    """Verify failed/aborted requests increment failure counters and never requests_completed_total."""
    engine = make_test_engine(total_blocks=4, block_size=4, max_batch_size=2)
    model_name = engine.metrics.model_name

    initial_completed = engine.metrics.requests_completed_total
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total

    # 1. Admission rejection (KV cache full)
    impossible_prompt = "hello " * 20
    req_rej = engine.add_request(impossible_prompt, max_tokens=10)
    with pytest.raises(MemoryError):
        engine.step()

    assert req_rej.status is RequestStatus.FAILED
    assert req_rej.stop_reason == "kv_cache_full"
    assert engine.metrics.requests_failed_total["kv_cache_full"] >= 1
    # Must NOT increment completed requests or token totals!
    assert engine.metrics.requests_completed_total == initial_completed
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens
    assert engine.metrics.generation_tokens_total == initial_gen_tokens

    # 2. Explicit cancellation / abort while waiting in queue (before generation)
    req_abort_waiting = engine.add_request("abort me while waiting", max_tokens=5)
    engine.abort_request(req_abort_waiting.req_id)

    assert req_abort_waiting.status is RequestStatus.FAILED
    assert req_abort_waiting.stop_reason == "abort"
    assert engine.metrics.requests_failed_total["abort"] >= 1
    # Waiting cancellation: 0 prompt/generation tokens processed
    assert engine.metrics.requests_completed_total == initial_completed
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens
    assert engine.metrics.generation_tokens_total == initial_gen_tokens

    # 3. Explicit cancellation / abort after generation started (credits work, does not increment completed)
    req_abort_running = engine.add_request("abort me while running", max_tokens=5)
    engine.step()  # admitted and prefilled + 1 token generated
    p_len = len(req_abort_running.prompt_token_ids)
    g_len = req_abort_running.num_generated
    assert g_len >= 1
    engine.abort_request(req_abort_running.req_id)

    assert req_abort_running.status is RequestStatus.FAILED
    assert req_abort_running.stop_reason == "abort"
    assert engine.metrics.requests_completed_total == initial_completed
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens + p_len
    assert engine.metrics.generation_tokens_total == initial_gen_tokens + g_len

    prom_text = engine.metrics.export_prometheus_text()
    assert f'vllm_requests_completed_total{{model="{model_name}"}} {float(initial_completed)}' in prom_text
    assert f'vllm_requests_failed_total{{error_type="kv_cache_full",model="{model_name}"}}' in prom_text
    assert f'vllm_requests_failed_total{{error_type="abort",model="{model_name}"}}' in prom_text


# =========================================================================
# Test K: Exposition Format, Block Capacity Gauges, and Cardinality Bounds
# =========================================================================
def test_prometheus_exposition_format_and_cardinality():
    """Verify exposition format compliance, KV capacity gauges, and strictly bounded labels."""
    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=2)
    model_name = engine.metrics.model_name

    # Run one request to populate stats
    engine.add_request("test metric exposition", max_tokens=2)
    while engine.step():
        pass

    prom_text = engine.metrics.export_prometheus_text()

    # 1. Exposition headers compliance
    assert "# HELP vllm_gpu_cache_allocated_blocks" in prom_text
    assert "# TYPE vllm_gpu_cache_allocated_blocks gauge" in prom_text
    assert "# HELP vllm_gpu_cache_total_blocks" in prom_text
    assert "# TYPE vllm_gpu_cache_total_blocks gauge" in prom_text
    assert "# HELP vllm_requests_failed_total" in prom_text
    assert "# TYPE vllm_requests_failed_total counter" in prom_text
    assert "# HELP vllm_time_to_first_token_seconds" in prom_text
    assert "# TYPE vllm_time_to_first_token_seconds histogram" in prom_text

    # 2. KV capacity gauges reflect pool
    assert f'vllm_gpu_cache_total_blocks{{model="{model_name}"}} 8' in prom_text
    assert f'vllm_gpu_cache_allocated_blocks{{model="{model_name}"}} 0' in prom_text

    # 3. Cardinality bounds check: no request IDs or dynamic text in labels
    allowed_label_keys = {"model", "error_type", "le"}
    for line in prom_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if "{" in line and "}" in line:
            label_part = line[line.find("{") + 1 : line.find("}")]
            labels = [kv.split("=")[0].strip() for kv in label_part.split(",") if "=" in kv]
            for key in labels:
                assert key in allowed_label_keys, f"Disallowed high-cardinality label key '{key}' in: {line}"

    # 4. Latency non-negativity
    for lat in engine.metrics.ttft_records_ms:
        assert lat >= 0.0
    for lat in engine.metrics.tpot_records_ms:
        assert lat >= 0.0
    for lat in engine.metrics.e2e_records_ms:
        assert lat >= 0.0


# =========================================================================
# Test L: Dedicated Token Counter Semantics Tests
# =========================================================================
def test_token_counter_semantics_successful_completion():
    """Verify successful request increments completed counter, prompt tokens, and generation tokens."""
    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=2)
    initial_completed = engine.metrics.requests_completed_total
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total

    req = engine.add_request("successful prompt test", max_tokens=4, sampling_params=SamplingParams.greedy())
    p_len = len(req.prompt_token_ids)

    while engine.step():
        pass

    assert req.status is RequestStatus.FINISHED
    assert req.num_generated > 0
    # Requests completed counter must increment by 1
    assert engine.metrics.requests_completed_total == initial_completed + 1
    # Prompt and generation work must be credited
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens + p_len
    assert engine.metrics.generation_tokens_total == initial_gen_tokens + req.num_generated
    # Failure counters must remain 0
    assert sum(engine.metrics.requests_failed_total.values()) == 0


def test_token_counter_semantics_rejection_before_generation():
    """Verify rejection before generation increments failure counter and leaves token work and completed at 0."""
    engine = make_test_engine(total_blocks=2, block_size=4, max_batch_size=2)
    initial_completed = engine.metrics.requests_completed_total
    initial_failed_kv = engine.metrics.requests_failed_total["kv_cache_full"]
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total

    # Impossible capacity prompt
    req = engine.add_request("hello " * 20, max_tokens=10)
    with pytest.raises(MemoryError):
        engine.step()

    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "kv_cache_full"
    # Completed requests counter MUST NOT increment
    assert engine.metrics.requests_completed_total == initial_completed
    # Failure counter MUST increment
    assert engine.metrics.requests_failed_total["kv_cache_full"] == initial_failed_kv + 1
    # No prompt or generation work performed
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens
    assert engine.metrics.generation_tokens_total == initial_gen_tokens


def test_token_counter_semantics_cancellation_before_first_token():
    """Verify cancellation before first token increments abort counter with 0 prompt and generation tokens."""
    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=2)
    initial_completed = engine.metrics.requests_completed_total
    initial_failed_abort = engine.metrics.requests_failed_total["abort"]
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total

    req = engine.add_request("abort me before admission", max_tokens=5)
    # Abort while in waiting queue (before step)
    aborted = engine.abort_request(req.req_id)
    assert aborted is True

    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "abort"
    assert engine.metrics.requests_completed_total == initial_completed
    assert engine.metrics.requests_failed_total["abort"] == initial_failed_abort + 1
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens
    assert engine.metrics.generation_tokens_total == initial_gen_tokens


def test_token_counter_semantics_cancellation_after_partial_output():
    """Verify cancellation after partial output preserves generated tokens in generation_tokens_total."""
    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=2)
    initial_completed = engine.metrics.requests_completed_total
    initial_failed_abort = engine.metrics.requests_failed_total["abort"]
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total

    req = engine.add_request("partial generation test", max_tokens=10, sampling_params=SamplingParams.greedy())
    p_len = len(req.prompt_token_ids)

    # Step 1: prefill + first generation
    engine.step()
    tokens_after_step1 = req.num_generated
    assert tokens_after_step1 >= 1

    # Step 2: further generation
    engine.step()
    tokens_after_step2 = req.num_generated
    assert tokens_after_step2 > tokens_after_step1

    # Now abort after partial output
    aborted = engine.abort_request(req.req_id)
    assert aborted is True

    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "abort"
    # Completed requests counter MUST NOT increment
    assert engine.metrics.requests_completed_total == initial_completed
    # Failure counter MUST increment
    assert engine.metrics.requests_failed_total["abort"] == initial_failed_abort + 1
    # Prompt tokens MUST be accounted for since prefill completed
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens + p_len
    # All generated tokens prior to abort MUST be accounted for in total generation work
    assert engine.metrics.generation_tokens_total == initial_gen_tokens + tokens_after_step2


def test_token_counter_semantics_decode_failure():
    """Verify decode failure preserves tokens generated prior to failure."""
    engine = make_test_engine(total_blocks=4, block_size=4, max_batch_size=2)
    initial_completed = engine.metrics.requests_completed_total
    initial_failed_kv = engine.metrics.requests_failed_total["kv_cache_full"]
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total

    # Prompt with 2 tokens: prefill (2) + first token (1) + step 0 decode (1) = 4 tokens (fills block 0)
    # Subsequent decode step requires block 1 and triggers allocator failure
    req = engine.add_request("hello world", max_tokens=8, sampling_params=SamplingParams.greedy())
    p_len = len(req.prompt_token_ids)

    # Step 0: prefill + first generation (fills 4-token block)
    engine.step()
    tokens_before_failure = req.num_generated
    assert tokens_before_failure >= 1

    # Inject allocator failure for subsequent decode step
    def failing_allocate():
        raise MemoryError("Out of KV Cache physical blocks!")

    orig_allocate = engine._allocate_block
    engine._allocate_block = failing_allocate
    try:
        engine.step()
    finally:
        engine._allocate_block = orig_allocate

    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "kv_cache_full"
    # Completed requests counter MUST NOT increment
    assert engine.metrics.requests_completed_total == initial_completed
    # Failure counter MUST increment
    assert engine.metrics.requests_failed_total["kv_cache_full"] == initial_failed_kv + 1
    # Prompt tokens MUST be accounted for
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens + p_len
    # The tokens generated prior to failure MUST be preserved
    assert engine.metrics.generation_tokens_total == initial_gen_tokens + tokens_before_failure


# =========================================================================
# Test M: TTFT Boundary Includes Controlled Queue Delay
# =========================================================================
def test_ttft_boundary_includes_controlled_queue_delay():
    """Verify TTFT measures from request arrival time, including queueing delay."""
    engine = make_test_engine(total_blocks=8, block_size=4, max_batch_size=2)
    model_name = engine.metrics.model_name

    # Add request and simulate artificial queue delay by placing arrival_time in past
    req = engine.add_request("controlled queue delay prompt", max_tokens=3)
    controlled_delay_seconds = 0.150  # 150 ms
    simulated_arrival = time.perf_counter() - controlled_delay_seconds
    req.arrival_time = simulated_arrival
    req.arrival_time_ns = int(simulated_arrival * 1_000_000_000)

    # Step engine to admit, prefill, and emit first token
    engine.step()

    assert req.ttft_ms is not None
    # User-facing TTFT MUST be >= queue delay (150ms)
    assert req.ttft_ms >= 140.0, f"Expected TTFT >= 140ms including queue delay, got {req.ttft_ms}ms"
    assert len(engine.metrics.ttft_records_ms) >= 1
    assert engine.metrics.ttft_records_ms[-1] >= 140.0

    prom_text = engine.metrics.export_prometheus_text()
    assert f'vllm_time_to_first_token_seconds_count{{model="{model_name}"}} 1.0' in prom_text


# =========================================================================
# Test N: Chunked Prefill Token Accounting on Failure or Cancellation
# =========================================================================
def test_chunked_prefill_failure_after_partial_chunks_accounts_tokens_exactly_once():
    """
    Verify prompt-token accounting when chunked prefill completes one or more chunks
    and then fails or is cancelled before the entire prompt prefill finishes:
    1. Completed chunks must be credited to prompt_tokens_total exactly once.
    2. Failed requests must increment requests_failed_total and not requests_completed_total.
    3. Blocks must be safely returned to the physical pool with zero memory leak.
    """
    # 16 blocks total, 4 tokens per block, chunk size = 4 tokens
    engine = make_test_engine(
        total_blocks=16,
        block_size=4,
        max_batch_size=2,
        chunked_prefill=True,
        prefill_chunk_size=4,
    )
    initial_free_blocks = engine.num_free_blocks
    initial_completed = engine.metrics.requests_completed_total
    initial_failed_error = engine.metrics.requests_failed_total["error"]
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total

    # Prompt with 12 tokens -> requires 3 chunks (4 + 4 + 4)
    prompt_text = "t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11"
    req = engine.add_request(prompt_text, max_tokens=4, sampling_params=SamplingParams.greedy())
    assert len(req.prompt_token_ids) == 12

    # Step 0: Admits req and executes Chunk 0 (tokens 0..4)
    engine.step()
    assert req.num_prefilled_tokens == 4
    assert req.is_prefill_done is False
    assert req.status is RequestStatus.RUNNING

    # Inject execution failure during Chunk 1 model forward pass
    orig_forward = engine.model.side_effect

    def failing_forward(*args, **kwargs):
        raise RuntimeError("Simulated mid-prefill execution failure in chunk 1")

    engine.model.side_effect = failing_forward

    # Step 1: Chunk 1 fails during model forward pass
    with pytest.raises(RuntimeError, match="Simulated mid-prefill execution failure"):
        engine.step()

    # Verify request transitioned to FAILED with error
    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "error"
    assert "Simulated mid-prefill execution failure" in req.error
    assert req in engine.completed_requests
    assert len(engine.running_batch) == 0

    # Verify completed requests counter is untouched
    assert engine.metrics.requests_completed_total == initial_completed
    # Verify failure counter incremented exactly once
    assert engine.metrics.requests_failed_total["error"] == initial_failed_error + 1

    # Verify Chunk 0 (4 tokens) was counted exactly once into prompt_tokens_total
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens + 4
    # No decode tokens generated
    assert engine.metrics.generation_tokens_total == initial_gen_tokens

    # Verify 100% of physical blocks are recovered
    assert engine.num_free_blocks == initial_free_blocks

    # Now verify cancellation / abort after 2 chunks completed
    engine.model.side_effect = orig_forward
    initial_failed_abort = engine.metrics.requests_failed_total["abort"]
    cur_prompt_tokens = engine.metrics.prompt_tokens_total

    req2 = engine.add_request(prompt_text, max_tokens=4, sampling_params=SamplingParams.greedy())
    # Step: Chunk 0 (4 tokens)
    engine.step()
    assert req2.num_prefilled_tokens == 4
    # Step: Chunk 1 (4 tokens -> total 8 tokens prefilled)
    engine.step()
    assert req2.num_prefilled_tokens == 8
    assert req2.is_prefill_done is False

    # Abort before Chunk 2 (the final chunk)
    aborted = engine.abort_request(req2.req_id)
    assert aborted is True
    assert req2.status is RequestStatus.FAILED
    assert req2.stop_reason == "abort"
    assert req2 in engine.completed_requests
    assert len(engine.running_batch) == 0

    # Exactly 8 tokens from the 2 completed chunks must be credited
    assert engine.metrics.prompt_tokens_total == cur_prompt_tokens + 8
    assert engine.metrics.requests_completed_total == initial_completed
    assert engine.metrics.requests_failed_total["abort"] == initial_failed_abort + 1
    assert engine.num_free_blocks == initial_free_blocks


# =========================================================================
# Test O: Prefix Caching Reused with Subsequent Partial Prefill Failure
# =========================================================================
def test_prefix_cache_reuse_with_subsequent_partial_prefill_failure():
    """
    Verify prompt-token accounting and prefix-cache metrics when a request reuses
    a cached prefix, partially completes subsequent uncached prefill chunks, and then fails:
    1. Request 1 seeds the prefix cache with a 4-token block and completes successfully.
    2. Request 2 shares the 4-token prefix, admits with prefix hit (reusing cached block),
       completes 1 uncached chunk (4 tokens), and fails on the second uncached chunk.
    3. Exactly 8 prompt tokens (4 cached prefix + 4 completed uncached chunk) are credited
       to prompt_tokens_total for Request 2 (total 12 across both requests).
    4. Request 2 is recorded as failed, not completed, and all blocks are cleanly reclaimed.
    5. Prefix cache hit rate metric accurately reflects prefix token reuse.
    """
    engine = make_test_engine(
        total_blocks=16,
        block_size=4,
        max_batch_size=2,
        prefix=True,
        chunked_prefill=True,
        prefill_chunk_size=4,
    )
    initial_free_blocks = engine.num_free_blocks

    # 1. Request 1: Seed prefix cache with 4 tokens (1 full block)
    req1 = engine.add_request("p0 p1 p2 p3", max_tokens=2, sampling_params=SamplingParams.greedy())
    assert len(req1.prompt_token_ids) == 4

    while engine.step():
        pass

    assert req1.status is RequestStatus.FINISHED
    assert req1.num_generated > 0
    assert engine.metrics.requests_completed_total == 1
    assert engine.metrics.prompt_tokens_total == 4
    gen_tokens_after_req1 = engine.metrics.generation_tokens_total
    assert gen_tokens_after_req1 > 0

    # Prefix cache should now hold the block for ("p0", "p1", "p2", "p3")
    assert len(engine.prefix_cache.hash_to_block_id) == 1

    # 2. Request 2: Prompt with 12 tokens sharing the first 4 tokens
    # Tokens: [p0, p1, p2, p3] (cached, 4 tok) + [u4, u5, u6, u7] (uncached chunk 1, 4 tok) + [u8, u9, u10, u11] (uncached chunk 2, 4 tok)
    prompt_text = "p0 p1 p2 p3 u4 u5 u6 u7 u8 u9 u10 u11"
    req2 = engine.add_request(prompt_text, max_tokens=4, sampling_params=SamplingParams.greedy())
    assert len(req2.prompt_token_ids) == 12

    # Step 0: Admits req2 with prefix hit of 4 tokens, then executes uncached Chunk 1 (tokens 4..8)
    engine.step()
    assert req2.prefix_hit_tokens == 4
    # Prefilled tokens is now: 4 (cached prefix) + 4 (Chunk 1) = 8 tokens
    assert req2.num_prefilled_tokens == 8
    assert req2.is_prefill_done is False
    assert req2.status is RequestStatus.RUNNING
    # Prefix hit rate reflects 4 hit tokens out of 12 total prompt tokens (0.3333)
    assert abs(engine.metrics.prefix_cache_hit_rate - (4.0 / 12.0)) < 1e-4

    # Inject failure during uncached Chunk 2 model forward pass
    orig_forward = engine.model.side_effect

    def failing_forward(*args, **kwargs):
        raise RuntimeError("Simulated execution failure in uncached prefill chunk 2")

    engine.model.side_effect = failing_forward

    # Step 1: Uncached Chunk 2 fails
    with pytest.raises(RuntimeError, match="Simulated execution failure in uncached prefill chunk 2"):
        engine.step()

    # Invariants for Request 2 failure
    assert req2.status is RequestStatus.FAILED
    assert req2.stop_reason == "error"
    assert "Simulated execution failure in uncached prefill chunk 2" in req2.error
    assert req2 in engine.completed_requests
    assert len(engine.running_batch) == 0

    # Metrics assertions:
    # Requests completed counter remains 1 (from req1 only; req2 is NOT counted as completed)
    assert engine.metrics.requests_completed_total == 1
    # Requests failed counter increments by 1
    assert engine.metrics.requests_failed_total["error"] == 1
    # Total prompt tokens must be 4 (req1) + 8 (req2: 4 cached prefix + 4 completed Chunk 1) = 12
    assert engine.metrics.prompt_tokens_total == 12
    # Generation tokens remain unchanged (no decode tokens emitted by req2)
    assert engine.metrics.generation_tokens_total == gen_tokens_after_req1

    # Block cleanup: req2's physical blocks are freed; cached block from req1 remains cached with ref_count 0
    # Available free blocks (free list + eviction queue) must equal initial free blocks
    assert engine.num_free_blocks == initial_free_blocks

    # 3. Request 3: Cancellation after prefix reuse (8 cached tokens) and 1 uncached chunk
    engine.model.side_effect = orig_forward
    prompt_text_req3 = "p0 p1 p2 p3 u4 u5 u6 u7 v8 v9 v10 v11 w12 w13 w14 w15"
    req3 = engine.add_request(prompt_text_req3, max_tokens=4, sampling_params=SamplingParams.greedy())
    assert len(req3.prompt_token_ids) == 16

    # Step 0: Admits req3 with 8 cached tokens and completes Chunk 2 (4 tokens -> 12 total)
    engine.step()
    assert req3.prefix_hit_tokens == 8
    assert req3.num_prefilled_tokens == 12
    assert req3.is_prefill_done is False

    aborted = engine.abort_request(req3.req_id)
    assert aborted is True
    assert req3.status is RequestStatus.FAILED
    assert req3.stop_reason == "abort"
    assert req3 in engine.completed_requests
    assert len(engine.running_batch) == 0

    # Total prompt tokens increments by exactly 12 (now 12 + 12 = 24)
    assert engine.metrics.prompt_tokens_total == 24
    assert engine.metrics.requests_completed_total == 1
    assert engine.metrics.requests_failed_total["abort"] == 1
    assert engine.num_free_blocks == initial_free_blocks
