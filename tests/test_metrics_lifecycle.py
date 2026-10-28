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
