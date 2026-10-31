"""
Integration Test Suite: OpenAI-Compatible REST API Server
---------------------------------------------------------
Validates:
  1. Root & Health probes (`GET /`, `GET /health`)
  2. Model catalog exposition (`GET /v1/models`)
  3. Non-streaming Chat Completions (`POST /v1/chat/completions`)
  4. Streaming SSE Chat Completions (`POST /v1/chat/completions` with `stream=true`)
  5. Legacy Text Completions (`POST /v1/completions`)
  6. Prometheus Telemetry Endpoint (`GET /metrics`)
  7. Official OpenAI Python SDK compatibility (via ASGI transport)
"""

import os
import sys
import json
import time
import pytest
import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_default_model, get_default_device
from src.server.serving_engine import AsyncServingEngine
from src.server.app import create_app


# Shared module-level engine fixture
_engine = None
_client = None
_app = None


def get_test_client():
    global _engine, _client, _app
    if _client is None:
        model_name = get_default_model()
        device = get_default_device()
        print(f"\n[Test Setup] Initializing AsyncServingEngine ({model_name} on {device})...")
        _engine = AsyncServingEngine(model_name=model_name, device=device, force=True)
        _app = create_app(_engine)
        _client = TestClient(_app)
    return _client, _engine, _app


# =========================================================================
# 1. Health & Meta Endpoints
# =========================================================================
def test_server_root_endpoint():
    client, engine, _ = get_test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["name"] == "Mini-vLLM Inference Server"
    assert data["model"] == engine.model_name
    assert "metrics" in data


def test_server_health_probe():
    client, engine, _ = get_test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model"] == engine.model_name
    assert "running_requests" in data


def test_v1_models_catalog():
    client, engine, _ = get_test_client()
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    model_ids = [m["id"] for m in data["data"]]
    assert engine.model_name in model_ids or "mini-vllm" in model_ids


# =========================================================================
# 2. Chat Completions: Non-Streaming
# =========================================================================
def test_chat_completions_non_streaming():
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [
            {"role": "system", "content": "You are a concise AI assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one word."}
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"].strip()) > 0
    assert choice["finish_reason"] in ["stop", "length"]

    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


# =========================================================================
# 3. Chat Completions: Streaming SSE
# =========================================================================
def test_chat_completions_streaming_sse():
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [
            {"role": "user", "content": "Count from 1 to 3."}
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert len(lines) >= 3, f"Expected multiple SSE lines, got: {lines}"

    # Verify first chunk sets role
    first_data_line = lines[0]
    assert first_data_line.startswith("data: ")
    first_json = json.loads(first_data_line[6:])
    assert first_json["object"] == "chat.completion.chunk"
    assert first_json["choices"][0]["delta"].get("role") == "assistant"

    # Verify final chunk is data: [DONE]
    assert lines[-1] == "data: [DONE]"

    # Verify intermediate content chunks
    collected_tokens = []
    for line in lines[1:-1]:
        if line.startswith("data: "):
            chunk = json.loads(line[6:])
            assert chunk["object"] == "chat.completion.chunk"
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                collected_tokens.append(delta["content"])

    full_streamed_text = "".join(collected_tokens)
    assert len(full_streamed_text.strip()) > 0


# =========================================================================
# 4. Legacy Text Completions
# =========================================================================
def test_legacy_text_completions():
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False
    }

    resp = client.post("/v1/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert len(data["choices"][0]["text"].strip()) > 0
    assert data["usage"]["total_tokens"] > 0


# =========================================================================
# 5. Prometheus Observability
# =========================================================================
def test_prometheus_metrics_endpoint():
    client, engine, _ = get_test_client()
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text

    assert "vllm_requests_completed_total" in text or "vllm_generation_tokens_total" in text
    assert "vllm_time_to_first_token_seconds" in text
    assert "vllm_time_per_output_token_seconds" in text
    assert "vllm_requests_failed_total" in text
    assert "vllm_gpu_cache_allocated_blocks" in text
    assert "vllm_gpu_cache_total_blocks" in text


# =========================================================================
# 6. Official OpenAI Python SDK Integration (via TestClient Transport)
# =========================================================================
def test_official_openai_sdk_integration():
    from openai import OpenAI

    client, engine, app = get_test_client()

    # Configure official OpenAI client targeting our FastAPI app in-process
    openai_client = OpenAI(
        base_url="http://testserver/v1",
        api_key="none",
        http_client=client
    )

    # 1. Test models list
    models = openai_client.models.list()
    assert len(models.data) >= 1

    # 2. Test non-streaming chat completion
    chat_resp = openai_client.chat.completions.create(
        model=engine.model_name,
        messages=[
            {"role": "user", "content": "Say hello!"}
        ],
        max_tokens=10,
        temperature=0.0
    )
    assert chat_resp.choices[0].message.content is not None
    assert len(chat_resp.choices[0].message.content.strip()) > 0

    # 3. Test streaming chat completion
    stream_resp = openai_client.chat.completions.create(
        model=engine.model_name,
        messages=[
            {"role": "user", "content": "What is the color of the sky?"}
        ],
        max_tokens=12,
        stream=True
    )

    tokens = []
    for chunk in stream_resp:
        if chunk.choices and chunk.choices[0].delta.content:
            tokens.append(chunk.choices[0].delta.content)

    streamed_reply = "".join(tokens)
    assert len(streamed_reply.strip()) > 0


# =========================================================================
# 7. Error Hygiene & Request Validation (OpenAI Envelope)
# =========================================================================
def test_validation_error_empty_messages():
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [],
        "max_tokens": 16
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["code"] == 422
    assert "messages" in data["error"]["message"].lower()


def test_validation_error_invalid_max_tokens():
    client, engine, _ = get_test_client()
    for bad_val in [0, -1, -50]:
        payload = {
            "model": engine.model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": bad_val
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == 422
        assert "max_tokens" in data["error"]["message"]


def test_validation_error_invalid_sampling_params():
    client, engine, _ = get_test_client()
    # Out of range temperature
    resp = client.post("/v1/chat/completions", json={
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 3.5
    })
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert "temperature" in data["error"]["message"]

    # Out of range top_p
    resp = client.post("/v1/chat/completions", json={
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 1.5
    })
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert "top_p" in data["error"]["message"]

    # Invalid repetition_penalty
    resp = client.post("/v1/chat/completions", json={
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "repetition_penalty": 0.0
    })
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert "repetition_penalty" in data["error"]["message"]


def test_not_found_openai_error():
    client, _, _ = get_test_client()
    resp = client.get("/v1/nonexistent_route")
    assert resp.status_code == 404
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == 404
    assert data["error"]["type"] == "invalid_request_error"
    assert "not found" in data["error"]["message"].lower()


def test_server_error_envelope_sanitization():
    from unittest.mock import patch
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 10
    }
    with patch.object(engine, "generate_complete", side_effect=RuntimeError("Internal kernel fault")):
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == 500
        assert data["error"]["type"] == "api_error"
        assert "Traceback" not in data["error"]["message"]
        assert "/home/" not in data["error"]["message"]


def test_metrics_running_counter_resilience():
    import asyncio
    from src.engine.sampling import SamplingParams
    client, engine, _ = get_test_client()
    assert engine.metrics.num_requests_running == 0

    # Test premature generator close (client disconnect)
    gen = engine.generate_stream(
        "Say something",
        sampling_params=SamplingParams(temperature=0.0),
        max_tokens=10
    )
    loop = asyncio.new_event_loop()
    try:
        item = loop.run_until_complete(gen.__anext__())
        assert item is not None
        loop.run_until_complete(gen.aclose())
    finally:
        loop.close()

    assert engine.metrics.num_requests_running == 0


def test_cuda_oom_error_semantics():
    from unittest.mock import patch
    import torch
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "Trigger OOM"}],
        "max_tokens": 16
    }
    with patch.object(engine, "generate_complete", side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory")):
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "cuda_out_of_memory"
        assert data["error"]["type"] == "server_error"
        assert "GPU out of memory" in data["error"]["message"]
        assert "Reduce max_tokens or prompt length" in data["error"]["message"]
        assert "Traceback" not in data["error"]["message"]


def test_sse_stream_error_event_mid_stream():
    from unittest.mock import patch
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "Trigger stream error"}],
        "max_tokens": 16,
        "stream": True
    }
    async def mock_stream(prompt, sampling_params, max_tokens):
        yield ("token1", True, False, None)
        raise RuntimeError("Kernel failed mid-stream")

    with patch.object(engine, "generate_stream", side_effect=mock_stream):
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            lines = [line.strip() for line in resp.iter_lines() if line.strip()]

        assert "data: [DONE]" not in lines
        error_lines = [l for l in lines if '"error"' in l]
        assert len(error_lines) == 1
        err_data = json.loads(error_lines[0].replace("data: ", ""))
        assert "error" in err_data
        assert err_data["error"]["code"] == 500
        assert err_data["error"]["type"] == "server_error"
        assert "RuntimeError" in err_data["error"]["message"] or "Kernel failed" in err_data["error"]["message"]


def test_internal_stream_exception_cleanup():
    import asyncio
    from unittest.mock import patch
    from src.engine.sampling import SamplingParams
    _, engine, _ = get_test_client()

    assert engine.metrics.num_requests_running == 0
    initial_completed = engine.metrics.requests_completed_total

    loop = asyncio.new_event_loop()
    try:
        with patch.object(engine.model, "forward", side_effect=RuntimeError("Prefill kernel fault")):
            gen = engine.generate_stream(
                "Trigger internal exception",
                sampling_params=SamplingParams(temperature=0.0),
                max_tokens=10
            )
            with pytest.raises(RuntimeError, match="Prefill kernel fault"):
                loop.run_until_complete(gen.__anext__())
    finally:
        loop.close()

    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.requests_completed_total == initial_completed


def test_serving_engine_failure_before_prefill():
    """Verify failure before prefill (e.g. tokenization error) records error without crediting prompt/generation tokens."""
    import asyncio
    from unittest.mock import patch
    from src.engine.sampling import SamplingParams
    _, engine, _ = get_test_client()

    initial_completed = engine.metrics.requests_completed_total
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total
    initial_failed = engine.metrics.requests_failed_total.get("error", 0)

    loop = asyncio.new_event_loop()
    try:
        with patch.object(engine.tokenizer, "encode", side_effect=ValueError("Tokenization fault")):
            gen = engine.generate_stream(
                "Trigger before-prefill exception",
                sampling_params=SamplingParams(temperature=0.0),
                max_tokens=10
            )
            with pytest.raises(ValueError, match="Tokenization fault"):
                loop.run_until_complete(gen.__anext__())
    finally:
        loop.close()

    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.requests_completed_total == initial_completed
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens
    assert engine.metrics.generation_tokens_total == initial_gen_tokens
    assert engine.metrics.requests_failed_total.get("error", 0) == initial_failed + 1


def test_serving_engine_cuda_oom_during_prefill():
    """Verify CUDA OOM during prefill records failure without crediting prompt or generation tokens."""
    import asyncio
    from unittest.mock import patch
    import torch
    from src.engine.sampling import SamplingParams
    _, engine, _ = get_test_client()

    initial_completed = engine.metrics.requests_completed_total
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total
    initial_failed = engine.metrics.requests_failed_total.get("cuda_oom", 0)

    loop = asyncio.new_event_loop()
    try:
        with patch.object(engine.model, "forward", side_effect=torch.cuda.OutOfMemoryError("CUDA OOM during prefill")):
            gen = engine.generate_stream(
                "Trigger prefill CUDA OOM",
                sampling_params=SamplingParams(temperature=0.0),
                max_tokens=10
            )
            with pytest.raises(torch.cuda.OutOfMemoryError, match="CUDA OOM during prefill"):
                loop.run_until_complete(gen.__anext__())
    finally:
        loop.close()

    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.requests_completed_total == initial_completed
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens
    assert engine.metrics.generation_tokens_total == initial_gen_tokens
    assert engine.metrics.requests_failed_total.get("cuda_oom", 0) == initial_failed + 1


def test_serving_engine_partial_generation_accounting_on_abort():
    """Verify partial generation tokens are credited exactly once when stream is aborted mid-flight."""
    import asyncio
    from src.engine.sampling import SamplingParams
    _, engine, _ = get_test_client()

    initial_completed = engine.metrics.requests_completed_total
    initial_prompt_tokens = engine.metrics.prompt_tokens_total
    initial_gen_tokens = engine.metrics.generation_tokens_total
    initial_failed_abort = engine.metrics.requests_failed_total.get("abort", 0)

    loop = asyncio.new_event_loop()
    try:
        gen = engine.generate_stream(
            "Hello world partial test prompt",
            sampling_params=SamplingParams(temperature=0.0),
            max_tokens=10
        )
        # Consume first 2 tokens (prefill chunk + 1 decode step)
        t1 = loop.run_until_complete(gen.__anext__())
        assert t1[1] is True  # is_first
        t2 = loop.run_until_complete(gen.__anext__())
        assert t2[1] is False

        # Now explicitly close / abort the generator mid-flight
        loop.run_until_complete(gen.aclose())
    finally:
        loop.close()

    expected_p_tokens = len(engine.tokenizer.encode("Hello world partial test prompt"))
    assert engine.metrics.num_requests_running == 0
    assert engine.metrics.requests_completed_total == initial_completed
    assert engine.metrics.requests_failed_total.get("abort", 0) == initial_failed_abort + 1
    # Prompt tokens accounted because prefill completed
    assert engine.metrics.prompt_tokens_total == initial_prompt_tokens + expected_p_tokens
    # Generated tokens accounted exactly once (2 tokens emitted before aclose)
    assert engine.metrics.generation_tokens_total == initial_gen_tokens + 2


def test_async_task_cancellation_releases_lock_and_counter():
    import asyncio
    from src.engine.sampling import SamplingParams
    _, engine, _ = get_test_client()

    assert engine.metrics.num_requests_running == 0
    assert not engine._lock.locked()

    async def run_streaming_consumer():
        async for token, is_first, is_last, reason in engine.generate_stream(
            "Long prompt for cancellation test",
            sampling_params=SamplingParams(temperature=0.0),
            max_tokens=50
        ):
            await asyncio.sleep(0.01)

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(run_streaming_consumer())
        loop.run_until_complete(asyncio.sleep(0.05))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            loop.run_until_complete(task)
        # Allow loop to process the async generator closure scheduled upon cancellation
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        loop.close()

    assert engine.metrics.num_requests_running == 0
    assert not engine._lock.locked()


def test_completion_empty_prompt_validation():
    client, engine, _ = get_test_client()

    # 1. Empty string prompt -> HTTP 422
    resp = client.post("/v1/completions", json={"model": engine.model_name, "prompt": ""})
    assert resp.status_code == 422
    assert "prompt" in resp.json()["error"]["message"].lower()

    # 2. Whitespace-only prompt -> HTTP 422
    resp = client.post("/v1/completions", json={"model": engine.model_name, "prompt": "   "})
    assert resp.status_code == 422
    assert "prompt" in resp.json()["error"]["message"].lower()

    # 3. Empty list prompt -> HTTP 422
    resp = client.post("/v1/completions", json={"model": engine.model_name, "prompt": []})
    assert resp.status_code == 422
    assert "prompt" in resp.json()["error"]["message"].lower()

    # 4. List containing empty string -> HTTP 422
    resp = client.post("/v1/completions", json={"model": engine.model_name, "prompt": ["valid", ""]})
    assert resp.status_code == 422
    assert "prompt" in resp.json()["error"]["message"].lower()

    # 5. Valid non-empty string prompt -> HTTP 200
    resp = client.post("/v1/completions", json={"model": engine.model_name, "prompt": "Paris is", "max_tokens": 5})
    assert resp.status_code == 200
    assert len(resp.json()["choices"]) == 1

    # 6. Valid non-empty list prompt -> HTTP 200
    resp = client.post("/v1/completions", json={"model": engine.model_name, "prompt": ["Paris is", "London is"], "max_tokens": 5})
    assert resp.status_code == 200
    assert len(resp.json()["choices"]) == 1


def test_chat_streaming_omitted_stream_options():
    """Verify chat streaming with omitted stream_options does not emit usage chunk."""
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    for line in lines[:-1]:
        chunk = json.loads(line[6:])
        assert "usage" not in chunk or chunk["usage"] is None


def test_chat_streaming_empty_stream_options():
    """Verify chat streaming with stream_options={} defaults include_usage to False."""
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {},
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    for line in lines[:-1]:
        chunk = json.loads(line[6:])
        assert "usage" not in chunk or chunk["usage"] is None


def test_chat_streaming_include_usage_false():
    """Verify chat streaming with stream_options.include_usage=False does not emit usage chunk."""
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    for line in lines[:-1]:
        chunk = json.loads(line[6:])
        assert "usage" not in chunk or chunk["usage"] is None


def test_chat_streaming_with_stream_options_usage():
    """Verify chat streaming with stream_options.include_usage=True emits final usage chunk before [DONE]."""
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "messages": [
            {"role": "user", "content": "Say hello in one word."}
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    usage_line = lines[-2]
    assert usage_line.startswith("data: ")
    usage_chunk = json.loads(usage_line[6:])

    assert usage_chunk["object"] == "chat.completion.chunk"
    assert usage_chunk["choices"] == []
    assert "usage" in usage_chunk
    assert usage_chunk["usage"]["prompt_tokens"] > 0
    assert usage_chunk["usage"]["completion_tokens"] > 0
    assert usage_chunk["usage"]["total_tokens"] == usage_chunk["usage"]["prompt_tokens"] + usage_chunk["usage"]["completion_tokens"]

    content_chunks = [json.loads(l[6:]) for l in lines[:-2] if l.startswith("data: ")]
    # Chunk 0 is the initial role chunk; chunks 1..N correspond to generated tokens (including EOS finish chunk)
    token_chunks = content_chunks[1:]
    assert usage_chunk["usage"]["completion_tokens"] == len(token_chunks)


def test_legacy_streaming_omitted_stream_options():
    """Verify legacy completions streaming with omitted stream_options does not emit usage chunk."""
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "prompt": "Hello world",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
    }
    with client.stream("POST", "/v1/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    for line in lines[:-1]:
        chunk = json.loads(line[6:])
        assert "usage" not in chunk or chunk["usage"] is None


def test_legacy_streaming_empty_stream_options():
    """Verify legacy completions streaming with stream_options={} defaults include_usage to False."""
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "prompt": "Hello world",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {},
    }
    with client.stream("POST", "/v1/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    for line in lines[:-1]:
        chunk = json.loads(line[6:])
        assert "usage" not in chunk or chunk["usage"] is None


def test_legacy_streaming_include_usage_false():
    """Verify legacy completions streaming with stream_options.include_usage=False does not emit usage chunk."""
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "prompt": "Hello world",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    with client.stream("POST", "/v1/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    for line in lines[:-1]:
        chunk = json.loads(line[6:])
        assert "usage" not in chunk or chunk["usage"] is None


def test_legacy_streaming_with_stream_options_usage():
    """Verify legacy completions streaming with stream_options.include_usage=True emits final usage chunk before [DONE]."""
    client, engine, _ = get_test_client()
    payload = {
        "model": engine.model_name,
        "prompt": "Hello world",
        "max_tokens": 6,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    with client.stream("POST", "/v1/completions", json=payload) as resp:
        assert resp.status_code == 200
        lines = [line.strip() for line in resp.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    usage_line = lines[-2]
    assert usage_line.startswith("data: ")
    usage_chunk = json.loads(usage_line[6:])

    assert usage_chunk["object"] == "text_completion"
    assert usage_chunk["choices"] == []
    assert "usage" in usage_chunk
    expected_prompt_tokens = len(engine.tokenizer.encode(payload["prompt"]))
    assert usage_chunk["usage"]["prompt_tokens"] == expected_prompt_tokens
    assert usage_chunk["usage"]["completion_tokens"] > 0
    assert usage_chunk["usage"]["total_tokens"] == usage_chunk["usage"]["prompt_tokens"] + usage_chunk["usage"]["completion_tokens"]

    content_chunks = [json.loads(l[6:]) for l in lines[:-2] if l.startswith("data: ")]
    assert usage_chunk["usage"]["completion_tokens"] == len(content_chunks)


# =========================================================================
# Standalone CLI Test Runner
# =========================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("🚀 Mini-vLLM OpenAI-Compatible Server: Integration Test Suite")
    print("=" * 75)

    test_functions = [
        ("Health & Root Endpoint", test_server_root_endpoint),
        ("Health Probe (/health)", test_server_health_probe),
        ("Model Catalog (/v1/models)", test_v1_models_catalog),
        ("Chat Completions (JSON Non-Streaming)", test_chat_completions_non_streaming),
        ("Chat Completions (SSE Streaming)", test_chat_completions_streaming_sse),
        ("Chat Streaming: Omitted stream_options", test_chat_streaming_omitted_stream_options),
        ("Chat Streaming: Empty stream_options={}", test_chat_streaming_empty_stream_options),
        ("Chat Streaming: include_usage=False", test_chat_streaming_include_usage_false),
        ("Chat Streaming: include_usage=True", test_chat_streaming_with_stream_options_usage),
        ("Legacy Text Completions (/v1/completions)", test_legacy_text_completions),
        ("Legacy Streaming: Omitted stream_options", test_legacy_streaming_omitted_stream_options),
        ("Legacy Streaming: Empty stream_options={}", test_legacy_streaming_empty_stream_options),
        ("Legacy Streaming: include_usage=False", test_legacy_streaming_include_usage_false),
        ("Legacy Streaming: include_usage=True", test_legacy_streaming_with_stream_options_usage),
        ("Prometheus Metrics (/metrics)", test_prometheus_metrics_endpoint),
        ("Official OpenAI Python SDK Integration", test_official_openai_sdk_integration),
        ("Validation: Empty Messages", test_validation_error_empty_messages),
        ("Validation: Invalid max_tokens", test_validation_error_invalid_max_tokens),
        ("Validation: Invalid Sampling Params", test_validation_error_invalid_sampling_params),
        ("404 Not Found OpenAI Envelope", test_not_found_openai_error),
        ("500 Internal Error Envelope Sanitization", test_server_error_envelope_sanitization),
        ("Metrics Running Counter Resilience", test_metrics_running_counter_resilience),
        ("CUDA OOM Error Semantics (503)", test_cuda_oom_error_semantics),
        ("SSE Stream Mid-Flight Error Event", test_sse_stream_error_event_mid_stream),
        ("Internal Stream Exception Metric Cleanup", test_internal_stream_exception_cleanup),
        ("Serving Engine: Failure Before Prefill", test_serving_engine_failure_before_prefill),
        ("Serving Engine: CUDA OOM During Prefill", test_serving_engine_cuda_oom_during_prefill),
        ("Serving Engine: Partial Gen Accounting on Abort", test_serving_engine_partial_generation_accounting_on_abort),
        ("Async Task Cancellation Lock & Counter", test_async_task_cancellation_releases_lock_and_counter),
        ("Legacy Completion Empty Prompt Validation", test_completion_empty_prompt_validation),
    ]

    passed = 0
    failed = 0
    start_all = time.perf_counter()

    for idx, (name, fn) in enumerate(test_functions, 1):
        t0 = time.perf_counter()
        try:
            fn()
            dur = (time.perf_counter() - t0) * 1000
            print(f"  [{idx}/{len(test_functions)}] ✓ {name:<46} PASSED ({dur:6.1f} ms)")
            passed += 1
        except Exception as e:
            dur = (time.perf_counter() - t0) * 1000
            print(f"  [{idx}/{len(test_functions)}] ✗ {name:<46} FAILED ({dur:6.1f} ms): {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    total_dur = time.perf_counter() - start_all
    print("=" * 75)
    if failed == 0:
        print(f"  🎉 ALL {passed}/{len(test_functions)} SERVER TESTS PASSED! ({total_dur:.2f}s)")
    else:
        print(f"  ⚠️  SERVER TESTS COMPLETED: {passed} PASSED, {failed} FAILED ({total_dur:.2f}s)")
    print("=" * 75)
    sys.exit(0 if failed == 0 else 1)
