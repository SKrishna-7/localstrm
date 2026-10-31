"""
Controlled CLI Integration and API Integration Test Suite (Phase 8.3.2)
-----------------------------------------------------------------------
Validates:
  1. CLI parser preserves the current default engine selection ('hf').
  2. Explicit Mini-vLLM selection chooses the adapter-backed engine path ('mini-vllm').
  3. Mini-vLLM configuration options (--batch-size, --total-blocks, --vram-budget, --kv-cache-dtype)
     are parsed correctly.
  4. Mutually exclusive options (--total-blocks and --vram-budget) fail at the CLI boundary.
  5. Both serving modes avoid unnecessary initialization of the unused backend.
  6. Root, health (/health), models catalog (/v1/models), and Prometheus (/metrics) endpoints
     remain fully compatible under the adapter.
  7. Chat completion (/v1/chat/completions) functions with both streaming SSE and non-streaming responses.
  8. Legacy completion (/v1/completions) functions with both streaming SSE and non-streaming responses.
  9. Request validation errors preserve the standardized OpenAI error format.
  10. Admission failures (e.g. MemoryError) reach the client without hanging the stream.
  11. Metrics remain consistent with request lifecycle states.
  12. Application shutdown terminates the adapter worker thread cleanly.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli import build_parser, cmd_serve
from src.engine.llm_engine import RequestStatus
from src.server.app import create_app
from src.server.serving_adapter import AsyncServingAdapter


class FakeTokenizer:
    """Deterministic in-memory tokenizer for testing."""
    def __init__(self):
        self.eos_token_id = 2
        self.chat_template = None

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text] or [1]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(t) if 32 <= t <= 126 else f"<{t}>" for t in token_ids)


class FakeRequest:
    """Mock EngineRequest tracking generated tokens and terminal state."""
    def __init__(self, req_id: int, prompt: str, max_tokens: int, sampling_params: Any = None):
        self.req_id = req_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.sampling_params = sampling_params
        self.output_token_ids: List[int] = []
        self.status = RequestStatus.WAITING
        self.stop_reason: Optional[str] = None
        self.error: Optional[str] = None

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)


class FakeMetrics:
    def __init__(self):
        self.num_requests_running = 0

    def export_prometheus_text(self) -> str:
        return "# HELP requests_total Total requests\nrequests_total 42\n"


class FakeEngine:
    """Deterministic in-memory FakeEngine mimicking MiniVLLMEngine."""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        fail_with_memory_error: bool = False
    ):
        self.model_name = model_name
        self.device = "cpu"
        self.waiting_queue: List[FakeRequest] = []
        self.running_batch: List[FakeRequest] = []
        self.completed_requests: List[FakeRequest] = []
        self.requests_by_id: dict[int, FakeRequest] = {}
        self.tokenizer = FakeTokenizer()
        self.metrics = FakeMetrics()
        self.current_step = 0
        self._next_req_id = 1
        self.fail_with_memory_error = fail_with_memory_error

    def add_request(self, prompt: str, max_tokens: int, sampling_params: Any = None) -> FakeRequest:
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        req = FakeRequest(
            req_id=self._next_req_id,
            prompt=prompt,
            max_tokens=max_tokens,
            sampling_params=sampling_params
        )
        self._next_req_id += 1
        self.waiting_queue.append(req)
        self.requests_by_id[req.req_id] = req
        return req

    def get_request(self, req_id: int) -> Optional[FakeRequest]:
        return self.requests_by_id.get(req_id)

    def abort_request(self, req_id: int) -> bool:
        for q in [self.waiting_queue, self.running_batch]:
            for r in list(q):
                if r.req_id == req_id:
                    q.remove(r)
                    r.status = RequestStatus.FAILED
                    r.stop_reason = "abort"
                    self.completed_requests.append(r)
                    return True
        return False

    def step(self) -> bool:
        while self.waiting_queue:
            req = self.waiting_queue.pop(0)
            if self.fail_with_memory_error:
                req.status = RequestStatus.FAILED
                req.stop_reason = "kv_cache_full"
                req.error = "request capacity exceeded total pool"
                self.completed_requests.append(req)
                err = MemoryError(req.error)
                err.request = req
                raise err
            req.status = RequestStatus.RUNNING
            self.running_batch.append(req)

        finished_this_step = []
        for req in list(self.running_batch):
            next_tok = 65 + (len(req.output_token_ids) % 26)  # 'A', 'B', 'C', ...
            req.output_token_ids.append(next_tok)

            if len(req.output_token_ids) >= req.max_tokens:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "length"
                finished_this_step.append(req)

        for req in finished_this_step:
            if req in self.running_batch:
                self.running_batch.remove(req)
            if req not in self.completed_requests:
                self.completed_requests.append(req)

        self.current_step += 1
        return bool(self.waiting_queue or self.running_batch)


# =============================================================================
# 1. CLI Parsing & Option Validation Tests
# =============================================================================

def test_cli_parser_default_engine_selection():
    """1. Validates that CLI parser defaults to 'hf' when --engine is omitted."""
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.subcommand == "serve"
    assert args.engine == "hf"
    assert args.batch_size == 4
    assert args.total_blocks is None
    assert args.vram_budget is None
    assert args.kv_cache_dtype == "fp16"
    assert args.host == "127.0.0.1"


def test_cli_parser_serve_default_host():
    """Validates that cli.py serve defaults to localhost (127.0.0.1) for security."""
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.host == "127.0.0.1"


def test_cli_parser_explicit_minivllm_selection():
    """2. Validates legacy --engine mini-vllm selection and parameter parsing."""
    parser = build_parser()
    args = parser.parse_args([
        "serve",
        "--engine", "mini-vllm",
        "--batch-size", "8",
        "--kv-cache-dtype", "int8",
        "--total-blocks", "128"
    ])
    assert args.subcommand == "serve"
    assert args.engine == "mini-vllm"
    assert args.batch_size == 8
    assert args.kv_cache_dtype == "int8"
    assert args.total_blocks == 128
    assert args.vram_budget is None


def test_cli_parser_explicit_localstrm_selection():
    """Validates canonical --engine localstrm selection and parameter parsing."""
    parser = build_parser()
    args = parser.parse_args([
        "serve",
        "--engine", "localstrm",
        "--batch-size", "8",
        "--kv-cache-dtype", "int8",
        "--total-blocks", "128"
    ])
    assert args.subcommand == "serve"
    assert args.engine == "localstrm"
    assert args.batch_size == 8
    assert args.kv_cache_dtype == "int8"
    assert args.total_blocks == 128
    assert args.vram_budget is None


def test_cli_parser_invalid_engine_rejected(capsys):
    """Validates that invalid --engine values are rejected by the parser with a useful error."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["serve", "--engine", "unsupported-engine"])
    assert exc_info.value.code == 2


def test_cli_parser_vram_budget_parsing():
    """3. Validates that --vram-budget parses correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "serve",
        "--engine", "localstrm",
        "--vram-budget", "3.5GiB"
    ])
    assert args.vram_budget == "3.5GiB"
    assert args.total_blocks is None


def test_cmd_serve_mutually_exclusive_blocks_and_budget_fails():
    """4. Validates that specifying both --total-blocks and --vram-budget exits with code 2."""
    parser = build_parser()
    args = parser.parse_args([
        "serve",
        "--engine", "localstrm",
        "--total-blocks", "64",
        "--vram-budget", "2GiB"
    ])

    with patch("src.memory.preflight.enforce_preflight_check"):
        with pytest.raises(SystemExit) as exc_info:
            cmd_serve(args)
        assert exc_info.value.code == 2


def test_cmd_serve_hf_ignores_minivllm_flags_with_warning(capsys):
    """5. Validates warning printed when LocalStrm flags are passed with default hf engine."""
    parser = build_parser()
    args = parser.parse_args([
        "serve",
        "--total-blocks", "64"
    ])

    with patch("src.memory.preflight.enforce_preflight_check"), \
         patch("src.server.serving_engine.AsyncServingEngine"), \
         patch("src.server.app.create_app"), \
         patch("uvicorn.run"):
        cmd_serve(args)

    captured = capsys.readouterr().out
    assert "Warning: Options --total-blocks, --vram-budget, and --kv-cache-dtype are only applicable" in captured


def test_cmd_serve_initializes_selected_backend_only():
    """6. Validates that only the chosen backend is imported and initialized."""
    parser = build_parser()

    # Case A: Default 'hf' engine
    args_hf = parser.parse_args(["serve"])
    with patch("src.memory.preflight.enforce_preflight_check"), \
         patch("src.server.serving_engine.AsyncServingEngine") as mock_hf_engine, \
         patch("src.server.app.create_app"), \
         patch("uvicorn.run"):
        cmd_serve(args_hf)
        mock_hf_engine.assert_called_once()

    # Case B: Legacy alias 'mini-vllm' engine
    args_mvllm = parser.parse_args(["serve", "--engine", "mini-vllm"])
    with patch("src.memory.preflight.enforce_preflight_check"), \
         patch("src.engine.llm_engine.MiniVLLMEngine") as mock_core_mv, \
         patch("src.server.serving_adapter.AsyncServingAdapter") as mock_adapter_mv, \
         patch("src.server.app.create_app"), \
         patch("uvicorn.run"):
        cmd_serve(args_mvllm)
        mock_core_mv.assert_called_once()
        mock_adapter_mv.assert_called_once()

    # Case C: Canonical 'localstrm' engine
    args_localstrm = parser.parse_args(["serve", "--engine", "localstrm"])
    with patch("src.memory.preflight.enforce_preflight_check"), \
         patch("src.engine.llm_engine.MiniVLLMEngine") as mock_core_ls, \
         patch("src.server.serving_adapter.AsyncServingAdapter") as mock_adapter_ls, \
         patch("src.server.app.create_app"), \
         patch("uvicorn.run"):
        cmd_serve(args_localstrm)
        mock_core_ls.assert_called_once()
        mock_adapter_ls.assert_called_once()


def test_cmd_serve_unsupported_engine_exits_code_2(capsys):
    """Validates that cmd_serve exits with code 2 if an unknown engine is provided."""
    import argparse
    args = argparse.Namespace(
        model="Qwen/Qwen2.5-0.5B",
        device="cpu",
        quant="none",
        host="127.0.0.1",
        port=8000,
        engine="completely_unknown_engine",
        verbose=False
    )
    with patch("src.memory.preflight.enforce_preflight_check"):
        with pytest.raises(SystemExit) as exc_info:
            cmd_serve(args)
        assert exc_info.value.code == 2
    captured = capsys.readouterr().out
    assert "Unsupported engine 'completely_unknown_engine'" in captured


# =============================================================================
# 2. FastAPI Endpoints Integration Tests (with Adapter & FakeEngine)
# =============================================================================

@pytest.fixture
def test_app_and_adapter():
    fake_engine = FakeEngine(model_name="Qwen/Qwen2.5-0.5B")
    adapter = AsyncServingAdapter(fake_engine)
    app = create_app(adapter)
    return app, adapter


def test_api_server_health_and_models(test_app_and_adapter):
    """7. Validates health, root, and models catalog endpoints with AsyncServingAdapter."""
    app, adapter = test_app_and_adapter
    with TestClient(app) as client:
        # GET /
        resp_root = client.get("/")
        assert resp_root.status_code == 200
        data_root = resp_root.json()
        assert data_root["status"] == "online"
        assert data_root["model"] == "Qwen/Qwen2.5-0.5B"

        # GET /health
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        data_health = resp_health.json()
        assert data_health["status"] == "healthy"
        assert "running_requests" in data_health

        # GET /v1/models
        resp_models = client.get("/v1/models")
        assert resp_models.status_code == 200
        data_models = resp_models.json()
        assert data_models["object"] == "list"
        model_ids = [m["id"] for m in data_models["data"]]
        assert "Qwen/Qwen2.5-0.5B" in model_ids

        # GET /metrics
        resp_metrics = client.get("/metrics")
        assert resp_metrics.status_code == 200
        assert "requests_total 42" in resp_metrics.text


def test_api_chat_completions_non_streaming(test_app_and_adapter):
    """8. Validates non-streaming POST /v1/chat/completions."""
    app, adapter = test_app_and_adapter
    with TestClient(app) as client:
        payload = {
            "model": "Qwen/Qwen2.5-0.5B",
            "messages": [
                {"role": "user", "content": "Hello world!"}
            ],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": False
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Qwen/Qwen2.5-0.5B"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["finish_reason"] == "length"
        assert data["choices"][0]["message"]["content"] == "ABCD"
        assert data["usage"]["completion_tokens"] == 4


def test_api_chat_completions_streaming_sse(test_app_and_adapter):
    """9. Validates streaming SSE POST /v1/chat/completions."""
    app, adapter = test_app_and_adapter
    with TestClient(app) as client:
        payload = {
            "model": "Qwen/Qwen2.5-0.5B",
            "messages": [
                {"role": "user", "content": "Stream me"}
            ],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": True
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = [line.strip() for line in resp.text.split("\n") if line.strip()]
        assert "data: [DONE]" in lines

        # Verify chunk structure
        data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
        assert len(data_lines) >= 3
        first_payload = json.loads(data_lines[0][6:])
        assert first_payload["choices"][0]["delta"]["role"] == "assistant"


def test_api_legacy_completions(test_app_and_adapter):
    """10. Validates POST /v1/completions with both stream=False and stream=True."""
    app, adapter = test_app_and_adapter
    with TestClient(app) as client:
        # Non-streaming
        resp = client.post("/v1/completions", json={
            "model": "Qwen/Qwen2.5-0.5B",
            "prompt": "Test legacy",
            "max_tokens": 2,
            "stream": False
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["text"] == "AB"

        # Streaming
        resp_stream = client.post("/v1/completions", json={
            "model": "Qwen/Qwen2.5-0.5B",
            "prompt": "Test legacy stream",
            "max_tokens": 2,
            "stream": True
        })
        assert resp_stream.status_code == 200
        assert "data: [DONE]" in resp_stream.text


def test_api_validation_error_format(test_app_and_adapter):
    """11. Validates that request validation error conforms to OpenAI error specification."""
    app, adapter = test_app_and_adapter
    with TestClient(app) as client:
        # Missing required 'messages' field in chat completion
        resp = client.post("/v1/chat/completions", json={
            "model": "Qwen/Qwen2.5-0.5B"
        })
        assert resp.status_code == 422
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["code"] == 422


def test_api_admission_failure_reaches_client_without_hanging():
    """12. Validates that an engine KV-cache capacity failure surfaces cleanly as 503 kv_cache_full."""
    fake_engine = FakeEngine(fail_with_memory_error=True)
    adapter = AsyncServingAdapter(fake_engine)
    app = create_app(adapter)

    with TestClient(app) as client:
        # Non-streaming chat completion
        resp = client.post("/v1/chat/completions", json={
            "model": "Qwen/Qwen2.5-0.5B",
            "messages": [{"role": "user", "content": "Exhaust pool"}],
            "max_tokens": 4,
            "stream": False
        })
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == "kv_cache_full"
        assert "KV cache capacity exceeded" in data["error"]["message"]


def test_api_shutdown_terminates_worker_cleanly(test_app_and_adapter):
    """13. Validates that server shutdown stops the adapter worker cleanly."""
    app, adapter = test_app_and_adapter

    # Attach shutdown event handler as cmd_serve does
    @app.on_event("shutdown")
    async def _shutdown():
        await adapter.shutdown()

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert adapter._worker_thread.is_alive()

    # After exiting the context manager, Starlette executes shutdown handlers
    assert not adapter._worker_thread.is_alive()
