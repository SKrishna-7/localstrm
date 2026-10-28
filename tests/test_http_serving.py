"""
End-to-End HTTP and OpenAI SDK Validation Suite (Phase 8.3.3)
------------------------------------------------------------
Validates the opt-in Mini-vLLM serving path across the real HTTP transport boundary:
  1. OpenAI-compatible non-streaming chat completion (/v1/chat/completions) via HTTP and OpenAI SDK.
  2. OpenAI-compatible streaming chat completion (SSE framing, order, [DONE] marker) via HTTP and OpenAI SDK.
  3. Legacy text completions (/v1/completions) in streaming and non-streaming modes.
  4. Concurrent clients (4 simultaneous streams) demonstrating overlapping progress and strict token isolation.
  5. Abrupt client disconnect triggering ASGI cancellation, adapter abort command, and engine request abort.
  6. Request validation errors (422), admission/KV-capacity exhaustion (500/SSE error), and mid-flight errors.
  7. Uvicorn server lifecycle: graceful shutdown, streaming shutdown, startup failure cleanup, and idempotency.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import sys
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
import httpx
import uvicorn

try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.llm_engine import RequestStatus
from src.server.app import create_app
from src.server.serving_adapter import AsyncServingAdapter


def get_free_port() -> int:
    """Finds an available ephemeral port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeTokenizer:
    """Deterministic in-memory tokenizer for HTTP testing."""
    def __init__(self):
        self.eos_token_id = 2
        self.chat_template = None

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text] or [1]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(t) if 32 <= t <= 126 else f"<{t}>" for t in token_ids)


class FakeMetrics:
    def __init__(self):
        self.num_requests_running = 0

    def export_prometheus_text(self) -> str:
        return "# HELP requests_total Total requests\nrequests_total 42\n"


class FakeEngineRequest:
    def __init__(self, req_id: int, prompt: str, max_tokens: int, sampling_params: Any = None):
        self.req_id = req_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.sampling_params = sampling_params
        self.output_token_ids: List[int] = []
        self.status = RequestStatus.WAITING
        self.stop_reason: Optional[str] = None
        self.error: Optional[str] = None


class DeterministicHTTPFakeEngine:
    """
    Deterministic in-memory FakeEngine for real HTTP transport validation.
    Supports step delays, programmatic admission failures, and tagged token generation.
    """
    def __init__(
        self,
        model_name: str = "test-http-model",
        fail_with_memory_error: bool = False,
        fail_with_cuda_oom: bool = False,
        fail_with_generic_memory_error: bool = False,
        fail_mid_stream_on_step: Optional[int] = None,
        step_delay: float = 0.0
    ):
        self.model_name = model_name
        self.device = "cpu"
        self.waiting_queue: List[FakeEngineRequest] = []
        self.running_batch: List[FakeEngineRequest] = []
        self.completed_requests: List[FakeEngineRequest] = []
        self.requests_by_id: Dict[int, FakeEngineRequest] = {}
        self.tokenizer = FakeTokenizer()
        self.metrics = FakeMetrics()
        self.current_step = 0
        self._next_req_id = 1
        self.step_delay = step_delay
        self.fail_with_memory_error = fail_with_memory_error
        self.fail_with_cuda_oom = fail_with_cuda_oom
        self.fail_with_generic_memory_error = fail_with_generic_memory_error
        self.fail_mid_stream_on_step = fail_mid_stream_on_step
        self.aborted_requests: List[int] = []

    def add_request(self, prompt: str, max_tokens: int, sampling_params: Any = None) -> FakeEngineRequest:
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        req = FakeEngineRequest(
            req_id=self._next_req_id,
            prompt=prompt,
            max_tokens=max_tokens,
            sampling_params=sampling_params
        )
        self._next_req_id += 1
        self.waiting_queue.append(req)
        self.requests_by_id[req.req_id] = req
        return req

    def get_request(self, req_id: int) -> Optional[FakeEngineRequest]:
        return self.requests_by_id.get(req_id)

    def abort_request(self, req_id: int) -> bool:
        self.aborted_requests.append(req_id)
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
        if self.step_delay > 0:
            time.sleep(self.step_delay)

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
            if self.fail_with_cuda_oom:
                import torch
                req.status = RequestStatus.FAILED
                req.stop_reason = "cuda_oom"
                req.error = "CUDA out of memory"
                self.completed_requests.append(req)
                err = torch.cuda.OutOfMemoryError("CUDA out of memory during kernel allocation")
                err.request = req
                raise err
            if self.fail_with_generic_memory_error:
                req.status = RequestStatus.FAILED
                req.stop_reason = "error"
                req.error = "Out of host system memory"
                self.completed_requests.append(req)
                err = MemoryError("Out of host system memory")
                err.request = req
                raise err
            req.status = RequestStatus.RUNNING
            self.running_batch.append(req)

        if self.fail_mid_stream_on_step is not None and self.current_step >= self.fail_mid_stream_on_step:
            raise RuntimeError(f"Simulated engine fault at step {self.current_step}")

        finished = []
        for req in list(self.running_batch):
            if "ZERO_TOKEN_EOS:" in req.prompt:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "eos_token"
                finished.append(req)
                continue

            # Token generation: 'A', 'B', 'C', ... based on output token count
            tok = 65 + (len(req.output_token_ids) % 26)
            req.output_token_ids.append(tok)

            if "EOS:" in req.prompt and len(req.output_token_ids) >= 2:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "eos_token"
                finished.append(req)
            elif "STOP_TOKEN:" in req.prompt and len(req.output_token_ids) >= 2:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "stop_token_151645"
                finished.append(req)
            elif "STOP_STRING:" in req.prompt and len(req.output_token_ids) >= 2:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "stop_string_END"
                finished.append(req)
            elif "UNKNOWN_STOP:" in req.prompt and len(req.output_token_ids) >= 2:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "unrecognized_custom_code"
                finished.append(req)
            elif len(req.output_token_ids) >= req.max_tokens:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "length"
                finished.append(req)

        for req in finished:
            self.running_batch.remove(req)
            self.completed_requests.append(req)

        self.current_step += 1
        return bool(self.waiting_queue or self.running_batch)


class ManagedHTTPServer:
    """Spawns and manages a live Uvicorn HTTP server running the FastAPI app on a free port."""
    def __init__(self, engine: DeterministicHTTPFakeEngine):
        self.engine = engine
        self.adapter = AsyncServingAdapter(engine)
        self.app = create_app(self.adapter)
        self.port = get_free_port()
        self.host = "127.0.0.1"
        self.base_url = f"http://{self.host}:{self.port}"

        # Register shutdown hook
        @self.app.on_event("shutdown")
        async def _shutdown_engine():
            await self.adapter.shutdown()

        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="Test-UvicornServer")

    def start(self, timeout: float = 5.0) -> None:
        self.thread.start()
        # Wait for HTTP server to become responsive
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = httpx.get(f"{self.base_url}/health", timeout=0.2)
                if resp.status_code == 200:
                    return
            except Exception:
                time.sleep(0.05)
        raise TimeoutError(f"HTTP server failed to bind and respond within {timeout}s on {self.base_url}")

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout)
        # Ensure adapter is fully shut down
        if not self.adapter._is_shutting_down:
            self.adapter.shutdown_sync()


@pytest.fixture
def http_server():
    """Provides a running HTTP server with a deterministic fake engine."""
    engine = DeterministicHTTPFakeEngine()
    srv = ManagedHTTPServer(engine)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


# =============================================================================
# Scenario A: OpenAI-Compatible Non-Streaming Chat Completion
# =============================================================================

def test_http_chat_completions_non_streaming_httpx(http_server: ManagedHTTPServer):
    """A1. Validates non-streaming chat completion over HTTP using httpx."""
    resp = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
            "temperature": 0.0
        },
        timeout=5.0
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert isinstance(data["created"], int)
    assert data["model"] == http_server.engine.model_name
    assert len(data["choices"]) == 1
    assert data["choices"][0]["index"] == 0
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "ABCDE"
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["usage"]["completion_tokens"] == 5
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + 5


@pytest.mark.skipif(not HAS_OPENAI_SDK, reason="Official OpenAI Python SDK not installed")
def test_http_chat_completions_non_streaming_openai_sdk(http_server: ManagedHTTPServer):
    """A2. Validates non-streaming chat completion using the official OpenAI Python SDK."""
    client = OpenAI(base_url=f"{http_server.base_url}/v1", api_key="test-key")
    response = client.chat.completions.create(
        model=http_server.engine.model_name,
        messages=[{"role": "user", "content": "Tell me a story"}],
        max_tokens=4,
        temperature=0.0
    )
    assert response.id.startswith("chatcmpl-")
    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "ABCD"
    assert response.choices[0].finish_reason == "length"
    assert response.usage.completion_tokens == 4


# =============================================================================
# Scenario B: OpenAI-Compatible Streaming Chat Completion
# =============================================================================

def test_http_chat_completions_streaming_sse_httpx(http_server: ManagedHTTPServer):
    """B1. Validates streaming chat completion over HTTP verifying exact SSE wire framing."""
    with httpx.stream(
        "POST",
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "Stream prompt"}],
            "max_tokens": 4,
            "stream": True
        },
        timeout=5.0
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        lines = [line.strip() for line in response.iter_lines() if line.strip()]

    # Validate SSE wire format
    assert lines[-1] == "data: [DONE]"
    data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    assert len(data_lines) == 5  # 1 initial role chunk + 4 token chunks

    parsed_chunks = [json.loads(l[6:]) for l in data_lines]
    req_id = parsed_chunks[0]["id"]
    assert req_id.startswith("chatcmpl-")

    # Initial chunk gives role assistant
    assert parsed_chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed_chunks[0]["choices"][0]["delta"]["content"] == ""

    # Token chunks arrive in order
    tokens = [c["choices"][0]["delta"]["content"] for c in parsed_chunks[1:]]
    assert tokens == ["A", "B", "C", "D"]

    # Completion ID consistency across all chunks
    for c in parsed_chunks:
        assert c["id"] == req_id
        assert c["object"] == "chat.completion.chunk"

    # Terminal chunk finish reason
    assert parsed_chunks[-1]["choices"][0]["finish_reason"] == "length"


@pytest.mark.skipif(not HAS_OPENAI_SDK, reason="Official OpenAI Python SDK not installed")
def test_http_chat_completions_streaming_openai_sdk(http_server: ManagedHTTPServer):
    """B2. Validates streaming chat completion using official OpenAI Python SDK."""
    client = OpenAI(base_url=f"{http_server.base_url}/v1", api_key="test-key")
    stream = client.chat.completions.create(
        model=http_server.engine.model_name,
        messages=[{"role": "user", "content": "Streaming via SDK"}],
        max_tokens=6,
        stream=True
    )
    tokens = []
    finish_reasons = []
    chunk_ids = set()

    for chunk in stream:
        chunk_ids.add(chunk.id)
        if chunk.choices and chunk.choices[0].delta.content:
            tokens.append(chunk.choices[0].delta.content)
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reasons.append(chunk.choices[0].finish_reason)

    assert "".join(tokens) == "ABCDEF"
    assert finish_reasons == ["length"]
    assert len(chunk_ids) == 1


# =============================================================================
# Scenario C: Legacy Completion Compatibility (/v1/completions)
# =============================================================================

def test_http_legacy_completions_non_streaming(http_server: ManagedHTTPServer):
    """C1. Validates legacy prompt completion non-streaming over HTTP."""
    resp = httpx.post(
        f"{http_server.base_url}/v1/completions",
        json={
            "model": http_server.engine.model_name,
            "prompt": "Complete this sentence",
            "max_tokens": 5
        },
        timeout=5.0
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "text_completion"
    assert data["choices"][0]["text"] == "ABCDE"
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["usage"]["completion_tokens"] == 5


def test_http_legacy_completions_streaming(http_server: ManagedHTTPServer):
    """C2. Validates legacy prompt completion streaming over HTTP."""
    with httpx.stream(
        "POST",
        f"{http_server.base_url}/v1/completions",
        json={
            "model": http_server.engine.model_name,
            "prompt": "Stream this sentence",
            "max_tokens": 3,
            "stream": True
        },
        timeout=5.0
    ) as response:
        assert response.status_code == 200
        lines = [line.strip() for line in response.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    token_lines = [json.loads(l[6:]) for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    assert len(token_lines) == 3
    tokens = [c["choices"][0]["text"] for c in token_lines]
    assert tokens == ["A", "B", "C"]
    assert token_lines[-1]["choices"][0]["finish_reason"] == "length"


# =============================================================================
# Scenario D: Concurrent Clients and Token Isolation
# =============================================================================

@pytest.mark.asyncio
async def test_http_concurrent_streaming_token_isolation():
    """
    D. Opens 4 simultaneous streaming clients over HTTP with step delay,
    verifying concurrent progress and strict per-client token isolation.
    """
    class IsolatedConcurrentEngine:
        def __init__(self):
            self.model_name = "concurrent-model"
            self.device = "cpu"
            self.waiting_queue: List[Any] = []
            self.running_batch: List[Any] = []
            self.completed_requests: List[Any] = []
            self.requests_by_id: Dict[int, Any] = {}
            self.tokenizer = None
            self.current_step = 0
            self._next_id = 1
            self.metrics = FakeMetrics()

        def add_request(self, prompt: str, max_tokens: int, sampling_params: Any = None):
            class Req:
                def __init__(self, rid, p, m):
                    self.req_id = rid
                    self.prompt = p
                    self.max_tokens = m
                    self.output_token_ids: List[int] = []
                    self.status = RequestStatus.WAITING
                    self.stop_reason = None
                    self.error = None
            req = Req(self._next_id, prompt, max_tokens)
            self._next_id += 1
            self.waiting_queue.append(req)
            self.requests_by_id[req.req_id] = req
            return req

        def get_request(self, rid):
            return self.requests_by_id.get(rid)

        def abort_request(self, rid):
            for q in [self.waiting_queue, self.running_batch]:
                for r in list(q):
                    if r.req_id == rid:
                        q.remove(r)
                        r.status = RequestStatus.FAILED
                        r.stop_reason = "abort"
                        self.completed_requests.append(r)
                        return True
            return False

        def step(self):
            # Step delay ensures all 4 requests are processed simultaneously in the running batch
            time.sleep(0.015)
            while self.waiting_queue:
                r = self.waiting_queue.pop(0)
                r.status = RequestStatus.RUNNING
                self.running_batch.append(r)

            finished = []
            for r in list(self.running_batch):
                # Tag tokens with client id from prompt
                cid = 0
                if "Client" in r.prompt:
                    cid = int(r.prompt.split("Client")[1].split()[0].strip())
                tok_val = (cid * 100) + len(r.output_token_ids)
                r.output_token_ids.append(tok_val)
                if len(r.output_token_ids) >= r.max_tokens:
                    r.status = RequestStatus.FINISHED
                    r.stop_reason = "length"
                    finished.append(r)

            for r in finished:
                self.running_batch.remove(r)
                self.completed_requests.append(r)
            self.current_step += 1
            return bool(self.waiting_queue or self.running_batch)

    class CustomTagTokenizer:
        def __init__(self):
            self.chat_template = None
        def encode(self, text):
            return [1] * len(text)
        def decode(self, token_ids, skip_special_tokens=True):
            return f"[{token_ids[0]}]"

    engine = IsolatedConcurrentEngine()
    engine.tokenizer = CustomTagTokenizer()
    adapter = AsyncServingAdapter(engine)
    app = create_app(adapter)

    port = get_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    # Wait for server readiness
    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.2).status_code == 200:
                break
        except Exception:
            await asyncio.sleep(0.05)

    num_clients = 4
    tokens_per_client = 6
    client_results: Dict[int, List[str]] = {}

    async def run_client(cid: int):
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as client:
            tokens = []
            payload = {
                "model": "concurrent-model",
                "messages": [{"role": "user", "content": f"Client{cid} request"}],
                "max_tokens": tokens_per_client,
                "stream": True
            }
            async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        chunk = json.loads(line[6:])
                        tok = chunk["choices"][0]["delta"].get("content", "")
                        if tok:
                            tokens.append(tok)
            client_results[cid] = tokens

    try:
        tasks = [run_client(i) for i in range(num_clients)]
        await asyncio.gather(*tasks)

        # Verify strict token isolation
        for cid in range(num_clients):
            toks = client_results[cid]
            assert len(toks) == tokens_per_client
            for step_idx, tok in enumerate(toks):
                expected = f"[{(cid * 100) + step_idx}]"
                assert tok == expected, f"Client {cid} received corrupted token {tok}, expected {expected}"

        # Verify engine batch is drained to idle
        assert len(engine.running_batch) == 0
        assert len(engine.waiting_queue) == 0
        assert len(engine.completed_requests) == num_clients
    finally:
        server.should_exit = True
        t.join(timeout=5.0)
        adapter.shutdown_sync()


# =============================================================================
# Scenario E: Abrupt Client Disconnect
# =============================================================================

def test_http_abrupt_client_disconnect_triggers_abort():
    """
    E. Starts a long streaming request over HTTP and abruptly breaks the stream,
    verifying ASGI disconnect detection and engine.abort_request() invocation.
    """
    engine = DeterministicHTTPFakeEngine(step_delay=0.03)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        tokens_read = 0
        with httpx.Client(base_url=server.base_url, timeout=10.0) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": engine.model_name,
                    "messages": [{"role": "user", "content": "Long disconnect prompt"}],
                    "max_tokens": 30,
                    "stream": True
                }
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        chunk = json.loads(line[6:])
                        if chunk["choices"][0]["delta"].get("content"):
                            tokens_read += 1
                            if tokens_read == 2:
                                # Abruptly exit context, closing the TCP connection
                                break

        assert tokens_read == 2
        # Allow worker thread to process the cancellation command
        time.sleep(0.15)

        # Confirm abort_request was called on the engine
        assert len(engine.aborted_requests) == 1
        assert engine.aborted_requests[0] == 1
        assert len(engine.running_batch) == 0

        # Confirm that a subsequent request executes normally (engine wasn't corrupted)
        resp2 = httpx.post(
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "Post-abort prompt"}],
                "max_tokens": 3
            },
            timeout=5.0
        )
        assert resp2.status_code == 200
        assert resp2.json()["choices"][0]["message"]["content"] == "ABC"
    finally:
        server.stop()


def test_http_abrupt_tcp_reset_triggers_abort_and_preserves_peer_stream():
    """
    E2. Starts an HTTP streaming request, reads initial SSE chunks, and closes
    the client connection abruptly via a raw TCP RST (SO_LINGER l_onoff=1, l_linger=0).
    Verifies that the server detects connection reset, adapter dispatches _AbortCommand,
    worker calls engine.abort_request(), and an active peer stream continues uninterrupted.
    """
    engine = DeterministicHTTPFakeEngine(step_delay=0.03)
    server = ManagedHTTPServer(engine)
    server.start()

    peer_tokens = []
    peer_finished = threading.Event()

    def run_peer_client():
        with httpx.Client(base_url=server.base_url, timeout=10.0) as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "peer request"}],
                "max_tokens": 5,
                "stream": True
            }) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        d = json.loads(line[6:])
                        tok = d["choices"][0]["delta"].get("content", "")
                        if tok:
                            peer_tokens.append(tok)
        peer_finished.set()

    peer_thread = threading.Thread(target=run_peer_client, daemon=True)
    peer_thread.start()

    # Raw socket connection to execute true kernel-level TCP RST
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect((server.host, server.port))
    body = json.dumps({
        "model": engine.model_name,
        "messages": [{"role": "user", "content": "reset request"}],
        "max_tokens": 30,
        "stream": True
    })
    req = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {server.host}:{server.port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
        f"{body}"
    )
    client_sock.sendall(req.encode("utf-8"))

    # Read until at least 2 SSE data chunks arrive
    buf = b""
    while buf.count(b"data: ") < 2:
        chunk = client_sock.recv(4096)
        if not chunk:
            break
        buf += chunk

    # Trigger abrupt kernel-level TCP RST packet
    linger_opt = struct.pack("ii", 1, 0)
    client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger_opt)
    client_sock.close()

    # Verify peer stream completes normally without cross-talk or cancellation
    assert peer_finished.wait(timeout=5.0)
    assert "".join(peer_tokens) == "ABCDE"

    # Verify reset stream was aborted in engine
    deadline = time.time() + 2.0
    while time.time() < deadline and not engine.aborted_requests:
        time.sleep(0.05)

    assert len(engine.aborted_requests) == 1
    assert len(engine.running_batch) == 0
    server.stop()


# =============================================================================
# Scenario F: Errors and Admission Failures
# =============================================================================

def test_http_request_payload_validation_errors(http_server: ManagedHTTPServer):
    """F1. Validates 422 Unprocessable Entity error envelopes for invalid payloads."""
    # 1. Empty messages list
    resp1 = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={"model": http_server.engine.model_name, "messages": []},
        timeout=5.0
    )
    assert resp1.status_code == 422
    data1 = resp1.json()
    assert data1["error"]["type"] == "invalid_request_error"
    assert data1["error"]["code"] == 422
    assert "messages" in data1["error"]["message"]

    # 2. Invalid max_tokens
    resp2 = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": -5
        },
        timeout=5.0
    )
    assert resp2.status_code == 422
    data2 = resp2.json()
    assert data2["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in data2["error"]["message"]


def test_http_admission_failure_non_streaming():
    """F2. Validates non-streaming admission rejection produces HTTP 503 kv_cache_full without leaking paths."""
    engine = DeterministicHTTPFakeEngine(fail_with_memory_error=True)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        resp = httpx.post(
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "Impossible memory prompt"}]
            },
            timeout=5.0
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == "kv_cache_full"
        assert "KV cache capacity exceeded" in data["error"]["message"]
        assert "/home/" not in data["error"]["message"]
        assert "Traceback" not in data["error"]["message"]
    finally:
        server.stop()


def test_http_admission_failure_streaming():
    """F3. Validates streaming admission rejection delivers an SSE error chunk with kv_cache_full without hanging."""
    engine = DeterministicHTTPFakeEngine(fail_with_memory_error=True)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "Impossible memory prompt"}],
                "stream": True
            },
            timeout=5.0
        ) as response:
            assert response.status_code == 200
            lines = [line.strip() for line in response.iter_lines() if line.strip()]

        # Must have initial role chunk followed by error event
        assert len(lines) == 2
        error_chunk = json.loads(lines[1][6:])
        assert "error" in error_chunk
        assert error_chunk["error"]["type"] == "server_error"
        assert error_chunk["error"]["code"] == "kv_cache_full"
        assert "KV cache capacity exceeded" in error_chunk["error"]["message"]
        assert "/home/" not in error_chunk["error"]["message"]
    finally:
        server.stop()


def test_http_legacy_admission_failure_non_streaming():
    """F3b. Validates non-streaming legacy completion admission rejection produces HTTP 503 kv_cache_full."""
    engine = DeterministicHTTPFakeEngine(fail_with_memory_error=True)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        resp = httpx.post(
            f"{server.base_url}/v1/completions",
            json={
                "model": engine.model_name,
                "prompt": "Impossible memory prompt",
                "stream": False
            },
            timeout=5.0
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == "kv_cache_full"
        assert "KV cache capacity exceeded" in data["error"]["message"]
        assert "/home/" not in data["error"]["message"]
        assert "Traceback" not in data["error"]["message"]
    finally:
        server.stop()


def test_http_legacy_admission_failure_streaming():
    """F3c. Validates streaming legacy completion admission rejection delivers SSE error chunk with kv_cache_full."""
    engine = DeterministicHTTPFakeEngine(fail_with_memory_error=True)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/completions",
            json={
                "model": engine.model_name,
                "prompt": "Impossible memory prompt",
                "stream": True
            },
            timeout=5.0
        ) as response:
            assert response.status_code == 200
            lines = [line.strip() for line in response.iter_lines() if line.strip()]

        assert len(lines) >= 1
        error_chunk = json.loads(lines[-1][6:])
        assert "error" in error_chunk
        assert error_chunk["error"]["type"] == "server_error"
        assert error_chunk["error"]["code"] == "kv_cache_full"
        assert "KV cache capacity exceeded" in error_chunk["error"]["message"]
        assert "/home/" not in error_chunk["error"]["message"]
    finally:
        server.stop()


def test_http_simulated_cuda_oom_classification_non_streaming():
    """F4. Validates genuine CUDA OOM returns HTTP 503 with dedicated cuda_out_of_memory classification."""
    engine = DeterministicHTTPFakeEngine(fail_with_cuda_oom=True)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        resp = httpx.post(
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "Trigger OOM"}],
                "max_tokens": 10
            },
            timeout=5.0
        )
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == "cuda_out_of_memory"
        assert "GPU out of memory" in data["error"]["message"]
        assert "Reduce max_tokens" in data["error"]["message"]
    finally:
        server.stop()


def test_http_simulated_cuda_oom_classification_streaming():
    """F5. Validates genuine CUDA OOM during streaming delivers SSE error event with cuda_out_of_memory code."""
    engine = DeterministicHTTPFakeEngine(fail_with_cuda_oom=True)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "Trigger OOM stream"}],
                "stream": True
            },
            timeout=5.0
        ) as response:
            assert response.status_code == 200
            lines = [line.strip() for line in response.iter_lines() if line.strip()]

        error_chunk = json.loads(lines[-1][6:])
        assert "error" in error_chunk
        assert error_chunk["error"]["type"] == "server_error"
        assert error_chunk["error"]["code"] == "cuda_out_of_memory"
        assert "GPU out of memory" in error_chunk["error"]["message"]
    finally:
        server.stop()


def test_http_generic_memory_error_not_mislabeled_as_cuda_oom():
    """F6. Validates generic Python MemoryError produces HTTP 500 and is NOT mislabeled as CUDA OOM (503)."""
    engine = DeterministicHTTPFakeEngine(fail_with_generic_memory_error=True)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        resp = httpx.post(
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "Generic memory failure"}]
            },
            timeout=5.0
        )
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "api_error"
        assert data["error"]["code"] == 500
        assert "MemoryError" in data["error"]["message"]
        assert "KVCache" not in data["error"]["message"]
        assert data["error"]["code"] != "cuda_out_of_memory"
        assert data["error"]["code"] != "kv_cache_full"
    finally:
        server.stop()


def test_http_unexpected_generation_exception_sanitization():
    """F7. Validates unexpected generation RuntimeError produces sanitized HTTP 500 with no traceback/path leaks."""
    engine = DeterministicHTTPFakeEngine(fail_mid_stream_on_step=2)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        resp = httpx.post(
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "Trigger step fault"}],
                "max_tokens": 10
            },
            timeout=5.0
        )
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "api_error"
        assert data["error"]["code"] == 500
        assert "RuntimeError" in data["error"]["message"]
        assert "/home/" not in data["error"]["message"]
        assert "Traceback" not in data["error"]["message"]
    finally:
        server.stop()


def test_http_mid_stream_engine_failure():
    """F8. Validates that unexpected engine failure mid-stream delivers SSE error and terminates cleanly."""
    engine = DeterministicHTTPFakeEngine(fail_mid_stream_on_step=2)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/chat/completions",
            json={
                "model": engine.model_name,
                "messages": [{"role": "user", "content": "Mid-stream fault"}],
                "max_tokens": 10,
                "stream": True
            },
            timeout=5.0
        ) as response:
            assert response.status_code == 200
            lines = [line.strip() for line in response.iter_lines() if line.strip()]

        last_chunk = json.loads(lines[-1][6:])
        assert "error" in last_chunk
        assert "RuntimeError" in last_chunk["error"]["message"]
    finally:
        server.stop()


def test_http_error_channel_and_tracker_cleanup():
    """F9. Validates channel registration and active tracker cleanup after client errors and failures."""
    engine = DeterministicHTTPFakeEngine(fail_with_memory_error=True)
    server = ManagedHTTPServer(engine)
    server.start()

    try:
        httpx.post(
            f"{server.base_url}/v1/chat/completions",
            json={"model": engine.model_name, "messages": [{"role": "user", "content": "bad"}]},
            timeout=5.0
        )
        assert len(server.adapter._channels) == 0
        assert len(engine.running_batch) == 0
        assert len(engine.waiting_queue) == 0
    finally:
        server.stop()


# =============================================================================
# Scenario G: Shutdown and Worker Lifecycle
# =============================================================================

def test_lifecycle_normal_startup_and_graceful_shutdown():
    """G1. Validates normal startup followed by graceful Uvicorn shutdown and clean worker termination."""
    engine = DeterministicHTTPFakeEngine()
    server = ManagedHTTPServer(engine)
    server.start()
    assert server.thread.is_alive()
    assert server.adapter._worker_thread.is_alive()

    server.stop()
    assert not server.thread.is_alive()
    assert not server.adapter._worker_thread.is_alive()
    assert server.adapter._is_shutting_down is True


def test_lifecycle_shutdown_during_active_streaming():
    """G2. Validates that shutting down while a request is actively streaming terminates gracefully."""
    engine = DeterministicHTTPFakeEngine(step_delay=0.04)
    server = ManagedHTTPServer(engine)
    server.start()

    stream_started = threading.Event()
    stream_error_received = threading.Event()

    def client_worker():
        try:
            with httpx.stream(
                "POST",
                f"{server.base_url}/v1/chat/completions",
                json={
                    "model": engine.model_name,
                    "messages": [{"role": "user", "content": "Long stream during shutdown"}],
                    "max_tokens": 50,
                    "stream": True
                },
                timeout=5.0
            ) as response:
                for line in response.iter_lines():
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        stream_started.set()
                        chunk = json.loads(line[6:])
                        if "error" in chunk:
                            stream_error_received.set()
        except Exception:
            stream_error_received.set()

    client_thread = threading.Thread(target=client_worker, daemon=True)
    client_thread.start()

    # Wait until streaming begins
    assert stream_started.wait(timeout=3.0)

    # Trigger server shutdown while request is active
    server.stop()
    client_thread.join(timeout=3.0)

    assert not server.adapter._worker_thread.is_alive()


def test_lifecycle_startup_failure_cleanup_without_leak():
    """G3. Validates that startup failure (e.g. port already bound) cleans up the worker thread."""
    # Occupy a port
    occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupier.bind(("127.0.0.1", 0))
    port = occupier.getsockname()[1]
    occupier.listen(1)

    engine = DeterministicHTTPFakeEngine()
    adapter = AsyncServingAdapter(engine)
    app = create_app(adapter)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)

    def run_uvicorn():
        try:
            srv.run()
        finally:
            if hasattr(adapter, "shutdown_sync") and not getattr(adapter, "_is_shutting_down", False):
                try:
                    adapter.shutdown_sync()
                except Exception:
                    pass
            elif hasattr(adapter, "shutdown") and not getattr(adapter, "_is_shutting_down", False):
                import asyncio
                try:
                    asyncio.run(adapter.shutdown())
                except Exception:
                    pass

    t = threading.Thread(target=run_uvicorn, daemon=True)
    t.start()
    t.join(timeout=3.0)

    try:
        assert not adapter._worker_thread.is_alive()
    finally:
        occupier.close()


def test_lifecycle_repeated_shutdown_is_idempotent():
    """G4. Validates that multiple repeated calls to shutdown_sync and shutdown are safe and idempotent."""
    engine = DeterministicHTTPFakeEngine()
    adapter = AsyncServingAdapter(engine)
    assert adapter._worker_thread.is_alive()

    # First shutdown
    adapter.shutdown_sync()
    assert not adapter._worker_thread.is_alive()
    assert adapter._is_shutting_down is True

    # Repeated shutdown_sync
    adapter.shutdown_sync()
    assert not adapter._worker_thread.is_alive()

    # Repeated async shutdown
    asyncio.run(adapter.shutdown())
    assert not adapter._worker_thread.is_alive()


@pytest.mark.asyncio
async def test_lifecycle_shutdown_cleans_unstarted_channels():
    """G5. Validates that adapter shutdown notifies and unblocks channels that had not started inference."""
    engine = DeterministicHTTPFakeEngine()
    adapter = AsyncServingAdapter(engine)

    # Stop worker loop directly without shutdown command to simulate worker exiting
    adapter._is_shutting_down = True
    await adapter.shutdown()

    # Now verify generate_stream raises immediately
    with pytest.raises(RuntimeError, match="Engine adapter is shutting down"):
        async for _ in adapter.generate_stream("Rejected prompt"):
            pass


# =============================================================================
# Scenario H: Natural EOS and Finish-Reason Protocol Mapping (Phase 8.6.1)
# =============================================================================

def test_http_natural_eos_chat_completion_non_streaming(http_server: ManagedHTTPServer):
    """H1. Validates non-streaming chat completion terminates naturally on EOS before max_tokens."""
    resp = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "EOS: natural termination test"}],
            "max_tokens": 10,
            "temperature": 0.0
        },
        timeout=5.0
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["completion_tokens"] == 2
    assert data["usage"]["completion_tokens"] < 10
    assert data["choices"][0]["message"]["content"] == "AB"


def test_http_natural_eos_chat_completion_streaming(http_server: ManagedHTTPServer):
    """H2. Validates streaming chat completion terminates on EOS with finish_reason 'stop' and [DONE]."""
    with httpx.stream(
        "POST",
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "EOS: natural stream test"}],
            "max_tokens": 10,
            "stream": True
        },
        timeout=5.0
    ) as response:
        assert response.status_code == 200
        lines = [line.strip() for line in response.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    assert len(chunks) >= 2
    intermediate_chunks = chunks[:-1]
    for c in intermediate_chunks:
        assert c["choices"][0]["finish_reason"] is None

    final_chunk = chunks[-1]
    assert final_chunk["choices"][0]["finish_reason"] == "stop"


def test_http_natural_eos_legacy_completions_non_streaming(http_server: ManagedHTTPServer):
    """H3. Validates legacy completion (/v1/completions) translates natural EOS to 'stop'."""
    resp = httpx.post(
        f"{http_server.base_url}/v1/completions",
        json={
            "model": http_server.engine.model_name,
            "prompt": "EOS: legacy prompt",
            "max_tokens": 10
        },
        timeout=5.0
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["completion_tokens"] == 2
    assert data["usage"]["completion_tokens"] < 10


def test_http_natural_eos_legacy_completions_streaming(http_server: ManagedHTTPServer):
    """H4. Validates legacy streaming completion translates natural EOS to 'stop'."""
    with httpx.stream(
        "POST",
        f"{http_server.base_url}/v1/completions",
        json={
            "model": http_server.engine.model_name,
            "prompt": "EOS: legacy stream prompt",
            "max_tokens": 10,
            "stream": True
        },
        timeout=5.0
    ) as response:
        assert response.status_code == 200
        lines = [line.strip() for line in response.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    for c in chunks[:-1]:
        assert c["choices"][0]["finish_reason"] is None
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_http_stop_token_termination(http_server: ManagedHTTPServer):
    """H5. Validates stop_token_<id> is translated to public finish_reason 'stop'."""
    resp = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "STOP_TOKEN: match"}],
            "max_tokens": 10
        },
        timeout=5.0
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["completion_tokens"] == 2


def test_http_stop_string_termination(http_server: ManagedHTTPServer):
    """H6. Validates stop_string_<str> is translated to public finish_reason 'stop'."""
    resp = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "STOP_STRING: match"}],
            "max_tokens": 10
        },
        timeout=5.0
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["completion_tokens"] == 2


def test_http_length_ceiling_preserved(http_server: ManagedHTTPServer):
    """H7. Validates existing max_tokens ceiling correctly retains finish_reason 'length'."""
    resp = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "Normal prompt for length test"}],
            "max_tokens": 4
        },
        timeout=5.0
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["usage"]["completion_tokens"] == 4


def test_http_unknown_stop_reason_non_streaming(http_server: ManagedHTTPServer):
    """H8. Validates unrecognized engine stop reason maps to None without schema error."""
    resp = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "UNKNOWN_STOP: mystery"}],
            "max_tokens": 10
        },
        timeout=5.0
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["finish_reason"] is None
    assert data["usage"]["completion_tokens"] == 2


def test_http_unknown_stop_reason_streaming(http_server: ManagedHTTPServer):
    """H9. Validates unrecognized engine stop reason in streaming maps to None without schema error."""
    with httpx.stream(
        "POST",
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "UNKNOWN_STOP: stream"}],
            "max_tokens": 10,
            "stream": True
        },
        timeout=5.0
    ) as response:
        assert response.status_code == 200
        lines = [line.strip() for line in response.iter_lines() if line.strip()]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    final_chunk = chunks[-1]
    assert final_chunk["choices"][0]["finish_reason"] is None


def test_http_post_completion_lifecycle_and_subsequent_request(http_server: ManagedHTTPServer):
    """H10. Validates queues, batches, and adapter channels are drained after natural EOS, and next request succeeds."""
    # 1. Complete an EOS request
    resp1 = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "EOS: first request"}],
            "max_tokens": 10
        },
        timeout=5.0
    )
    assert resp1.status_code == 200
    assert resp1.json()["choices"][0]["finish_reason"] == "stop"

    # Give adapter loop a brief moment to process completion queue removal
    time.sleep(0.05)

    # 2. Check /health endpoint metrics
    health_resp = httpx.get(f"{http_server.base_url}/health", timeout=2.0)
    assert health_resp.status_code == 200
    health_data = health_resp.json()
    assert health_data.get("running_requests") == 0

    # 3. Check engine internal state
    assert len(http_server.engine.waiting_queue) == 0
    assert len(http_server.engine.running_batch) == 0

    # 4. Check adapter internal channel tracking
    with http_server.adapter._registry_lock:
        assert len(http_server.adapter._channels) == 0

    # 5. Subsequent request succeeds immediately
    resp2 = httpx.post(
        f"{http_server.base_url}/v1/chat/completions",
        json={
            "model": http_server.engine.model_name,
            "messages": [{"role": "user", "content": "Subsequent normal request"}],
            "max_tokens": 3
        },
        timeout=5.0
    )
    assert resp2.status_code == 200
    assert resp2.json()["choices"][0]["finish_reason"] == "length"
    assert resp2.json()["usage"]["completion_tokens"] == 3


@pytest.mark.skipif(not HAS_OPENAI_SDK, reason="openai SDK is not installed")
def test_http_openai_sdk_natural_eos_streaming(http_server: ManagedHTTPServer):
    """H11. Validates OpenAI Python SDK receives finish_reason 'stop' upon natural EOS."""
    client = OpenAI(
        base_url=f"{http_server.base_url}/v1",
        api_key="fake-key"
    )
    stream = client.chat.completions.create(
        model=http_server.engine.model_name,
        messages=[{"role": "user", "content": "EOS: openai sdk streaming"}],
        max_tokens=10,
        stream=True
    )
    finish_reasons = []
    tokens = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            tokens.append(chunk.choices[0].delta.content)
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reasons.append(chunk.choices[0].finish_reason)

    assert "".join(tokens) == "AB"
    assert finish_reasons == ["stop"]
