"""
Unit and Integration Tests for Asynchronous Serving Adapter Core (Phase 8.3.1)
-----------------------------------------------------------------------------
Validates:
  1. Single request receives ordered token deltas and a terminal event.
  2. Concurrent requests receive isolated token streams without cross-talk.
  3. Adapter accepts submissions while inference work is actively pending.
  4. Cancellation is processed safely on the worker thread, not the event loop.
  5. A request failure is isolated and does not abort healthy concurrent requests.
  6. Synchronous engine steps never execute on the asyncio event-loop thread.
  7. Output channel registration prevents early-token delivery races.
  8. Empty-work operation conserves CPU and does not busy-spin.
  9. Graceful shutdown terminates active requests and stops the worker cleanly.
  10. Unsupported or invalid sampling parameters fail explicitly with ValueError.
  11. Non-streaming generate_complete() returns complete assembled text and usage.
  12. format_chat_prompt() correctly formats conversation messages.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from typing import Any, List, Optional, Tuple

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.llm_engine import RequestStatus
from src.engine.sampling import SamplingParams
from src.server.serving_adapter import AsyncServingAdapter, validate_sampling_params


class FakeTokenizer:
    """Deterministic in-memory tokenizer for testing."""
    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return [ord(c) for c in text]

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


class FakeEngine:
    """
    Deterministic in-memory FakeEngine mimicking MiniVLLMEngine.
    Tracks calling threads, step executions, and allows programmatic failure injection.
    """
    def __init__(
        self,
        fail_on_req_id: Optional[int] = None,
        fail_with_memory_error: bool = False,
        step_delay: float = 0.0
    ):
        self.waiting_queue: List[FakeRequest] = []
        self.running_batch: List[FakeRequest] = []
        self.completed_requests: List[FakeRequest] = []
        self.requests_by_id: dict[int, FakeRequest] = {}
        self.tokenizer = FakeTokenizer()
        self.current_step = 0
        self._next_req_id = 1
        self.step_count = 0
        self.calling_threads: List[int] = []
        self.abort_invoked = False
        self.abort_thread_id: Optional[int] = None
        self.fail_on_req_id = fail_on_req_id
        self.fail_with_memory_error = fail_with_memory_error
        self.step_delay = step_delay
        self.device = torch.device("cpu")

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
        self.abort_invoked = True
        self.abort_thread_id = threading.get_ident()
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
        self.step_count += 1
        self.calling_threads.append(threading.get_ident())
        if self.step_delay > 0:
            time.sleep(self.step_delay)

        # 1. Admission phase
        while self.waiting_queue:
            req = self.waiting_queue.pop(0)
            if self.fail_on_req_id == req.req_id and self.fail_with_memory_error:
                req.status = RequestStatus.FAILED
                req.stop_reason = "kv_cache_full"
                req.error = f"request #{req.req_id} capacity exceeded total pool"
                self.completed_requests.append(req)
                err = MemoryError(req.error)
                err.request = req
                raise err
            req.status = RequestStatus.RUNNING
            self.running_batch.append(req)

        # 2. Execution phase
        finished_this_step = []
        for req in list(self.running_batch):
            if self.fail_on_req_id == req.req_id and not self.fail_with_memory_error:
                req.status = RequestStatus.FAILED
                req.stop_reason = "kv_cache_full"
                req.error = "Out of KV Cache physical blocks!"
                finished_this_step.append(req)
                continue

            # Deterministic token emission
            if req.prompt.startswith("ZERO_TOKEN_EOS:"):
                req.status = RequestStatus.FINISHED
                req.stop_reason = "eos_token"
                finished_this_step.append(req)
                continue

            if req.prompt.startswith("Alpha:"):
                next_tok = 65 + len(req.output_token_ids)  # 'A', 'B', 'C', ...
            elif req.prompt.startswith("Beta:"):
                next_tok = 97 + len(req.output_token_ids)  # 'a', 'b', 'c', ...
            else:
                next_tok = 48 + (len(req.output_token_ids) % 10)  # '0', '1', ...

            req.output_token_ids.append(next_tok)

            if req.prompt.startswith("EOS:") and len(req.output_token_ids) >= 2:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "eos_token"
                finished_this_step.append(req)
            elif req.prompt.startswith("STOP_TOKEN:") and len(req.output_token_ids) >= 2:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "stop_token_151645"
                finished_this_step.append(req)
            elif req.prompt.startswith("STOP_STRING:") and len(req.output_token_ids) >= 2:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "stop_string_END"
                finished_this_step.append(req)
            elif req.prompt.startswith("UNKNOWN_STOP:") and len(req.output_token_ids) >= 2:
                req.status = RequestStatus.FINISHED
                req.stop_reason = "unrecognized_custom_code"
                finished_this_step.append(req)
            elif len(req.output_token_ids) >= req.max_tokens:
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
# Test Suite
# =============================================================================

@pytest.mark.asyncio
async def test_single_request_ordered_deltas_and_terminal_event():
    """1. Validates that one request receives ordered token deltas and a terminal event."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        chunks: List[Tuple[str, bool, bool, Optional[str]]] = []
        async for chunk in adapter.generate_stream("Hello", max_tokens=3):
            chunks.append(chunk)

        assert len(chunks) == 3

        # Chunk 0: first token
        text0, is_first0, is_last0, finish0 = chunks[0]
        assert text0 == "0"
        assert is_first0 is True
        assert is_last0 is False
        assert finish0 is None

        # Chunk 1: intermediate token
        text1, is_first1, is_last1, finish1 = chunks[1]
        assert text1 == "1"
        assert is_first1 is False
        assert is_last1 is False
        assert finish1 is None

        # Chunk 2: terminal token
        text2, is_first2, is_last2, finish2 = chunks[2]
        assert text2 == "2"
        assert is_first2 is False
        assert is_last2 is True
        assert finish2 == "length"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_concurrent_requests_isolated_event_delivery():
    """2. Validates that two concurrent requests receive only their own events without cross-talk."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        async def collect_stream(prompt: str, max_tokens: int) -> List[str]:
            tokens = []
            async for text, _, _, _ in adapter.generate_stream(prompt, max_tokens=max_tokens):
                tokens.append(text)
            return tokens

        # Launch concurrent requests
        results = await asyncio.gather(
            collect_stream("Alpha: First sequence", max_tokens=3),
            collect_stream("Beta: Second sequence", max_tokens=3)
        )

        tokens_alpha, tokens_beta = results
        # Alpha should receive uppercase 'A', 'B', 'C'
        assert "".join(tokens_alpha) == "ABC"
        # Beta should receive lowercase 'a', 'b', 'c'
        assert "".join(tokens_beta) == "abc"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_adapter_accepts_submissions_while_inference_pending():
    """3. Validates that the adapter accepts submissions while inference work is pending."""
    engine = FakeEngine(step_delay=0.01)
    adapter = AsyncServingAdapter(engine)

    try:
        tokens_1: List[str] = []
        tokens_2: List[str] = []

        async def run_req1():
            async for text, _, _, _ in adapter.generate_stream("Alpha: Stream 1", max_tokens=4):
                tokens_1.append(text)
                if len(tokens_1) == 1:
                    # Enqueue request 2 while request 1 is actively running
                    asyncio.create_task(run_req2())

        async def run_req2():
            async for text, _, _, _ in adapter.generate_stream("Beta: Stream 2", max_tokens=2):
                tokens_2.append(text)

        await run_req1()
        # Allow request 2 to complete
        for _ in range(20):
            if len(tokens_2) == 2:
                break
            await asyncio.sleep(0.01)

        assert "".join(tokens_1) == "ABCD"
        assert "".join(tokens_2) == "ab"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_cancellation_processed_by_worker_thread():
    """4. Validates that cancellation is processed by the worker thread rather than mutating engine from event loop."""
    engine = FakeEngine(step_delay=0.01)
    adapter = AsyncServingAdapter(engine)

    event_loop_thread_id = threading.get_ident()

    try:
        gen = adapter.generate_stream("Long prompt", max_tokens=10)
        # Receive first token
        first_chunk = await gen.__anext__()
        assert first_chunk[1] is True

        # Trigger client-side cancellation by closing generator
        await gen.aclose()

        # Allow worker thread to process abort command
        for _ in range(20):
            if engine.abort_invoked:
                break
            await asyncio.sleep(0.01)

        assert engine.abort_invoked is True
        # Verify abort was executed on the worker thread, NEVER the event loop thread
        assert engine.abort_thread_id == adapter._worker_thread.ident
        assert engine.abort_thread_id != event_loop_thread_id
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_request_failure_isolated_from_unrelated_requests():
    """5. Validates that a failing request does not terminate healthy concurrent requests."""
    # Configure fake engine to fail request #1 on pre-admission with MemoryError
    engine = FakeEngine(fail_on_req_id=1, fail_with_memory_error=True)
    adapter = AsyncServingAdapter(engine)

    try:
        async def run_failing_req():
            chunks = []
            async for c in adapter.generate_stream("Fail me", max_tokens=3):
                chunks.append(c)
            return chunks

        async def run_healthy_req():
            chunks = []
            async for c in adapter.generate_stream("Alpha: Healthy", max_tokens=3):
                chunks.append(c)
            return chunks

        res_fail, res_healthy = await asyncio.gather(
            run_failing_req(),
            run_healthy_req(),
            return_exceptions=True
        )

        # Request 1 must fail with MemoryError
        assert isinstance(res_fail, MemoryError)

        # Request 2 must complete successfully
        assert not isinstance(res_healthy, Exception)
        assert len(res_healthy) == 3
        assert "".join(c[0] for c in res_healthy) == "ABC"
        assert res_healthy[-1][2] is True  # is_last
        assert res_healthy[-1][3] == "length"  # finish_reason
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_no_synchronous_engine_steps_on_event_loop_thread():
    """6. Validates that the adapter never executes synchronous engine.step() on the event-loop thread."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    event_loop_thread_id = threading.get_ident()

    try:
        async for _ in adapter.generate_stream("Test execution thread", max_tokens=3):
            pass

        assert len(engine.calling_threads) > 0
        for tid in engine.calling_threads:
            # Must run on the dedicated worker thread
            assert tid == adapter._worker_thread.ident
            # Must NEVER run on the asyncio event-loop thread
            assert tid != event_loop_thread_id
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_output_channel_registration_prevents_early_token_loss():
    """7. Validates that output channel registration cannot lose an early token event."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        chunks = []
        async for chunk in adapter.generate_stream("Immediate token test", max_tokens=1):
            chunks.append(chunk)

        assert len(chunks) == 1
        text, is_first, is_last, finish_reason = chunks[0]
        assert is_first is True
        assert is_last is True
        assert finish_reason == "length"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_empty_work_operation_does_not_busy_spin():
    """8. Validates that the worker thread blocks and does not busy-spin when there is no work."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        # Worker is running with empty queues
        step_count_before = engine.step_count
        await asyncio.sleep(0.08)
        step_count_after = engine.step_count

        # Zero step iterations must occur when queues are empty
        assert step_count_before == 0
        assert step_count_after == 0
        assert adapter._worker_thread.is_alive()
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_shutdown_handles_pending_and_active_requests():
    """9. Validates that shutdown handles pending/active requests and terminates the worker cleanly."""
    engine = FakeEngine(step_delay=0.02)
    adapter = AsyncServingAdapter(engine)

    async def run_long_stream():
        tokens = []
        async for text, _, _, _ in adapter.generate_stream("Long work", max_tokens=10):
            tokens.append(text)
        return tokens

    task = asyncio.create_task(run_long_stream())
    await asyncio.sleep(0.03)

    # Initiate shutdown
    await adapter.shutdown(timeout=1.0)

    # Verify subsequent requests are immediately rejected
    with pytest.raises(RuntimeError, match="Engine adapter is shutting down"):
        async for _ in adapter.generate_stream("Rejected prompt"):
            pass

    # Verify worker thread has cleanly terminated
    assert not adapter._worker_thread.is_alive()

    # Active stream task must have been terminated with an error
    res = await asyncio.gather(task, return_exceptions=True)
    assert isinstance(res[0], RuntimeError)


def test_unsupported_sampling_parameters_fail_explicitly():
    """10. Validates that invalid or unsupported sampling parameters fail explicitly with ValueError."""
    # Valid parameters
    validate_sampling_params(SamplingParams(temperature=0.7, top_p=0.9), max_tokens=10)

    # Negative max_tokens
    with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
        validate_sampling_params(SamplingParams(), max_tokens=0)

    # Negative temperature
    with pytest.raises(ValueError, match="temperature must be non-negative"):
        validate_sampling_params(SamplingParams(temperature=-0.5), max_tokens=10)

    # Invalid top_p
    with pytest.raises(ValueError, match="top_p must be in"):
        validate_sampling_params(SamplingParams(top_p=0.0), max_tokens=10)
    with pytest.raises(ValueError, match="top_p must be in"):
        validate_sampling_params(SamplingParams(top_p=1.5), max_tokens=10)

    # Invalid top_k
    with pytest.raises(ValueError, match="top_k must be -1"):
        validate_sampling_params(SamplingParams(top_k=-5), max_tokens=10)

    # Invalid min_p
    with pytest.raises(ValueError, match="min_p must be in"):
        validate_sampling_params(SamplingParams(min_p=1.2), max_tokens=10)

    # Invalid repetition penalty
    with pytest.raises(ValueError, match="repetition_penalty must be positive"):
        validate_sampling_params(SamplingParams(repetition_penalty=0.0), max_tokens=10)


@pytest.mark.asyncio
async def test_generate_complete_contract():
    """11. Validates that generate_complete() returns complete assembled text and usage counts."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        full_text, p_tokens, gen_tokens, finish_reason = await adapter.generate_complete("Hello", max_tokens=3)
        assert full_text == "012"
        assert p_tokens == len("Hello")
        assert gen_tokens == 3
        assert finish_reason == "length"
    finally:
        await adapter.shutdown()


def test_format_chat_prompt():
    """12. Validates format_chat_prompt formats conversation messages into string."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    formatted = adapter.format_chat_prompt(messages)
    assert "System: You are a helpful assistant." in formatted
    assert "User: Hello!" in formatted
    assert "Assistant: " in formatted


# =========================================================================
# Phase 8.6.1: Focused Stop Reason & Natural Termination Tests
# =========================================================================
def test_map_engine_stop_reason_to_finish_reason_direct():
    """13. Directly verifies mapping of all internal stop reason variants to schema finish reasons."""
    from src.server.serving_adapter import map_engine_stop_reason_to_finish_reason

    # Natural EOS
    assert map_engine_stop_reason_to_finish_reason("eos_token") == "stop"
    assert map_engine_stop_reason_to_finish_reason("stop") == "stop"

    # Stop tokens
    assert map_engine_stop_reason_to_finish_reason("stop_token_151645") == "stop"
    assert map_engine_stop_reason_to_finish_reason("stop_token_2") == "stop"

    # Stop strings
    assert map_engine_stop_reason_to_finish_reason("stop_string_END") == "stop"
    assert map_engine_stop_reason_to_finish_reason("stop_string_\n\n") == "stop"

    # Length
    assert map_engine_stop_reason_to_finish_reason("length") == "length"

    # Abort
    assert map_engine_stop_reason_to_finish_reason("abort") == "abort"

    # None
    assert map_engine_stop_reason_to_finish_reason(None) is None

    # Unknown / unexpected stop reasons
    assert map_engine_stop_reason_to_finish_reason("unrecognized_code") is None
    assert map_engine_stop_reason_to_finish_reason("foo_bar") is None
    assert map_engine_stop_reason_to_finish_reason(9999) is None


@pytest.mark.asyncio
async def test_adapter_natural_eos_termination():
    """14. Verifies natural EOS (stopping before max_tokens) yields finish_reason='stop' in streaming and complete."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        # Prompt "EOS:test" terminates after 2 tokens even though max_tokens=10
        chunks = []
        async for chunk in adapter.generate_stream("EOS:streaming_prompt", max_tokens=10):
            chunks.append(chunk)

        assert len(chunks) == 2  # exactly 2 tokens emitted before EOS
        assert chunks[-1][2] is True  # is_last
        assert chunks[-1][3] == "stop"  # finish_reason mapped from 'eos_token' to 'stop'

        # Test generate_complete
        full_text, p_tokens, gen_tokens, finish_reason = await adapter.generate_complete("EOS:complete_prompt", max_tokens=10)
        assert gen_tokens == 2
        assert finish_reason == "stop"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_adapter_stop_token_termination():
    """15. Verifies internal 'stop_token_...' reasons map to finish_reason='stop'."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        full_text, p_tokens, gen_tokens, finish_reason = await adapter.generate_complete("STOP_TOKEN:test", max_tokens=10)
        assert gen_tokens == 2
        assert finish_reason == "stop"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_adapter_stop_string_termination():
    """16. Verifies internal 'stop_string_...' reasons map to finish_reason='stop'."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        full_text, p_tokens, gen_tokens, finish_reason = await adapter.generate_complete("STOP_STRING:test", max_tokens=10)
        assert gen_tokens == 2
        assert finish_reason == "stop"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_adapter_length_termination():
    """17. Verifies requests hitting max_tokens retain finish_reason='length'."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        full_text, p_tokens, gen_tokens, finish_reason = await adapter.generate_complete("Normal:length_test", max_tokens=4)
        assert gen_tokens == 4
        assert finish_reason == "length"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_adapter_unknown_stop_reason_handling():
    """18. Verifies unknown engine stop reasons map to None without crashing or raising."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        chunks = []
        async for chunk in adapter.generate_stream("UNKNOWN_STOP:test", max_tokens=10):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[-1][2] is True  # is_last
        assert chunks[-1][3] is None  # mapped to None safely

        full_text, p_tokens, gen_tokens, finish_reason = await adapter.generate_complete("UNKNOWN_STOP:test2", max_tokens=10)
        assert finish_reason is None
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_adapter_zero_token_eos_termination():
    """19. Verifies zero-token completion with EOS maps to finish_reason='stop'."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        chunks = []
        async for chunk in adapter.generate_stream("ZERO_TOKEN_EOS:test", max_tokens=5):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0][2] is True  # is_last
        assert chunks[0][3] == "stop"  # finish_reason mapped to stop

        full_text, p_tokens, gen_tokens, finish_reason = await adapter.generate_complete("ZERO_TOKEN_EOS:test2", max_tokens=5)
        assert gen_tokens == 0
        assert finish_reason == "stop"
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_adapter_natural_eos_channel_cleanup():
    """20. Verifies that output channels and active request tracking cleanly purge on EOS."""
    engine = FakeEngine()
    adapter = AsyncServingAdapter(engine)

    try:
        async for _ in adapter.generate_stream("EOS:cleanup_test", max_tokens=10):
            pass

        # Verify channels dictionary is completely clean
        with adapter._registry_lock:
            assert len(adapter._channels) == 0

        # Verify running batch in engine is empty
        assert len(engine.running_batch) == 0
    finally:
        await adapter.shutdown()
