"""
Asynchronous Serving Adapter Core
---------------------------------
Bridges the asynchronous FastAPI serving layer with MiniVLLMEngine using a
single-owner worker thread architecture with per-request asynchronous output channels.

Key Invariants:
  1. Single-Owner Execution: All mutations and step calls to MiniVLLMEngine are
     executed strictly by a dedicated background worker thread.
  2. Asynchronous Output Channels: Each HTTP request receives a dedicated
     asyncio.Queue registered prior to command submission.
  3. Thread-Safe Delivery: Tokens and completion events are transferred across
     the thread boundary using loop.call_soon_threadsafe().
  4. Non-Blocking Event Loop: Synchronous GPU forward passes and CPU tensor
     computations in engine.step() never execute on the asyncio event-loop thread.
  5. Idle Conservation: The worker thread blocks on a thread-safe queue when no
     requests are active, consuming 0% CPU.
  6. Idempotent Cancellation: Disconnections are dispatched through the worker
     command queue, invoking engine.abort_request() safely on the worker thread.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import itertools
import os
import queue
import sys
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import get_logger
from src.engine.llm_engine import EngineRequest, MiniVLLMEngine, RequestStatus
from src.engine.sampling import SamplingParams

logger = get_logger("mini_vllm.serving_adapter")


from src.server.protocol import KVCacheMemoryError


def map_engine_stop_reason_to_finish_reason(stop_reason: Optional[str]) -> Optional[str]:
    """
    Explicitly translates an internal engine stop_reason to a public OpenAI-compatible finish_reason.

    Allowed public schema values (defined in src/server/protocol.py):
      Literal["stop", "length", "abort"] or None.

    Translation Rules:
      - "eos_token" -> "stop" (natural EOS token emitted by model)
      - "stop" -> "stop" (standard stop)
      - "stop_token_<id>" -> "stop" (explicit stop token ID matched)
      - "stop_string_<str>" -> "stop" (explicit stop string matched)
      - "length" -> "length" (reached max_tokens ceiling)
      - "abort" -> "abort" (request aborted / cancelled)
      - None -> None
      - Any unrecognized/unknown stop reason -> None (logged with warning, preventing
        accidental schema ValidationError while avoiding hiding unknown engine states)
    """
    if stop_reason is None:
        return None
    if not isinstance(stop_reason, str):
        logger.warning("Unexpected non-string stop_reason: %r; mapping to None", stop_reason)
        return None

    # 1. Exact matches
    if stop_reason in ("eos_token", "stop"):
        return "stop"
    if stop_reason == "length":
        return "length"
    if stop_reason == "abort":
        return "abort"

    # 2. Structured prefix matches (explicit patterns from check_stop_criteria)
    if stop_reason.startswith("stop_token_") or stop_reason.startswith("stop_string_"):
        return "stop"

    # 3. Unknown stop reasons: Log warning and map to None so schema validation does not fail
    logger.warning("Unrecognized engine stop_reason %r encountered; mapping to None", stop_reason)
    return None


@dataclass
class OutputChunk:
    """Represents a streaming generation delta delivered to an asyncio output channel."""
    text: str = ""
    is_first: bool = False
    is_last: bool = False
    finish_reason: Optional[str] = None
    token_id: Optional[int] = None
    error: Optional[Exception] = None


@dataclass
class _SubmitCommand:
    adapter_req_id: int
    prompt: str
    sampling_params: SamplingParams
    max_tokens: int


@dataclass
class _AbortCommand:
    adapter_req_id: int


@dataclass
class _ShutdownCommand:
    drain: bool = False


@dataclass
class _RequestTracker:
    adapter_req_id: int
    loop: asyncio.AbstractEventLoop
    output_queue: asyncio.Queue[OutputChunk]
    engine_req_id: Optional[int] = None
    seen_tokens: int = 0
    is_first_emitted: bool = False
    is_last_emitted: bool = False
    is_finished: bool = False
    is_cancelled: bool = False


def validate_sampling_params(params: SamplingParams, max_tokens: int) -> None:
    """Explicitly validates sampling parameters, rejecting unsupported or invalid options."""
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be greater than 0, got {max_tokens}")
    if params.temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {params.temperature}")
    if params.top_p <= 0.0 or params.top_p > 1.0:
        raise ValueError(f"top_p must be in (0.0, 1.0], got {params.top_p}")
    if params.top_k < -1 or params.top_k == 0:
        raise ValueError(f"top_k must be -1 (disabled) or >= 1, got {params.top_k}")
    if params.min_p < 0.0 or params.min_p > 1.0:
        raise ValueError(f"min_p must be in [0.0, 1.0], got {params.min_p}")
    if params.repetition_penalty <= 0.0:
        raise ValueError(f"repetition_penalty must be positive, got {params.repetition_penalty}")


def _deliver_chunk(
    loop: asyncio.AbstractEventLoop,
    async_queue: asyncio.Queue[OutputChunk],
    chunk: OutputChunk
) -> None:
    """Thread-safe event-loop bridge delivering OutputChunk from worker thread to asyncio."""
    if not loop.is_closed():
        try:
            loop.call_soon_threadsafe(async_queue.put_nowait, chunk)
        except RuntimeError:
            # Event loop was closed concurrently during delivery
            pass


class AsyncServingAdapter:
    """
    Thread-safe asynchronous serving adapter wrapping MiniVLLMEngine.
    All engine mutations, scheduling, and forward passes are owned solely
    by a dedicated worker thread.
    """
    def __init__(self, engine: Any):
        self.engine = engine
        self._command_queue: queue.Queue[Union[_SubmitCommand, _AbortCommand, _ShutdownCommand]] = queue.Queue()
        self._id_counter = itertools.count(1)

        self._registry_lock = threading.Lock()
        self._channels: Dict[int, _RequestTracker] = {}

        self._is_shutting_down = False

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="MiniVLLM-EngineWorker"
        )
        self._worker_thread.start()

    @property
    def model_name(self) -> str:
        if hasattr(self.engine, "config") and hasattr(self.engine.config, "model"):
            return self.engine.config.model.model_name_or_path
        return getattr(self.engine, "model_name", "mini-vllm")

    @property
    def device_str(self) -> str:
        return str(getattr(self.engine, "device", "cpu"))

    @property
    def tokenizer(self) -> Any:
        return getattr(self.engine, "tokenizer", None)

    @property
    def metrics(self) -> Any:
        return getattr(self.engine, "metrics", None)

    def format_chat_prompt(self, messages: List[dict]) -> str:
        """Formats multi-turn messages into a single prompt string using chat template."""
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback chat format
        lines = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("Assistant: ")
        return "\n".join(lines)

    async def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        max_tokens: int = 128
    ) -> AsyncGenerator[Tuple[str, bool, bool, Optional[str]], None]:
        """
        Asynchronously generates tokens and yields streaming deltas:
        Yields: (token_str, is_first, is_last, finish_reason)
        """
        if self._is_shutting_down:
            raise RuntimeError("Engine adapter is shutting down")

        if sampling_params is None:
            sampling_params = SamplingParams.greedy()

        validate_sampling_params(sampling_params, max_tokens)

        adapter_req_id = next(self._id_counter)
        loop = asyncio.get_running_loop()
        output_queue: asyncio.Queue[OutputChunk] = asyncio.Queue()

        tracker = _RequestTracker(
            adapter_req_id=adapter_req_id,
            loop=loop,
            output_queue=output_queue
        )

        # 1. Register channel before submitting command to prevent early-token delivery races
        with self._registry_lock:
            self._channels[adapter_req_id] = tracker

        # 2. Submit command to the worker thread
        self._command_queue.put(_SubmitCommand(
            adapter_req_id=adapter_req_id,
            prompt=prompt,
            sampling_params=sampling_params,
            max_tokens=max_tokens
        ))

        try:
            while True:
                chunk = await output_queue.get()
                if chunk.error is not None:
                    raise chunk.error
                yield (chunk.text, chunk.is_first, chunk.is_last, chunk.finish_reason)
                if chunk.is_last:
                    break
        except (GeneratorExit, asyncio.CancelledError):
            # Client disconnected or generator was cancelled
            tracker.is_cancelled = True
            self._command_queue.put(_AbortCommand(adapter_req_id=adapter_req_id))
            raise
        finally:
            with self._registry_lock:
                self._channels.pop(adapter_req_id, None)

    async def generate_complete(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        max_tokens: int = 128
    ) -> Tuple[str, int, int, Optional[str]]:
        """
        Executes complete generation and returns:
        (generated_text, prompt_token_count, generated_token_count, finish_reason)
        """
        full_chunks = []
        finish_reason: Optional[str] = None
        prompt_len = len(self.tokenizer.encode(prompt)) if self.tokenizer is not None else 0

        async for chunk_text, _, is_last, f_reason in self.generate_stream(prompt, sampling_params, max_tokens):
            full_chunks.append(chunk_text)
            if is_last:
                finish_reason = f_reason

        full_text = "".join(full_chunks)
        gen_tokens = len(self.tokenizer.encode(full_text)) if self.tokenizer is not None else 0
        return full_text, prompt_len, gen_tokens, finish_reason

    def shutdown_sync(self, timeout: float = 5.0) -> None:
        """Synchronously shuts down the adapter and worker thread."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        self._command_queue.put(_ShutdownCommand(drain=False))
        self._worker_thread.join(timeout)

        with self._registry_lock:
            for tracker in list(self._channels.values()):
                _deliver_chunk(
                    tracker.loop,
                    tracker.output_queue,
                    OutputChunk(error=RuntimeError("Engine adapter shut down"), is_last=True)
                )
            self._channels.clear()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shuts down the adapter and worker thread."""
        if self._is_shutting_down:
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.shutdown_sync, timeout)
        except RuntimeError:
            self.shutdown_sync(timeout)

    def _worker_loop(self) -> None:
        """
        Dedicated single-owner worker loop running on a background thread.
        Owns all MiniVLLMEngine state mutations, admissions, step iterations, and cancellations.
        """
        active_trackers: Dict[int, _RequestTracker] = {}
        adapter_to_engine_id: Dict[int, int] = {}
        cancelled_adapter_ids: set[int] = set()

        try:
            while True:
                # -----------------------------------------------------------------
                # 1. Command Ingestion & Idle Synchronization Phase
                # -----------------------------------------------------------------
                has_active_work = bool(self.engine.waiting_queue or self.engine.running_batch)

                # Idle Conservation: block on queue if no active work
                # If waiting_queue has requests but running_batch is empty (e.g. backpressure),
                # use a 10ms timeout to avoid busy-spinning while preserving responsive admission.
                timeout = 0.01 if (has_active_work and not self.engine.running_batch) else None

                try:
                    cmd = self._command_queue.get(block=(not has_active_work or timeout is not None), timeout=timeout)
                    commands = [cmd]
                    # Drain all other pending commands without blocking
                    while True:
                        try:
                            commands.append(self._command_queue.get_nowait())
                        except queue.Empty:
                            break
                except queue.Empty:
                    commands = []

                should_shutdown = False
                for cmd in commands:
                    if isinstance(cmd, _ShutdownCommand):
                        should_shutdown = True
                        break
                    elif isinstance(cmd, _SubmitCommand):
                        if cmd.adapter_req_id in cancelled_adapter_ids:
                            cancelled_adapter_ids.remove(cmd.adapter_req_id)
                            continue

                        with self._registry_lock:
                            tracker = self._channels.get(cmd.adapter_req_id)
                        if tracker is None or tracker.is_cancelled:
                            continue

                        try:
                            eng_req = self.engine.add_request(
                                prompt=cmd.prompt,
                                max_tokens=cmd.max_tokens,
                                sampling_params=cmd.sampling_params
                            )
                            tracker.engine_req_id = eng_req.req_id
                            active_trackers[eng_req.req_id] = tracker
                            adapter_to_engine_id[cmd.adapter_req_id] = eng_req.req_id
                        except Exception as add_exc:
                            _deliver_chunk(
                                tracker.loop,
                                tracker.output_queue,
                                OutputChunk(error=add_exc, is_last=True)
                            )
                    elif isinstance(cmd, _AbortCommand):
                        eng_id = adapter_to_engine_id.pop(cmd.adapter_req_id, None)
                        if eng_id is not None:
                            self.engine.abort_request(eng_id)
                            active_trackers.pop(eng_id, None)
                        else:
                            cancelled_adapter_ids.add(cmd.adapter_req_id)

                if should_shutdown:
                    for eng_id, tracker in list(active_trackers.items()):
                        try:
                            self.engine.abort_request(eng_id)
                        except Exception:
                            pass
                        _deliver_chunk(
                            tracker.loop,
                            tracker.output_queue,
                            OutputChunk(error=RuntimeError("Engine adapter shut down"), is_last=True)
                        )
                    active_trackers.clear()
                    adapter_to_engine_id.clear()
                    break

                # -----------------------------------------------------------------
                # 2. Engine Step Execution Phase
                # -----------------------------------------------------------------
                if not (self.engine.waiting_queue or self.engine.running_batch):
                    continue

                step_exc = None
                try:
                    self.engine.step()
                except MemoryError as mem_err:
                    step_exc = mem_err
                except Exception as exc:
                    step_exc = exc

                # -----------------------------------------------------------------
                # 3. Output Delta & Terminal State Dispatch Phase
                # -----------------------------------------------------------------
                if step_exc is not None:
                    failed_req = getattr(step_exc, "request", None)
                    if failed_req is not None and failed_req.req_id in active_trackers:
                        # Specific isolated request failure (e.g. MemoryError during admission)
                        tracker = active_trackers.pop(failed_req.req_id)
                        adapter_to_engine_id.pop(tracker.adapter_req_id, None)
                        is_kv_full = (
                            getattr(failed_req, "stop_reason", None) == "kv_cache_full"
                            or "kv" in str(step_exc).lower()
                            or "pool" in str(step_exc).lower()
                        )
                        err_to_deliver = step_exc
                        if is_kv_full:
                            err_to_deliver = KVCacheMemoryError(str(step_exc))
                        _deliver_chunk(
                            tracker.loop,
                            tracker.output_queue,
                            OutputChunk(
                                is_first=not tracker.is_first_emitted,
                                is_last=True,
                                finish_reason="kv_cache_full" if is_kv_full else getattr(failed_req, "stop_reason", "error"),
                                error=err_to_deliver
                            )
                        )
                    else:
                        # General exception during execution: notify all affected failed requests
                        failed_any = False
                        for eng_id, tracker in list(active_trackers.items()):
                            req = self.engine.get_request(eng_id)
                            if req is not None and req.status == RequestStatus.FAILED:
                                active_trackers.pop(eng_id, None)
                                adapter_to_engine_id.pop(tracker.adapter_req_id, None)
                                failed_any = True
                                _deliver_chunk(
                                    tracker.loop,
                                    tracker.output_queue,
                                    OutputChunk(
                                        is_first=not tracker.is_first_emitted,
                                        is_last=True,
                                        finish_reason=req.stop_reason or "error",
                                        error=step_exc
                                    )
                                )
                        if not failed_any:
                            # Catastrophic engine step failure: abort and notify all active batch requests
                            for eng_id, tracker in list(active_trackers.items()):
                                try:
                                    self.engine.abort_request(eng_id)
                                except Exception:
                                    pass
                                active_trackers.pop(eng_id, None)
                                adapter_to_engine_id.pop(tracker.adapter_req_id, None)
                                _deliver_chunk(
                                    tracker.loop,
                                    tracker.output_queue,
                                    OutputChunk(
                                        is_first=not tracker.is_first_emitted,
                                        is_last=True,
                                        finish_reason="error",
                                        error=step_exc
                                    )
                                )

                # Inspect all active trackers for new token deltas and completions
                for eng_id, tracker in list(active_trackers.items()):
                    req = self.engine.get_request(eng_id)
                    if req is None:
                        continue

                    total_tokens = len(req.output_token_ids)
                    is_terminal = req.status in (RequestStatus.FINISHED, RequestStatus.FAILED)

                    if total_tokens > tracker.seen_tokens:
                        new_ids = req.output_token_ids[tracker.seen_tokens:total_tokens]
                        for idx, tok_id in enumerate(new_ids):
                            is_first = not tracker.is_first_emitted
                            tracker.is_first_emitted = True

                            is_last_token = (idx == len(new_ids) - 1) and is_terminal
                            finish_reason = map_engine_stop_reason_to_finish_reason(req.stop_reason) if is_last_token else None

                            tok_str = self.tokenizer.decode([tok_id], skip_special_tokens=True) if self.tokenizer else str(tok_id)
                            chunk = OutputChunk(
                                text=tok_str,
                                is_first=is_first,
                                is_last=is_last_token,
                                finish_reason=finish_reason,
                                token_id=tok_id
                            )
                            if is_last_token:
                                tracker.is_last_emitted = True
                            _deliver_chunk(tracker.loop, tracker.output_queue, chunk)
                        tracker.seen_tokens = total_tokens

                    if is_terminal and not tracker.is_finished:
                        tracker.is_finished = True
                        active_trackers.pop(eng_id, None)
                        adapter_to_engine_id.pop(tracker.adapter_req_id, None)

                        if req.status == RequestStatus.FAILED:
                            err_msg = req.error or f"Request failed with reason {req.stop_reason}"
                            if req.stop_reason == "cuda_oom":
                                err = torch.cuda.OutOfMemoryError(err_msg)
                            elif req.stop_reason == "kv_cache_full":
                                err = KVCacheMemoryError(err_msg)
                            else:
                                err = RuntimeError(err_msg)
                            tracker.is_last_emitted = True
                            _deliver_chunk(
                                tracker.loop,
                                tracker.output_queue,
                                OutputChunk(
                                    is_first=not tracker.is_first_emitted,
                                    is_last=True,
                                    finish_reason=map_engine_stop_reason_to_finish_reason(req.stop_reason),
                                    error=err
                                )
                            )
                        elif total_tokens == 0:
                            # Finished with 0 tokens (e.g. prompt already stopped)
                            tracker.is_last_emitted = True
                            _deliver_chunk(
                                tracker.loop,
                                tracker.output_queue,
                                OutputChunk(
                                    is_first=not tracker.is_first_emitted,
                                    is_last=True,
                                    finish_reason=map_engine_stop_reason_to_finish_reason(req.stop_reason)
                                )
                            )
                        elif not tracker.is_last_emitted:
                            # Request terminated after seen_tokens without final is_last chunk
                            tracker.is_last_emitted = True
                            _deliver_chunk(
                                tracker.loop,
                                tracker.output_queue,
                                OutputChunk(
                                    is_first=not tracker.is_first_emitted,
                                    is_last=True,
                                    finish_reason=map_engine_stop_reason_to_finish_reason(req.stop_reason)
                                )
                            )

        except Exception as fatal_worker_exc:
            logger.error("Fatal worker loop exception: %s", fatal_worker_exc, exc_info=True)
            for eng_id, tracker in list(active_trackers.items()):
                _deliver_chunk(
                    tracker.loop,
                    tracker.output_queue,
                    OutputChunk(error=fatal_worker_exc, is_last=True)
                )
            active_trackers.clear()
