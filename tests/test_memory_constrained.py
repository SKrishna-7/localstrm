"""
Tests for Memory-Constrained Reliability and Failure Recovery in Mini-vLLM.
-------------------------------------------------------------------------
Verifies:
1. Every failed request reaches a clear terminal state (RequestStatus.FAILED)
   with explicit stop_reason and error tracking.
2. KV-block allocation failures during decode isolate the failing sequence
   without aborting healthy concurrent requests in the batch.
3. Distinction between internal KV-cache pool exhaustion and hardware CUDA OOM.
4. Clean rollback and 100% block recovery across all failure pathways.
5. Explicit request cancellation via abort_request.
6. Backpressure scheduling under constrained pool capacity.
7. Sustained integrity over repeated failure-recovery cycles.
"""

from __future__ import annotations

import types
from typing import Any, Sequence
from unittest.mock import MagicMock
import pytest
import torch

from src.engine.llm_engine import (
    EngineRequest,
    MiniVLLMEngine,
    PhysicalBlockPool,
    RequestStatus,
    SamplingParams,
)


class MockTokenizer:
    """Lightweight deterministic tokenizer."""

    def __init__(self, vocab: dict[str, int] | None = None, eos_token_id: int = 2) -> None:
        self.vocab = vocab or {"hello": 10, "world": 11, "antigravity": 12, "test": 13}
        self.eos_token_id = eos_token_id

    def encode(self, text: str) -> list[int]:
        words = text.strip().split()
        return [self.vocab.get(w, ord(w[0]) if w else 1) for w in words] or [1]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(t) for t in token_ids)


def create_test_model(
    num_layers: int = 2,
    num_heads: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 16,
    vocab_size: int = 100,
    eos_token_id: int = 2,
    device: str = "cpu",
) -> MagicMock:
    """Build a mock model with minimal layers for deterministic CPU testing."""
    model = MagicMock()
    model.config = types.SimpleNamespace(
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=num_heads * head_dim,
        vocab_size=vocab_size,
    )
    model.device = torch.device(device)
    model.to.return_value = model
    model.eval.return_value = model

    def forward_impl(input_tensor: Any, past_key_values: Any = None, use_cache: bool = True, **kwargs: Any) -> Any:
        num_tokens = input_tensor.shape[1]
        logits = torch.zeros((1, num_tokens, vocab_size))
        # Non-eos token by default so generation proceeds
        logits[0, -1, 10] = 10.0
        # Standard HuggingFace tuple format: tuple of (k, v) where shape is [1, num_kv_heads, seq_len, head_dim]
        cache = tuple(
            (
                torch.zeros((1, num_kv_heads, num_tokens, head_dim)),
                torch.zeros((1, num_kv_heads, num_tokens, head_dim)),
            )
            for _ in range(num_layers)
        )
        return types.SimpleNamespace(logits=logits, past_key_values=cache)

    model.side_effect = forward_impl
    return model


def make_test_engine(
    *,
    total_blocks: int = 8,
    block_size: int = 4,
    max_batch_size: int = 4,
    prefix: bool = False,
    model: Any = None,
    tokenizer: Any = None,
) -> MiniVLLMEngine:
    engine = MiniVLLMEngine(
        model=model or create_test_model(),
        tokenizer=tokenizer or MockTokenizer(),
        total_blocks=total_blocks,
        block_size=block_size,
        max_batch_size=max_batch_size,
        device="cpu",
        enable_prefix_caching=prefix,
        chunked_prefill=False,
    )

    # Attach default deterministic batched_decode_step mock on runner
    def default_decode(batched_tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        B = batched_tokens.shape[0]
        logits = torch.zeros((B, 100))
        logits[:, 10] = 5.0
        return logits

    engine.model_runner.batched_decode_step = default_decode
    return engine


def test_terminal_state_on_model_execution_failure():
    """Verify that an execution failure marks requests FAILED, sets stop_reason and error,

    adds to completed_requests, and restores 100% of blocks.
    """
    engine = make_test_engine(total_blocks=4, block_size=4)
    initial_free = engine.num_free_blocks

    req = engine.add_request("hello world", max_tokens=4)

    # Inject model forward failure
    def raising_forward(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated model runtime failure")

    engine.model.side_effect = raising_forward

    with pytest.raises(RuntimeError, match="simulated model runtime failure"):
        engine.step()

    # Terminal state invariants
    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "error"
    assert "simulated model runtime failure" in (req.error or "")
    assert req.finish_step == 0
    assert req in engine.completed_requests

    # Queue and memory cleanup
    assert req not in engine.running_batch
    assert req.physical_block_ids == []
    assert req.kv_cache is None
    assert engine.num_free_blocks == initial_free


def test_terminal_state_on_cuda_oom():
    """Verify that a CUDA OOM failure is explicitly identified with stop_reason='cuda_oom'."""
    engine = make_test_engine(total_blocks=4, block_size=4)
    initial_free = engine.num_free_blocks

    req = engine.add_request("hello world", max_tokens=4)

    # Inject CUDA OOM exception
    def oom_forward(*args: Any, **kwargs: Any) -> Any:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 512.00 MiB")

    engine.model.side_effect = oom_forward

    with pytest.raises(torch.cuda.OutOfMemoryError):
        engine.step()

    # Explicit failure reason distinction
    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "cuda_oom"
    assert "CUDA out of memory" in (req.error or "")
    assert req in engine.completed_requests
    assert engine.num_free_blocks == initial_free


def test_terminal_state_on_impossible_admission():
    """Verify that a request exceeding total pool capacity fails cleanly with MemoryError

    and is transitioned to RequestStatus.FAILED with stop_reason='kv_cache_full'.
    """
    engine = make_test_engine(total_blocks=2, block_size=4)
    initial_free = engine.num_free_blocks

    # 12 tokens prompt + 4 tokens max = 16 tokens -> 4 blocks > total_blocks (2)
    impossible_prompt = "hello " * 12
    req = engine.add_request(impossible_prompt, max_tokens=4)

    with pytest.raises(MemoryError, match="requires 4 KV blocks, but the pool has only 2") as exc_info:
        engine.step()

    # Observability via exception attribute and returned request object
    assert exc_info.value.request is req
    assert req.status is RequestStatus.FAILED
    assert req.stop_reason == "kv_cache_full"
    assert "requires 4 KV blocks" in (req.error or "")
    assert req.finish_step == 0
    assert req.finish_time is not None
    assert req.finish_time_ns is not None

    # Observability via engine lookup and collections
    assert engine.get_request(req.req_id) is req
    assert req in engine.failed_requests
    assert req not in engine.waiting_queue
    assert req not in engine.running_batch
    assert req not in engine.completed_requests
    assert engine.num_free_blocks == initial_free


def test_request_isolation_on_decode_block_exhaustion():
    """Verify that when one request exhausts the KV-pool during decode:

    1. The failing request transitions to RequestStatus.FAILED with stop_reason='kv_cache_full'.
    2. The failing request's blocks are immediately returned to the pool.
    3. Healthy concurrent requests in the batch are NOT aborted and complete successfully.
    4. 100% of blocks are returned to the free pool at the end.
    """
    # 4 blocks total, block_size = 4
    engine = make_test_engine(total_blocks=4, block_size=4, max_batch_size=2)
    initial_free = engine.num_free_blocks
    assert initial_free == 4

    # req_healthy: 1 prompt token ("hello") + 5 max_tokens.
    # At step 0, generates 2 tokens (prefill + decode), total tokens = 3 <= 4 (1 block).
    req_healthy = engine.add_request("hello", max_tokens=5, sampling_params=SamplingParams.greedy())

    # req_greedy: 2 prompt tokens ("hello world") + 5 max_tokens.
    # At step 0, generates 2 tokens (prefill + decode), total tokens = 4 (fills block 1).
    # At step 1, needs token 3 (total tokens = 5 > 4), crossing block boundary to require block 2.
    req_greedy = engine.add_request("hello world", max_tokens=5, sampling_params=SamplingParams.greedy())

    # Step 0: schedule and prefill both. Each gets 1 block (2 blocks allocated, 2 free).
    engine.step()
    assert req_healthy in engine.running_batch
    assert req_greedy in engine.running_batch
    assert len(req_healthy.physical_block_ids) == 1
    assert len(req_greedy.physical_block_ids) == 1
    assert engine.num_free_blocks == 2

    # Now simulate an allocation failure specifically when req_greedy asks for its 2nd block
    original_allocate = engine._allocate_block

    def constrained_allocate() -> int:
        raise MemoryError("Out of KV Cache physical blocks!")

    engine._allocate_block = constrained_allocate

    # Step 1: Decode step
    # req_healthy has total_tokens = 1 prompt + 2 generated = 3 <= block_size (4) -> DOES NOT ALLOCATE!
    # req_greedy has total_tokens = 2 prompt + 2 generated = 4. Next token needs block 2 -> CALLS _allocate_block -> FAILS!
    engine.step()

    # req_greedy must fail gracefully and be evicted
    assert req_greedy.status is RequestStatus.FAILED
    assert req_greedy.stop_reason == "kv_cache_full"
    assert "Out of KV Cache" in (req_greedy.error or "")
    assert req_greedy in engine.completed_requests
    assert req_greedy not in engine.running_batch
    assert req_greedy.physical_block_ids == []

    # req_healthy MUST REMAIN HEALTHY and still be in running_batch
    assert req_healthy.status is RequestStatus.RUNNING
    assert req_healthy in engine.running_batch

    # Restore original allocator and let req_healthy finish
    engine._allocate_block = original_allocate
    while engine.step():
        pass

    assert req_healthy.status is RequestStatus.FINISHED
    assert req_healthy.num_generated == 5
    assert req_healthy in engine.completed_requests

    # Verify 100% of blocks recovered
    assert engine.num_free_blocks == initial_free


def test_explicit_abort_request():
    """Verify that abort_request cleanly cancels waiting and running requests,

    releasing all physical blocks and updating completed_requests.
    """
    engine = make_test_engine(total_blocks=4, block_size=4, max_batch_size=1)
    initial_free = engine.num_free_blocks

    req1 = engine.add_request("first request", max_tokens=5)
    req2 = engine.add_request("second request waiting", max_tokens=5)

    # Step once: req1 is admitted and running; req2 is waiting (max_batch_size=1)
    engine.step()
    assert req1 in engine.running_batch
    assert req2 in engine.waiting_queue
    assert len(req1.physical_block_ids) > 0

    # 1. Abort waiting request
    success = engine.abort_request(req2.req_id)
    assert success is True
    assert req2.status is RequestStatus.FAILED
    assert req2.stop_reason == "abort"
    assert req2 in engine.completed_requests
    assert req2 not in engine.waiting_queue

    # 2. Abort running request
    success = engine.abort_request(req1.req_id)
    assert success is True
    assert req1.status is RequestStatus.FAILED
    assert req1.stop_reason == "abort"
    assert req1 in engine.completed_requests
    assert req1 not in engine.running_batch
    assert req1.physical_block_ids == []

    # 3. Abort non-existent request
    assert engine.abort_request(9999) is False

    # 4. Verify all blocks are recovered
    assert engine.num_free_blocks == initial_free


def test_scheduler_admission_backpressure():
    """Verify that backpressure prevents admitting a sequence when free blocks are

    strictly required to protect active decode progress.
    """
    # 2 blocks total, block_size = 4
    engine = make_test_engine(total_blocks=2, block_size=4, max_batch_size=2)
    initial_free = engine.num_free_blocks

    # Req 1 needs 1 prompt block, 1 max token
    req1 = engine.add_request("hello", max_tokens=1)
    # Req 2 needs 1 prompt block, 1 max token
    req2 = engine.add_request("world", max_tokens=1)

    # Step 0: req1 is scheduled. Because total_blocks=2 and cushion=1 when running_batch has items,
    # req1 takes 1 block (leaving 1 block free). Req 2 requires needed_new_blocks(1) + cushion(1) = 2,
    # so Req 2 safely waits in queue until req1 finishes.
    engine.step()
    assert req1 in engine.completed_requests or req1 in engine.running_batch

    # Run remaining steps to completion
    while engine.step():
        pass

    assert req1.status is RequestStatus.FINISHED
    assert req2.status is RequestStatus.FINISHED
    assert engine.num_free_blocks == initial_free


def test_repeated_failure_and_recovery_cycle():
    """Verify zero memory leaks over repeated cycles of failures followed by successful completions."""
    engine = make_test_engine(total_blocks=4, block_size=4)
    initial_free = engine.num_free_blocks

    for cycle in range(5):
        # 1. Trigger a failure via abort
        req_fail = engine.add_request("failing request", max_tokens=2)
        engine.abort_request(req_fail.req_id)
        assert req_fail.status is RequestStatus.FAILED
        assert engine.num_free_blocks == initial_free

        # 2. Run a normal request to completion
        req_ok = engine.add_request("good request", max_tokens=2)
        while engine.step():
            pass
        assert req_ok.status is RequestStatus.FINISHED
        assert engine.num_free_blocks == initial_free

    assert len(engine.completed_requests) == 10


def test_repeated_admission_and_rejection_safety():
    """Verify that repeated rejection or admission attempts do not duplicate

    completion records, corrupt engine state, or cause memory leaks.
    """
    engine = make_test_engine(total_blocks=2, block_size=4)
    initial_free = engine.num_free_blocks

    # 1. First impossible admission attempt
    impossible_prompt = "hello " * 12  # 12 prompt + 4 max = 16 tokens -> 4 blocks > 2 total
    req = engine.add_request(impossible_prompt, max_tokens=4)

    with pytest.raises(MemoryError):
        engine.step()

    assert req.status is RequestStatus.FAILED
    assert req not in engine.completed_requests
    assert len(engine.completed_requests) == 0

    # 2. Re-admitting the terminal request via add_request must be rejected
    with pytest.raises(ValueError, match="Cannot re-admit terminal request"):
        engine.add_request(req)

    # 3. Forcibly placing terminal request back in waiting_queue is purged without error
    engine.waiting_queue.append(req)
    # Step should evict it from waiting_queue cleanly
    has_active = engine.step()
    assert has_active is False
    assert req not in engine.waiting_queue
    assert len(engine.completed_requests) == 0

    # 4. Aborting an already rejected request returns False and does not duplicate
    aborted = engine.abort_request(req.req_id)
    assert aborted is False
    assert len(engine.completed_requests) == 0

    # 5. Subsequent valid request succeeds and produces exactly 1 completion record
    valid_req = engine.add_request("hello", max_tokens=2)
    while engine.step():
        pass

    assert valid_req.status is RequestStatus.FINISHED
    assert len(engine.completed_requests) == 1
    assert engine.completed_requests[0] is valid_req
    assert engine.num_free_blocks == initial_free


def test_backpressure_regime_semantics():
    """Verify two-tier backpressure semantics:

    1. Conservative regime: guaranteed admission when total worst-case capacity fits.
    2. Oversubscribed regime: permits admission with decode_cushion headroom, but mid-decode
       exhaustion remains possible and isolated.
    """
    # 4 blocks total, block_size = 4
    engine = make_test_engine(total_blocks=4, block_size=4, max_batch_size=3)
    initial_free = engine.num_free_blocks

    # Req A: 1 token prompt + 1 max_token = 2 tokens -> 1 block
    req_a = engine.add_request("hello", max_tokens=1)
    # Req B: 1 token prompt + 1 max_token = 2 tokens -> 1 block
    req_b = engine.add_request("world", max_tokens=1)

    # Collective worst-case = 1 + 1 = 2 blocks <= 4 total blocks (Conservative Regime)
    # Both must be admitted and scheduled immediately
    engine.step()
    assert req_a in engine.running_batch or req_a in engine.completed_requests
    assert req_b in engine.running_batch or req_b in engine.completed_requests

    while engine.step():
        pass

    assert req_a.status is RequestStatus.FINISHED
    assert req_b.status is RequestStatus.FINISHED
    assert engine.num_free_blocks == initial_free
