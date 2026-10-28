"""Bounded lifecycle, allocation, and failure-recovery checks for Mini-vLLM."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine.llm_engine import MiniVLLMEngine, PhysicalBlockPool, RequestStatus
from src.memory.prefix_cache import compute_block_hash


class FakeTokenizer:
    eos_token_id = 9

    def __init__(self, tokens=None):
        self.tokens = tokens

    def encode(self, prompt):
        if self.tokens is not None:
            return list(self.tokens)
        return [1] if prompt else []


class FakeModel:
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        hidden_size=4,
        head_dim=4,
        vocab_size=16,
    )

    def to(self, device):
        return self

    def eval(self):
        return self


class RaisingModel(FakeModel):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("injected model failure")


def make_engine(*, tokenizer=None, total_blocks=4, block_size=2, max_batch_size=2, prefix=False):
    return MiniVLLMEngine(
        model_name="fake-model",
        device="cpu",
        tokenizer=tokenizer or FakeTokenizer(),
        model=FakeModel(),
        max_batch_size=max_batch_size,
        total_blocks=total_blocks,
        block_size=block_size,
        enable_prefix_caching=prefix,
        chunked_prefill=False,
    )


def test_physical_pool_exact_capacity_release_and_invalid_references():
    pool = PhysicalBlockPool(
        num_layers=1,
        total_blocks=2,
        block_size=2,
        num_kv_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
    )
    first = pool.allocate_block()
    second = pool.allocate_block()
    assert {first, second} == {0, 1}
    assert pool.num_free == 0

    with pytest.raises(MemoryError, match="Out of KV Cache"):
        pool.allocate_block()

    pool.free_block(first)
    assert pool.num_free == 1
    with pytest.raises(ValueError, match="already free"):
        pool.free_block(first)
    with pytest.raises(ValueError, match="invalid"):
        pool.free_block(99)
    assert pool.allocate_block() == first


def test_engine_rejects_invalid_configuration_and_empty_prompt():
    with pytest.raises(ValueError, match="max_batch_size"):
        make_engine(max_batch_size=0)
    with pytest.raises(ValueError, match="total_blocks"):
        make_engine(total_blocks=0)
    with pytest.raises(ValueError, match="block_size"):
        make_engine(block_size=0)

    engine = make_engine(tokenizer=FakeTokenizer())
    with pytest.raises(ValueError, match="max_tokens"):
        engine.add_request("valid", max_tokens=0)
    with pytest.raises(ValueError, match="prompt"):
        engine.add_request("", max_tokens=1)


def test_impossible_request_fails_instead_of_leaving_run_stuck():
    engine = make_engine(tokenizer=FakeTokenizer(tokens=[1, 2, 3]), total_blocks=1, block_size=2)
    engine.add_request("too-large", max_tokens=1)

    with pytest.raises(MemoryError, match="requires"):
        engine.run()
    assert not engine.running_batch
    assert not engine.waiting_queue
    assert engine.num_free_blocks == 1


def test_admission_allocation_failure_rolls_back_partial_state():
    engine = make_engine(tokenizer=FakeTokenizer(tokens=[1, 2, 3]), total_blocks=3, block_size=2)
    req = engine.add_request("two-block-prompt", max_tokens=1)
    original_allocate = engine._allocate_block
    calls = 0

    def fail_on_second_allocation():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected allocation failure")
        return original_allocate()

    engine._allocate_block = fail_on_second_allocation
    with pytest.raises(MemoryError, match="injected"):
        engine.step()

    assert req in engine.waiting_queue
    assert not engine.running_batch
    assert req.physical_block_ids == []
    assert engine.num_free_blocks == 3


def test_execution_failure_releases_running_request_and_blocks():
    engine = make_engine(total_blocks=3, block_size=2)
    req = engine.add_request("request", max_tokens=1)
    engine.model = RaisingModel()
    with pytest.raises(RuntimeError, match="injected model failure"):
        engine.step()

    assert req not in engine.running_batch
    assert req.kv_cache is None
    assert req.physical_block_ids == []
    assert engine.num_free_blocks == 3
    replacement = engine.add_request("replacement", max_tokens=1)
    assert replacement.req_id != req.req_id


def test_decode_failure_releases_running_request_and_blocks():
    engine = make_engine(total_blocks=3, block_size=2)
    req = engine.add_request("request", max_tokens=2)
    engine._schedule(current_step=0)
    req.num_prefilled_tokens = len(req.prompt_token_ids)
    req.output_token_ids = [1]

    def fail_decode(*args, **kwargs):
        raise RuntimeError("injected decode failure")

    engine.model_runner.batched_decode_step = fail_decode
    with pytest.raises(RuntimeError, match="injected decode failure"):
        engine.step()

    assert req not in engine.running_batch
    assert req.kv_cache is None
    assert req.physical_block_ids == []
    assert engine.num_free_blocks == 3


def test_cached_eos_request_finishes_without_an_extra_decode():
    tokens = list(range(16))
    engine = make_engine(
        tokenizer=FakeTokenizer(tokens=tokens),
        total_blocks=2,
        block_size=16,
        prefix=True,
    )
    block_id = engine.prefix_cache.allocate_new_block()
    block_hash = compute_block_hash(None, tuple(tokens))
    block = engine.prefix_cache.blocks[block_id]
    block.block_hash = block_hash
    block.ref_count = 0
    block.tokens = tokens
    engine.prefix_cache.hash_to_block_id[block_hash] = block_id
    logits = torch.full((1, 16), -100.0)
    logits[0, engine.tokenizer.eos_token_id] = 100.0
    engine.prefix_cache.hash_to_logits[block_hash] = logits

    req = engine.add_request("cached", max_tokens=8)
    completed = engine.run()

    assert completed == [req]
    assert req.status is RequestStatus.FINISHED
    assert req.output_token_ids == [engine.tokenizer.eos_token_id]
    assert req.stop_reason == "eos_token"
    assert not engine.running_batch
    assert engine.num_free_blocks == engine.total_blocks
