"""
Tests for Sampling Parity:
Verifies that batched GPU argmax decode sampling produces identical output tokens
to sequential sample_next_token across batch sizes, prompts, and KV boundaries.
"""

import pytest
import torch
from src.engine.llm_engine import MiniVLLMEngine
from src.engine.sampling import SamplingParams
from src.config import get_default_model


@pytest.fixture(scope="module")
def model_path():
    return get_default_model()


def test_sampling_parity_greedy_identical_tokens(model_path):
    """Verify that batched argmax and sequential sampling produce 100% identical tokens."""
    prompts = [
        "The capital of France is",
        "Python is a popular programming language because",
        "Artificial intelligence and machine learning are",
        "In physics, Newton's third law of motion states",
    ]
    max_tokens = 16

    # 1. Run with sequential sampling
    engine_seq = MiniVLLMEngine(
        model_name=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_batch_size=len(prompts),
        total_blocks=64,
        block_size=16,
        batched_decode=True,
        enable_batched_sampling=False,
        verbose=False,
    )
    for p in prompts:
        engine_seq.add_request(prompt=p, max_tokens=max_tokens, sampling_params=SamplingParams.greedy())
    completed_seq = engine_seq.run()
    seq_tokens = {r.prompt: r.output_token_ids for r in completed_seq}
    assert engine_seq.num_free_blocks == engine_seq.total_blocks

    # 2. Run with batched sampling
    engine_bat = MiniVLLMEngine(
        model_name=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_batch_size=len(prompts),
        total_blocks=64,
        block_size=16,
        batched_decode=True,
        enable_batched_sampling=True,
        verbose=False,
    )
    for p in prompts:
        engine_bat.add_request(prompt=p, max_tokens=max_tokens, sampling_params=SamplingParams.greedy())
    completed_bat = engine_bat.run()
    bat_tokens = {r.prompt: r.output_token_ids for r in completed_bat}
    assert engine_bat.num_free_blocks == engine_bat.total_blocks

    # 3. Assert exact match
    for p in prompts:
        assert seq_tokens[p] == bat_tokens[p], f"Mismatch for prompt '{p}':\nSeq: {seq_tokens[p]}\nBat: {bat_tokens[p]}"


def test_sampling_parity_single_request(model_path):
    """Verify that single request (concurrency 1) behaves identically."""
    prompt = "Deep learning architectures have revolutionized"
    max_tokens = 12

    engine_seq = MiniVLLMEngine(
        model_name=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_batch_size=1,
        total_blocks=32,
        block_size=16,
        batched_decode=True,
        enable_batched_sampling=False,
        verbose=False,
    )
    engine_seq.add_request(prompt=prompt, max_tokens=max_tokens, sampling_params=SamplingParams.greedy())
    completed_seq = engine_seq.run()

    engine_bat = MiniVLLMEngine(
        model_name=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_batch_size=1,
        total_blocks=32,
        block_size=16,
        batched_decode=True,
        enable_batched_sampling=True,
        verbose=False,
    )
    engine_bat.add_request(prompt=prompt, max_tokens=max_tokens, sampling_params=SamplingParams.greedy())
    completed_bat = engine_bat.run()

    assert completed_seq[0].output_token_ids == completed_bat[0].output_token_ids
    assert engine_seq.num_free_blocks == engine_seq.total_blocks
    assert engine_bat.num_free_blocks == engine_bat.total_blocks
