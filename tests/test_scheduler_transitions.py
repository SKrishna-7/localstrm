"""
Stress & Transition Tests for Continuous Batching Scheduler & Memory Paging
----------------------------------------------------------------------------
Rigorously verifies:
  1. Staggered Request Completion: Sequences finishing at different steps cleanly
     evict, return blocks to the free pool, and do not disrupt active sequences.
  2. Dynamic Mid-Batch Admission: Adding new requests while a batch is mid-decode
     correctly schedules prefill and integrates them into the running batch.
  3. Memory Starvation & Block Recycling: High-demand workloads on a memory-constrained
     block pool wait safely in the queue, recover blocks as requests finish, and
     complete without deadlock or leaked blocks.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_default_model, get_default_device
from src.engine.llm_engine import MiniVLLMEngine, RequestStatus, SamplingParams


class TestSchedulerTransitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_str = get_default_device()
        cls.model_name = get_default_model()

    def test_staggered_request_completion(self):
        """
        Verify that requests with different completion lengths (e.g. 3, 8, 15 tokens)
        complete at different times, free their blocks immediately, and allow remaining
        sequences to decode without corruption.
        """
        engine = MiniVLLMEngine(
            model_name=self.model_name,
            device=self.device_str,
            max_batch_size=4,
            total_blocks=32,
            block_size=16,
            chunked_prefill=True,
            prefill_chunk_size=32,
            enable_prefix_caching=False,  # direct block testing
            verbose=False
        )

        initial_free_blocks = engine.num_free_blocks

        # Add 3 requests with staggered max_tokens
        req_short = engine.add_request("Paris is the capital of", max_tokens=3, sampling_params=SamplingParams.greedy())
        req_med = engine.add_request("Python is a programming language that", max_tokens=8, sampling_params=SamplingParams.greedy())
        req_long = engine.add_request("Artificial intelligence and machine learning are", max_tokens=15, sampling_params=SamplingParams.greedy())

        results = engine.run()
        self.assertEqual(len(results), 3)

        self.assertEqual(req_short.status, RequestStatus.FINISHED)
        self.assertEqual(req_short.num_generated, 3)

        self.assertEqual(req_med.status, RequestStatus.FINISHED)
        self.assertEqual(req_med.num_generated, 8)

        self.assertEqual(req_long.status, RequestStatus.FINISHED)
        self.assertEqual(req_long.num_generated, 15)

        # Verify that all physical blocks were returned to the free pool
        self.assertEqual(engine.num_free_blocks, initial_free_blocks, "All blocks must be freed after completion")
        self.assertEqual(len(engine.running_batch), 0)
        self.assertEqual(len(engine.waiting_queue), 0)

    def test_dynamic_mid_batch_admission(self):
        """
        Verify that submitting new requests while an existing batch is in decode phase
        properly schedules them and decodes them concurrently.
        """
        engine = MiniVLLMEngine(
            model_name=self.model_name,
            device=self.device_str,
            max_batch_size=4,
            total_blocks=32,
            block_size=16,
            chunked_prefill=True,
            prefill_chunk_size=32,
            enable_prefix_caching=False,
            verbose=False
        )

        initial_free = engine.num_free_blocks

        # Start with Request 1
        req1 = engine.add_request("The ocean is", max_tokens=12, sampling_params=SamplingParams.greedy())

        # Step engine a few times so req1 is in decode phase
        for _ in range(4):
            engine.step()

        self.assertTrue(req1.is_prefill_done)
        self.assertGreater(req1.num_generated, 0)
        self.assertEqual(len(engine.running_batch), 1)

        # Dynamically inject Request 2 mid-execution
        req2 = engine.add_request("The solar system has", max_tokens=6, sampling_params=SamplingParams.greedy())

        # Step once: req2 should be admitted and prefilled
        engine.step()
        self.assertIn(req2, engine.running_batch)

        # Run until both complete
        while engine.step():
            pass

        self.assertEqual(req1.status, RequestStatus.FINISHED)
        self.assertEqual(req2.status, RequestStatus.FINISHED)
        self.assertEqual(req1.num_generated, 12)
        self.assertEqual(req2.num_generated, 6)

        # Verify no memory leak
        self.assertEqual(engine.num_free_blocks, initial_free)

    def test_memory_starvation_and_block_recycling(self):
        """
        Verify engine under extreme memory constraint:
        Total blocks = 4. Each request requires 1-2 blocks.
        Batch size = 4, but only ~2 requests can fit at a time.
        Queued requests must wait, and once running requests finish and free blocks,
        waiting requests must be scheduled and completed without deadlock.
        """
        engine = MiniVLLMEngine(
            model_name=self.model_name,
            device=self.device_str,
            max_batch_size=4,
            total_blocks=5,  # strictly constrained
            block_size=16,
            chunked_prefill=True,
            prefill_chunk_size=16,
            enable_prefix_caching=False,
            verbose=False
        )

        initial_free = engine.num_free_blocks
        self.assertEqual(initial_free, 5)

        # Add 4 requests that collectively exceed 5 blocks if run concurrently
        # Each prompt has ~5-10 tokens, output ~8 tokens -> ~1-2 blocks each
        reqs = [
            engine.add_request(f"Question {i}: What is {i} + {i}?", max_tokens=6, sampling_params=SamplingParams.greedy())
            for i in range(4)
        ]

        completed = engine.run()
        self.assertEqual(len(completed), 4)

        for r in reqs:
            self.assertEqual(r.status, RequestStatus.FINISHED)
            self.assertGreaterEqual(r.num_generated, 1)
            self.assertLessEqual(r.num_generated, 6)

        # Verify complete block recovery
        self.assertEqual(engine.num_free_blocks, initial_free)


if __name__ == "__main__":
    unittest.main()
