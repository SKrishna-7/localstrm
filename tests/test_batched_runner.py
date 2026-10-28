"""
Unit & Equivalence Tests for BatchedModelRunner (Milestone 2)
-------------------------------------------------------------
Verifies:
  1. Batched QKV projections produce identical predictions to HuggingFace reference.
  2. Ragged / variable sequence lengths operate without padding or cross-sequence corruption.
  3. Independent physical block tables maintain total isolation across sequences.
  4. Both FP16 and INT8 KV cache modes function correctly.
"""

import os
import sys
import copy
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import get_default_model, get_default_device
from src.engine.model_runner import BatchedModelRunner
from src.engine.llm_engine import MiniVLLMEngine, PhysicalBlockPool


class TestBatchedModelRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_str = get_default_device()
        cls.device = torch.device(cls.device_str)
        cls.model_name = get_default_model()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        dtype = torch.float32 if cls.device.type == "cpu" else torch.float16
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, torch_dtype=dtype).to(cls.device)
        cls.model.eval()

    def test_batched_decode_numerical_equivalence(self):
        """Tests that batched QKV decode matches single-sequence generation under greedy argmax."""
        prompt1 = "What is the capital of France?"
        prompt2 = "Write a quick Python hello world script."

        tok1 = self.tokenizer.encode(prompt1, return_tensors="pt").to(self.device)
        tok2 = self.tokenizer.encode(prompt2, return_tensors="pt").to(self.device)

        len1 = tok1.shape[1]
        len2 = tok2.shape[1]
        self.assertNotEqual(len1, len2, "Prompts should have different lengths for ragged test")

        # 1. HuggingFace Reference Decode
        with torch.no_grad():
            out1 = self.model(tok1, use_cache=True)
            ref_first1 = torch.argmax(out1.logits[:, -1, :], dim=-1, keepdim=True)
            ref_step1 = self.model(ref_first1, past_key_values=out1.past_key_values, use_cache=True)
            ref_tok1 = torch.argmax(ref_step1.logits[:, -1, :], dim=-1, keepdim=True)

            out2 = self.model(tok2, use_cache=True)
            ref_first2 = torch.argmax(out2.logits[:, -1, :], dim=-1, keepdim=True)
            ref_step2 = self.model(ref_first2, past_key_values=out2.past_key_values, use_cache=True)
            ref_tok2 = torch.argmax(ref_step2.logits[:, -1, :], dim=-1, keepdim=True)

        # 2. Setup Physical KV Pool for BatchedModelRunner
        block_size = 16
        total_blocks = 20
        runner = BatchedModelRunner(self.model, device=self.device, block_size=block_size, kv_cache_dtype="fp16")

        kv_pool = PhysicalBlockPool(
            num_layers=runner.num_layers,
            total_blocks=total_blocks,
            block_size=block_size,
            num_kv_heads=runner.num_kv_heads,
            head_dim=runner.head_dim,
            device=self.device,
            kv_cache_dtype="fp16"
        )

        # Populate prompt tokens for seq 1 and seq 2 into separate blocks
        # Seq 1 assigned blocks [1, 3, 5]
        # Seq 2 assigned blocks [2, 4, 6]
        block_tables = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.long, device=self.device)
        seq_lens = torch.tensor([len1 + 1, len2 + 1], dtype=torch.long, device=self.device)

        for l in range(runner.num_layers):
            k1 = out1.past_key_values.layers[l].keys[0]  # [num_kv, len1, head_dim]
            v1 = out1.past_key_values.layers[l].values[0]
            for t in range(len1):
                b_id = block_tables[0, t // block_size].item()
                b_off = t % block_size
                kv_pool.k_pool[l, b_id, b_off] = k1[:, t, :]
                kv_pool.v_pool[l, b_id, b_off] = v1[:, t, :]

            k2 = out2.past_key_values.layers[l].keys[0]
            v2 = out2.past_key_values.layers[l].values[0]
            for t in range(len2):
                b_id = block_tables[1, t // block_size].item()
                b_off = t % block_size
                kv_pool.k_pool[l, b_id, b_off] = k2[:, t, :]
                kv_pool.v_pool[l, b_id, b_off] = v2[:, t, :]

        # Batched input: [2, 1]
        batched_tokens = torch.cat([ref_first1, ref_first2], dim=0)

        # Execute Batched Decode Step
        batched_logits = runner.batched_decode_step(batched_tokens, block_tables, seq_lens, kv_pool)
        self.assertEqual(batched_logits.shape, (2, self.model.config.vocab_size))

        pred_tok1 = torch.argmax(batched_logits[0:1], dim=-1, keepdim=True)
        pred_tok2 = torch.argmax(batched_logits[1:2], dim=-1, keepdim=True)

        self.assertEqual(pred_tok1.item(), ref_tok1.item(), "Sequence 1 prediction must match HuggingFace reference")
        self.assertEqual(pred_tok2.item(), ref_tok2.item(), "Sequence 2 prediction must match HuggingFace reference")

    def test_batched_decode_int8_kv(self):
        """Tests that batched decode operates cleanly under INT8 quantized KV cache."""
        runner = BatchedModelRunner(self.model, device=self.device, block_size=16, kv_cache_dtype="int8")
        total_blocks = 10
        kv_pool = PhysicalBlockPool(
            num_layers=runner.num_layers,
            total_blocks=total_blocks,
            block_size=16,
            num_kv_heads=runner.num_kv_heads,
            head_dim=runner.head_dim,
            device=self.device,
            kv_cache_dtype="int8"
        )

        batched_tokens = torch.tensor([[100], [200]], device=self.device)
        block_tables = torch.tensor([[0], [1]], dtype=torch.long, device=self.device)
        seq_lens = torch.tensor([1, 1], dtype=torch.long, device=self.device)

        logits = runner.batched_decode_step(batched_tokens, block_tables, seq_lens, kv_pool)
        self.assertEqual(logits.shape, (2, self.model.config.vocab_size))
        self.assertFalse(torch.isnan(logits).any(), "INT8 decode logits must not contain NaNs")

    def test_real_qwen_multi_request_greedy_outputs_match_huggingface(self):
        """Unrelated co-batched requests must not change greedy token IDs."""
        prompts = ["Hello", "Explain why the sky appears blue.", "2 + 2 ="]
        budget = 8
        references = {}

        with torch.no_grad():
            for prompt in prompts:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                generated = []
                past = None
                current = input_ids
                for _ in range(budget):
                    outputs = self.model(
                        current,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = outputs.past_key_values
                    token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
                    generated.append(token)
                    current = torch.tensor([[token]], dtype=torch.long, device=self.device)
                references[prompt] = generated

        def run_batch(request_order):
            engine = MiniVLLMEngine(
                model_name=self.model_name,
                device=str(self.device),
                tokenizer=self.tokenizer,
                model=self.model,
                max_batch_size=3,
                total_blocks=32,
                chunked_prefill=False,
                enable_prefix_caching=False,
                batched_decode=False,
            )
            for prompt in request_order:
                engine.add_request(prompt=prompt, max_tokens=budget)
            completed = engine.run()
            return [list(req.output_token_ids) for req in completed]

        self.assertEqual(
            run_batch(prompts),
            [references[prompt] for prompt in prompts],
        )
        reversed_prompts = list(reversed(prompts))
        self.assertEqual(
            run_batch(reversed_prompts),
            [references[prompt] for prompt in reversed_prompts],
        )

    def test_true_batched_decode_parallel_execution(self):
        """Tests that batched_decode=True executes in true parallel mode, frees all blocks, and is permutation invariant."""
        prompts = [
            "The capital of France is",
            "Write a Python function to compute fibonacci numbers.",
            "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
        ]
        budget = 12

        def run_parallel(request_order):
            engine = MiniVLLMEngine(
                model_name=self.model_name,
                device=str(self.device),
                tokenizer=self.tokenizer,
                model=self.model,
                max_batch_size=len(prompts),
                total_blocks=64,
                chunked_prefill=False,
                enable_prefix_caching=False,
                batched_decode=True,
            )
            for prompt in request_order:
                engine.add_request(prompt=prompt, max_tokens=budget)
            completed = engine.run()
            self.assertEqual(engine.kv_pool.num_free, engine.kv_pool.total_blocks, "All KV blocks must be freed")
            return {req.prompt: list(req.output_token_ids) for req in completed}

        out_forward = run_parallel(prompts)
        out_reversed = run_parallel(list(reversed(prompts)))

        for prompt in prompts:
            self.assertEqual(len(out_forward[prompt]), budget, f"Request for '{prompt}' should generate {budget} tokens")
            self.assertEqual(out_forward[prompt], out_reversed[prompt], f"Parallel execution must be permutation invariant for '{prompt}'")

    def test_gqa_zero_copy_mixed_lengths_and_cleanup(self):
        """Tests GQA zero-copy execution with ragged/mixed sequence lengths and KV block recovery in FP16."""
        prompts = [
            "Hi",
            "What is the airspeed velocity of an unladen swallow?",
            "Solve: x^2 - 4x + 4 = 0",
            "Machine learning on consumer GPUs requires predictable memory management.",
        ]
        budget = 8

        engine = MiniVLLMEngine(
            model_name=self.model_name,
            device=str(self.device),
            tokenizer=self.tokenizer,
            model=self.model,
            max_batch_size=len(prompts),
            total_blocks=64,
            chunked_prefill=True,
            prefill_chunk_size=16,
            enable_prefix_caching=False,
            batched_decode=True,
            kv_cache_dtype="fp16",
        )
        for prompt in prompts:
            engine.add_request(prompt=prompt, max_tokens=budget)
        completed = engine.run()

        self.assertEqual(len(completed), len(prompts))
        self.assertEqual(engine.kv_pool.num_free, engine.kv_pool.total_blocks, "All KV blocks must be freed")
        for req in completed:
            self.assertEqual(len(req.output_token_ids), budget)

    def test_gqa_zero_copy_int8_kv_cache(self):
        """Tests GQA zero-copy execution with INT8 quantized KV cache and verifies cleanup."""
        prompts = [
            "Explain quantization in neural networks.",
            "List three prime numbers greater than 50.",
        ]
        budget = 10

        engine = MiniVLLMEngine(
            model_name=self.model_name,
            device=str(self.device),
            tokenizer=self.tokenizer,
            model=self.model,
            max_batch_size=len(prompts),
            total_blocks=48,
            chunked_prefill=True,
            prefill_chunk_size=32,
            enable_prefix_caching=False,
            batched_decode=True,
            kv_cache_dtype="int8",
        )
        for prompt in prompts:
            engine.add_request(prompt=prompt, max_tokens=budget)
        completed = engine.run()

        self.assertEqual(len(completed), len(prompts))
        self.assertEqual(engine.kv_pool.num_free, engine.kv_pool.total_blocks, "All INT8 KV blocks must be freed")
        for req in completed:
            self.assertEqual(len(req.output_token_ids), budget)


if __name__ == "__main__":
    unittest.main()
