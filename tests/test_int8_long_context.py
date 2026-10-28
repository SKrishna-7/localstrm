"""
Long-Context INT8 KV Cache Evaluation & Numerical Scaling Test Suite
--------------------------------------------------------------------
Evaluates INT8 KV Cache behavior as sequence length scales up to 1024+ tokens:
  1. Memory Footprint Scaling: Evaluates analytical and physical tensor memory
     reduction across 64, 128, 256, 512, 1024, and 2048 tokens.
  2. Long-Context Logit Error: Measures relative L_inf error and greedy token
     agreement across multi-step decode on a 512-token context.
  3. Numerical Stability: Confirms zero NaNs, Infs, or scale degradation over long contexts.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_default_model, get_default_device
from src.engine.llm_engine import MiniVLLMEngine, PhysicalBlockPool, SamplingParams, RequestStatus
from src.engine.model_runner import BatchedModelRunner
from benchmarks.instrumentation import calculate_analytical_kv_cache_mb
from transformers import AutoModelForCausalLM, AutoTokenizer


class TestINT8LongContext(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_str = get_default_device()
        cls.device = torch.device(cls.device_str)
        cls.model_name = get_default_model()

    def test_analytical_and_physical_memory_scaling(self):
        """
        Verifies that INT8 KV cache halves tensor byte consumption across
        increasing context lengths (64, 128, 256, 512, 1024, 2048 tokens).
        """
        num_layers = 24
        num_kv_heads = 2
        head_dim = 64
        token_counts = [64, 128, 256, 512, 1024, 2048]

        for num_tokens in token_counts:
            mb_fp16 = calculate_analytical_kv_cache_mb(num_layers, num_kv_heads, head_dim, num_tokens, "fp16")
            mb_int8 = calculate_analytical_kv_cache_mb(num_layers, num_kv_heads, head_dim, num_tokens, "int8")

            # INT8 is 1 byte per element + 2-byte scale per head_dim slice (64 elements)
            # scale overhead = 2 / 64 = 3.125%
            # Thus INT8 should be ~51.5% to 53.5% of FP16 memory
            ratio = mb_int8 / mb_fp16
            self.assertLess(ratio, 0.55, f"INT8 memory ratio {ratio:.3f} should be < 0.55 of FP16")
            self.assertGreater(ratio, 0.49, f"INT8 memory ratio {ratio:.3f} should be > 0.49 of FP16")

        # Physical pool memory test
        block_size = 16
        total_blocks = 128  # 2048 tokens capacity
        pool_fp16 = PhysicalBlockPool(
            num_layers=num_layers,
            total_blocks=total_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=self.device,
            kv_cache_dtype="fp16"
        )
        pool_int8 = PhysicalBlockPool(
            num_layers=num_layers,
            total_blocks=total_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=self.device,
            kv_cache_dtype="int8"
        )

        bytes_fp16 = pool_fp16.k_pool.numel() * pool_fp16.k_pool.element_size() * 2
        bytes_int8 = (pool_int8.k_pool.numel() * pool_int8.k_pool.element_size() * 2) + \
                     (pool_int8.k_scales.numel() * pool_int8.k_scales.element_size() * 2)

        physical_ratio = bytes_int8 / bytes_fp16
        print(f"\n[Physical Pool Memory Scaling (2048 tokens)]")
        print(f"  • FP16 Total Bytes: {bytes_fp16 / (1024**2):.2f} MB")
        print(f"  • INT8 Total Bytes: {bytes_int8 / (1024**2):.2f} MB (Ratio: {physical_ratio:.3f})")

        self.assertLess(physical_ratio, 0.55)

    def test_long_context_multi_step_precision(self):
        """
        Validates that INT8 KV Cache preserves high numerical fidelity
        and token agreement over a 512-token context prefix across 15 decode steps.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device)
        model.eval()

        # Build a 512-token prompt
        base_sentence = "Artificial intelligence and distributed high throughput serving systems require efficient memory caching. "
        prompt = base_sentence * 25
        encoded = tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"][:, :512]
        prompt_len = input_ids.shape[1]

        # Prefill via HuggingFace
        with torch.no_grad():
            hf_out = model(input_ids, use_cache=True)
            first_token = torch.argmax(hf_out.logits[:, -1, :], dim=-1, keepdim=True)

        block_size = 16
        needed_blocks = (prompt_len + 32 + block_size - 1) // block_size
        assigned_blocks = list(range(needed_blocks))
        block_tables = torch.tensor([assigned_blocks], dtype=torch.long, device=self.device)

        runner_fp16 = BatchedModelRunner(model, self.device, block_size=block_size, kv_cache_dtype="fp16")
        pool_fp16 = PhysicalBlockPool(
            num_layers=runner_fp16.num_layers,
            total_blocks=needed_blocks + 10,
            block_size=block_size,
            num_kv_heads=runner_fp16.num_kv_heads,
            head_dim=runner_fp16.head_dim,
            device=self.device,
            kv_cache_dtype="fp16"
        )

        runner_int8 = BatchedModelRunner(model, self.device, block_size=block_size, kv_cache_dtype="int8")
        pool_int8 = PhysicalBlockPool(
            num_layers=runner_int8.num_layers,
            total_blocks=needed_blocks + 10,
            block_size=block_size,
            num_kv_heads=runner_int8.num_kv_heads,
            head_dim=runner_int8.head_dim,
            device=self.device,
            kv_cache_dtype="int8"
        )

        # Populate prefilled KV
        for l in range(runner_fp16.num_layers):
            k = hf_out.past_key_values.layers[l].keys[0]
            v = hf_out.past_key_values.layers[l].values[0]
            for t in range(prompt_len):
                b_id = assigned_blocks[t // block_size]
                b_off = t % block_size
                k_vec = k[:, t, :]
                v_vec = v[:, t, :]

                # FP16
                pool_fp16.k_pool[l, b_id, b_off] = k_vec
                pool_fp16.v_pool[l, b_id, b_off] = v_vec

                # INT8
                sk = torch.clamp(k_vec.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
                sv = torch.clamp(v_vec.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
                pool_int8.k_pool[l, b_id, b_off] = torch.clamp(torch.round(k_vec / sk), -128, 127).to(torch.int8)
                pool_int8.v_pool[l, b_id, b_off] = torch.clamp(torch.round(v_vec / sv), -128, 127).to(torch.int8)
                pool_int8.k_scales[l, b_id, b_off] = sk
                pool_int8.v_scales[l, b_id, b_off] = sv

        # Decode for 15 steps
        curr_tok_fp16 = first_token
        curr_tok_int8 = first_token
        matching_tokens = 0
        rel_errors = []

        for step in range(15):
            seq_lens = torch.tensor([prompt_len + 1 + step], dtype=torch.long, device=self.device)

            logits_fp16 = runner_fp16.batched_decode_step(curr_tok_fp16, block_tables, seq_lens, pool_fp16)
            logits_int8 = runner_int8.batched_decode_step(curr_tok_int8, block_tables, seq_lens, pool_int8)

            self.assertFalse(torch.isnan(logits_int8).any(), "NaN detected in INT8 logits")
            self.assertFalse(torch.isinf(logits_int8).any(), "Inf detected in INT8 logits")

            abs_diff = torch.abs(logits_fp16 - logits_int8)
            fp16_scale = logits_fp16.abs().max().item()
            rel_err = abs_diff.max().item() / max(fp16_scale, 1e-5)
            rel_errors.append(rel_err)

            tok_fp16 = torch.argmax(logits_fp16, dim=-1, keepdim=True)
            tok_int8 = torch.argmax(logits_int8, dim=-1, keepdim=True)

            if tok_fp16.item() == tok_int8.item():
                matching_tokens += 1

            curr_tok_fp16 = tok_fp16
            curr_tok_int8 = tok_int8

        avg_rel = sum(rel_errors) / len(rel_errors)
        max_rel = max(rel_errors)
        agreement_pct = (matching_tokens / 15) * 100.0

        print(f"\n[Long-Context INT8 Multi-Step Numerical Precision (512 prompt tokens)]")
        print(f"  • Decode Steps:       15")
        print(f"  • Token Agreement:    {matching_tokens}/15 ({agreement_pct:.1f}%)")
        print(f"  • Avg Relative Error: {avg_rel:.4e}")
        print(f"  • Max Relative Error: {max_rel:.4e}")

        self.assertLess(max_rel, 0.08, f"Max relative error {max_rel:.4e} exceeds 8%")
        self.assertGreaterEqual(agreement_pct, 80.0, f"Greedy agreement {agreement_pct:.1f}% below 80%")

    def test_long_context_end_to_end_engine(self):
        """
        Executes an end-to-end MiniVLLMEngine run with INT8 KV Cache
        over a ~500-token prompt generating 15 tokens.
        """
        base_sentence = "Deep neural networks are composed of multiple layers of linear matrix multiplications and non-linearities. "
        prompt = base_sentence * 25

        engine = MiniVLLMEngine(
            model_name=self.model_name,
            device=self.device_str,
            max_batch_size=2,
            total_blocks=128,
            block_size=16,
            kv_cache_dtype="int8",
            chunked_prefill=True,
            prefill_chunk_size=64,
            enable_prefix_caching=False,
            verbose=False
        )

        req = engine.add_request(prompt, max_tokens=15, sampling_params=SamplingParams.greedy())
        engine.run()

        self.assertEqual(req.status, RequestStatus.FINISHED)
        self.assertEqual(req.num_generated, 15)
        text = engine.tokenizer.decode(req.output_token_ids)
        print(f"\n[Long-Context INT8 E2E Engine Run]")
        print(f"  • Prompt Tokens:    {len(req.prompt_token_ids)}")
        print(f"  • Generated Tokens: {req.num_generated}")
        print(f"  • Decoded Output:   {text}")
        self.assertGreater(len(text.strip()), 0)


if __name__ == "__main__":
    unittest.main()
