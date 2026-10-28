"""
Comprehensive GQA Generalization & Robustness Test Suite (Gate 2).
Covers:
1. Multiple query-head and KV-head configurations (Q=14/KV=2, Q=32/KV=8, Q=8/KV=2, Q=8/KV=8 MHA).
2. Correct mapping of query heads to corresponding KV heads (contiguous vs interleaved).
3. Rejection of invalid head configurations and unsupported options.
4. Mixed sequence lengths and ragged attention masks.
5. Physical KV block boundaries and block allocation/cleanup.
6. Supported cache dtypes: FP16, INT8, and BF16 (on supported hardware).
7. Error behavior on invalid dtype/scale combinations.
8. Deterministic repeatability and request-order permutations.
9. Numerical equivalence with trusted reference implementations (0.0 difference).
"""

import math
import unittest
import torch
import torch.nn.functional as F

from src.kernels.paged_attention import paged_attention_decode
from src.engine.llm_engine import MiniVLLMEngine, PhysicalBlockPool
from src.config import get_default_device, get_default_model


class TestGQAGeneralization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_str = get_default_device()
        cls.device = torch.device(cls.device_str)
        cls.model_name = get_default_model()

    def _ground_truth_attention(self, query, k_pool, v_pool, block_tables, seq_lens, block_size, head_mapping="contiguous", k_scales=None, v_scales=None):
        """Reference un-grouped standard attention for mathematical equivalence verification."""
        B, num_heads, _, head_dim = query.shape
        num_kv_heads = k_pool.shape[2]
        G = num_heads // num_kv_heads
        max_sl = int(seq_lens.max().item())
        max_blocks = (max_sl + block_size - 1) // block_size

        # Gather and dequantize
        sub_tables = block_tables[:, :max_blocks]
        keys = k_pool[sub_tables]
        values = v_pool[sub_tables]
        if k_scales is not None:
            keys = keys.to(query.dtype) * k_scales[sub_tables]
        if v_scales is not None:
            values = values.to(query.dtype) * v_scales[sub_tables]

        flat_keys = keys.view(B, -1, num_kv_heads, head_dim)[:, :max_sl].permute(0, 2, 1, 3)
        flat_values = values.view(B, -1, num_kv_heads, head_dim)[:, :max_sl].permute(0, 2, 1, 3)

        # Expand KV heads to match query heads explicitly via reference repeat_interleave
        if G > 1:
            if head_mapping == "contiguous":
                ref_k = flat_keys.repeat_interleave(G, dim=1)
                ref_v = flat_values.repeat_interleave(G, dim=1)
                ref_q = query
            else: # interleaved
                ref_k = flat_keys.repeat(1, G, 1, 1)
                ref_v = flat_values.repeat(1, G, 1, 1)
                ref_q = query
        else:
            ref_k = flat_keys
            ref_v = flat_values
            ref_q = query

        mask = (torch.arange(max_sl, device=query.device).unsqueeze(0) < seq_lens.unsqueeze(1)).view(B, 1, 1, max_sl)
        return F.scaled_dot_product_attention(ref_q, ref_k, ref_v, attn_mask=mask)

    def test_multiple_head_configurations_fp16(self):
        """Validates zero-copy GQA across various head topologies: Qwen (14/2), LLaMA (32/8), small (8/2), MHA (8/8)."""
        configs = [
            (14, 2, 64),   # Qwen2.5-0.5B (G=7)
            (32, 8, 128),  # LLaMA-3-8B style (G=4)
            (8, 2, 64),    # Small GQA (G=4)
            (6, 3, 32),    # Small GQA (G=2)
            (8, 8, 64),    # MHA (G=1)
        ]

        for num_heads, num_kv_heads, head_dim in configs:
            for B in [1, 3, 6]:
                seq_lens = torch.randint(8, 60, (B,), device=self.device)
                max_sl = int(seq_lens.max().item())
                block_size = 16
                max_blocks = (max_sl + block_size - 1) // block_size
                total_blocks = max_blocks * B + 4

                q = torch.randn(B, num_heads, 1, head_dim, device=self.device, dtype=torch.float16)
                k_pool = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=self.device, dtype=torch.float16)
                v_pool = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=self.device, dtype=torch.float16)
                block_tables = torch.randint(0, total_blocks, (B, max_blocks), device=self.device)

                ref_out = self._ground_truth_attention(q, k_pool, v_pool, block_tables, seq_lens, block_size)
                opt_out = paged_attention_decode(
                    query=q,
                    k_cache_pool=k_pool,
                    v_cache_pool=v_pool,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    block_size=block_size,
                )

                diff = (ref_out - opt_out).abs().max().item()
                self.assertEqual(diff, 0.0, f"Mismatch for config H={num_heads}, KV={num_kv_heads}, B={B}: max diff {diff}")

    def test_interleaved_head_mapping(self):
        """Tests that 'interleaved' head mapping correctly matches reference interleaved expansion."""
        B, num_heads, num_kv_heads, head_dim = 2, 8, 2, 64
        seq_lens = torch.tensor([12, 28], device=self.device)
        block_size = 16
        total_blocks = 10
        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.long, device=self.device)

        q = torch.randn(B, num_heads, 1, head_dim, device=self.device, dtype=torch.float16)
        k_pool = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=self.device, dtype=torch.float16)
        v_pool = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=self.device, dtype=torch.float16)

        ref_out = self._ground_truth_attention(q, k_pool, v_pool, block_tables, seq_lens, block_size, head_mapping="interleaved")
        opt_out = paged_attention_decode(
            query=q,
            k_cache_pool=k_pool,
            v_cache_pool=v_pool,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
            head_mapping="interleaved",
        )

        diff = (ref_out - opt_out).abs().max().item()
        self.assertEqual(diff, 0.0, f"Interleaved head mapping mismatch: {diff}")

    def test_architectural_validation_and_errors(self):
        """Verifies clear error rejection on invalid head topologies, dtypes, and configurations."""
        block_size = 16
        total_blocks = 8
        seq_lens = torch.tensor([10], device=self.device)
        block_tables = torch.tensor([[0]], device=self.device)

        k_pool = torch.randn(total_blocks, block_size, 4, 64, device=self.device, dtype=torch.float16)
        v_pool = torch.randn(total_blocks, block_size, 4, 64, device=self.device, dtype=torch.float16)

        # 1. num_heads not divisible by num_kv_heads (14 not divisible by 4)
        q_indivisible = torch.randn(1, 14, 1, 64, device=self.device, dtype=torch.float16)
        with self.assertRaises(ValueError) as ctx:
            paged_attention_decode(q_indivisible, k_pool, v_pool, block_tables, seq_lens, block_size)
        self.assertIn("must be divisible", str(ctx.exception))

        # 2. num_heads < num_kv_heads
        q_fewer = torch.randn(1, 2, 1, 64, device=self.device, dtype=torch.float16)
        with self.assertRaises(ValueError) as ctx:
            paged_attention_decode(q_fewer, k_pool, v_pool, block_tables, seq_lens, block_size)
        self.assertIn("cannot be less than", str(ctx.exception))

        # 3. Unsupported head_mapping
        q_valid = torch.randn(1, 8, 1, 64, device=self.device, dtype=torch.float16)
        with self.assertRaises(ValueError) as ctx:
            paged_attention_decode(q_valid, k_pool, v_pool, block_tables, seq_lens, block_size, head_mapping="unsupported_custom")
        self.assertIn("Unsupported head_mapping", str(ctx.exception))

        # 4. INT8 KV cache without scales raises clear ValueError
        k_int8 = torch.randint(-128, 127, (total_blocks, block_size, 4, 64), device=self.device, dtype=torch.int8)
        v_int8 = torch.randint(-128, 127, (total_blocks, block_size, 4, 64), device=self.device, dtype=torch.int8)
        with self.assertRaises(ValueError) as ctx:
            paged_attention_decode(q_valid, k_int8, v_int8, block_tables, seq_lens, block_size, k_scales=None)
        self.assertIn("requires both k_scales and v_scales", str(ctx.exception))

        # 5. Dimension mismatch
        q_3d = torch.randn(1, 8, 64, device=self.device, dtype=torch.float16)
        with self.assertRaises(ValueError) as ctx:
            paged_attention_decode(q_3d, k_pool, v_pool, block_tables, seq_lens, block_size)
        self.assertIn("Expected query to be a 4D tensor", str(ctx.exception))

    def test_int8_quantized_kv_gqa(self):
        """Verifies INT8 quantized KV cache operates with zero-copy GQA across ragged batch."""
        B, num_heads, num_kv_heads, head_dim = 3, 14, 2, 64
        seq_lens = torch.tensor([12, 35, 52], device=self.device)
        block_size = 16
        total_blocks = 20
        block_tables = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], device=self.device)

        q = torch.randn(B, num_heads, 1, head_dim, device=self.device, dtype=torch.float16)
        k_pool = torch.randint(-128, 127, (total_blocks, block_size, num_kv_heads, head_dim), device=self.device, dtype=torch.int8)
        v_pool = torch.randint(-128, 127, (total_blocks, block_size, num_kv_heads, head_dim), device=self.device, dtype=torch.int8)
        k_scales = torch.rand(total_blocks, block_size, num_kv_heads, 1, device=self.device, dtype=torch.float16)
        v_scales = torch.rand(total_blocks, block_size, num_kv_heads, 1, device=self.device, dtype=torch.float16)

        ref_out = self._ground_truth_attention(q, k_pool, v_pool, block_tables, seq_lens, block_size, k_scales=k_scales, v_scales=v_scales)
        opt_out = paged_attention_decode(
            query=q,
            k_cache_pool=k_pool,
            v_cache_pool=v_pool,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
            k_scales=k_scales,
            v_scales=v_scales,
        )

        diff = (ref_out - opt_out).abs().max().item()
        self.assertEqual(diff, 0.0, f"INT8 GQA mismatch: {diff}")

    def test_bfloat16_kv_gqa_if_supported(self):
        """Verifies BF16 execution on hardware that natively supports BF16."""
        if not (self.device.type == "cuda" and torch.cuda.is_bf16_supported()):
            self.skipTest("BF16 not supported on this CUDA device.")

        B, num_heads, num_kv_heads, head_dim = 2, 14, 2, 64
        seq_lens = torch.tensor([15, 30], device=self.device)
        block_size = 16
        total_blocks = 10
        block_tables = torch.tensor([[0, 1], [2, 3]], device=self.device)

        q = torch.randn(B, num_heads, 1, head_dim, device=self.device, dtype=torch.bfloat16)
        k_pool = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)
        v_pool = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=self.device, dtype=torch.bfloat16)

        ref_out = self._ground_truth_attention(q, k_pool, v_pool, block_tables, seq_lens, block_size)
        opt_out = paged_attention_decode(
            query=q,
            k_cache_pool=k_pool,
            v_cache_pool=v_pool,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
        )

        diff = (ref_out - opt_out).abs().max().item()
        self.assertEqual(diff, 0.0, f"BF16 GQA mismatch: {diff}")

    def test_repeatability_and_request_order(self):
        """Verifies repeated calls and batch permutations produce 100% deterministic identical outputs."""
        prompts = [
            "Test prompt A for GQA repeatability.",
            "Test prompt B with longer context to exercise variable length.",
            "Test prompt C.",
        ]
        budget = 8

        def run_engine(order):
            engine = MiniVLLMEngine(
                model_name=self.model_name,
                device=str(self.device),
                max_batch_size=len(order),
                total_blocks=48,
                chunked_prefill=True,
                enable_prefix_caching=False,
                batched_decode=True,
            )
            for p in order:
                engine.add_request(prompt=p, max_tokens=budget)
            completed = engine.run()
            self.assertEqual(engine.kv_pool.num_free, engine.kv_pool.total_blocks, "All KV blocks must be freed")
            return {r.prompt: list(r.output_token_ids) for r in completed}

        run1 = run_engine(prompts)
        run2 = run_engine(prompts)
        run_rev = run_engine(list(reversed(prompts)))

        for p in prompts:
            self.assertEqual(run1[p], run2[p], f"Non-deterministic execution on prompt: {p}")
            self.assertEqual(run1[p], run_rev[p], f"Order permutation mismatch on prompt: {p}")


if __name__ == "__main__":
    unittest.main()
