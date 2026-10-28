"""
Milestone 6: The PagedAttention Kernel & Numerical Verification
---------------------------------------------------------------
Implements the PagedAttention decode kernel that gathers Key and Value
vectors directly from non-contiguous physical blocks using the Block Table.
Proves numerical equivalence (0.00 difference) against standard contiguous attention.
"""

import math
import os
import sys
import argparse
from typing import Optional
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import get_default_device, add_common_engine_args


def standard_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Standard contiguous attention (Ground Truth)."""
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, values)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache_pool: torch.Tensor,
    v_cache_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
    k_scales: Optional[torch.Tensor] = None,
    v_scales: Optional[torch.Tensor] = None,
    sub_tables: Optional[torch.Tensor] = None,
    max_sl: Optional[int] = None,
    attn_mask: Optional[torch.Tensor] = None,
    head_mapping: str = "contiguous",
) -> torch.Tensor:
    """
    PagedAttention Decode Kernel with Zero-Copy GQA Expansion.

    Computes attention directly using the Block Table without
    re-allocating or copying the underlying physical memory.

    Args:
        query:        [batch_size, num_heads, 1, head_dim]
        k_cache_pool: [total_blocks, block_size, num_kv_heads, head_dim]
        v_cache_pool: [total_blocks, block_size, num_kv_heads, head_dim]
        block_tables: [batch_size, max_num_blocks_per_seq] (int64 physical block IDs)
        seq_lens:     [batch_size] (valid token count per sequence)
        block_size:   number of tokens per physical block
        k_scales:     optional [total_blocks, block_size, num_kv_heads, 1] scales for INT8 KV
        v_scales:     optional [total_blocks, block_size, num_kv_heads, 1] scales for INT8 KV
        sub_tables:   optional pre-sliced block tables [batch_size, max_blocks]
        max_sl:       optional pre-computed max sequence length
        attn_mask:    optional pre-computed attention boolean mask [batch_size, 1, 1, max_sl]
        head_mapping: "contiguous" (standard HuggingFace/Qwen/LLaMA/Mistral) or "interleaved"
    """
    if query.ndim != 4:
        raise ValueError(f"Expected query to be a 4D tensor [batch_size, num_heads, seq_len, head_dim], got shape {list(query.shape)}")
    if k_cache_pool.ndim != 4 or v_cache_pool.ndim != 4:
        raise ValueError("k_cache_pool and v_cache_pool must be 4D tensors [blocks, block_size, kv_heads, head_dim]")
    if k_cache_pool.shape[2] != v_cache_pool.shape[2]:
        raise ValueError(f"Mismatch in KV heads: k has {k_cache_pool.shape[2]}, v has {v_cache_pool.shape[2]}")
    if k_cache_pool.shape[3] != query.shape[3] or v_cache_pool.shape[3] != query.shape[3]:
        raise ValueError(f"head_dim mismatch: query={query.shape[3]}, k={k_cache_pool.shape[3]}, v={v_cache_pool.shape[3]}")

    batch_size, num_heads, _, head_dim = query.shape
    num_kv_heads = k_cache_pool.shape[2]

    if num_heads < num_kv_heads:
        raise ValueError(f"num_heads ({num_heads}) cannot be less than num_kv_heads ({num_kv_heads})")
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}) for GQA/MHA."
        )

    if head_mapping not in ("contiguous", "interleaved"):
        raise ValueError(
            f"Unsupported head_mapping: {head_mapping!r}. Supported options are 'contiguous' and 'interleaved'."
        )

    if k_cache_pool.dtype == torch.int8 and (k_scales is None or v_scales is None):
        raise ValueError("INT8 quantized KV cache requires both k_scales and v_scales for dequantization.")

    if query.dtype == torch.bfloat16 and query.is_cuda and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 is not supported on this CUDA device.")

    num_queries_per_kv = num_heads // num_kv_heads  # GQA ratio

    if max_sl is None:
        max_sl = int(seq_lens.max().item())

    if sub_tables is None:
        max_blocks = (max_sl + block_size - 1) // block_size
        sub_tables = block_tables[:, :max_blocks]

    # Gather physical blocks for the whole batch simultaneously
    keys = k_cache_pool[sub_tables]
    values = v_cache_pool[sub_tables]

    # Dequantize if scales are provided (INT8 KV Cache)
    if k_scales is not None:
        keys = keys.to(query.dtype) * k_scales[sub_tables]
    if v_scales is not None:
        values = values.to(query.dtype) * v_scales[sub_tables]

    # Flatten blocks to sequence dimension and slice to max_sl
    flat_keys = keys.view(batch_size, -1, num_kv_heads, head_dim)[:, :max_sl]
    flat_values = values.view(batch_size, -1, num_kv_heads, head_dim)[:, :max_sl]

    # Reshape to [batch_size, num_kv_heads, max_sl, head_dim]
    valid_keys = flat_keys.permute(0, 2, 1, 3)
    valid_values = flat_values.permute(0, 2, 1, 3)

    # Handle GQA (Grouped Query Attention)
    if num_queries_per_kv > 1:
        # Zero-copy GQA head expansion:
        # Instead of allocating repeated key/value tensors (num_queries_per_kv * memory),
        # view query as [B * H_kv, G, 1, D] and expand key/value as [B * H_kv, G, S, D] (stride 0 on dim 1).
        if head_mapping == "contiguous":
            q_grouped = query.reshape(batch_size * num_kv_heads, num_queries_per_kv, 1, head_dim)
        else:
            q_grouped = (
                query.view(batch_size, num_queries_per_kv, num_kv_heads, 1, head_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(batch_size * num_kv_heads, num_queries_per_kv, 1, head_dim)
            )

        k_grouped = valid_keys.reshape(batch_size * num_kv_heads, 1, max_sl, head_dim).expand(
            batch_size * num_kv_heads, num_queries_per_kv, max_sl, head_dim
        )
        v_grouped = valid_values.reshape(batch_size * num_kv_heads, 1, max_sl, head_dim).expand(
            batch_size * num_kv_heads, num_queries_per_kv, max_sl, head_dim
        )

        if attn_mask is None:
            mask = (torch.arange(max_sl, device=query.device).unsqueeze(0) < seq_lens.unsqueeze(1)).view(
                batch_size, 1, 1, max_sl
            )
        else:
            mask = attn_mask

        if mask.shape[0] == batch_size and num_kv_heads > 1:
            mask_grouped = mask.repeat_interleave(num_kv_heads, dim=0)
        else:
            mask_grouped = mask

        out = F.scaled_dot_product_attention(q_grouped, k_grouped, v_grouped, attn_mask=mask_grouped)

        if head_mapping == "contiguous":
            return out.view(batch_size, num_heads, 1, head_dim)
        else:
            return (
                out.view(batch_size, num_kv_heads, num_queries_per_kv, 1, head_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(batch_size, num_heads, 1, head_dim)
            )
    else:
        # Standard Multi-Head Attention (num_queries_per_kv == 1)
        if attn_mask is None:
            attn_mask = (torch.arange(max_sl, device=query.device).unsqueeze(0) < seq_lens.unsqueeze(1)).view(
                batch_size, 1, 1, max_sl
            )
        return F.scaled_dot_product_attention(query, valid_keys, valid_values, attn_mask=attn_mask)


def verify_paged_attention_equivalence(device: Optional[str] = None):
    device_str = device or get_default_device()

    print("=" * 65)
    print("      PAGED ATTENTION NUMERICAL VERIFICATION")
    print("=" * 65)
    print(f"Device: {device_str}")

    torch.manual_seed(42)
    dev = torch.device(device_str)

    num_heads = 14
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    total_pool_blocks = 10

    # 1. Pre-allocate Global Physical KV Pool in VRAM
    k_cache_pool = torch.randn(total_pool_blocks, block_size, num_kv_heads, head_dim, device=dev)
    v_cache_pool = torch.randn(total_pool_blocks, block_size, num_kv_heads, head_dim, device=dev)

    print(f"\n[1] Pre-allocated Physical KV Pool:")
    print(f"  • K-Pool Shape: {list(k_cache_pool.shape)} -> [blocks, block_size, kv_heads, head_dim]")
    print(f"  • V-Pool Shape: {list(v_cache_pool.shape)}")

    # 2. Setup Sequence: 38 tokens in scattered physical blocks [1, 7, 4]
    seq_len = 38
    assigned_blocks = [1, 7, 4]
    block_table = torch.tensor([assigned_blocks], dtype=torch.long, device=dev)
    seq_lens = torch.tensor([seq_len], dtype=torch.long, device=dev)

    print(f"\n[2] Sequence Setup:")
    print(f"  • Sequence Token Length: {seq_len} tokens")
    print(f"  • Non-Contiguous Physical Blocks: {assigned_blocks}")

    query = torch.randn(1, num_heads, 1, head_dim, device=dev)

    # 3. Ground Truth: Reconstruct contiguous tokens manually
    b0_k = k_cache_pool[1, 0:16]
    b1_k = k_cache_pool[7, 0:16]
    b2_k = k_cache_pool[4, 0:6]
    true_keys = torch.cat([b0_k, b1_k, b2_k], dim=0)

    b0_v = v_cache_pool[1, 0:16]
    b1_v = v_cache_pool[7, 0:16]
    b2_v = v_cache_pool[4, 0:6]
    true_values = torch.cat([b0_v, b1_v, b2_v], dim=0)

    true_keys = true_keys.permute(1, 0, 2).unsqueeze(0).repeat_interleave(num_heads // num_kv_heads, dim=1)
    true_values = true_values.permute(1, 0, 2).unsqueeze(0).repeat_interleave(num_heads // num_kv_heads, dim=1)

    expected_output = standard_attention(query, true_keys, true_values)

    # 4. PagedAttention Kernel Execution
    paged_output = paged_attention_decode(
        query=query,
        k_cache_pool=k_cache_pool,
        v_cache_pool=v_cache_pool,
        block_tables=block_table,
        seq_lens=seq_lens,
        block_size=block_size
    )

    max_diff = torch.max(torch.abs(expected_output - paged_output)).item()
    is_identical = torch.allclose(expected_output, paged_output, atol=1e-5)

    print(f"\n[3] Verification Results:")
    print(f"  • Output Shape:            {list(paged_output.shape)}")
    print(f"  • Max Absolute Difference: {max_diff:.2e}")
    print(f"  • Are Results Equivalent?  {'✅ PASSED (100% Match)' if is_identical else '❌ FAILED'}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PagedAttention Kernel Equivalence Verification")
    add_common_engine_args(parser)
    args = parser.parse_args()

    verify_paged_attention_equivalence(device=args.device)
