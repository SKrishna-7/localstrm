"""
Milestone 2: Batched Model Runner
----------------------------------
Replaces sequential per-request model forward calls with batched QKV projections,
independent per-sequence RoPE, and PagedAttention decode over non-contiguous block tables.

Key Architectural Capabilities:
  1. Batched QKV Projections: All active requests in running_batch are projected
     simultaneously in a single batched GEMM, reading model weights from VRAM once.
  2. Ragged / Variable Sequence Lengths: Evaluates sequences with different context lengths
     without padding or wasted bubble slots.
  3. Independent Block Tables: Gathers non-contiguous KV blocks per sequence directly
     from the global pre-allocated Physical KV Cache Pool.
  4. Precision Support: Operates natively with FP16 and INT8 quantized KV cache pools.
"""

import math
from typing import Optional, List, Tuple, Any
import torch
import torch.nn.functional as F

from src.kernels.paged_attention import paged_attention_decode


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to query and key tensors.

    Args:
        q: [batch_size, num_heads, 1, head_dim]
        k: [batch_size, num_kv_heads, 1, head_dim]
        cos: [batch_size, 1, head_dim]
        sin: [batch_size, 1, head_dim]
    """
    cos = cos.unsqueeze(1)  # [batch_size, 1, 1, head_dim]
    sin = sin.unsqueeze(1)  # [batch_size, 1, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BatchedModelRunner:
    """
    Batched Model Runner executing true multi-request parallel decoding
    coupled with PagedAttention and PhysicalBlockPool.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        block_size: int = 16,
        kv_cache_dtype: str = "fp16",
        batched_decode: bool = True,
    ):
        self.model = model
        self.device = device
        self.block_size = block_size
        self.kv_cache_dtype = kv_cache_dtype.lower()
        self.batched_decode = batched_decode

        cfg = model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.hidden_size = cfg.hidden_size
        self.head_dim = getattr(cfg, "head_dim", self.hidden_size // self.num_heads)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    @torch.no_grad()
    def batched_decode_step(
        self,
        batched_tokens: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_pool: Any
    ) -> torch.Tensor:
        """
        Executes a single decode step for all active sequences in parallel.

        Args:
            batched_tokens: [B, 1] input token IDs (last generated token of each request)
            block_tables:   [B, max_blocks] physical block IDs per sequence
            seq_lens:       [B] total sequence length for each request (including the new token)
            kv_pool:        PhysicalBlockPool instance holding global K and V blocks

        Returns:
            batched_logits: [B, vocab_size] next-token prediction logits
        """
        batch_size = batched_tokens.shape[0]

        # By default (batched_decode=True), all sequences are decoded in parallel
        # via a single batched GEMM and PagedAttention SDPA call.
        # If batched_decode=False, decode each row sequentially to eliminate
        # cross-row GEMM reduction order variance for strict contract testing.
        if batch_size > 1 and not self.batched_decode:
            return torch.cat(
                [
                    self.batched_decode_step(
                        batched_tokens[row : row + 1],
                        block_tables[row : row + 1],
                        seq_lens[row : row + 1],
                        kv_pool,
                    )
                    for row in range(batch_size)
                ],
                dim=0,
            )

        # 1. Batched Token Embeddings: [B, 1] -> [B, 1, hidden_size]
        hidden_states = self.model.model.embed_tokens(batched_tokens)

        # 2. Per-Sequence Position IDs & Rotary Embeddings (RoPE)
        # Sequence b has position = seq_lens[b] - 1
        pos_ids = (seq_lens - 1).unsqueeze(1)  # [B, 1]
        position_embeddings = self.model.model.rotary_emb(hidden_states, pos_ids)
        cos, sin = position_embeddings

        # Precompute block indexing & attention mask ONCE for all layers
        pos = seq_lens - 1
        block_indices = pos // self.block_size
        offsets = pos % self.block_size
        blk_ids = block_tables[torch.arange(batch_size, device=self.device), block_indices]

        max_sl = int(seq_lens.max().item())
        max_blocks = (max_sl + self.block_size - 1) // self.block_size
        sub_tables = block_tables[:, :max_blocks]
        attn_mask = (torch.arange(max_sl, device=self.device).unsqueeze(0) < seq_lens.unsqueeze(1)).view(
            batch_size, 1, 1, max_sl
        )
        if self.num_kv_heads > 1 and self.num_queries_per_kv > 1:
            attn_mask = attn_mask.repeat_interleave(self.num_kv_heads, dim=0)

        # 3. Transformer Layer Loop
        for l_idx in range(self.num_layers):
            layer = self.model.model.layers[l_idx]
            residual = hidden_states

            # Input RMSNorm
            normed_states = layer.input_layernorm(hidden_states)

            # Batched Q, K, V Projections: single GEMM reads layer weights once
            q = layer.self_attn.q_proj(normed_states).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(normed_states).view(batch_size, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(normed_states).view(batch_size, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Write newly computed Key & Value vectors directly into Physical Block Pool
            k_vecs = k[:, :, 0, :]  # [batch_size, num_kv_heads, head_dim]
            v_vecs = v[:, :, 0, :]  # [batch_size, num_kv_heads, head_dim]

            if self.kv_cache_dtype == "int8":
                scale_k = torch.clamp(k_vecs.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
                scale_v = torch.clamp(v_vecs.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
                k_int8 = torch.clamp(torch.round(k_vecs / scale_k), -128, 127).to(torch.int8)
                v_int8 = torch.clamp(torch.round(v_vecs / scale_v), -128, 127).to(torch.int8)

                kv_pool.k_pool[l_idx, blk_ids, offsets] = k_int8
                kv_pool.v_pool[l_idx, blk_ids, offsets] = v_int8
                kv_pool.k_scales[l_idx, blk_ids, offsets] = scale_k
                kv_pool.v_scales[l_idx, blk_ids, offsets] = scale_v
            else:
                kv_pool.k_pool[l_idx, blk_ids, offsets] = k_vecs
                kv_pool.v_pool[l_idx, blk_ids, offsets] = v_vecs

            # PagedAttention over independent block tables and variable sequence lengths
            k_scales_layer = kv_pool.k_scales[l_idx] if self.kv_cache_dtype == "int8" else None
            v_scales_layer = kv_pool.v_scales[l_idx] if self.kv_cache_dtype == "int8" else None

            attn_output = paged_attention_decode(
                query=q,
                k_cache_pool=kv_pool.k_pool[l_idx],
                v_cache_pool=kv_pool.v_pool[l_idx],
                block_tables=block_tables,
                seq_lens=seq_lens,
                block_size=self.block_size,
                k_scales=k_scales_layer,
                v_scales=v_scales_layer,
                sub_tables=sub_tables,
                max_sl=max_sl,
                attn_mask=attn_mask
            )  # [B, num_heads, 1, head_dim]

            # Output Projection & Residual
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_size)
            attn_output = layer.self_attn.o_proj(attn_output)
            hidden_states = residual + attn_output

            # Post-Attention Layernorm & Batched MLP (Gate, Up, Down Projections)
            residual = hidden_states
            normed_mlp = layer.post_attention_layernorm(hidden_states)
            mlp_output = layer.mlp(normed_mlp)
            hidden_states = residual + mlp_output

        # 4. Final RMSNorm & LM Head Projection
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states[:, -1, :])  # [B, vocab_size]

        return logits
