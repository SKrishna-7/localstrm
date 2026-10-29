"""
Unified Mini-vLLM Engine
------------------------
A complete, self-contained inference engine that fuses:
  1. Real Qwen2.5-0.5B model weights.
  2. A pre-allocated global Physical KV Cache pool across all 24 layers.
  3. The Paged Block Manager (Virtual Memory addressing).
  4. The PagedAttention Decode Kernel.
  5. Continuous (Iteration-Level) Batching Scheduler.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Tuple, Any, Union
import math
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

# Support running directly or as a package module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig, get_default_model, get_default_device, add_common_engine_args, get_logger
from src.kernels.quantization import apply_optional_quantization, load_and_quantize_model
from src.engine.sampling import SamplingParams, sample_next_token, check_stop_criteria
from src.engine.model_runner import BatchedModelRunner
from src.memory.prefix_cache import PrefixCacheManager, compute_block_hash, clone_hf_cache
from src.observability.metrics import EngineMetricsTracker

logger = get_logger("mini_vllm.engine")


class RequestStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"



def _sample_paired_timestamp(device: Optional[torch.device] = None) -> Tuple[float, int]:
    """Synchronize the target CUDA device, then capture a paired timestamp."""
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    t_ns = time.perf_counter_ns()
    return t_ns / 1_000_000_000.0, t_ns


@dataclass
class EngineRequest:
    """Represents a request inside the unified engine."""
    req_id: int
    prompt: str
    max_tokens: int
    arrival_step: int
    device: Optional[torch.device] = None

    status: RequestStatus = RequestStatus.WAITING
    prompt_token_ids: List[int] = field(default_factory=list)
    output_token_ids: List[int] = field(default_factory=list)

    # Block Table: list of physical block IDs assigned to this sequence
    physical_block_ids: List[int] = field(default_factory=list)

    start_step: Optional[int] = None
    finish_step: Optional[int] = None
    kv_cache: Optional[object] = None
    num_prefilled_tokens: int = 0
    prefix_hit_tokens: int = 0
    sampling_params: Optional[SamplingParams] = None

    arrival_time: float = -1.0
    schedule_time: Optional[float] = None
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    ttft_ms: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    arrival_time_ns: Optional[int] = None
    schedule_time_ns: Optional[int] = None
    prefill_start_time_ns: Optional[int] = None
    prefill_end_time_ns: Optional[int] = None
    first_token_time_ns: Optional[int] = None
    finish_time_ns: Optional[int] = None

    stop_reason: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.arrival_time_ns is None:
            if self.arrival_time < 0.0:
                t_s, t_ns = _sample_paired_timestamp(self.device)
                self.arrival_time = t_s
                self.arrival_time_ns = t_ns
            else:
                self.arrival_time_ns = int(round(self.arrival_time * 1_000_000_000))
        elif self.arrival_time < 0.0:
            self.arrival_time = self.arrival_time_ns / 1_000_000_000.0

    @property
    def is_prefill_done(self) -> bool:
        return self.num_prefilled_tokens >= len(self.prompt_token_ids)

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)


class PhysicalBlockPool:
    """
    Global pre-allocated Physical KV Cache Pool in VRAM/RAM.
    Allocated ONCE at engine startup. Zero dynamic tensor allocation.
    Supports standard FP16 and on-demand INT8 compression (50% VRAM reduction).
    """
    def __init__(
        self,
        num_layers: int,
        total_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        kv_cache_dtype: str = "fp16"
    ):
        if total_blocks <= 0:
            raise ValueError("total_blocks must be greater than 0")
        if block_size <= 0:
            raise ValueError("block_size must be greater than 0")

        self.num_layers = num_layers
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.kv_cache_dtype = kv_cache_dtype.lower()

        # If INT8 KV Cache is active: store 8-bit ints + per-token FP16 dynamic scale factors
        if self.kv_cache_dtype == "int8":
            self.k_pool = torch.zeros(
                (num_layers, total_blocks, block_size, num_kv_heads, head_dim),
                dtype=torch.int8,
                device=device
            )
            self.v_pool = torch.zeros(
                (num_layers, total_blocks, block_size, num_kv_heads, head_dim),
                dtype=torch.int8,
                device=device
            )
            self.k_scales = torch.ones(
                (num_layers, total_blocks, block_size, num_kv_heads, 1),
                dtype=torch.float16,
                device=device
            )
            self.v_scales = torch.ones(
                (num_layers, total_blocks, block_size, num_kv_heads, 1),
                dtype=torch.float16,
                device=device
            )
        else:
            self.k_pool = torch.zeros(
                (num_layers, total_blocks, block_size, num_kv_heads, head_dim),
                dtype=dtype,
                device=device
            )
            self.v_pool = torch.zeros(
                (num_layers, total_blocks, block_size, num_kv_heads, head_dim),
                dtype=dtype,
                device=device
            )
            self.k_scales = None
            self.v_scales = None

        # Free list of block IDs
        self.free_blocks: List[int] = list(range(total_blocks))

    def write_kv(
        self,
        layer_idx: int,
        blk_idx: int,
        blk_offset: int,
        k_vec: torch.Tensor,
        v_vec: torch.Tensor
    ):
        """Writes K and V vectors into the physical pool (handling dynamic INT8 quantization if enabled)."""
        if self.kv_cache_dtype == "int8":
            # Per-token symmetric quantization
            scale_k = torch.clamp(k_vec.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
            scale_v = torch.clamp(v_vec.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)

            quant_k = torch.clamp((k_vec / scale_k).round(), -128, 127).to(torch.int8)
            quant_v = torch.clamp((v_vec / scale_v).round(), -128, 127).to(torch.int8)

            self.k_pool[layer_idx, blk_idx, blk_offset] = quant_k
            self.v_pool[layer_idx, blk_idx, blk_offset] = quant_v
            self.k_scales[layer_idx, blk_idx, blk_offset] = scale_k
            self.v_scales[layer_idx, blk_idx, blk_offset] = scale_v
        else:
            self.k_pool[layer_idx, blk_idx, blk_offset] = k_vec
            self.v_pool[layer_idx, blk_idx, blk_offset] = v_vec

    def write_kv_all_layers(
        self,
        blk_idx: int,
        blk_offset: int,
        k_all: torch.Tensor,
        v_all: torch.Tensor
    ):
        """Vectorized write of all layers in one operation to minimize Python interpreter overhead."""
        if self.kv_cache_dtype == "int8":
            scale_k = torch.clamp(k_all.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
            scale_v = torch.clamp(v_all.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
            quant_k = torch.clamp((k_all / scale_k).round(), -128, 127).to(torch.int8)
            quant_v = torch.clamp((v_all / scale_v).round(), -128, 127).to(torch.int8)
            self.k_pool[:, blk_idx, blk_offset] = quant_k
            self.v_pool[:, blk_idx, blk_offset] = quant_v
            self.k_scales[:, blk_idx, blk_offset] = scale_k
            self.v_scales[:, blk_idx, blk_offset] = scale_v
        else:
            self.k_pool[:, blk_idx, blk_offset] = k_all
            self.v_pool[:, blk_idx, blk_offset] = v_all

    def write_kv_chunk(
        self,
        block_ids: List[int],
        start_token_idx: int,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor
    ):
        """
        Vectorized block-level write of a prefill chunk across all layers.
        k_chunk: [num_layers, chunk_len, num_kv, head_dim]
        v_chunk: [num_layers, chunk_len, num_kv, head_dim]
        """
        chunk_len = k_chunk.shape[1]
        offset = 0
        while offset < chunk_len:
            global_pos = start_token_idx + offset
            blk_id = block_ids[global_pos // self.block_size]
            blk_off = global_pos % self.block_size
            tokens_in_blk = min(chunk_len - offset, self.block_size - blk_off)

            sub_k = k_chunk[:, offset : offset + tokens_in_blk]
            sub_v = v_chunk[:, offset : offset + tokens_in_blk]

            if self.kv_cache_dtype == "int8":
                scale_k = torch.clamp(sub_k.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
                scale_v = torch.clamp(sub_v.abs().max(dim=-1, keepdim=True)[0] / 127.0, min=1e-8).to(torch.float16)
                quant_k = torch.clamp((sub_k / scale_k).round(), -128, 127).to(torch.int8)
                quant_v = torch.clamp((sub_v / scale_v).round(), -128, 127).to(torch.int8)

                self.k_pool[:, blk_id, blk_off : blk_off + tokens_in_blk] = quant_k
                self.v_pool[:, blk_id, blk_off : blk_off + tokens_in_blk] = quant_v
                self.k_scales[:, blk_id, blk_off : blk_off + tokens_in_blk] = scale_k
                self.v_scales[:, blk_id, blk_off : blk_off + tokens_in_blk] = scale_v
            else:
                self.k_pool[:, blk_id, blk_off : blk_off + tokens_in_blk] = sub_k
                self.v_pool[:, blk_id, blk_off : blk_off + tokens_in_blk] = sub_v

            offset += tokens_in_blk

    def read_kv(
        self,
        layer_idx: int,
        phys_blocks: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reads and dequantizes physical blocks directly for PagedAttention."""
        k_pool_layer = self.k_pool[layer_idx]
        v_pool_layer = self.v_pool[layer_idx]

        if self.kv_cache_dtype == "int8":
            k_scales_layer = self.k_scales[layer_idx]
            v_scales_layer = self.v_scales[layer_idx]
            phys_k = k_pool_layer[phys_blocks].to(torch.float16) * k_scales_layer[phys_blocks]
            phys_v = v_pool_layer[phys_blocks].to(torch.float16) * v_scales_layer[phys_blocks]
        else:
            phys_k = k_pool_layer[phys_blocks]
            phys_v = v_pool_layer[phys_blocks]

        return phys_k, phys_v

    def copy_block(self, src_block_id: int, dst_block_id: int):
        """Duplicates src_block into dst_block for Copy-on-Write prefix cache branching."""
        self.k_pool[:, dst_block_id] = self.k_pool[:, src_block_id].clone()
        self.v_pool[:, dst_block_id] = self.v_pool[:, src_block_id].clone()
        if self.kv_cache_dtype == "int8":
            self.k_scales[:, dst_block_id] = self.k_scales[:, src_block_id].clone()
            self.v_scales[:, dst_block_id] = self.v_scales[:, src_block_id].clone()

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("Out of KV Cache physical blocks!")
        return self.free_blocks.pop(0)

    def free_block(self, block_id: int):
        if not isinstance(block_id, int) or not 0 <= block_id < self.total_blocks:
            raise ValueError(f"invalid physical block id: {block_id}")
        if block_id in self.free_blocks:
            raise ValueError(f"physical block {block_id} is already free")
        self.free_blocks.append(block_id)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


class MiniVLLMEngine:
    """
    Unified Inference Engine coordinating scheduling, memory paging,
    and model execution.
    """
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        model_name: Optional[str] = None,
        max_batch_size: int = 4,
        total_blocks: Optional[int] = None,
        vram_budget: Optional[Union[int, float, str]] = None,
        block_size: int = 16,
        device: Optional[str] = None,
        quantization: str = "none",
        chunked_prefill: bool = True,
        prefill_chunk_size: int = 32,
        kv_cache_dtype: str = "fp16",
        enable_prefix_caching: bool = True,
        batched_decode: Optional[bool] = None,
        enable_batched_sampling: bool = True,
        verbose: bool = False,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
        trust_remote_code: bool = False,
    ):
        if total_blocks is not None and vram_budget is not None:
            raise ValueError(
                f"Conflicting KV-cache configuration: both total_blocks ({total_blocks}) "
                f"and vram_budget ({vram_budget!r}) were provided. "
                "Specify total_blocks OR vram_budget, not both."
            )

        if config is None:
            resolved_model = model_name or get_default_model()
            resolved_device = device or get_default_device()

            vram_budget_bytes = None
            if vram_budget is not None:
                from src.memory.preflight import parse_vram_budget_bytes
                vram_budget_bytes = parse_vram_budget_bytes(vram_budget)

            self.config = EngineConfig(
                model=ModelConfig.from_str(resolved_model, device=resolved_device),
                cache=CacheConfig(
                    block_size=block_size,
                    total_blocks=total_blocks,
                    vram_budget_bytes=vram_budget_bytes,
                    kv_cache_dtype=kv_cache_dtype,
                ),
                scheduler=SchedulerConfig(
                    max_batch_size=max_batch_size,
                    enable_chunked_prefill=chunked_prefill,
                    prefill_chunk_size=prefill_chunk_size,
                    enable_prefix_caching=enable_prefix_caching,
                    batched_decode=batched_decode if batched_decode is not None else True,
                ),
                device=resolved_device,
                quantization=quantization,
                trust_remote_code=trust_remote_code,
            )
        else:
            self.config = config
            if trust_remote_code:
                self.config.trust_remote_code = True
            if batched_decode is not None:
                self.config.scheduler.batched_decode = batched_decode
            if total_blocks is not None and self.config.cache.vram_budget_bytes is not None:
                raise ValueError(
                    f"Conflicting KV-cache configuration: explicit total_blocks ({total_blocks}) "
                    f"conflicts with config.cache.vram_budget_bytes ({self.config.cache.vram_budget_bytes})."
                )
            if vram_budget is not None and self.config.cache.total_blocks is not None:
                raise ValueError(
                    f"Conflicting KV-cache configuration: explicit vram_budget ({vram_budget!r}) "
                    f"conflicts with config.cache.total_blocks ({self.config.cache.total_blocks})."
                )
            if vram_budget is not None:
                from src.memory.preflight import parse_vram_budget_bytes
                self.config.cache.vram_budget_bytes = parse_vram_budget_bytes(vram_budget)
            elif total_blocks is not None:
                self.config.cache.total_blocks = total_blocks

        if self.config.scheduler.max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")
        if self.config.cache.block_size <= 0:
            raise ValueError("block_size must be greater than 0")

        self.verbose = verbose
        self.device = torch.device(self.config.device)
        self.max_batch_size = self.config.scheduler.max_batch_size
        self.block_size = self.config.cache.block_size
        self.chunked_prefill = getattr(self.config.scheduler, "enable_chunked_prefill", True)
        self.prefill_chunk_size = getattr(self.config.scheduler, "prefill_chunk_size", 32)
        self.enable_prefix_caching = getattr(self.config.scheduler, "enable_prefix_caching", True)
        self.batched_decode = getattr(self.config.scheduler, "batched_decode", True)
        self.enable_batched_sampling = enable_batched_sampling
        self.kv_cache_dtype = getattr(self.config.cache, "kv_cache_dtype", "fp16")
        target_model = self.config.model.model_name_or_path

        # Resolve VRAM budget before loading model weights
        self.budget_resolution: Optional[Any] = None
        if self.config.cache.vram_budget_bytes is not None:
            from src.memory.preflight import estimate_vram_budget, ModelMetadata
            meta = None
            if model is not None and hasattr(model, "config"):
                cfg = model.config
                meta = ModelMetadata(
                    model_id=target_model,
                    is_local=True,
                    num_layers=getattr(cfg, "num_hidden_layers", None),
                    num_heads=getattr(cfg, "num_attention_heads", None),
                    num_kv_heads=getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", None)),
                    head_dim=getattr(cfg, "head_dim", 64),
                    hidden_size=getattr(cfg, "hidden_size", None),
                    vocab_size=getattr(cfg, "vocab_size", None),
                )
            budget_est = estimate_vram_budget(
                model_id=target_model,
                vram_budget=self.config.cache.vram_budget_bytes,
                device=str(self.device),
                quantization=self.config.quantization,
                block_size=self.config.cache.block_size,
                kv_cache_dtype=self.config.cache.kv_cache_dtype,
                chunked_prefill=self.chunked_prefill,
                metadata=meta,
                raise_on_error=True,
                trust_remote_code=getattr(self.config, "trust_remote_code", False),
            )
            self.budget_resolution = budget_est
            self.total_blocks = budget_est.total_blocks
            self.config.cache.total_blocks = budget_est.total_blocks
        else:
            self.total_blocks = self.config.cache.total_blocks

        if self.total_blocks is None or self.total_blocks <= 0:
            raise ValueError(f"total_blocks must be positive, got {self.total_blocks}")

        if self.verbose:
            logger.info("Initializing MiniVLLMEngine on %s (model: %s, quant: %s, kv_cache: %s, prefix_cache=%s, chunked_prefill=%s [%d tok/chunk], total_blocks=%d)",
                        self.device, target_model, self.config.quantization, self.kv_cache_dtype.upper(), self.enable_prefix_caching, self.chunked_prefill, self.prefill_chunk_size, self.total_blocks)
            if self.budget_resolution is not None:
                logger.info("VRAM Budget Resolution: %s", self.budget_resolution.summary())

        trust_remote = getattr(self.config, "trust_remote_code", False)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=trust_remote)

        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        if model is not None:
            self.model = model.to(self.device) if hasattr(model, "to") else model
            if hasattr(self.model, "eval"):
                self.model.eval()
        else:
            self.model, _, _ = load_and_quantize_model(
                target_model,
                device=self.device,
                quantization=self.config.quantization,
                dtype=dtype,
                logger=logger,
                trust_remote_code=trust_remote
            )

        model_cfg = self.model.config
        if getattr(model_cfg, "num_hidden_layers", None) is None or getattr(model_cfg, "num_attention_heads", None) is None:
            raise ValueError(
                f"Unsupported model architecture '{model_cfg.__class__.__name__}'. "
                f"MiniVLLMEngine requires decoder-only transformer models with standard "
                f"attention heads and layers (e.g. LLaMA, Qwen2, Mistral)."
            )
        self.num_layers = model_cfg.num_hidden_layers
        self.num_heads = model_cfg.num_attention_heads
        self.num_kv_heads = getattr(model_cfg, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(model_cfg, "head_dim", model_cfg.hidden_size // self.num_heads)

        # 1. Initialize Global Pre-allocated Physical KV Pool (with optional INT8 compression)
        self.kv_pool = PhysicalBlockPool(
            num_layers=self.num_layers,
            total_blocks=self.total_blocks,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=self.device,
            dtype=dtype,
            kv_cache_dtype=self.kv_cache_dtype
        )

        # 2. Initialize Prefix Cache Manager for content-addressed block sharing
        self.prefix_cache = PrefixCacheManager(
            total_blocks=self.total_blocks,
            block_size=self.block_size
        ) if self.enable_prefix_caching else None

        # 3. Observability & Prometheus Telemetry Tracker
        self.metrics = EngineMetricsTracker(model_name=target_model, enable_prometheus=True)

        # 4. Batched Model Runner (Milestone 2: Batched QKV projections + PagedAttention)
        self.model_runner = BatchedModelRunner(
            model=self.model,
            device=self.device,
            block_size=self.block_size,
            kv_cache_dtype=self.kv_cache_dtype,
            batched_decode=self.batched_decode,
        )

        # Queues & Tracking
        self.waiting_queue: List[EngineRequest] = []
        self.running_batch: List[EngineRequest] = []
        self.completed_requests: List[EngineRequest] = []
        self.requests_by_id: Dict[int, EngineRequest] = {}
        self.current_step: int = 0
        self._next_req_id: int = 1
        self._update_queue_and_cache_metrics()

    def add_request(
        self,
        prompt: Union[str, EngineRequest],
        max_tokens: Optional[int] = None,
        arrival_step: int = 0,
        sampling_params: Optional[SamplingParams] = None
    ) -> EngineRequest:
        if isinstance(prompt, EngineRequest):
            req = prompt
            if req.status in (RequestStatus.FINISHED, RequestStatus.FAILED):
                raise ValueError(f"Cannot re-admit terminal request #{req.req_id} (status={req.status})")
            if req in self.waiting_queue or req in self.running_batch:
                raise ValueError(f"Request #{req.req_id} is already queued or running")
            self.waiting_queue.append(req)
            self.requests_by_id[req.req_id] = req
            self._update_queue_and_cache_metrics()
            return req

        if max_tokens is None or max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        if sampling_params is None:
            sampling_params = SamplingParams.greedy()

        prompt_token_ids = self.tokenizer.encode(prompt)
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")

        req = EngineRequest(
            req_id=self._next_req_id,
            prompt=prompt,
            max_tokens=max_tokens,
            arrival_step=arrival_step,
            prompt_token_ids=prompt_token_ids,
            device=self.device,
            sampling_params=sampling_params
        )
        self._next_req_id += 1
        self.waiting_queue.append(req)
        self.requests_by_id[req.req_id] = req
        self._update_queue_and_cache_metrics()
        return req

    def get_request(self, req_id: int) -> Optional[EngineRequest]:
        """Retrieve any request tracked by the engine by its unique request ID."""
        return self.requests_by_id.get(req_id)

    @property
    def failed_requests(self) -> List[EngineRequest]:
        """List of all requests that reached RequestStatus.FAILED."""
        return [r for r in self.requests_by_id.values() if r.status == RequestStatus.FAILED]

    def abort_request(self, req_id: int) -> bool:
        """Explicitly cancels an active or waiting request by ID, releasing its blocks."""
        # Check waiting queue
        for req in list(self.waiting_queue):
            if req.req_id == req_id:
                self.waiting_queue.remove(req)
                req.status = RequestStatus.FAILED
                req.stop_reason = "abort"
                req.finish_step = self.current_step
                req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
                if req not in self.completed_requests:
                    self.completed_requests.append(req)
                self._update_queue_and_cache_metrics()
                return True

        # Check running batch
        for req in list(self.running_batch):
            if req.req_id == req_id:
                self.running_batch.remove(req)
                req.status = RequestStatus.FAILED
                req.stop_reason = "abort"
                req.finish_step = self.current_step
                req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
                block_ids = list(req.physical_block_ids)
                req.physical_block_ids.clear()
                req.kv_cache = None
                for b_id in block_ids:
                    try:
                        self._free_block(b_id)
                    except Exception:
                        logger.exception("Failed to release block %s on abort", b_id)
                if req not in self.completed_requests:
                    self.completed_requests.append(req)
                self.metrics.record_request_completed(
                    prompt_tokens=len(req.prompt_token_ids),
                    generated_tokens=req.num_generated,
                    e2e_latency_ms=(time.perf_counter() - req.arrival_time) * 1000.0
                )
                self._update_queue_and_cache_metrics()
                return True

        return False

    @property
    def num_free_blocks(self) -> int:
        if self.enable_prefix_caching and self.prefix_cache:
            return self.prefix_cache.num_free
        return self.kv_pool.num_free

    def _allocate_block(self) -> int:
        if self.enable_prefix_caching and self.prefix_cache:
            return self.prefix_cache.allocate_new_block()
        return self.kv_pool.allocate_block()

    def _free_block(self, block_id: int):
        if self.enable_prefix_caching and self.prefix_cache:
            self.prefix_cache.free_block(block_id)
        else:
            self.kv_pool.free_block(block_id)

    def _update_queue_and_cache_metrics(self) -> None:
        """Synchronizes engine metrics tracker with current queue depth and KV-cache pool state."""
        if hasattr(self, "metrics") and self.metrics is not None:
            allocated_blocks = max(0, self.total_blocks - self.num_free_blocks)
            self.metrics.record_cache_usage(allocated_blocks=allocated_blocks, total_blocks=self.total_blocks)
            self.metrics.record_queue_state(running=len(self.running_batch), waiting=len(self.waiting_queue))

    def _schedule(self, current_step: int):
        """Scheduler: Admits waiting requests into running slots using Prefix Caching and Paged memory."""
        eligible = [r for r in self.waiting_queue if r.arrival_step <= current_step]

        while eligible and len(self.running_batch) < self.max_batch_size:
            req = eligible[0]
            if req.status in (RequestStatus.FINISHED, RequestStatus.FAILED):
                eligible.pop(0)
                if req in self.waiting_queue:
                    self.waiting_queue.remove(req)
                continue

            num_tokens = len(req.prompt_token_ids)
            cached_blocks = []
            prefix_hit_tokens = 0
            last_hit_hash = None

            # 1. Automatic Prefix Caching Check (Content-addressed block reuse)
            if self.enable_prefix_caching and self.prefix_cache:
                num_full_blocks = num_tokens // self.block_size
                parent_hash = None
                for b_i in range(num_full_blocks):
                    chunk = req.prompt_token_ids[b_i * self.block_size : (b_i + 1) * self.block_size]
                    b_hash = compute_block_hash(parent_hash, tuple(chunk))
                    if b_hash in self.prefix_cache.hash_to_block_id:
                        hit_b_id = self.prefix_cache.hash_to_block_id[b_hash]
                        self.prefix_cache.blocks[hit_b_id].ref_count += 1
                        if hit_b_id in self.prefix_cache.lru_eviction_queue:
                            self.prefix_cache.lru_eviction_queue.remove(hit_b_id)
                        cached_blocks.append(hit_b_id)
                        prefix_hit_tokens += self.block_size
                        parent_hash = b_hash
                        last_hit_hash = b_hash
                    else:
                        break  # Prefix continuity broken

            remaining_tokens = num_tokens - prefix_hit_tokens
            needed_new_blocks = (remaining_tokens + self.block_size - 1) // self.block_size if remaining_tokens > 0 else 0

            # Calculate total blocks needed to guarantee execution for this request plus active running requests
            req_full_blocks = (num_tokens + req.max_tokens + self.block_size - 1) // self.block_size - len(cached_blocks)
            running_reserved_blocks = sum(
                max(0, (len(r.prompt_token_ids) + r.max_tokens + self.block_size - 1) // self.block_size - len(r.physical_block_ids))
                for r in self.running_batch
            )

            if req_full_blocks > self.total_blocks:
                # A request that needs more new blocks than the entire pool can
                # provide can never become schedulable; fail rather than spin.
                self.waiting_queue.remove(req)
                for b_id in cached_blocks:
                    self._free_block(b_id)
                req.status = RequestStatus.FAILED
                req.stop_reason = "kv_cache_full"
                req.error = (
                    f"request #{req.req_id} requires {req_full_blocks} KV blocks, "
                    f"but the pool has only {self.total_blocks}"
                )
                req.finish_step = current_step
                req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
                self._update_queue_and_cache_metrics()
                err = MemoryError(req.error)
                err.request = req
                raise err

            # Admission headroom and two-tier backpressure policy:
            # 1. Conservative capacity regime (req_full_blocks + running_reserved_blocks <= total_blocks):
            #    Worst-case capacity (prompt + max_tokens) for this candidate request plus all
            #    active running sequences fits entirely within the total pool capacity. Under this
            #    regime, completion without mid-decode block exhaustion is guaranteed (assuming
            #    no concurrent external allocation).
            # 2. Oversubscribed regime (req_full_blocks + running_reserved_blocks > total_blocks):
            #    The collective worst-case capacity exceeds the total physical pool. To maintain
            #    practical concurrency without premature starvation, admission is permitted if
            #    there are enough free blocks for the new prefill plus a dynamic decode cushion
            #    (at least 1 free block per currently running decode sequence).
            #    NOTE: In this oversubscribed regime, requests are NOT guaranteed to complete;
            #    if multiple active sequences generate to full length simultaneously, mid-generation
            #    KV-pool exhaustion can occur, which will isolate and fail the crossing request
            #    (stop_reason="kv_cache_full") while allowing peers to proceed.
            if req_full_blocks + running_reserved_blocks <= self.total_blocks:
                min_required_blocks = req_full_blocks + running_reserved_blocks
            else:
                decode_cushion = max(1, len(self.running_batch)) if self.total_blocks > 1 else 0
                min_required_blocks = needed_new_blocks + decode_cushion

            # Check if pool has enough free blocks
            if self.num_free_blocks < min_required_blocks:
                # Rollback ref_counts on cached blocks
                for b_id in cached_blocks:
                    self._free_block(b_id)
                break  # Memory full: wait for running sequences to finish

            eligible.pop(0)
            self.waiting_queue.remove(req)

            # Allocate physical blocks for remaining prompt tokens
            new_blocks = []
            try:
                for _ in range(needed_new_blocks):
                    b_id = self._allocate_block()
                    new_blocks.append(b_id)
            except Exception:
                for b_id in new_blocks:
                    try:
                        self._free_block(b_id)
                    except Exception:
                        logger.exception("Failed to roll back block %s after admission failure", b_id)
                for b_id in cached_blocks:
                    try:
                        self._free_block(b_id)
                    except Exception:
                        logger.exception("Failed to roll back cached block %s after admission failure", b_id)
                self.waiting_queue.insert(0, req)
                raise

            req.physical_block_ids = cached_blocks + new_blocks
            req.num_prefilled_tokens = prefix_hit_tokens
            req.prefix_hit_tokens = prefix_hit_tokens
            req.status = RequestStatus.RUNNING
            req.start_step = current_step
            if req.schedule_time is None:
                req.schedule_time, req.schedule_time_ns = _sample_paired_timestamp(self.device)

            # If we had a prefix hit, restore the cached KV state so HuggingFace starts at offset!
            if prefix_hit_tokens > 0 and self.prefix_cache and last_hit_hash in self.prefix_cache.hash_to_kv_cache:
                req.kv_cache = clone_hf_cache(self.prefix_cache.hash_to_kv_cache[last_hit_hash], prefix_hit_tokens)

            completed_from_cache = False
            # If 100% of the prompt was cached (prefix_hit_tokens == num_tokens):
            if req.is_prefill_done:
                if req.prefill_start_time is None:
                    req.prefill_start_time = req.schedule_time
                    req.prefill_start_time_ns = req.schedule_time_ns
                if req.prefill_end_time is None:
                    req.prefill_end_time = req.schedule_time
                    req.prefill_end_time_ns = req.schedule_time_ns
                if self.prefix_cache and last_hit_hash in self.prefix_cache.hash_to_logits:
                    cached_logits = self.prefix_cache.hash_to_logits[last_hit_hash]
                    first_token = sample_next_token(cached_logits, req.sampling_params, past_token_ids=req.output_token_ids)
                    req.output_token_ids.append(first_token)
                    now_t, now_t_ns = _sample_paired_timestamp(self.device)
                    req.first_token_time = now_t
                    req.first_token_time_ns = now_t_ns
                    ref_time = req.schedule_time if req.schedule_time is not None else req.arrival_time
                    req.ttft_ms = max(0.1, (now_t - ref_time) * 1000.0)
                    self.metrics.record_ttft(req.ttft_ms)
                    should_stop, reason = check_stop_criteria(
                        req.output_token_ids,
                        self.tokenizer,
                        req.sampling_params.stop_token_ids if req.sampling_params else [],
                        req.sampling_params.stop_strings if req.sampling_params else []
                    )
                    if should_stop:
                        req.stop_reason = reason
                    elif req.num_generated >= req.max_tokens:
                        req.stop_reason = "length"
                    if should_stop or req.num_generated >= req.max_tokens:
                        req.status = RequestStatus.FINISHED
                        req.finish_step = current_step
                        req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
                        self.metrics.record_request_completed(
                            prompt_tokens=len(req.prompt_token_ids),
                            generated_tokens=req.num_generated,
                            e2e_latency_ms=(req.finish_time - req.arrival_time) * 1000.0,
                        )
                        if req not in self.completed_requests:
                            self.completed_requests.append(req)
                        for b_id in req.physical_block_ids:
                            self._free_block(b_id)
                        req.physical_block_ids.clear()
                        req.kv_cache = None
                        completed_from_cache = True
                else:
                    # If logits not cached, force computing last token so first output token is generated
                    req.num_prefilled_tokens = max(0, num_tokens - 1)
            elif req.prefill_start_time is None:
                req.prefill_start_time, req.prefill_start_time_ns = _sample_paired_timestamp(self.device)

            if not completed_from_cache:
                self.running_batch.append(req)

            if self.verbose:
                if prefix_hit_tokens > 0:
                    logger.info("[Step %2d] ⚡ PREFIX CACHE HIT Req #%d: Reused %d tokens (%d blocks)! (0ms prefill)",
                                current_step, req.req_id, prefix_hit_tokens, len(cached_blocks))
                logger.debug("[Step %2d] + ADMITTED Req #%d (Blocks: %s, Prefilled: %d/%d)",
                             current_step, req.req_id, req.physical_block_ids, req.num_prefilled_tokens, num_tokens)

        # Update real-time queue and memory gauges
        self._update_queue_and_cache_metrics()

    def _paged_attention_layer(
        self,
        query: torch.Tensor,
        layer_idx: int,
        block_tables: List[List[int]],
        seq_lens: List[int]
    ) -> torch.Tensor:
        """
        Executes PagedAttention across non-contiguous physical blocks
        for a specific transformer layer.
        """
        batch_size = len(seq_lens)
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        scale = 1.0 / math.sqrt(self.head_dim)
        outputs = []

        k_pool_layer = self.kv_pool.k_pool[layer_idx]
        v_pool_layer = self.kv_pool.v_pool[layer_idx]

        for b in range(batch_size):
            seq_len = seq_lens[b]
            phys_blocks = block_tables[b]

            # Gather and dequantize physical blocks directly via kv_pool (FP16 or INT8)
            phys_k, phys_v = self.kv_pool.read_kv(layer_idx, phys_blocks)

            flat_k = phys_k.view(-1, self.num_kv_heads, self.head_dim)[:seq_len]
            flat_v = phys_v.view(-1, self.num_kv_heads, self.head_dim)[:seq_len]

            # [1, num_kv_heads, seq_len, head_dim]
            valid_k = flat_k.permute(1, 0, 2).unsqueeze(0)
            valid_v = flat_v.permute(1, 0, 2).unsqueeze(0)

            if num_queries_per_kv > 1:
                valid_k = valid_k.repeat_interleave(num_queries_per_kv, dim=1)
                valid_v = valid_v.repeat_interleave(num_queries_per_kv, dim=1)

            q_b = query[b : b + 1]  # [1, num_heads, 1, head_dim]
            scores = torch.matmul(q_b, valid_k.transpose(-1, -2)) * scale
            probs = F.softmax(scores, dim=-1)
            out_b = torch.matmul(probs, valid_v)
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)

    def _execute_step(self, current_step: int):
        """Executes 1 iteration step: chunked prefill for new sequences + batched decode for running sequences."""
        finished_this_step = []

        # ---------------------------------------------------------------------
        # 1. PREFILL PHASE: Chunked prompt processing for admitting sequences
        # ---------------------------------------------------------------------
        prefill_reqs = [r for r in self.running_batch if not r.is_prefill_done]
        for req in prefill_reqs:
            rem_tokens = len(req.prompt_token_ids) - req.num_prefilled_tokens
            if self.chunked_prefill and self.prefill_chunk_size > 0:
                chunk_len = min(self.prefill_chunk_size, rem_tokens)
            else:
                chunk_len = rem_tokens

            start_idx = req.num_prefilled_tokens
            chunk_tokens = req.prompt_token_ids[start_idx : start_idx + chunk_len]
            input_tensor = torch.tensor([chunk_tokens], device=self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_tensor,
                    past_key_values=req.kv_cache,
                    use_cache=True
                )

            req.kv_cache = outputs.past_key_values
            hf_cache = outputs.past_key_values

            # Store chunk KV vectors into our pre-allocated physical pool (handling INT8/FP16)
            if hasattr(hf_cache, "layers"):
                k_all_layers = torch.stack([hf_cache.layers[l].keys[0] for l in range(self.num_layers)])
                v_all_layers = torch.stack([hf_cache.layers[l].values[0] for l in range(self.num_layers)])
            else:
                k_all_layers = torch.stack([hf_cache[l][0][0] for l in range(self.num_layers)])
                v_all_layers = torch.stack([hf_cache[l][1][0] for l in range(self.num_layers)])

            k_chunk = k_all_layers[:, :, start_idx : start_idx + chunk_len, :].permute(0, 2, 1, 3)
            v_chunk = v_all_layers[:, :, start_idx : start_idx + chunk_len, :].permute(0, 2, 1, 3)
            self.kv_pool.write_kv_chunk(
                block_ids=req.physical_block_ids,
                start_token_idx=start_idx,
                k_chunk=k_chunk,
                v_chunk=v_chunk
            )

            req.num_prefilled_tokens += chunk_len

            # Register completed blocks in prefix cache for future requests
            if self.enable_prefix_caching and self.prefix_cache:
                num_blocks_now = req.num_prefilled_tokens // self.block_size
                parent_h = None
                for b_i in range(num_blocks_now):
                    chunk_b = req.prompt_token_ids[b_i * self.block_size : (b_i + 1) * self.block_size]
                    b_hash = compute_block_hash(parent_h, tuple(chunk_b))
                    parent_h = b_hash
                    b_id = req.physical_block_ids[b_i]
                    self.prefix_cache.hash_to_block_id[b_hash] = b_id
                    self.prefix_cache.blocks[b_id].block_hash = b_hash
                    self.prefix_cache.blocks[b_id].tokens = list(chunk_b)
                    tokens_up_to_b = (b_i + 1) * self.block_size
                    if b_hash not in self.prefix_cache.hash_to_kv_cache:
                        self.prefix_cache.hash_to_kv_cache[b_hash] = clone_hf_cache(req.kv_cache, tokens_up_to_b)
                    if b_i == num_blocks_now - 1 and req.is_prefill_done:
                        self.prefix_cache.hash_to_logits[b_hash] = outputs.logits[:, -1, :].detach().clone()

            # Emit the first output token when the entire prompt has finished prefilling
            if req.is_prefill_done:
                t_prefill_done, t_prefill_done_ns = _sample_paired_timestamp(self.device)
                if req.prefill_end_time is None:
                    req.prefill_end_time = t_prefill_done
                    req.prefill_end_time_ns = t_prefill_done_ns

                next_token = sample_next_token(
                    outputs.logits[:, -1, :],
                    req.sampling_params,
                    past_token_ids=req.output_token_ids
                )
                req.output_token_ids.append(next_token)

                t_first_token, t_first_token_ns = _sample_paired_timestamp(self.device)
                if req.first_token_time is None:
                    req.first_token_time = t_first_token
                    req.first_token_time_ns = t_first_token_ns
                if req.ttft_ms is None:
                    ref_time = req.schedule_time if req.schedule_time is not None else req.arrival_time
                    req.ttft_ms = max(0.1, (t_first_token - ref_time) * 1000.0)
                    self.metrics.record_ttft(req.ttft_ms)

                # Check stop criteria on first token
                should_stop, reason = check_stop_criteria(
                    req.output_token_ids,
                    self.tokenizer,
                    req.sampling_params.stop_token_ids if req.sampling_params else [],
                    req.sampling_params.stop_strings if req.sampling_params else []
                )
                if should_stop:
                    req.stop_reason = reason
                    req.status = RequestStatus.FINISHED
                    req.finish_step = current_step
                    req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
                    finished_this_step.append(req)
                elif req.num_generated >= req.max_tokens:
                    req.stop_reason = "length"
                    req.status = RequestStatus.FINISHED
                    req.finish_step = current_step
                    req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
                    finished_this_step.append(req)

                if self.verbose:
                    logger.debug("[Step %2d] ✓ Prefill finished for Req #%d (%d tokens), emitted token %d",
                                 current_step, req.req_id, len(req.prompt_token_ids), next_token)
            else:
                if self.verbose:
                    logger.debug("[Step %2d] 🧩 Chunked prefill Req #%d (%d/%d tokens processed)",
                                 current_step, req.req_id, req.num_prefilled_tokens, len(req.prompt_token_ids))

        # ---------------------------------------------------------------------
        # 2. BATCHED DECODE PHASE: Single batched forward pass for all active sequences
        # ---------------------------------------------------------------------
        decode_reqs = [r for r in self.running_batch if r.is_prefill_done and r.output_token_ids and r not in finished_this_step]
        if decode_reqs:
            t_decode_start = time.perf_counter()

            # 2a. Memory Allocation & Copy-on-Write for all decode sequences with per-request isolation
            valid_decode_reqs = []
            for req in decode_reqs:
                curr_token_count = req.total_tokens
                needed_blocks = (curr_token_count + 1 + self.block_size - 1) // self.block_size
                try:
                    if needed_blocks > len(req.physical_block_ids):
                        new_b_id = self._allocate_block()
                        req.physical_block_ids.append(new_b_id)

                    blk_idx = req.physical_block_ids[curr_token_count // self.block_size]
                    if self.enable_prefix_caching and self.prefix_cache:
                        blk = self.prefix_cache.blocks[blk_idx]
                        if blk.ref_count > 1:
                            new_b_id = self._allocate_block()
                            self.kv_pool.copy_block(blk_idx, new_b_id)
                            self.prefix_cache.blocks[blk_idx].ref_count -= 1
                            self.prefix_cache.blocks[new_b_id].ref_count = 1
                            self.prefix_cache.blocks[new_b_id].tokens = list(blk.tokens)
                            req.physical_block_ids[curr_token_count // self.block_size] = new_b_id
                    valid_decode_reqs.append(req)
                except (MemoryError, torch.cuda.OutOfMemoryError) as alloc_err:
                    is_cuda_oom = (
                        isinstance(alloc_err, (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", ())))
                        or "out of memory" in str(alloc_err).lower()
                    )
                    stop_reason = "cuda_oom" if is_cuda_oom else "kv_cache_full"
                    logger.warning("[Step %2d] Decode block allocation failed for Req #%d (%s): %s",
                                   current_step, req.req_id, stop_reason, alloc_err)
                    req.status = RequestStatus.FAILED
                    req.stop_reason = stop_reason
                    req.error = str(alloc_err)
                    req.finish_step = current_step
                    req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)

                    for b_id in req.physical_block_ids:
                        try:
                            self._free_block(b_id)
                        except Exception:
                            logger.exception("Failed to release block %s for failed request #%d", b_id, req.req_id)
                    req.physical_block_ids.clear()
                    req.kv_cache = None

                    if req in self.running_batch:
                        self.running_batch.remove(req)
                    if req not in self.completed_requests:
                        self.completed_requests.append(req)

                    e2e_ms = (time.perf_counter() - req.arrival_time) * 1000.0
                    self.metrics.record_request_completed(
                        prompt_tokens=len(req.prompt_token_ids),
                        generated_tokens=req.num_generated,
                        e2e_latency_ms=e2e_ms
                    )
                    if is_cuda_oom and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._update_queue_and_cache_metrics()

            decode_reqs = valid_decode_reqs

            if decode_reqs:
                # 2b. Assemble batched inputs for parallel model forward pass
                B = len(decode_reqs)
                batched_tokens = torch.tensor([[r.output_token_ids[-1]] for r in decode_reqs], device=self.device)
                max_blocks = max(len(r.physical_block_ids) for r in decode_reqs)
                block_tables = torch.zeros((B, max_blocks), dtype=torch.long, device=self.device)
                for b, r in enumerate(decode_reqs):
                    block_tables[b, :len(r.physical_block_ids)] = torch.tensor(r.physical_block_ids, dtype=torch.long, device=self.device)
                seq_lens = torch.tensor([r.total_tokens for r in decode_reqs], dtype=torch.long, device=self.device)

                # Single Batched Forward Pass: GEMM across all active sequences!
                batched_logits = self.model_runner.batched_decode_step(
                    batched_tokens=batched_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    kv_pool=self.kv_pool
                )

                tpot_batch_ms = (time.perf_counter() - t_decode_start) * 1000.0

                # 2c. Sample next token and verify stopping criteria for each sequence
                all_greedy = all(
                    (r.sampling_params is None or r.sampling_params.temperature <= 0.0)
                    and (r.sampling_params is None or (
                        r.sampling_params.repetition_penalty == 1.0
                        and r.sampling_params.frequency_penalty == 0.0
                        and r.sampling_params.presence_penalty == 0.0
                    ))
                    for r in decode_reqs
                )

                if self.enable_batched_sampling and all_greedy and len(decode_reqs) > 1:
                    # Batched GPU argmax with single host transfer to eliminate B sequential CUDA synchronization stalls
                    sampled_tokens = torch.argmax(batched_logits, dim=-1).cpu().tolist()
                else:
                    sampled_tokens = None

                for b, req in enumerate(decode_reqs):
                    if sampled_tokens is not None:
                        next_token = sampled_tokens[b]
                    else:
                        next_token = sample_next_token(
                            batched_logits[b : b + 1],
                            req.sampling_params,
                            past_token_ids=req.output_token_ids
                        )
                    req.output_token_ids.append(next_token)
                    self.metrics.record_tpot(tpot_batch_ms)

                    should_stop, reason = check_stop_criteria(
                        req.output_token_ids,
                        self.tokenizer,
                        req.sampling_params.stop_token_ids if req.sampling_params else [],
                        req.sampling_params.stop_strings if req.sampling_params else []
                    )
                    if should_stop:
                        req.stop_reason = reason
                        req.status = RequestStatus.FINISHED
                        req.finish_step = current_step
                        req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
                        finished_this_step.append(req)
                    elif req.num_generated >= req.max_tokens:
                        req.stop_reason = "length"
                        req.status = RequestStatus.FINISHED
                        req.finish_step = current_step
                        req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
                        finished_this_step.append(req)

        # Evict finished sequences and release physical blocks back to pool
        for req in finished_this_step:
            self.running_batch.remove(req)
            req.kv_cache = None  # Free PyTorch tensor memory
            for b_id in req.physical_block_ids:
                self._free_block(b_id)

            e2e_ms = (time.perf_counter() - req.arrival_time) * 1000.0
            self.metrics.record_request_completed(
                prompt_tokens=len(req.prompt_token_ids),
                generated_tokens=req.num_generated,
                e2e_latency_ms=e2e_ms
            )

            if req not in self.completed_requests:
                self.completed_requests.append(req)
            if self.verbose:
                logger.debug("[Step %2d] - FINISHED Req #%d (Tokens: %d, Freed: %s, E2E: %.1fms)",
                             current_step, req.req_id, req.num_generated, req.physical_block_ids, e2e_ms)

        if finished_this_step:
            self._update_queue_and_cache_metrics()

    def _abort_running_requests(self, exc: Optional[BaseException] = None):
        """Release resources for requests interrupted by an execution error."""
        failed_requests = list(self.running_batch)
        self.running_batch.clear()
        self.waiting_queue[:] = [req for req in self.waiting_queue if req not in failed_requests]

        is_cuda_oom = exc is not None and (
            isinstance(exc, (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", ())))
            or "out of memory" in str(exc).lower()
        )
        is_kv_pool_full = exc is not None and (
            isinstance(exc, MemoryError)
            or "kv cache" in str(exc).lower()
            or "physical block" in str(exc).lower()
        )
        if is_cuda_oom:
            stop_reason = "cuda_oom"
        elif is_kv_pool_full:
            stop_reason = "kv_cache_full"
        else:
            stop_reason = "error"

        err_msg = str(exc) if exc is not None else "Execution aborted"

        for req in failed_requests:
            req.status = RequestStatus.FAILED
            req.stop_reason = stop_reason
            req.error = err_msg
            req.finish_step = self.current_step
            req.finish_time, req.finish_time_ns = _sample_paired_timestamp(self.device)
            if req not in self.completed_requests:
                self.completed_requests.append(req)

            block_ids = list(req.physical_block_ids)
            req.physical_block_ids.clear()
            req.kv_cache = None
            for block_id in block_ids:
                try:
                    self._free_block(block_id)
                except Exception:
                    # Never replace the execution exception with cleanup noise.
                    logger.exception("Failed to release block %s after request execution error", block_id)

        if is_cuda_oom and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._update_queue_and_cache_metrics()

    def step(self) -> bool:
        """Executes a single engine iteration step. Returns True if requests remain active/waiting."""
        if not (self.waiting_queue or self.running_batch):
            return False

        self._schedule(current_step=self.current_step)

        if not self.running_batch:
            self.current_step += 1
            return bool(self.waiting_queue or self.running_batch)

        if self.verbose:
            active_info = [f"#{r.req_id}({r.num_generated}/{r.max_tokens})" for r in self.running_batch]
            logger.debug("Step %2d | Active (%d/%d): %s | Free Blocks: %d",
                         self.current_step, len(self.running_batch), self.max_batch_size,
                         ', '.join(active_info), self.kv_pool.num_free)

        try:
            self._execute_step(current_step=self.current_step)
        except Exception as exc:
            self._abort_running_requests(exc=exc)
            raise
        self.current_step += 1
        return bool(self.waiting_queue or self.running_batch)

    def run(self) -> List[EngineRequest]:
        """Runs the engine loop until all requests in the queues are complete."""
        t0 = time.perf_counter()

        while self.step():
            pass

        self.last_run_time = time.perf_counter() - t0
        self._update_queue_and_cache_metrics()
        return self.completed_requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Mini-vLLM Engine with PagedAttention & Continuous Batching")
    add_common_engine_args(parser)
    parser.add_argument("--prompt", type=str, action="append", help="Prompt text to process (can specify multiple times)")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens per response (default: 32)")
    parser.add_argument("--batch-size", type=int, default=4, help="Max continuous batch size (default: 4)")
    parser.add_argument("--total-blocks", type=int, default=64, help="Total physical KV cache blocks (default: 64)")
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per physical block (default: 16)")
    parser.add_argument("--quant", type=str, choices=["none", "int8", "int4", "hybrid_int8"], default="none", help="Optional weight quantization")
    parser.add_argument("--kv-cache-dtype", type=str, choices=["fp16", "int8"], default="fp16", help="KV cache storage dtype (default: fp16)")
    parser.add_argument("--chunked-prefill", action="store_true", default=True, help="Enable chunked prefill (default: True)")
    parser.add_argument("--no-chunked-prefill", action="store_false", dest="chunked_prefill", help="Disable chunked prefill")
    parser.add_argument("--chunk-size", type=int, default=32, help="Tokens per prefill chunk (default: 32)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-P nucleus sampling (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=-1, help="Top-K sampling (default: -1 disabled)")
    parser.add_argument("--min-p", type=float, default=0.0, help="Min-P dynamic probability threshold (default: 0.0)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)")
    parser.add_argument("--stop", type=str, action="append", default=[], help="Stop strings to terminate generation early")
    parser.add_argument("--metrics", action="store_true", default=True, help="Display telemetry dashboard (default: True)")
    parser.add_argument("--no-metrics", action="store_false", dest="metrics", help="Disable telemetry dashboard")
    args = parser.parse_args()

    prompts = args.prompt
    if not prompts:
        if not sys.stdin.isatty():
            prompts = [line.strip() for line in sys.stdin if line.strip()]
        else:
            try:
                user_p = input("Enter prompt: ").strip()
                if user_p:
                    prompts = [user_p]
            except (KeyboardInterrupt, EOFError):
                pass

    if not prompts:
        print("No prompts provided. Use --prompt 'your text' or pipe input via stdin.")
        sys.exit(0)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        stop_strings=args.stop or []
    )

    engine = MiniVLLMEngine(
        model_name=args.model,
        max_batch_size=args.batch_size,
        total_blocks=args.total_blocks,
        block_size=args.block_size,
        device=args.device,
        quantization=args.quant,
        kv_cache_dtype=args.kv_cache_dtype,
        chunked_prefill=args.chunked_prefill,
        prefill_chunk_size=args.chunk_size,
        verbose=args.verbose
    )

    for i, p in enumerate(prompts):
        engine.add_request(prompt=p, max_tokens=args.max_tokens, arrival_step=0, sampling_params=sampling_params)

    completed = engine.run()

    total_tokens = sum(r.num_generated for r in completed)
    total_time = getattr(engine, "last_run_time", 0.0)
    tps = total_tokens / total_time if total_time > 0 else 0.0

    for req in sorted(completed, key=lambda r: r.req_id):
        text = engine.tokenizer.decode(req.output_token_ids, skip_special_tokens=True)
        print(f"\n[Request #{req.req_id}]")
        print(f"Prompt: {req.prompt}")
        print(f"Output: {text}")
        print(f"Tokens: {req.num_generated} | Duration: {req.finish_step - req.start_step + 1} steps")

    print(f"\nCompleted {len(completed)} request(s) | Generated {total_tokens} tokens in {total_time:.2f}s ({tps:.1f} tok/s)")
    if getattr(args, "metrics", True):
        print("\n" + engine.metrics.render_ascii_dashboard(elapsed_seconds=total_time))
