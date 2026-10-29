"""
Engine Configuration Module
----------------------------
Production-grade configuration dataclasses inspired by vLLM / HuggingFace TGI.
Eliminates all hardcoded model names and paths across the engine.
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


def get_default_model() -> str:
    """Returns the default model ID or path from environment variable or standard fallback."""
    return os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B")


def get_default_device() -> str:
    """Returns 'cuda' if GPU is available and initialized, else 'cpu'."""
    return os.getenv("ENGINE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelConfig:
    """Configuration for model architecture and weights."""
    model_name_or_path: str = field(default_factory=get_default_model)
    dtype: torch.dtype = torch.float16
    trust_remote_code: bool = True
    max_model_len: Optional[int] = None

    @classmethod
    def from_str(cls, model_str: Optional[str] = None, device: str = "cpu") -> "ModelConfig":
        model_name = model_str or get_default_model()
        dtype = torch.float32 if device == "cpu" else torch.float16
        return cls(model_name_or_path=model_name, dtype=dtype)


@dataclass
class CacheConfig:
    """Configuration for physical KV Cache memory pool."""
    block_size: int = 16
    total_blocks: Optional[int] = None
    vram_budget_bytes: Optional[int] = None
    gpu_memory_utilization: float = 0.85
    kv_cache_dtype: str = "fp16"  # "fp16" or "int8"

    def __post_init__(self) -> None:
        if self.total_blocks is not None and self.vram_budget_bytes is not None:
            raise ValueError(
                f"Conflicting KV-cache configuration: both total_blocks ({self.total_blocks}) "
                f"and vram_budget_bytes ({self.vram_budget_bytes}) were provided. "
                "Specify total_blocks OR vram_budget, not both."
            )
        if self.total_blocks is None and self.vram_budget_bytes is None:
            self.total_blocks = 64
        elif self.total_blocks is not None and self.total_blocks <= 0:
            raise ValueError(f"total_blocks must be positive, got {self.total_blocks}")
        elif self.vram_budget_bytes is not None and self.vram_budget_bytes <= 0:
            raise ValueError(f"vram_budget_bytes must be positive, got {self.vram_budget_bytes}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")


@dataclass
class SchedulerConfig:
    """Configuration for continuous batching and prefill policies."""
    max_batch_size: int = 4
    max_tokens_per_step: int = 512
    enable_chunked_prefill: bool = True
    prefill_chunk_size: int = 32
    enable_prefix_caching: bool = True
    batched_decode: bool = True


@dataclass
class EngineConfig:
    """Unified Engine Configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    device: str = field(default_factory=get_default_device)
    quantization: str = "none"
    trust_remote_code: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EngineConfig":
        """Builds configuration dynamically from CLI arguments and environment variables."""
        model_name = getattr(args, "model", None) or get_default_model()
        device = getattr(args, "device", None) or get_default_device()
        quant = getattr(args, "quant", "none") or "none"
        batch_size = getattr(args, "batch_size", 4)
        total_blocks = getattr(args, "total_blocks", None)
        vram_budget = getattr(args, "vram_budget", None)
        block_size = getattr(args, "block_size", 16)
        trust_remote_code = getattr(args, "trust_remote_code", False)

        vram_budget_bytes = None
        if vram_budget is not None:
            from src.memory.preflight import parse_vram_budget_bytes
            vram_budget_bytes = parse_vram_budget_bytes(vram_budget)

        model_cfg = ModelConfig.from_str(model_name, device=device)
        cache_cfg = CacheConfig(
            block_size=block_size,
            total_blocks=total_blocks,
            vram_budget_bytes=vram_budget_bytes,
            kv_cache_dtype=getattr(args, "kv_cache_dtype", "fp16")
        )
        sched_cfg = SchedulerConfig(
            max_batch_size=batch_size,
            enable_chunked_prefill=getattr(args, "chunked_prefill", True),
            prefill_chunk_size=getattr(args, "chunk_size", 32),
            enable_prefix_caching=getattr(args, "enable_prefix_caching", True),
            batched_decode=getattr(args, "batched_decode", True),
        )

        return cls(
            model=model_cfg,
            cache=cache_cfg,
            scheduler=sched_cfg,
            device=device,
            quantization=quant,
            trust_remote_code=trust_remote_code
        )


import logging


def get_logger(name: str = "mini_vllm") -> logging.Logger:
    """Returns a configured logger for engine subsystems."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
    return logger


def add_common_engine_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attaches standard production CLI flags across any engine entry point."""
    parser.add_argument(
        "--model",
        type=str,
        default=get_default_model(),
        help="Model name from HuggingFace Hub or local filesystem path (default: $MODEL_ID or Qwen/Qwen2.5-0.5B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        choices=["cuda", "cpu"],
        help="Compute device for execution (default: auto-detected cuda or cpu)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging during inference"
    )
    return parser
