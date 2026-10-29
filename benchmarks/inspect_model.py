"""
Milestone 1: Model Parameter Accounting & Memory Math
------------------------------------------------------
Inspects model parameters, weight footprints, GQA (Grouped Query Attention) ratios,
and calculates theoretical KV cache scaling for 6 GB VRAM budgets.
"""

import os
import sys
import argparse
from typing import Optional
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Support running directly or as a package module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import get_default_model, get_default_device, add_common_engine_args


def format_bytes(num_bytes: int) -> str:
    """Format raw bytes into human-readable B, KB, MB, or GB."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def print_memory_stats(label: str, device: torch.device):
    """Prints PyTorch CUDA memory allocator stats if running on GPU."""
    if device.type != "cuda":
        print(f"[{label}] Running on CPU (CUDA memory stats not applicable)")
        return
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    print(f"--- Memory Stats: {label} ---")
    print(f"  • Actually Allocated (Tensors): {format_bytes(allocated)}")
    print(f"  • Reserved by Caching Allocator: {format_bytes(reserved)}")
    print(f"  • Peak Memory Allocated:        {format_bytes(max_allocated)}")
    print(f"----------------------------------------")


def inspect_model_memory(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = False
):
    model_name = model_name or get_default_model()
    device_str = device or get_default_device()
    target_device = torch.device(device_str)
    device = target_device

    # Detect hardware device
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory
        print(f"Total GPU VRAM: {format_bytes(total_vram)}")
    else:
        print(f"Using Device: {device.type.upper()}")

    print_memory_stats("Before Loading Model", device)

    print(f"\nLoading model: '{model_name}' (dtype=torch.float16)...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device.type,
        trust_remote_code=trust_remote_code
    )
    model.eval()

    print_memory_stats("After Loading Weights", device)

    # 1. Parameter Accounting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    print("\n=== Parameter & Weight Breakdown ===")
    print(f"Total Parameters:      {total_params:,} ({total_params / 1e6:.1f} Million)")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Weights Memory Size:   {format_bytes(weight_bytes)}")

    # 2. Architecture Dimensions
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    gqa_ratio = num_heads // num_kv_heads

    print("\n=== Architecture Dimensions ===")
    print(f"Vocab Size:            {config.vocab_size:,}")
    print(f"Hidden Size (d_model): {config.hidden_size}")
    print(f"Num Layers:            {config.num_hidden_layers}")
    print(f"Num Attention Heads:   {num_heads}")
    print(f"Num KV Heads (GQA):    {num_kv_heads} ({gqa_ratio}x GQA reduction)")
    print(f"Head Dimension:        {head_dim}")
    print(f"Intermediate (MLP):    {config.intermediate_size}")

    # 3. KV Cache Theoretical Math
    bytes_per_elem = 2  # float16 = 2 bytes per element
    num_layers = config.num_hidden_layers
    kv_cache_per_token_bytes = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem

    print("\n=== KV Cache Scaling Calculations ===")
    print(f"KV Cache per single token:      {format_bytes(kv_cache_per_token_bytes)}")
    context_2k = kv_cache_per_token_bytes * 2048
    context_4k = kv_cache_per_token_bytes * 4096
    print(f"KV Cache for 2,048 tokens:      {format_bytes(context_2k)} per sequence")
    print(f"KV Cache for 4,096 tokens:      {format_bytes(context_4k)} per sequence")

    # Estimate capacity inside a 3.5 GB free VRAM budget
    free_cache_budget = 3.5 * (1024**3)
    max_2k_seqs = int(free_cache_budget // context_2k)
    print(f"Concurrent 2k-token streams possible in 3.5 GB: ~{max_2k_seqs} streams!")

    # 4. Prefill Forward Pass Test
    print("\nRunning sample prefill forward pass (512 prompt tokens)...")
    dummy_input = torch.randint(0, config.vocab_size, (1, 512), device=device)
    with torch.no_grad():
        _ = model(dummy_input, use_cache=True)

    print_memory_stats("After Prefill Forward Pass", device)
    print("\n[Milestone 1 Complete] Model memory inspected successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect LLM Architecture and Memory Footprint")
    add_common_engine_args(parser)
    args = parser.parse_args()

    inspect_model_memory(model_name=args.model, device=args.device)
