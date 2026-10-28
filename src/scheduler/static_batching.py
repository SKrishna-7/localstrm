"""
Milestone 3: Static Request Batching & The Execution Bubble
------------------------------------------------------------
Demonstrates traditional request-level batching with left-padding.
Empirically measures the execution bubble (wasted GPU compute) caused
by short requests being dragged along until the slowest request finishes.
"""

import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Support running directly or as a package module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import get_default_model, get_default_device, add_common_engine_args


@dataclass
class Request:
    """Represents an incoming user inference request."""
    id: int
    prompt: str
    max_tokens: int


def display_batch_visualization(requests: List[Request], prompt_lengths: List[int], max_prompt_len: int):
    """Prints an ASCII diagram showing how static batching pads prompts."""
    print("\n--- Static Batching Memory Layout (Left Padded) ---")
    for req, plen in zip(requests, prompt_lengths):
        pads = max_prompt_len - plen
        pad_str = "[PAD] " * pads
        token_str = "[TOKEN] " * plen
        print(f"Req #{req.id} (Len {plen:2d}): | {pad_str}{token_str}| -> Target: {req.max_tokens} tokens")
    print("-" * 55)


def run_static_batching(
    requests: List[Request],
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = False
):
    model_name = model_name or get_default_model()
    device_str = device or get_default_device()

    if verbose:
        print("=" * 65)
        print("      STATIC BATCHING & THE EXECUTION BUBBLE")
        print("=" * 65)
        print(f"Target Model: {model_name}")
        print(f"Device:       {device_str}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device_str == "cpu" else torch.float16,
    ).to(device_str)
    model.eval()

    # Left-padding is mandatory for decoder-only causal generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[2/4] Tokenizing and padding batch...")
    prompts = [r.prompt for r in requests]
    tokenized = tokenizer(prompts, padding=True, return_tensors="pt").to(device_str)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    batch_size, max_prompt_len = input_ids.shape
    prompt_lengths = [int(mask.sum().item()) for mask in attention_mask]

    display_batch_visualization(requests, prompt_lengths, max_prompt_len)

    print("[3/4] Running batched Prefill phase...")
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
    prefill_time = (time.perf_counter() - t_prefill_start) * 1000

    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values

    print(f"  • Prefill completed in: {prefill_time:.2f} ms")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Max prompt length: {max_prompt_len} tokens")

    print("\n[4/4] Running batched Decode phase (tracking execution bubbles)...")
    finished = [False] * batch_size
    generated_ids = [[tok.item()] for tok in next_tokens]
    actual_tokens_generated = [1] * batch_size
    finish_step = [0] * batch_size
    wasted_steps = [0] * batch_size

    max_decode_steps = max(r.max_tokens for r in requests)
    curr_input_ids = next_tokens

    t_decode_start = time.perf_counter()

    for step in range(1, max_decode_steps):
        with torch.no_grad():
            outputs = model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

        past_key_values = outputs.past_key_values
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        curr_input_ids = next_tokens

        # Check status for each request in batch
        for i in range(batch_size):
            if not finished[i]:
                actual_tokens_generated[i] += 1
                generated_ids[i].append(next_tokens[i].item())

                if actual_tokens_generated[i] >= requests[i].max_tokens or next_tokens[i].item() == tokenizer.eos_token_id:
                    finished[i] = True
                    finish_step[i] = step
            else:
                # Wasted idle computation step!
                wasted_steps[i] += 1

        if all(finished):
            break

    total_decode_time = (time.perf_counter() - t_decode_start) * 1000

    print("\n" + "=" * 65)
    print("                    PER-REQUEST RESULTS")
    print("=" * 65)
    for i, req in enumerate(requests):
        gen_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
        print(f"\n[Request #{req.id}]")
        print(f"  Prompt:             \"{req.prompt}\"")
        print(f"  Output:             \"{gen_text}\"")
        print(f"  Tokens Wanted:      {req.max_tokens}")
        print(f"  Tokens Generated:   {actual_tokens_generated[i]}")
        print(f"  Finished At Step:   {finish_step[i]}")
        print(f"  Idle/Wasted Steps:  {wasted_steps[i]} steps (dragged along by other requests)")

    total_useful_steps = sum(actual_tokens_generated)
    total_wasted_steps = sum(wasted_steps)
    total_slots = total_useful_steps + total_wasted_steps
    bubble_ratio = (total_wasted_steps / total_slots) * 100

    print("\n" + "=" * 65)
    print("             EFFICIENCY & BUBBLE METRICS")
    print("=" * 65)
    print(f"  • Total Useful Token Generations: {total_useful_steps}")
    print(f"  • Total Wasted Idle Computations: {total_wasted_steps}")
    print(f"  • Execution Bubble (Waste Ratio): {bubble_ratio:.1f}%")
    print(f"  • Total Decode Latency:           {total_decode_time:.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static Batching & Execution Bubble Benchmark")
    add_common_engine_args(parser)
    parser.add_argument("--prompt", type=str, action="append", help="Prompt text to batch (can specify multiple times)")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens per response (default: 20)")
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
        print("No prompts provided. Specify --prompt 'your text' or pipe input via stdin.")
        sys.exit(0)

    test_requests = [
        Request(id=i + 1, prompt=p, max_tokens=args.max_tokens if i > 0 else 5)
        for i, p in enumerate(prompts)
    ]

    run_static_batching(test_requests, model_name=args.model, device=args.device, verbose=args.verbose)
