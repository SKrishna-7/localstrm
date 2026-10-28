"""
Benchmark Workload Generator
----------------------------
Generates reproducible synthetic workloads for LLM serving benchmarks:
  - Workload A: Controlled Fixed-Length Workload (fixed prompt & fixed decode)
  - Workload B: Mixed-Length Workload (short, medium, long prompts & generation targets,
                with shared system prompts for prefix caching)
Supports closed-loop concurrency batches and open-loop Poisson / uniform arrival patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import random


@dataclass
class BenchmarkRequest:
    """Represents a single request in a benchmark workload."""
    request_id: str
    prompt: str
    target_tokens: int
    arrival_time_offset: float = 0.0  # seconds from benchmark start
    prompt_token_ids: List[int] = field(default_factory=list)
    system_prompt_tokens: int = 0
    workload_type: str = "mixed"

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


def generate_fixed_workload(
    tokenizer: Any,
    num_requests: int,
    prompt_tokens: int = 32,
    target_tokens: int = 20,
    seed: int = 42
) -> List[BenchmarkRequest]:
    """
    Workload A: Generates requests with exactly repeatable prompt token count
    and fixed target output token count.
    """
    rng = random.Random(seed)

    vocabulary_samples = [
        "In modern computer systems, the central processing unit coordinates instructions.",
        "Operating system kernels manage virtual memory through paged translation tables.",
        "Graphics processing units execute massive matrix multiplication with parallel tensor cores.",
        "High-performance computing clusters distribute computation across high-speed network fabrics.",
        "Distributed database storage engines ensure consistency using write-ahead transaction logs."
    ]

    requests = []
    for idx in range(num_requests):
        req_id = f"fixed-{idx+1:04d}"

        # Build prompt to exact target token count
        base_text = vocabulary_samples[idx % len(vocabulary_samples)]
        tokens = tokenizer.encode(base_text)

        # Extend or trim to match exact prompt_tokens
        while len(tokens) < prompt_tokens:
            extra = tokenizer.encode(" Additional contextual memory block for inference testing.")
            tokens.extend(extra)
        tokens = tokens[:prompt_tokens]
        prompt_text = tokenizer.decode(tokens, skip_special_tokens=True)

        requests.append(
            BenchmarkRequest(
                request_id=req_id,
                prompt=prompt_text,
                target_tokens=target_tokens,
                arrival_time_offset=0.0,
                prompt_token_ids=tokens,
                system_prompt_tokens=0,
                workload_type="fixed"
            )
        )

    return requests


def generate_mixed_workload(
    tokenizer: Any,
    num_requests: int,
    seed: int = 42,
    shared_prefix: bool = True
) -> List[BenchmarkRequest]:
    """
    Workload B: Generates diverse realistic requests with variable prompt lengths,
    variable generation targets, and shared system prefixes.
    """
    rng = random.Random(seed)

    system_prefix = (
        "System: You are an expert AI systems performance engineer. "
        "Write clean, memory-efficient, production-grade code.\n"
    )
    sys_tokens = len(tokenizer.encode(system_prefix)) if shared_prefix else 0

    templates = [
        # (user query, target output tokens, use_system_prefix)
        ("User: Write a python fibonacci generator function.\nAssistant:", 24, True),
        ("User: Explain what an operating system kernel does in one sentence.\nAssistant:", 12, False),
        ("User: Implement quicksort with in-place partition in Python.\nAssistant:", 36, True),
        ("User: What is the difference between concurrency and parallelism?\nAssistant:", 18, False),
        ("User: Write a binary search algorithm in Python.\nAssistant:", 28, True),
        ("User: What is GPU memory bandwidth saturation?\nAssistant:", 16, False),
        ("User: Write an LRU cache implementation using OrderedDict.\nAssistant:", 32, True),
        ("User: Summarize why GEMV is memory-bound during LLM decoding.\nAssistant:", 20, False),
        ("User: Implement a thread-safe producer-consumer queue in Python.\nAssistant:", 30, True),
        ("User: Describe PagedAttention and physical block management.\nAssistant:", 22, False),
        ("User: Write a breadth-first search graph traversal function.\nAssistant:", 34, True),
        ("User: What are the primary causes of execution bubbles in static batching?\nAssistant:", 14, False),
    ]

    requests = []
    for idx in range(num_requests):
        req_id = f"mixed-{idx+1:04d}"
        query, target_tok, use_sys = templates[idx % len(templates)]

        full_prompt = (system_prefix + query) if (shared_prefix and use_sys) else query
        tokens = tokenizer.encode(full_prompt)

        # Stagger arrival offset slightly if open-loop
        requests.append(
            BenchmarkRequest(
                request_id=req_id,
                prompt=full_prompt,
                target_tokens=target_tok,
                arrival_time_offset=idx * 0.05,
                prompt_token_ids=tokens,
                system_prompt_tokens=sys_tokens if (shared_prefix and use_sys) else 0,
                workload_type="mixed"
            )
        )

    return requests
