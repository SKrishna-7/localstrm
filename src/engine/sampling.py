"""
Production LLM Sampling Module
-------------------------------
Industrial-strength token sampling supporting:
  1. Temperature scaling (greedy at 0.0)
  2. Top-K filtering
  3. Top-P (Nucleus) sampling
  4. Min-P dynamic threshold sampling
  5. Repetition, Frequency, and Presence penalties
  6. Multi-token stop string and stop token id detection
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    """Production sampling parameters controlling probability shaping and termination."""
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_token_ids: List[int] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)

    @classmethod
    def greedy(cls) -> "SamplingParams":
        return cls(temperature=0.0)


def apply_repetition_penalties(
    logits: torch.Tensor,
    past_token_ids: List[int],
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Applies repetition, frequency, and presence penalties to logits in-place.
    """
    if not past_token_ids:
        return logits

    device = logits.device
    unique_tokens, counts = torch.unique(torch.tensor(past_token_ids, device=device), return_counts=True)

    # 1. Standard Multiplicative Repetition Penalty
    if repetition_penalty != 1.0:
        vals = logits[..., unique_tokens]
        # If logit > 0, divide by penalty. If logit < 0, multiply by penalty.
        logits[..., unique_tokens] = torch.where(
            vals > 0,
            vals / repetition_penalty,
            vals * repetition_penalty
        )

    # 2. Additive Frequency and Presence Penalties (OpenAI style)
    if frequency_penalty != 0.0 or presence_penalty != 0.0:
        logits[..., unique_tokens] -= frequency_penalty * counts.float() + presence_penalty

    return logits


def apply_top_k_top_p_min_p(
    logits: torch.Tensor,
    top_k: int = -1,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    Applies Top-K, Nucleus (Top-P), and Min-P filtering to vocabulary logits.
    """
    # 1. Top-K filtering
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        # Keep only top_k values, set rest to -infinity
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # 2. Min-P filtering: removes tokens with prob < min_p * max_prob
    if 0.0 < min_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs, dim=-1, keepdim=True)[0]
        threshold = min_p * max_prob
        indices_to_remove = probs < threshold
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # 3. Top-P (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = False
        else:
            # Keep at least 1 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def sample_next_token(
    logits: torch.Tensor,
    params: SamplingParams,
    past_token_ids: Optional[List[int]] = None
) -> int:
    """
    Selects the next token from logits following the given SamplingParams.
    """
    # Extract last token logits: shape [vocab_size]
    if logits.ndim > 1:
        logits = logits[-1]
    logits = logits.clone()

    # Greedy Sampling
    if params.temperature <= 0.0:
        return torch.argmax(logits, dim=-1).item()

    # 1. Penalties
    if past_token_ids:
        logits = apply_repetition_penalties(
            logits,
            past_token_ids,
            repetition_penalty=params.repetition_penalty,
            frequency_penalty=params.frequency_penalty,
            presence_penalty=params.presence_penalty
        )

    # 2. Temperature Scaling
    logits = logits / params.temperature

    # 3. Probability Filters (Top-K, Top-P, Min-P)
    logits = apply_top_k_top_p_min_p(
        logits,
        top_k=params.top_k,
        top_p=params.top_p,
        min_p=params.min_p
    )

    # 4. Categorical Sampling
    probs = F.softmax(logits, dim=-1)
    # Guard against NaN/inf
    if torch.isnan(probs).any() or (probs.sum() == 0):
        return torch.argmax(logits, dim=-1).item()

    sampled_token = torch.multinomial(probs, num_samples=1).item()
    return sampled_token


def check_stop_criteria(
    generated_ids: List[int],
    tokenizer: any,
    stop_token_ids: List[int],
    stop_strings: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Checks if generation should terminate due to EOS, stop_token_ids, or stop_strings.
    Returns: (should_stop, reason)
    """
    if not generated_ids:
        return False, None

    last_token = generated_ids[-1]

    # 1. Token ID check
    if last_token == tokenizer.eos_token_id:
        return True, "eos_token"
    if last_token in stop_token_ids:
        return True, f"stop_token_{last_token}"

    # 2. Multi-token stop string check
    if stop_strings:
        # Decode the last 20 tokens to check for string suffixes
        tail_ids = generated_ids[-20:] if len(generated_ids) > 20 else generated_ids
        decoded_tail = tokenizer.decode(tail_ids, skip_special_tokens=False)
        for stop_str in stop_strings:
            if stop_str and stop_str in decoded_tail:
                return True, f"stop_string_{stop_str}"

    return False, None
