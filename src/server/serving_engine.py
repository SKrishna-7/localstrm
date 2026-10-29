"""
Asynchronous Serving Engine
---------------------------
Coordinates model forward passes, KV caching, probability sampling,
and token streaming for the OpenAI-compatible REST API.
"""

import os
import sys
import time
import asyncio
from typing import AsyncGenerator, Tuple, List, Optional, Union
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import get_default_model, get_default_device, get_logger
from src.memory.preflight import enforce_preflight_check
from src.kernels.quantization import apply_optional_quantization, load_and_quantize_model
from src.engine.sampling import (
    SamplingParams,
    sample_next_token,
    check_stop_criteria
)
from src.observability.metrics import EngineMetricsTracker

logger = get_logger("mini_vllm.server_engine")


class AsyncServingEngine:
    """
    Thread-safe serving engine managing model execution,
    token-by-token streaming, and Prometheus telemetry.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        quantization: str = "none",
        force: bool = False,
        trust_remote_code: bool = False
    ):
        self.model_name = model_name or get_default_model()
        self.device_str = device or get_default_device()
        self.device = torch.device(self.device_str)
        self.quantization = quantization.lower()
        self.trust_remote_code = trust_remote_code

        # 1. Enforce memory safety
        enforce_preflight_check(
            self.model_name,
            device=self.device_str,
            quantization=self.quantization,
            force=force,
            trust_remote_code=self.trust_remote_code
        )

        logger.info("Initializing AsyncServingEngine with model '%s' on %s (quant=%s)...",
                    self.model_name, self.device_str, self.quantization)

        # 2. Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16

        self.model, _, _ = load_and_quantize_model(
            self.model_name,
            device=self.device,
            quantization=self.quantization,
            dtype=dtype,
            logger=logger,
            trust_remote_code=self.trust_remote_code
        )

        # 4. Observability Tracker
        self.metrics = EngineMetricsTracker(model_name=self.model_name, enable_prometheus=True)
        self._lock = asyncio.Lock()

    def format_chat_prompt(self, messages: List[dict]) -> str:
        """Formats multi-turn messages into a single prompt string using chat template."""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback chat format
        lines = []
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("Assistant: ")
        return "\n".join(lines)

    async def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        max_tokens: int = 1028
    ) -> AsyncGenerator[Tuple[str, bool, bool, Optional[str]], None]:
        """
        Asynchronously generates tokens and yields streaming deltas:
        Yields: (token_str, is_first, is_last, finish_reason)
        """
        async with self._lock:
            self.metrics.num_requests_running += 1
            completed_normally = False
            t_prefill_start = time.perf_counter()
            prompt_len = 0
            generated_ids = []
            try:
                # Tokenize prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                prompt_len = input_ids.shape[1]

                with torch.no_grad():
                    outputs = self.model(input_ids, use_cache=True)
                t_prefill_end = time.perf_counter()

                ttft_ms = (t_prefill_end - t_prefill_start) * 1000.0
                self.metrics.record_ttft(ttft_ms)

                past_key_values = outputs.past_key_values
                next_token = sample_next_token(outputs.logits[:, -1, :], sampling_params, past_token_ids=[])
                generated_ids = [next_token]

                first_text = self.tokenizer.decode([next_token], skip_special_tokens=True)

                if next_token == self.tokenizer.eos_token_id or max_tokens <= 1:
                    finish_reason = "stop" if next_token == self.tokenizer.eos_token_id else "length"
                    yield (first_text, True, True, finish_reason)
                    completed_normally = True
                    return

                yield (first_text, True, False, None)

                # Let other event loop tasks breathe
                await asyncio.sleep(0)

                curr_token = torch.tensor([[next_token]], device=self.device)
                finish_reason = "length"

                for step in range(1, max_tokens):
                    t_decode_start = time.perf_counter()
                    with torch.no_grad():
                        outputs = self.model(curr_token, past_key_values=past_key_values, use_cache=True)
                    t_decode_end = time.perf_counter()

                    tpot_ms = (t_decode_end - t_decode_start) * 1000.0
                    self.metrics.record_tpot(tpot_ms)

                    past_key_values = outputs.past_key_values
                    next_token = sample_next_token(
                        outputs.logits[:, -1, :],
                        sampling_params,
                        past_token_ids=generated_ids
                    )
                    generated_ids.append(next_token)

                    token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)

                    # Check stop criteria (EOS or stop strings)
                    should_stop, stop_reason = check_stop_criteria(
                        generated_ids=generated_ids,
                        tokenizer=self.tokenizer,
                        stop_token_ids=sampling_params.stop_token_ids or [self.tokenizer.eos_token_id],
                        stop_strings=sampling_params.stop_strings or []
                    )

                    if should_stop or next_token == self.tokenizer.eos_token_id:
                        finish_reason = "stop"
                        yield (token_text, False, True, finish_reason)
                        completed_normally = True
                        break
                    elif step == max_tokens - 1:
                        finish_reason = "length"
                        yield (token_text, False, True, finish_reason)
                        completed_normally = True
                        break
                    else:
                        yield (token_text, False, False, None)

                    curr_token = torch.tensor([[next_token]], device=self.device)
                    await asyncio.sleep(0)
            except (GeneratorExit, asyncio.CancelledError):
                logger.info("generate_stream cancelled or client disconnected")
                raise
            except (torch.cuda.OutOfMemoryError, getattr(torch, "OutOfMemoryError", ())) as e:
                logger.error("CUDA OOM during generate_stream: %s", e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            except Exception as e:
                logger.error("Error during generate_stream: %s", e)
                raise
            finally:
                self.metrics.num_requests_running = max(0, self.metrics.num_requests_running - 1)
                if completed_normally:
                    self.metrics.record_request_completed(
                        prompt_tokens=prompt_len,
                        generated_tokens=len(generated_ids),
                        e2e_latency_ms=(time.perf_counter() - t_prefill_start) * 1000.0
                    )

    async def generate_complete(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        max_tokens: int = 128
    ) -> Tuple[str, int, int, str]:
        """
        Executes complete generation and returns:
        (generated_text, prompt_token_count, generated_token_count, finish_reason)
        """
        full_chunks = []
        finish_reason = "stop"
        prompt_len = len(self.tokenizer.encode(prompt))

        async for chunk_text, _, is_last, f_reason in self.generate_stream(prompt, sampling_params, max_tokens):
            full_chunks.append(chunk_text)
            if is_last and f_reason:
                finish_reason = f_reason

        full_text = "".join(full_chunks)
        gen_tokens = len(self.tokenizer.encode(full_text))
        return full_text, prompt_len, gen_tokens, finish_reason
