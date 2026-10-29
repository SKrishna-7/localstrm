#!/usr/bin/env python3
"""
Mini-vLLM CLI: Production-Structured Command-Line Interface for LLM Inference.

Usage:
  # Chat Mode (Streaming interactive REPL with live metrics)
  python cli.py chat --model Qwen/Qwen2.5-0.5B
  python cli.py chat --quant int8
  python cli.py chat --quant int4

  # Run Unified Mini-vLLM Engine (PagedAttention + Continuous Batching)
  python cli.py engine --model Qwen/Qwen2.5-0.5B --batch-size 4

  # Run Continuous Batching Benchmark
  python cli.py benchmark --batch-size 4

  # OpenAI-Compatible Streaming REST Server
  python cli.py serve --port 8000
  python cli.py serve --port 8000 --quant int8
"""

import sys
import os
import warnings

# Suppress noisy third-party library warnings before importing heavy packages
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import argparse
import time
from typing import List, Optional, Set, Any, Dict

# Add project root to sys.path so src imports resolve cleanly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import get_default_model, get_default_device, EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from src.kernels.quantization import apply_optional_quantization, load_and_quantize_model


# Terminal ANSI Colors
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner():
    banner = f"""
{Colors.CYAN}{Colors.BOLD}╔═══════════════════════════════════════════════════════════════════════════╗
║                    🚀 MINI-VLLM INFERENCE ENGINE CLI                     ║
║          High-Throughput Paged LLM Serving from First Principles          ║
╚═══════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def cmd_inspect(args):
    """Subcommand: Inspect model architecture, parameter weights, and VRAM budget."""
    from benchmarks.inspect_model import inspect_model_memory
    inspect_model_memory(
        model_name=args.model,
        device=args.device,
        trust_remote_code=getattr(args, "trust_remote_code", False)
    )


def cmd_benchmark(args):
    """Subcommand: Run continuous batching and throughput benchmarks."""
    from src.memory.preflight import enforce_preflight_check
    from src.scheduler.continuous_scheduler import ContinuousBatchingEngine

    enforce_preflight_check(
        args.model,
        device=args.device,
        force=getattr(args, "force", False),
        trust_remote_code=getattr(args, "trust_remote_code", False)
    )

    prompts = args.prompt
    if not prompts:
        if not sys.stdin.isatty():
            prompts = [line.strip() for line in sys.stdin if line.strip()]
        else:
            try:
                user_p = input(f"{Colors.BOLD}{Colors.BLUE}Enter benchmark prompt > {Colors.RESET}").strip()
                if user_p:
                    prompts = [user_p]
            except (KeyboardInterrupt, EOFError):
                pass

    if not prompts:
        print(f"{Colors.RED}No prompts provided. Specify --prompt 'your query' or pipe input via stdin.{Colors.RESET}")
        return

    engine = ContinuousBatchingEngine(
        model_name=args.model,
        max_batch_size=args.batch_size,
        device=args.device,
        verbose=args.verbose
    )

    for i, p in enumerate(prompts):
        arrival = i * 2  # stagger arrival steps
        engine.add_request(prompt=p, max_new_tokens=args.max_tokens, arrival_step=arrival)

    engine.run_simulation()


def cmd_engine(args):
    """Subcommand: Run the full unified Mini-vLLM Engine with PagedAttention and physical pool."""
    from src.memory.preflight import enforce_preflight_check
    from src.engine.llm_engine import MiniVLLMEngine, RequestStatus
    from src.engine.sampling import SamplingParams

    enforce_preflight_check(
        args.model,
        device=args.device,
        quantization=args.quant,
        force=getattr(args, "force", False),
        trust_remote_code=getattr(args, "trust_remote_code", False)
    )

    prompts = args.prompt
    if not prompts:
        if not sys.stdin.isatty():
            prompts = [line.strip() for line in sys.stdin if line.strip()]
        else:
            try:
                user_p = input(f"{Colors.BOLD}{Colors.BLUE}Enter prompt > {Colors.RESET}").strip()
                if user_p:
                    prompts = [user_p]
            except (KeyboardInterrupt, EOFError):
                pass

    if not prompts:
        print(f"{Colors.RED}No prompts provided. Pass --prompt 'your query' or pipe input via stdin.{Colors.RESET}")
        return

    sampling_params = SamplingParams(
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        top_k=getattr(args, "top_k", -1),
        min_p=getattr(args, "min_p", 0.0),
        repetition_penalty=getattr(args, "repetition_penalty", 1.0),
        stop_strings=getattr(args, "stop", []) or []
    )

    engine = MiniVLLMEngine(
        model_name=args.model,
        max_batch_size=args.batch_size,
        total_blocks=args.total_blocks,
        vram_budget=getattr(args, "vram_budget", None),
        block_size=args.block_size,
        device=args.device,
        quantization=args.quant,
        kv_cache_dtype=getattr(args, "kv_cache_dtype", "fp16"),
        chunked_prefill=getattr(args, "chunked_prefill", True),
        prefill_chunk_size=getattr(args, "chunk_size", 32),
        enable_prefix_caching=getattr(args, "prefix_cache", True),
        verbose=args.verbose
    )

    for p in prompts:
        engine.add_request(prompt=p, max_tokens=args.max_tokens, arrival_step=0, sampling_params=sampling_params)

    try:
        completed = engine.run()
    except MemoryError as e:
        print(f"\n{Colors.RED}{Colors.BOLD}Error: Admission rejected due to insufficient memory.{Colors.RESET}")
        print(f"  {Colors.RED}{e}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}Hint: Increase --total-blocks or reduce --max-tokens / prompt length.{Colors.RESET}\n")
        sys.exit(1)

    total_tokens = sum(r.num_generated for r in completed)
    total_time = getattr(engine, "last_run_time", 0.0)
    tps = total_tokens / total_time if total_time > 0 else 0.0

    has_failures = False
    print(f"\n{Colors.BOLD}=== Engine Generation Results ==={Colors.RESET}")
    for req in sorted(completed, key=lambda r: r.req_id):
        is_failed = (
            req.status == RequestStatus.FAILED
            or req.error is not None
            or req.stop_reason in ("oom", "cuda_oom", "kv_cache_full", "exhaustion", "abort", "failed")
        )
        if is_failed:
            has_failures = True
            status_tag = f"{Colors.RED}[FAILED]{Colors.RESET}"
        else:
            status_tag = f"{Colors.GREEN}[SUCCESS]{Colors.RESET}"

        print(f"\n{Colors.BOLD}[Request #{req.req_id}] {status_tag}")
        print(f"  {Colors.CYAN}Prompt:{Colors.RESET} {req.prompt}")
        if is_failed:
            fail_reason = req.stop_reason or "error"
            print(f"  {Colors.RED}Failure Reason:{Colors.RESET} {fail_reason}")
            if req.error:
                print(f"  {Colors.RED}Details:{Colors.RESET} {req.error}")
        else:
            text = engine.tokenizer.decode(req.output_token_ids, skip_special_tokens=True)
            print(f"  {Colors.GREEN}Output:{Colors.RESET} {text}")

        if req.finish_step is not None and req.start_step is not None:
            dur_str = f"{req.finish_step - req.start_step + 1} steps"
        else:
            dur_str = "N/A"
        print(f"  {Colors.DIM}Tokens: {req.num_generated} | Duration: {dur_str}{Colors.RESET}")

    successful_count = sum(
        1 for r in completed
        if r.status != RequestStatus.FAILED
        and r.error is None
        and r.stop_reason not in ("oom", "cuda_oom", "kv_cache_full", "exhaustion", "abort", "failed")
    )
    failed_count = len(completed) - successful_count
    print(f"\n{Colors.BOLD}Completed {len(completed)} request(s) ({successful_count} succeeded, {failed_count} failed) | {total_tokens} tokens in {total_time:.2f}s ({tps:.1f} tok/s){Colors.RESET}\n")

    if getattr(args, "metrics", True):
        print(engine.metrics.render_ascii_dashboard(elapsed_seconds=total_time))

    if getattr(args, "export_prometheus", None):
        with open(args.export_prometheus, "w") as f:
            f.write(engine.metrics.export_prometheus_text())
        print(f"\n{Colors.GREEN}✓ Exported Prometheus metrics to: {args.export_prometheus}{Colors.RESET}\n")

    if has_failures:
        sys.exit(1)




def cmd_quantize(args):
    """Subcommand: Run optional weight quantization benchmark (int8/int4)."""
    from src.memory.preflight import enforce_preflight_check
    from src.kernels.quantization import run_quantization_benchmark

    enforce_preflight_check(
        args.model,
        device=args.device,
        quantization=args.quant,
        force=getattr(args, "force", False)
    )

    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            try:
                user_p = input(f"{Colors.BOLD}{Colors.BLUE}Enter prompt for quantization test > {Colors.RESET}").strip()
                if user_p:
                    prompt = user_p
            except (KeyboardInterrupt, EOFError):
                pass

    run_quantization_benchmark(
        selected_quant=args.quant,
        prompt=prompt,
        model_name=args.model,
        device=args.device
    )


def cmd_prefix_cache(args):
    """Subcommand: Run content-addressed prefix caching benchmark."""
    from src.memory.prefix_cache import run_prefix_caching_benchmark
    queries = args.query
    if not queries:
        if not sys.stdin.isatty():
            queries = [line.strip() for line in sys.stdin if line.strip()]
        else:
            try:
                user_q = input(f"{Colors.BOLD}{Colors.BLUE}Enter query for prefix cache test > {Colors.RESET}").strip()
                if user_q:
                    queries = [user_q]
            except (KeyboardInterrupt, EOFError):
                pass

    run_prefix_caching_benchmark(
        model_name=args.model,
        device=args.device,
        system_prompt=args.system_prompt,
        user_queries=queries,
        verbose=args.verbose
    )


def cmd_preempt(args):
    """Subcommand: Run memory starvation resilience and preemption benchmark."""
    from src.memory.preemption import StarvationResilientEngine
    prompts = args.prompt
    if not prompts and not sys.stdin.isatty():
        prompts = [line.strip() for line in sys.stdin if line.strip()]

    if not prompts:
        prompts = [
            "Primary high-priority analytical workload",
            "Secondary lower-priority background task"
        ]

    engine = StarvationResilientEngine(
        total_blocks=args.total_blocks,
        block_size=args.block_size,
        verbose=args.verbose
    )
    for i, p in enumerate(prompts):
        engine.add_request(prompt=p, target_tokens=args.tokens, arrival_step=0, priority=i)

    engine.run_simulation()


def cmd_chunked_prefill(args):
    """Subcommand: Benchmark Chunked Prefill vs Monolithic Prefill latency jitter."""
    from src.memory.preflight import enforce_preflight_check
    from src.scheduler.chunked_prefill import benchmark_monolithic_vs_chunked

    enforce_preflight_check(
        args.model,
        device=args.device,
        force=getattr(args, "force", False)
    )

    benchmark_monolithic_vs_chunked(
        model_name=args.model,
        chunk_size=args.chunk_size,
        device=args.device,
        prompt_short=args.prompt_short,
        prompt_long=args.prompt_long,
        verbose=args.verbose
    )


def cmd_list_models(args):
    """Subcommand/Flag: List all locally installed and cached Hugging Face models."""
    from src.memory.preflight import list_installed_models, print_installed_models_report
    models = list_installed_models()
    print_installed_models_report(models, verbose=getattr(args, "verbose", False))


def cmd_check(args):
    """Subcommand: Inspect whether a model will fit into system VRAM/RAM before downloading."""
    if getattr(args, "list_models", False):
        cmd_list_models(args)
        return
    from src.memory.preflight import enforce_preflight_check
    enforce_preflight_check(
        args.model,
        device=args.device,
        quantization=getattr(args, "quant", "none"),
        force=getattr(args, "force", False)
    )


def cmd_metrics(args):
    """Subcommand: Run inference workload and display live Prometheus telemetry dashboard."""
    from src.memory.preflight import enforce_preflight_check
    from src.engine.llm_engine import MiniVLLMEngine

    enforce_preflight_check(
        args.model,
        device=args.device,
        force=getattr(args, "force", False)
    )

    prompts = args.prompt or [
        "Explain quantum computing in one sentence.",
        "Explain what an operating system kernel does in one sentence.",
        "Write a python function to compute fibonacci numbers."
    ]

    print_banner()
    print(f"{Colors.BOLD}Starting Telemetry & Observability Benchmark...{Colors.RESET}")
    print(f"Model: {args.model} | Device: {args.device} | Prompts: {len(prompts)}")

    engine = MiniVLLMEngine(
        model_name=args.model,
        max_batch_size=args.batch_size,
        device=args.device,
        kv_cache_dtype=getattr(args, "kv_cache_dtype", "fp16"),
        chunked_prefill=True,
        enable_prefix_caching=True,
        verbose=args.verbose
    )

    if args.serve:
        port = args.port
        success = engine.metrics.start_http_server(port=port)
        if success:
            print(f"\n{Colors.GREEN}⚡ Prometheus scraping server active at http://localhost:{port}/metrics{Colors.RESET}\n")

    for p in prompts:
        engine.add_request(prompt=p, max_tokens=args.max_tokens, arrival_step=0)

    completed = engine.run()
    total_time = getattr(engine, "last_run_time", 0.0)

    print(f"\n{Colors.GREEN}{Colors.BOLD}=== Engine Workload Completed ==={Colors.RESET}")
    print(engine.metrics.render_ascii_dashboard(elapsed_seconds=total_time))

    if args.export:
        with open(args.export, "w") as f:
            f.write(engine.metrics.export_prometheus_text())
        print(f"\n{Colors.GREEN}✓ Exported Prometheus metrics to: {args.export}{Colors.RESET}")
    else:
        print(f"\n{Colors.CYAN}{Colors.BOLD}--- Prometheus Exposition Sample (/metrics) ---{Colors.RESET}")
        prom_sample = "\n".join([line for line in engine.metrics.export_prometheus_text().splitlines() if not line.startswith("#")][:15])
        print(f"{Colors.DIM}{prom_sample}{Colors.RESET}")

    if args.serve:
        print(f"\n{Colors.YELLOW}Prometheus server running on http://localhost:{args.port}/metrics. Press Ctrl+C to exit.{Colors.RESET}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down metrics server.")


def cmd_stress_test(args):
    """Subcommand: Run advanced head-to-head stress-test benchmark (Baseline vs. Mini-vLLM)."""
    from src.memory.preflight import enforce_preflight_check
    from benchmarks.benchmark_serving import run_comprehensive_benchmark

    enforce_preflight_check(
        args.model,
        device=args.device,
        force=getattr(args, "force", False)
    )

    run_comprehensive_benchmark(
        model_name=args.model,
        device_str=args.device,
        num_requests=args.requests,
        batch_size=args.batch_size,
        kv_cache_dtype=args.kv_cache_dtype
    )


def cmd_perf_benchmark(args):
    """Subcommand: Run rigorous performance benchmark across concurrency levels and workloads."""
    from benchmarks.run_benchmark import run_benchmark_suite
    concurrency_list = [int(c.strip()) for c in args.concurrencies.split(",")] if getattr(args, "concurrencies", None) else None
    workload_list = [w.strip() for w in args.workloads.split(",")] if getattr(args, "workloads", None) else None
    engine_list = [e.strip() for e in args.engines.split(",")] if getattr(args, "engines", None) else None

    run_benchmark_suite(
        model_name=args.model,
        device_str=args.device,
        profile=args.profile,
        concurrencies=concurrency_list,
        engines=engine_list,
        workloads=workload_list,
        num_requests_per_concurrency=getattr(args, "num_requests", None),
        repetitions=getattr(args, "repetitions", 1),
        output_dir=getattr(args, "output_dir", None)
    )


def cmd_test(args):
    """Subcommand: Run complete system regression test suite validating all subsystems."""
    import subprocess
    cmd = [sys.executable, "tests/test_all_features.py"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def cmd_serve(args):
    """Subcommand: Launch OpenAI-compatible streaming REST API server."""
    import uvicorn
    from src.memory.preflight import enforce_preflight_check
    from src.server.app import create_app

    enforce_preflight_check(
        args.model,
        device=args.device,
        quantization=args.quant,
        force=getattr(args, "force", False)
    )

    raw_engine = getattr(args, "engine", "hf")
    serving_engine_type = raw_engine.lower() if isinstance(raw_engine, str) else "hf"
    if serving_engine_type in ("mini-vllm", "minivllm"):
        serving_engine_type = "localstrm"

    if serving_engine_type == "localstrm":
        total_blocks = getattr(args, "total_blocks", None)
        vram_budget = getattr(args, "vram_budget", None)
        if total_blocks is not None and vram_budget is not None:
            print(f"{Colors.RED}Error: Conflicting KV-cache configuration: both --total-blocks ({total_blocks}) "
                  f"and --vram-budget ({vram_budget!r}) were provided. "
                  f"Specify --total-blocks OR --vram-budget, not both.{Colors.RESET}")
            sys.exit(2)
    elif serving_engine_type == "hf":
        total_blocks = getattr(args, "total_blocks", None)
        vram_budget = getattr(args, "vram_budget", None)
        kv_dtype = getattr(args, "kv_cache_dtype", "fp16")
        if total_blocks is not None or vram_budget is not None or kv_dtype != "fp16":
            print(f"{Colors.YELLOW}Warning: Options --total-blocks, --vram-budget, and --kv-cache-dtype are only applicable "
                  f"when --engine localstrm is selected. Ignoring for Hugging Face engine.{Colors.RESET}")
    else:
        print(f"{Colors.RED}Error: Unsupported engine '{args.engine}'. Supported engines are 'localstrm' (or legacy alias 'mini-vllm') and 'hf'.{Colors.RESET}")
        sys.exit(2)

    print_banner()
    print(f"{Colors.BOLD}{Colors.CYAN}Starting OpenAI-Compatible Inference Server...{Colors.RESET}")
    print(f"  Engine:       {Colors.GREEN}{'LocalStrm (PagedAttention + Continuous Batching)' if serving_engine_type == 'localstrm' else 'Hugging Face (Serial)'}{Colors.RESET}")
    print(f"  Model:        {Colors.GREEN}{args.model}{Colors.RESET}")
    print(f"  Device:       {Colors.GREEN}{args.device}{Colors.RESET}")
    print(f"  Quantization: {Colors.GREEN}{args.quant}{Colors.RESET}")
    print(f"  Host:         {Colors.GREEN}{args.host}{Colors.RESET}")
    print(f"  Port:         {Colors.GREEN}{args.port}{Colors.RESET}")
    if serving_engine_type == "localstrm":
        print(f"  KV Cache:     {Colors.GREEN}{getattr(args, 'kv_cache_dtype', 'fp16').upper()}{Colors.RESET}")
        if getattr(args, 'vram_budget', None):
            print(f"  VRAM Budget:  {Colors.GREEN}{args.vram_budget}{Colors.RESET}")
        elif getattr(args, 'total_blocks', None):
            print(f"  KV Blocks:    {Colors.GREEN}{args.total_blocks}{Colors.RESET}")
        print(f"  Batch Size:   {Colors.GREEN}{getattr(args, 'batch_size', 4)}{Colors.RESET}")

    print(f"\n{Colors.DIM}OpenAPI Docs:   http://{args.host}:{args.port}/docs{Colors.RESET}")
    print(f"{Colors.DIM}Chat Endpoint:  http://{args.host}:{args.port}/v1/chat/completions{Colors.RESET}")
    print(f"{Colors.DIM}Models List:    http://{args.host}:{args.port}/v1/models{Colors.RESET}")
    print(f"{Colors.DIM}Metrics:        http://{args.host}:{args.port}/metrics{Colors.RESET}\n")

    if serving_engine_type == "localstrm":
        from src.engine.llm_engine import MiniVLLMEngine
        from src.server.serving_adapter import AsyncServingAdapter

        core_engine = MiniVLLMEngine(
            model_name=args.model,
            max_batch_size=getattr(args, "batch_size", 4),
            total_blocks=getattr(args, "total_blocks", None),
            vram_budget=getattr(args, "vram_budget", None),
            device=args.device,
            quantization=args.quant,
            kv_cache_dtype=getattr(args, "kv_cache_dtype", "fp16"),
            verbose=args.verbose,
            trust_remote_code=getattr(args, "trust_remote_code", False)
        )
        engine = AsyncServingAdapter(core_engine)
    else:
        from src.server.serving_engine import AsyncServingEngine

        engine = AsyncServingEngine(
            model_name=args.model,
            device=args.device,
            quantization=args.quant,
            force=getattr(args, "force", False),
            trust_remote_code=getattr(args, "trust_remote_code", False)
        )

    app = create_app(engine)

    if hasattr(engine, "shutdown"):
        @app.on_event("shutdown")
        async def _shutdown_engine():
            await engine.shutdown()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if hasattr(engine, "shutdown_sync") and not getattr(engine, "_is_shutting_down", False):
            try:
                engine.shutdown_sync()
            except Exception:
                pass
        elif hasattr(engine, "shutdown") and not getattr(engine, "_is_shutting_down", False):
            import asyncio
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    asyncio.create_task(engine.shutdown())
                else:
                    asyncio.run(engine.shutdown())
            except Exception:
                pass



def get_stopping_criteria_token_ids(tokenizer: Any, model: Optional[Any] = None) -> Set[int]:
    """
    Extracts all valid end-of-sequence / stop token IDs for generation.
    Safely verifies that candidate special tokens actually exist in the tokenizer
    vocabulary and cannot accidentally alias unknown-token IDs, pad IDs, or unmapped tokens.
    """
    stop_token_ids = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_token_ids.update(tokenizer.eos_token_id)
        else:
            stop_token_ids.add(tokenizer.eos_token_id)

    gen_config = getattr(model, "generation_config", None) if model is not None else None
    if gen_config is not None and getattr(gen_config, "eos_token_id", None) is not None:
        cfg_eos = gen_config.eos_token_id
        if isinstance(cfg_eos, list):
            stop_token_ids.update(cfg_eos)
        elif isinstance(cfg_eos, int):
            stop_token_ids.add(cfg_eos)

    unk_id = getattr(tokenizer, "unk_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None

    for end_tok in ("<|im_end|>", "<|eot_id|>", "</s>"):
        try:
            if vocab is not None:
                if end_tok in vocab:
                    tid = vocab[end_tok]
                    if isinstance(tid, int) and tid != unk_id and tid != pad_id and tid >= 0:
                        stop_token_ids.add(tid)
            else:
                tid = tokenizer.convert_tokens_to_ids(end_tok)
                if isinstance(tid, int) and tid != unk_id and tid != pad_id and tid >= 0:
                    decoded = tokenizer.decode([tid], skip_special_tokens=False).strip()
                    if decoded == end_tok:
                        stop_token_ids.add(tid)
        except Exception:
            pass

    return stop_token_ids


def format_chat_prompt(conversation_history: List[Dict[str, str]], tokenizer: Any) -> str:
    """Formats multi-turn conversation history using tokenizer chat template or standard multi-turn fallback."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)

    parts = []
    for msg in conversation_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant: ")
    return "\n".join(parts)


def _run_chat_hf(args):
    """Interactive streaming chat REPL powered by Hugging Face sequential decode."""
    from src.memory.preflight import enforce_preflight_check

    device_str = args.device or get_default_device()
    device = torch.device(device_str)

    trust_remote = getattr(args, "trust_remote_code", False)
    enforce_preflight_check(
        args.model,
        device=device_str,
        quantization=args.quant,
        force=getattr(args, "force", False),
        trust_remote_code=trust_remote
    )

    print_banner()

    print(f"{Colors.BOLD}Device:{Colors.RESET}        {device.type.upper()}")
    print(f"{Colors.BOLD}Model:{Colors.RESET}         {args.model}")
    print(f"{Colors.BOLD}Engine:{Colors.RESET}        Hugging Face (Serial)")
    print(f"{Colors.BOLD}Quantization:{Colors.RESET}  {args.quant.upper()}")
    print(f"{Colors.BOLD}Max Tokens:{Colors.RESET}    {args.max_tokens}")
    print(f"{Colors.BOLD}Temperature:{Colors.RESET}   {args.temperature}")
    print(f"{Colors.DIM}Commands: /clear (reset history), /stats (memory info), /exit (quit){Colors.RESET}\n")

    # 1. Load Tokenizer & Model (with CPU-staged quantization if enabled)
    print(f"{Colors.YELLOW}Loading model weights...{Colors.RESET}", end="", flush=True)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust_remote)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model, orig_bytes, quant_bytes = load_and_quantize_model(
        args.model,
        device=device,
        quantization=args.quant,
        dtype=dtype,
        trust_remote_code=trust_remote
    )
    load_time = time.perf_counter() - t0
    print(f" {Colors.GREEN}Ready! ({load_time:.2f}s){Colors.RESET}")

    is_chat_model = any(k in args.model.lower() for k in ("instruct", "chat", "it", "conversational"))
    if not is_chat_model:
        print(f" {Colors.YELLOW}[Notice] '{args.model}' appears to be a base completion model, not instruction-tuned (e.g. '*-Instruct'). Interactive chat responses may produce code/text completion rather than conversation.{Colors.RESET}")

    if args.quant != "none":
        savings = (1.0 - (quant_bytes / orig_bytes)) * 100 if orig_bytes > 0 else 0.0
        print(f" {Colors.GREEN}Applied {args.quant.upper()} quantization ({savings:.1f}% VRAM freed){Colors.RESET}")

    # Build comprehensive set of stop token IDs (handles single int, list of ints, and verified template end tokens)
    stop_token_ids = get_stopping_criteria_token_ids(tokenizer, model)

    print(f"\n{Colors.GREEN}{Colors.BOLD}● Engine online! Type your message below:{Colors.RESET}\n")

    conversation_history = []
    if args.system:
        conversation_history.append({"role": "system", "content": args.system})

    # Interactive Loop
    while True:
        try:
            user_input = input(f"{Colors.BOLD}{Colors.BLUE}You > {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Mini-vLLM CLI. Goodbye!")
            break

        if not user_input:
            continue

        # Handle Slash Commands
        if user_input.lower() in ["/exit", "/quit", ":q"]:
            print("Exiting Mini-vLLM CLI. Goodbye!")
            break
        elif user_input.lower() == "/clear":
            conversation_history.clear()
            if args.system:
                conversation_history.append({"role": "system", "content": args.system})
            print(f"{Colors.DIM}Conversation history cleared.{Colors.RESET}")
            continue
        elif user_input.lower() == "/stats":
            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / (1024**2)
                res = torch.cuda.memory_reserved() / (1024**2)
                print(f"{Colors.DIM}[VRAM Stats] Allocated: {alloc:.1f} MB | Reserved: {res:.1f} MB{Colors.RESET}")
            else:
                print(f"{Colors.DIM}[Engine Stats] Running on CPU{Colors.RESET}")
            continue

        # Append to history
        conversation_history.append({"role": "user", "content": user_input})

        # Format input using model chat template if available, else standard multi-turn fallback
        prompt_text = format_chat_prompt(conversation_history, tokenizer)

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        # ==========================================
        # STREAMING INFERENCE WITH LIVE METRICS
        # ==========================================
        print(f"\n{Colors.BOLD}{Colors.GREEN}Mini-vLLM > {Colors.RESET}", end="", flush=True)

        t_prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
        t_prefill_end = time.perf_counter()
        ttft = (t_prefill_end - t_prefill_start) * 1000  # ms

        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        first_token_val = next_token_id.item()

        generated_ids = [first_token_val]
        decode_times = []
        curr_token = next_token_id

        if first_token_val not in stop_token_ids:
            first_token_str = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            sys.stdout.write(first_token_str)
            sys.stdout.flush()

            # Decode token-by-token with streaming
            for _ in range(1, args.max_tokens):
                t_step_start = time.perf_counter()
                with torch.no_grad():
                    outputs = model(curr_token, past_key_values=past_key_values, use_cache=True)
                t_step_end = time.perf_counter()
                decode_times.append((t_step_end - t_step_start) * 1000)

                past_key_values = outputs.past_key_values

                # Sampling with temperature
                if args.temperature > 0.0:
                    logits = outputs.logits[:, -1, :] / args.temperature
                    probs = F.softmax(logits, dim=-1)
                    curr_token = torch.multinomial(probs, num_samples=1)
                else:
                    curr_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

                token_id = curr_token.item()
                generated_ids.append(token_id)

                if token_id in stop_token_ids:
                    break

                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                sys.stdout.write(token_str)
                sys.stdout.flush()

        clean_ids = [t for t in generated_ids if t not in stop_token_ids]
        assistant_reply = tokenizer.decode(clean_ids, skip_special_tokens=True)
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        # Calculate metrics
        avg_tpot = sum(decode_times) / len(decode_times) if decode_times else 0.0
        total_gen_time = sum(decode_times) / 1000
        throughput = len(decode_times) / total_gen_time if total_gen_time > 0 else 0.0

        # Print live metrics pill
        print(f"\n\n{Colors.DIM}┌──────────────────────────────────────────────────────────────┐")
        print(f"│ ⚡ TTFT: {ttft:6.1f} ms  │  Avg TPOT: {avg_tpot:5.1f} ms/tok  │  Speed: {throughput:5.1f} tok/s  │")
        print(f"│ 📊 Tokens: {len(generated_ids):4d}   │  Prompt: {prompt_len:4d} tokens   │  Quant: {args.quant.upper():4s}       │")
        print(f"└──────────────────────────────────────────────────────────────┘{Colors.RESET}\n")


def _run_chat_localstrm(args):
    """Interactive streaming chat REPL powered by LocalStrm MiniVLLMEngine (PagedAttention)."""
    from src.memory.preflight import enforce_preflight_check
    from src.engine.llm_engine import MiniVLLMEngine, RequestStatus
    from src.engine.sampling import SamplingParams

    device_str = args.device or get_default_device()
    device = torch.device(device_str)

    trust_remote = getattr(args, "trust_remote_code", False)
    enforce_preflight_check(
        args.model,
        device=device_str,
        quantization=getattr(args, "quant", "none"),
        force=getattr(args, "force", False),
        trust_remote_code=trust_remote
    )

    print_banner()

    print(f"{Colors.BOLD}Device:{Colors.RESET}        {device.type.upper()}")
    print(f"{Colors.BOLD}Model:{Colors.RESET}         {args.model}")
    print(f"{Colors.BOLD}Engine:{Colors.RESET}        LocalStrm (PagedAttention + Continuous Batching)")
    print(f"{Colors.BOLD}Quantization:{Colors.RESET}  {args.quant.upper()}")
    print(f"{Colors.BOLD}Max Tokens:{Colors.RESET}    {args.max_tokens}")
    print(f"{Colors.BOLD}Temperature:{Colors.RESET}   {args.temperature}")
    kv_dtype = getattr(args, "kv_cache_dtype", "fp16")
    print(f"{Colors.BOLD}KV Cache:{Colors.RESET}      {kv_dtype.upper()}")
    if getattr(args, "vram_budget", None):
        print(f"{Colors.BOLD}VRAM Budget:{Colors.RESET}   {args.vram_budget}")
    elif getattr(args, "total_blocks", None):
        print(f"{Colors.BOLD}KV Blocks:{Colors.RESET}     {args.total_blocks}")
    print(f"{Colors.DIM}Commands: /clear (reset history), /stats (memory info), /exit (quit){Colors.RESET}\n")

    print(f"{Colors.YELLOW}Loading LocalStrm engine...{Colors.RESET}", end="", flush=True)
    t0 = time.perf_counter()

    # Note: MiniVLLMEngine validates model architecture and fails closed with a clear
    # ValueError if the architecture is incompatible with PagedAttention / standard transformer blocks.
    engine = MiniVLLMEngine(
        model_name=args.model,
        max_batch_size=getattr(args, "batch_size", 4),
        total_blocks=getattr(args, "total_blocks", None),
        vram_budget=getattr(args, "vram_budget", None),
        device=device_str,
        quantization=getattr(args, "quant", "none"),
        kv_cache_dtype=getattr(args, "kv_cache_dtype", "fp16"),
        verbose=getattr(args, "verbose", False),
        trust_remote_code=trust_remote
    )
    load_time = time.perf_counter() - t0
    print(f" {Colors.GREEN}Ready! ({load_time:.2f}s){Colors.RESET}")

    tokenizer = engine.tokenizer
    model = engine.model

    is_chat_model = any(k in args.model.lower() for k in ("instruct", "chat", "it", "conversational"))
    if not is_chat_model:
        print(f" {Colors.YELLOW}[Notice] '{args.model}' appears to be a base completion model, not instruction-tuned (e.g. '*-Instruct'). Interactive chat responses may produce code/text completion rather than conversation.{Colors.RESET}")

    stop_token_ids = get_stopping_criteria_token_ids(tokenizer, model)

    print(f"\n{Colors.GREEN}{Colors.BOLD}● Engine online! Type your message below:{Colors.RESET}\n")

    conversation_history = []
    if getattr(args, "system", None):
        conversation_history.append({"role": "system", "content": args.system})

    while True:
        try:
            user_input = input(f"{Colors.BOLD}{Colors.BLUE}You > {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting LocalStrm CLI. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "/quit", ":q"]:
            print("Exiting LocalStrm CLI. Goodbye!")
            break
        elif user_input.lower() == "/clear":
            conversation_history.clear()
            if getattr(args, "system", None):
                conversation_history.append({"role": "system", "content": args.system})
            print(f"{Colors.DIM}Conversation history cleared.{Colors.RESET}")
            continue
        elif user_input.lower() == "/stats":
            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / (1024**2)
                res = torch.cuda.memory_reserved() / (1024**2)
                print(f"{Colors.DIM}[VRAM Stats] Allocated: {alloc:.1f} MB | Reserved: {res:.1f} MB{Colors.RESET}")
            else:
                print(f"{Colors.DIM}[Engine Stats] Running on CPU{Colors.RESET}")
            free_b = engine.num_free_blocks
            tot_b = engine.total_blocks
            print(f"{Colors.DIM}[KV Cache Pool] Free blocks: {free_b}/{tot_b} ({(free_b/tot_b)*100:.1f}% free){Colors.RESET}")
            continue

        conversation_history.append({"role": "user", "content": user_input})

        prompt_text = format_chat_prompt(conversation_history, tokenizer)

        sampling_params = SamplingParams(
            temperature=getattr(args, "temperature", 0.7),
            top_p=getattr(args, "top_p", 1.0),
            top_k=getattr(args, "top_k", -1),
            min_p=getattr(args, "min_p", 0.0),
            repetition_penalty=getattr(args, "repetition_penalty", 1.0),
            stop_token_ids=list(stop_token_ids),
        )

        req = engine.add_request(
            prompt=prompt_text,
            max_tokens=args.max_tokens,
            arrival_step=engine.current_step,
            sampling_params=sampling_params
        )

        prompt_len = len(req.prompt_token_ids)
        print(f"\n{Colors.BOLD}{Colors.GREEN}LocalStrm > {Colors.RESET}", end="", flush=True)

        num_emitted = 0
        decode_times = []
        t_step_start = time.perf_counter()

        while True:
            active = engine.step()
            t_step_end = time.perf_counter()
            step_duration_ms = (t_step_end - t_step_start) * 1000

            if len(req.output_token_ids) > num_emitted:
                if num_emitted > 0:
                    decode_times.append(step_duration_ms)
                new_tokens = req.output_token_ids[num_emitted:]
                for tok_id in new_tokens:
                    if tok_id not in stop_token_ids:
                        token_str = tokenizer.decode([tok_id], skip_special_tokens=True)
                        sys.stdout.write(token_str)
                        sys.stdout.flush()
                num_emitted = len(req.output_token_ids)

            t_step_start = time.perf_counter()

            if req.status in (RequestStatus.FINISHED, RequestStatus.FAILED):
                break
            if not active:
                break

        if req.status == RequestStatus.FAILED:
            print(f"\n{Colors.RED}Generation failed: {req.error or req.stop_reason}{Colors.RESET}")

        clean_ids = [t for t in req.output_token_ids if t not in stop_token_ids]
        assistant_reply = tokenizer.decode(clean_ids, skip_special_tokens=True)
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        ttft = req.ttft_ms if req.ttft_ms is not None else 0.0
        avg_tpot = sum(decode_times) / len(decode_times) if decode_times else 0.0
        total_gen_time = sum(decode_times) / 1000
        throughput = len(decode_times) / total_gen_time if total_gen_time > 0 else 0.0

        print(f"\n\n{Colors.DIM}┌──────────────────────────────────────────────────────────────┐")
        print(f"│ ⚡ TTFT: {ttft:6.1f} ms  │  Avg TPOT: {avg_tpot:5.1f} ms/tok  │  Speed: {throughput:5.1f} tok/s  │")
        print(f"│ 📊 Tokens: {len(req.output_token_ids):4d}   │  Prompt: {prompt_len:4d} tokens   │  Quant: {getattr(args, 'quant', 'none').upper():4s}       │")
        print(f"└──────────────────────────────────────────────────────────────┘{Colors.RESET}\n")


def cmd_chat(args):
    """Subcommand: Interactive streaming chat REPL with selectable LocalStrm or Hugging Face engine."""
    engine_type = (getattr(args, "engine", "hf") or "hf").lower()
    if engine_type in ("mini-vllm", "minivllm"):
        engine_type = "localstrm"

    if engine_type == "localstrm":
        total_blocks = getattr(args, "total_blocks", None)
        vram_budget = getattr(args, "vram_budget", None)
        if total_blocks is not None and vram_budget is not None:
            print(f"{Colors.RED}Error: Conflicting KV-cache options: both --total-blocks ({total_blocks}) "
                  f"and --vram-budget ({vram_budget!r}) were provided. "
                  f"Specify --total-blocks OR --vram-budget, not both.{Colors.RESET}")
            sys.exit(2)
        _run_chat_localstrm(args)
    elif engine_type == "hf":
        total_blocks = getattr(args, "total_blocks", None)
        vram_budget = getattr(args, "vram_budget", None)
        kv_dtype = getattr(args, "kv_cache_dtype", "fp16")
        if total_blocks is not None or vram_budget is not None or kv_dtype != "fp16":
            print(f"{Colors.YELLOW}Warning: Options --total-blocks, --vram-budget, and --kv-cache-dtype are only applicable "
                  f"when --engine localstrm is selected. Ignoring for Hugging Face engine.{Colors.RESET}")
        _run_chat_hf(args)
    else:
        print(f"{Colors.RED}Error: Unsupported engine '{args.engine}'. Supported engines are 'localstrm' (or legacy alias 'mini-vllm') and 'hf'.{Colors.RESET}")
        sys.exit(2)


def build_parent_parser() -> argparse.ArgumentParser:
    """Shared parent parser providing common configuration arguments across all subcommands."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--model",
        type=str,
        default=get_default_model(),
        help="HuggingFace model repository ID or local weight path (default: $MODEL_ID or Qwen/Qwen2.5-0.5B)"
    )
    parent.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        choices=["cuda", "cpu"],
        help="Execution compute device (default: auto-detected cuda or cpu)"
    )
    parent.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging during inference"
    )
    parent.add_argument(
        "--force",
        action="store_true",
        help="Bypass pre-flight system memory and hardware capacity check"
    )
    parent.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow executing custom modeling code defined in HuggingFace model repositories (default: False)"
    )
    parent.add_argument(
        "-l", "--list-models", "--models",
        action="store_true",
        dest="list_models",
        help="List all locally installed and cached Hugging Face models and exit"
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    """Builds and returns the top-level argument parser for Mini-vLLM CLI."""
    parent_parser = build_parent_parser()

    parser = argparse.ArgumentParser(
        description="Mini-vLLM: Production-Structured LLM Inference Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent_parser]
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available engine subcommands")

    # Subcommand: chat
    parser_chat = subparsers.add_parser("chat", parents=[parent_parser], help="Start an interactive streaming chat session")
    parser_chat.add_argument("--quant", type=str, choices=["none", "int8", "int4", "hybrid_int8"], default="none", help="Optional quantization precision")
    parser_chat.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens per response")
    parser_chat.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 = greedy)")
    parser_chat.add_argument("--system", type=str, default=None, help="Custom system prompt")
    parser_chat.add_argument(
        "--engine",
        type=str,
        choices=["hf", "localstrm", "mini-vllm"],
        default="hf",
        help="Chat engine backend: 'localstrm' (LocalStrm PagedAttention continuous batching engine; legacy alias: 'mini-vllm') or 'hf' (default: Hugging Face serial engine)"
    )
    parser_chat.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum continuous batch capacity for LocalStrm engine (default: 4)"
    )
    parser_chat.add_argument(
        "--total-blocks",
        type=int,
        default=None,
        help="Total physical KV cache blocks in pool for LocalStrm engine"
    )
    parser_chat.add_argument(
        "--vram-budget",
        type=str,
        default=None,
        help="Total engine VRAM operational budget ceiling (e.g. '2GiB', '1400MiB') for LocalStrm engine"
    )
    parser_chat.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["fp16", "int8"],
        default="fp16",
        help="KV cache storage dtype for LocalStrm engine (default: fp16, int8 cuts KV memory by 50%%)"
    )

    # Subcommand: engine
    parser_engine = subparsers.add_parser("engine", parents=[parent_parser], help="Run unified PagedAttention + Continuous Batching engine")
    parser_engine.add_argument("--prompt", type=str, action="append", help="Prompt text to generate (can specify multiple times)")
    parser_engine.add_argument("--max-tokens", type=int, default=32, help="Max tokens per response (default: 32)")
    parser_engine.add_argument("--batch-size", type=int, default=4, help="Maximum continuous batch capacity")
    parser_engine.add_argument("--total-blocks", type=int, default=None, help="Total physical KV cache blocks in pool")
    parser_engine.add_argument("--vram-budget", type=str, default=None, help="Total engine VRAM operational budget ceiling (e.g. '2GiB', '1400MiB'). Sizes KV-cache capacity after deducting model weights, activation workspace, runtime overhead, and safety margin. Note: acts as a pre-allocation sizing ceiling, not a hard CUDA allocator trap.")
    parser_engine.add_argument("--block-size", type=int, default=16, help="Tokens per physical block")
    parser_engine.add_argument("--quant", type=str, choices=["none", "int8", "int4", "hybrid_int8"], default="none", help="Optional weight quantization")
    parser_engine.add_argument("--kv-cache-dtype", type=str, choices=["fp16", "int8"], default="fp16", help="KV cache storage dtype (default: fp16, int8 cuts KV memory by 50%%)")
    parser_engine.add_argument("--prefix-cache", action="store_true", default=True, help="Enable automatic chained block-hash prefix caching (default: enabled)")
    parser_engine.add_argument("--no-prefix-cache", action="store_false", dest="prefix_cache", help="Disable automatic chained block-hash prefix caching")
    parser_engine.add_argument("--chunked-prefill", action="store_true", default=True, help="Enable chunked prefill (default: enabled)")
    parser_engine.add_argument("--no-chunked-prefill", action="store_false", dest="chunked_prefill", help="Disable chunked prefill (use monolithic)")
    parser_engine.add_argument("--chunk-size", type=int, default=32, help="Tokens per prefill chunk (default: 32)")
    parser_engine.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 greedy)")
    parser_engine.add_argument("--top-p", type=float, default=1.0, help="Top-P nucleus sampling (default: 1.0)")
    parser_engine.add_argument("--top-k", type=int, default=-1, help="Top-K sampling (default: -1 disabled)")
    parser_engine.add_argument("--min-p", type=float, default=0.0, help="Min-P dynamic probability threshold (default: 0.0)")
    parser_engine.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)")
    parser_engine.add_argument("--stop", type=str, action="append", default=[], help="Stop strings to terminate generation early")
    parser_engine.add_argument("--metrics", action="store_true", default=True, help="Display telemetry dashboard (default: enabled)")
    parser_engine.add_argument("--no-metrics", action="store_false", dest="metrics", help="Hide telemetry dashboard")
    parser_engine.add_argument("--export-prometheus", type=str, default=None, help="File path to save Prometheus plaintext metrics")

    # Subcommand: benchmark
    parser_bench = subparsers.add_parser("benchmark", parents=[parent_parser], help="Run continuous batching and throughput benchmarks")
    parser_bench.add_argument("--prompt", type=str, action="append", help="Custom prompt for benchmark (can specify multiple times)")
    parser_bench.add_argument("--max-tokens", type=int, default=16, help="Maximum tokens per benchmark request")
    parser_bench.add_argument("--batch-size", type=int, default=4, help="Maximum continuous batch capacity")

    # Subcommand: inspect
    subparsers.add_parser("inspect", parents=[parent_parser], help="Inspect model parameters and VRAM budget")

    # Subcommand: quantize
    parser_quant = subparsers.add_parser("quantize", parents=[parent_parser], help="Benchmark optional weight quantization (INT8/INT4/HYBRID_INT8)")
    parser_quant.add_argument("--prompt", type=str, default=None, help="Input prompt to test quantized generation")
    parser_quant.add_argument("--quant", type=str, choices=["none", "int8", "int4", "hybrid_int8"], default="int8", help="Quantization mode (default: int8)")

    # Subcommand: prefix-cache
    parser_pc = subparsers.add_parser("prefix-cache", parents=[parent_parser], help="Benchmark automatic chained block-hash prefix caching & deduplication")
    parser_pc.add_argument("--system-prompt", type=str, default=None, help="Shared system prefix")
    parser_pc.add_argument("--query", type=str, action="append", help="User queries sharing prefix (can specify multiple times)")

    # Subcommand: preempt
    parser_preempt = subparsers.add_parser("preempt", parents=[parent_parser], help="Benchmark memory starvation preemption & recovery")
    parser_preempt.add_argument("--prompt", type=str, action="append", help="Request prompt (can specify multiple times)")
    parser_preempt.add_argument("--tokens", type=int, default=12, help="Tokens per sequence (default: 12)")
    parser_preempt.add_argument("--total-blocks", type=int, default=5, help="Total physical blocks (default: 5)")
    parser_preempt.add_argument("--block-size", type=int, default=4, help="Tokens per block (default: 4)")

    # Subcommand: check (preflight)
    parser_check = subparsers.add_parser("check", parents=[parent_parser], help="Verify if a model will fit in system VRAM/RAM before downloading")
    parser_check.add_argument("--quant", type=str, choices=["none", "int8", "int4", "hybrid_int8"], default="none", help="Target quantization mode to test")

    # Subcommand: chunked-prefill
    parser_cp = subparsers.add_parser("chunked-prefill", parents=[parent_parser], help="Benchmark Chunked Prefill vs Monolithic Prefill latency jitter")
    parser_cp.add_argument("--chunk-size", type=int, default=32, help="Chunk size in tokens (default: 32)")
    parser_cp.add_argument("--prompt-short", type=str, default=None, help="Short active decode stream prompt")
    parser_cp.add_argument("--prompt-long", type=str, default=None, help="Long prefill arrival prompt")

    # Subcommand: metrics
    parser_met = subparsers.add_parser("metrics", parents=[parent_parser], help="Run engine workload with live Prometheus telemetry & dashboard")
    parser_met.add_argument("--prompt", type=str, action="append", help="Prompt text to process (can specify multiple times)")
    parser_met.add_argument("--max-tokens", type=int, default=24, help="Max tokens per response (default: 24)")
    parser_met.add_argument("--batch-size", type=int, default=4, help="Maximum continuous batch capacity")
    parser_met.add_argument("--kv-cache-dtype", type=str, choices=["fp16", "int8"], default="fp16", help="KV cache storage dtype")
    parser_met.add_argument("--export", type=str, default=None, help="File path to save Prometheus plaintext metrics")
    parser_met.add_argument("--serve", action="store_true", help="Start background HTTP server exposing /metrics endpoint")
    parser_met.add_argument("--port", type=int, default=9090, help="HTTP port for /metrics server (default: 9090)")

    # Subcommand: stress-test
    parser_stress = subparsers.add_parser("stress-test", parents=[parent_parser], help="Run head-to-head stress benchmark: HuggingFace Static vs Mini-vLLM")
    parser_stress.add_argument("--requests", type=int, default=8, help="Number of concurrent requests in workload (default: 8)")
    parser_stress.add_argument("--batch-size", type=int, default=4, help="Maximum continuous batch capacity (default: 4)")
    parser_stress.add_argument("--kv-cache-dtype", type=str, choices=["fp16", "int8"], default="fp16", help="KV cache storage dtype for Mini-vLLM (default: fp16)")

    # Subcommand: serve
    parser_serve = subparsers.add_parser("serve", parents=[parent_parser], help="Start OpenAI-compatible streaming REST API server")
    parser_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser_serve.add_argument("--quant", type=str, choices=["none", "int8", "int4", "hybrid_int8"], default="none", help="Optional model quantization (default: none)")
    parser_serve.add_argument(
        "--engine",
        type=str,
        choices=["hf", "localstrm", "mini-vllm"],
        default="hf",
        help="Serving engine backend: 'localstrm' (LocalStrm PagedAttention continuous batching engine; legacy alias: 'mini-vllm') or 'hf' (default: Hugging Face serial engine)"
    )
    parser_serve.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum continuous batch capacity for LocalStrm engine (default: 4)"
    )
    parser_serve.add_argument(
        "--total-blocks",
        type=int,
        default=None,
        help="Total physical KV cache blocks in pool for LocalStrm engine"
    )
    parser_serve.add_argument(
        "--vram-budget",
        type=str,
        default=None,
        help="Total engine VRAM operational budget ceiling (e.g. '2GiB', '1400MiB') for LocalStrm engine"
    )
    parser_serve.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["fp16", "int8"],
        default="fp16",
        help="KV cache storage dtype for LocalStrm engine (default: fp16, int8 cuts KV memory by 50%%)"
    )

    # Subcommand: perf-benchmark
    parser_perf = subparsers.add_parser("perf-benchmark", parents=[parent_parser], help="Run rigorous performance benchmark suite (Static Baseline vs Mini-vLLM)")
    parser_perf.add_argument("--profile", type=str, choices=["smoke", "full"], default="smoke", help="Execution profile: smoke (~15s) or full")
    parser_perf.add_argument("--concurrencies", type=str, default=None, help="Comma-separated concurrency levels e.g. '1,2,4,8,16'")
    parser_perf.add_argument("--engines", type=str, default=None, help="Comma-separated engines e.g. 'baseline,minivllm-fp16,minivllm-int8'")
    parser_perf.add_argument("--workloads", type=str, default=None, help="Comma-separated workloads e.g. 'fixed,mixed'")
    parser_perf.add_argument("--num-requests", type=int, default=None, help="Requests per concurrency level")
    parser_perf.add_argument("--repetitions", type=int, default=1, help="Repetitions per test cell")
    parser_perf.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    # Subcommand: list-models / models
    subparsers.add_parser("list-models", parents=[parent_parser], help="List locally installed and cached Hugging Face models")
    subparsers.add_parser("models", parents=[parent_parser], help="Alias for list-models")

    # Subcommand: test
    subparsers.add_parser("test", help="Run comprehensive automated regression test suite across all 12 subsystems")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "list_models", False) or args.subcommand in ("list-models", "models"):
        cmd_list_models(args)
        return

    if args.subcommand == "chat":
        cmd_chat(args)
    elif args.subcommand == "serve":
        cmd_serve(args)
    elif args.subcommand == "perf-benchmark":
        cmd_perf_benchmark(args)
    elif args.subcommand == "engine":
        cmd_engine(args)
    elif args.subcommand == "benchmark":
        cmd_benchmark(args)
    elif args.subcommand == "inspect":
        cmd_inspect(args)
    elif args.subcommand == "quantize":
        cmd_quantize(args)
    elif args.subcommand == "prefix-cache":
        cmd_prefix_cache(args)
    elif args.subcommand == "preempt":
        cmd_preempt(args)
    elif args.subcommand == "check":
        cmd_check(args)
    elif args.subcommand == "chunked-prefill":
        cmd_chunked_prefill(args)
    elif args.subcommand == "metrics":
        cmd_metrics(args)
    elif args.subcommand == "stress-test":
        cmd_stress_test(args)
    elif args.subcommand == "test":
        cmd_test(args)
    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
