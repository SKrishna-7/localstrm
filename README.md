# LocalStrm

> A lightweight PyTorch LLM inference engine built from first principles to explore how modern serving systems such as vLLM work under the hood. Study PagedAttention, KV-cache memory management, continuous batching, prefix caching, and optional hybrid INT8 execution on memory-constrained consumer GPUs.

[![Python: 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](INSTALL.md)
[![PyTorch: 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Test Suite: 240 Tests](https://img.shields.io/badge/Test%20Suite-240%20Tests-brightgreen.svg)](tests/)

---

## On This Page

- [Overview](#overview)
- [Installation & Quick Start](#installation--quick-start)
- [Pre-Download Model & VRAM Preflight](#pre-download-model--vram-preflight)
- [Core Capabilities & Architecture](#core-capabilities--architecture)
- [How LocalStrm Works](#how-localstrm-works)
- [CLI Reference](#cli-reference)
- [Benchmark Studies](#benchmark-studies-concurrent-serving-scheduling-efficiency-and-quantized-inference)
- [Tests & Known Limitations](#tests-supported-configurations--known-limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

LocalStrm breaks LLM inference into the systems that make text generation possible: model loading, KV-cache memory management, request scheduling, token generation, and serving.
Instead of treating inference as a single `generate()` call, LocalStrm implements these components as separate, inspectable parts of a PyTorch inference engine. The goal is to make memory management and concurrent request scheduling easier to study in a smaller codebase.

> **Note:** The engine class name `MiniVLLMEngine` and the legacy CLI flag `--engine mini-vllm` are internal identifiers retained for backward compatibility with existing tests and scripts. The canonical user-facing engine selector is `--engine localstrm`.

---
## Installation & Quick Start

LocalStrm is verified on Linux, with Windows support untested/best-effort. Python 3.11 is the recommended starting point, and the following instructions cover virtual environment setup, PyTorch installation, hardware checks, inference, and optional HTTP serving.

### 1. Prerequisites

- **Operating System:** Linux verified (Ubuntu 22.04 LTS or newer), Windows untested/best-effort.
- **Python:** 3.11 recommended; Python 3.10 is also listed as supported.
- **GPU inference:** NVIDIA GPU with a compatible driver and CUDA-enabled PyTorch installation.
- **Git:** Required if you are cloning the repository.

Verify Python is installed:

**Linux (Terminal)**

```bash
python3.11 --version
```

**Windows (PowerShell)**

```powershell
py -3.11 --version
```

If Python is not installed, download it from the [official Python website](https://www.python.org/downloads/).

### 2. Clone the Repository

Skip this step if you already have the LocalStrm source code.

```bash
git clone https://github.com/SKrishna-7/localstrm.git
cd localstrm
```

### 3. Create and Activate a Virtual Environment

A virtual environment keeps LocalStrm's Python dependencies separate from other projects.

**Linux (Terminal)**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks virtual environment activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Alternatively, run `.venv\Scripts\python.exe` directly without activating the environment.

### 4. Install PyTorch

For GPU inference, install a PyTorch build compatible with your operating system and NVIDIA driver.

Visit the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) and select your operating system, Pip, Python, and an appropriate CUDA version. Run the generated installation command inside your activated virtual environment.

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda)"
```

For GPU inference, `torch.cuda.is_available()` should return `True`.

### 5. Install LocalStrm Dependencies

Run these commands from the repository root with the virtual environment activated.

**Core inference**

```bash
python -m pip install -r requirements.txt
```

**HTTP API serving**

Install the serving dependencies if you plan to expose the model through the REST API:

```bash
python -m pip install -r requirements-serving.txt
```

**Development, testing, and benchmarks**

For the full development and testing toolset:

```bash
python -m pip install -r requirements-dev.txt
```

Choose the dependency setup appropriate for your workflow. If you need multiple feature groups, check the dependency files for overlapping requirements.

### 6. Configure the Python Environment

Set `PYTHONPATH` for the current terminal session.

**Linux (Bash)**

```bash
export PYTHONPATH=.
```

**Windows (PowerShell)**

```powershell
$env:PYTHONPATH = "."
```

Verify that the CLI loads correctly:

```bash
python cli.py --help
```

### 7. Check Hardware and Model Memory

Before loading a model, run the preflight check:

```bash
python cli.py check --model Qwen/Qwen2.5-0.5B --device cuda
```

LocalStrm estimates model memory requirements, available GPU memory, host RAM, and other capacity constraints before model loading.

**Important:** Preflight is an estimate, not a guarantee that inference will succeed. Actual memory consumption depends on runtime conditions, and CUDA out-of-memory errors may still occur.

### 8. Run Your First Inference

Generate text with the LocalStrm inference engine:

```bash
python cli.py engine \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain virtual memory in two sentences." \
  --temperature 0.0
```

This uses greedy decoding for the selected prompt.

The first run may download model files if they are not already cached locally.

### 9. Start Interactive Chat

For quick interactive experimentation in the terminal:

```bash
# Run with LocalStrm native PagedAttention engine:
python cli.py chat --engine localstrm --model Qwen/Qwen2.5-Coder-0.5B-Instruct

# Or run with the Hugging Face serial backend (default):
python cli.py chat --engine hf --model Qwen/Qwen2.5-0.5B
```

**Note:** The `chat` command supports both `--engine localstrm` (LocalStrm's native PagedAttention engine; legacy alias: `--engine mini-vllm`) and `--engine hf` (Hugging Face serial engine, default).

### 10. Start the HTTP API (Optional)

Install the serving dependencies, then start the LocalStrm serving backend:

```bash
python cli.py serve \
  --engine localstrm \
  --model Qwen/Qwen2.5-0.5B \
  --host 127.0.0.1 \
  --port 8000
```

The server listens at `http://127.0.0.1:8000`.

Available endpoints include:

- `POST /v1/chat/completions` - Chat completions, with optional streaming.
- `POST /v1/completions` - Text completions.
- `GET /v1/models` - Model information.
- `GET /health` - Health status.
- `GET /metrics` - Prometheus metrics.
- `GET /docs` - Interactive API documentation.

The `localstrm` backend uses LocalStrm's native inference engine (`--engine mini-vllm` is supported as a legacy backwards-compatible alias). The default `hf` backend uses Hugging Face serving instead.

### 11. Platform Notes

- **Windows and Linux:** Run all commands from the repository root using the activated virtual environment.
- **GPU availability:** Use `--device cuda` only when CUDA is available in PyTorch. CPU execution is available only for supported commands and configurations.
- **Hybrid INT8:** The optional `hybrid_int8` mode requires an NVIDIA GPU with compute capability 8.0 or newer.
- **Memory constraints:** Use the preflight command before attempting larger models or more demanding configurations.
- **Troubleshooting:** For detailed configuration, supported architectures, and additional setup notes, see [`INSTALL.md`](INSTALL.md).
---

## Pre-Download Model & VRAM Preflight

One of LocalStrm's core capabilities is its **Pre-Flight Memory & Hardware Guard** ([`src/memory/preflight.py`](src/memory/preflight.py)), accessible via the CLI. The output below is an example from a recorded local preflight run, not a guarantee that the same model will fit on every system:

```bash
python cli.py check --model Qwen/Qwen2.5-3B --quant hybrid_int8 --device cuda
```

```text
================================================================================
                    PRE-FLIGHT HARDWARE & MEMORY AUDIT REPORT
================================================================================
Model: Qwen/Qwen2.5-3B | Device: cuda (NVIDIA GeForce RTX 4050 Laptop GPU)
Quantization Precision: hybrid_int8
[CAPACITY ESTIMATION]
  Available GPU VRAM:       5.71 GiB (with 5% safety margin: 5.42 GiB)
  Available Host RAM:       23.40 GiB
  Available Disk Space:     184.20 GiB
[PROJECTED MEMORY CONSUMPTION]
  Model Compressed Weights: ~3.16 GiB
  KV-Cache Footprint:       ~0.25 GiB (batch=1, context=2048)
  CUDA Runtime Overhead:    ~0.80 GiB (context + activation buffers)
  ------------------------------------------------------------------
  Total Projected VRAM:     ~4.21 GiB
[STATUS] PASS: Model is projected to fit safely in GPU VRAM.
================================================================================
```

### How Preflight Operates Before Downloading Weights

1. **Lightweight Metadata Fetching:**
   Before downloading multi-gigabyte tensor files, the preflight guard inspects the repository metadata via `huggingface_hub.HfApi.model_info()` or downloads only the lightweight `config.json` (~3 KB). For local directories, it scans file sizes on disk without loading weights into RAM or VRAM.
2. **Hardware Capacity Inspection:**
   Queries `torch.cuda.mem_get_info()` for free VRAM, `psutil.virtual_memory()` for available host RAM, and `shutil.disk_usage()` for target cache storage.
3. **Multi-Stage Fit Calculations:**
   - **Weight Memory:** Scaled by precision factor (1.0 for FP16, 0.50 for INT8, 0.55 for Hybrid INT8, 0.25 for INT4).
   - **KV-Cache Footprint:** Exact analytical formula: $2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes\_per\_elem} \times \text{max\_context} \times \text{batch\_size}$.
   - **Host RAM Staging Check:** When using quantized modes (`int8` or `hybrid_int8`), uncompressed FP16 weights are loaded into host RAM first for in-place quantization before transferring compressed weights to GPU. Preflight verifies host RAM can hold uncompressed weights before allowing download.
   - **Disk Space Check:** Verifies sufficient free disk space for uncompressed checkpoints plus a 10% safety buffer.
4. **Fail-Closed Safety:**
   If model metadata cannot be resolved (e.g. invalid repository ID, network failure, or inaccessible model), preflight aborts with **exit code 1** and actionable guidance rather than proceeding with unresolved model metadata.
5. **Estimate vs. Guarantee:**
   Preflight provides a **rigorous analytical estimate**, not an absolute operating guarantee. Dynamic activation memory varies with runtime batch composition, and CUDA allocator fragmentation can lead to allocation errors if background processes consume VRAM.

---

## Core Capabilities & Architecture

```text
Inference Engine Architecture
├── cli.py                     # Unified entry point (engine, chat, serve, check, benchmark)
├── src/
│   ├── config.py              # Engine configuration & environment overrides
│   ├── engine/
│   │   ├── llm_engine.py      # Core execution loop & physical memory binding
│   │   ├── model_runner.py    # Batched model execution & GQA pointer gathering
│   │   └── sampling.py        # Probability shaping (Top-P, Top-K, Min-P, penalties)
│   ├── memory/
│   │   ├── block_manager.py   # Virtual-to-physical block tables & pool allocator
│   │   ├── prefix_cache.py    # Content-addressed chained block-hash deduplication
│   │   ├── preemption.py      # Victim selection, eviction & recompute rollback
│   │   └── preflight.py       # Hardware capacity & metadata preflight checks
│   ├── kernels/
│   │   ├── paged_attention.py # Gathered SDPA attention for non-contiguous KV blocks
│   │   └── quantization.py    # Dynamic INT8 & Opt-In Hybrid INT8 (torch._int_mm)
│   ├── scheduler/
│   │   ├── continuous_scheduler.py # Iteration-level continuous batching scheduler
│   │   ├── chunked_prefill.py # Interleaved prompt slicing for latency smoothing
│   │   └── static_batching.py # Padded batching baseline for comparative benchmarks
│   ├── server/
│   │   ├── app.py             # FastAPI application & lifespan management
│   │   ├── protocol.py        # Pydantic v2 schemas for OpenAI REST protocol
│   │   ├── serving_adapter.py # Async serving queue & background worker adapter
│   │   └── serving_engine.py  # Thread-safe async streaming engine
│   └── observability/
│       └── metrics.py         # Prometheus metrics exporter & latency histograms
└── tests/                     # 19 automated regression & integration test modules
```

### Architectural Highlights

1. **PagedAttention & Physical Block Manager:**
   Stores KV-cache state in fixed-size physical blocks (16 tokens per block) and maps logical token positions to those blocks. This avoids requiring each request to own one large contiguous KV-cache allocation.
2. **Continuous (Iteration-Level) Batching:**
   Evaluates decode steps iteration by iteration. Completed requests depart immediately, and arriving requests are scheduled into active iterations without waiting for entire batch completion, reducing inactive sequence-slot scheduling waste relative to locked static batching (measured up to 28.8% on the tested mixed-length workload).
3. **Automatic Prefix Caching:**
   Computes SHA-256 content hashes for token blocks and reuses matching cached prefix state when available, avoiding repeated prefill computation for the reused portion.
4. **Hybrid INT8 Execution Mode:**
   An opt-in precision mode (`--quant hybrid_int8`). Attention projections remain in FP16, while MLP projections (`gate_proj`, `up_proj`, `down_proj`) use integer matrix multiplication via `torch._int_mm`. Tensor Core execution was not independently verified with hardware performance counters in the reported benchmark.
5. **CPU-Staged Quantized Loading:**
   When loading quantized models on CUDA, uncompressed FP16 weights are loaded into host RAM first, quantized in place, and only compressed weights are transferred to GPU VRAM. Full uncompressed weights never touch GPU memory.
6. **OpenAI-Compatible REST Server:**
   FastAPI application implementing `/v1/chat/completions` (supporting streaming Server-Sent Events and non-streaming responses), `/v1/models`, `/health`, and Prometheus `/metrics`.

---

## How LocalStrm Works

LocalStrm separates the work of inference into request handling, scheduling, KV-cache management, model execution, sampling, and response delivery. This makes the mechanics of a serving engine easier to inspect and test.


### End-to-End Request Lifecycle

1. **Request entry:** `cli.py engine` submits prompts to `MiniVLLMEngine`. `cli.py chat` supports both `--engine localstrm` (native PagedAttention engine) and `--engine hf` (default Hugging Face serial engine). For HTTP requests, the server backend depends on `--engine`: the default `hf` backend uses Hugging Face serving, while `--engine localstrm` (legacy alias: `--engine mini-vllm`) selects LocalStrm's engine and serving adapter.
2. **Tokenization and registration:** The ingress layer formats chat messages when applicable, tokenizes the prompt, and creates an `EngineRequest` with its sampling settings. New requests enter the waiting queue until the scheduler can admit them.
3. **Scheduling and prefix reuse:** The scheduler considers available KV-cache blocks, request length limits, and memory pressure. When prefix caching is enabled, matching 16-token blocks can be reused through chained SHA-256 hashes. A request that cannot fit in the pool fails rather than remaining queued indefinitely.
4. **Prompt prefill:** The engine processes prompt tokens to populate the KV cache. Optional chunked prefill divides longer prompts into smaller pieces so prefill work can be interleaved with decode work.
5. **Decode and attention:** `batched_decode=True` is the standard execution path. Ready sequences can be decoded while other requests are still completing chunked prefill. The model runner performs batched model operations and reads cached keys and values through the physical block mapping. The opt-in serial mode (`batched_decode=False`) supports numerical-contract checks.
6. **Token sampling:** The engine selects the next token using greedy or stochastic sampling. Batched greedy argmax is a conditional fast path for multiple sequences when batched sampling is enabled and every request uses unpenalized greedy settings; other configurations use per-request sampling.
7. **Stopping and memory reclamation:** Generation stops when a configured stopping condition is reached, including EOS, a length limit, or supported stop criteria. Unshared blocks are released; cached prefix blocks may remain available for reuse until eviction is needed.
8. **Response delivery:** `cli.py engine` prints completed output after execution. The applicable HTTP serving path can stream OpenAI-compatible Server-Sent Events. Token-by-token terminal streaming belongs to the separate `cli.py chat` REPL, not `cli.py engine`.

This architecture reflects the current implementation rather than a claim of feature or performance parity with vLLM. See the CLI reference and benchmark sections for supported options and measured trade-offs.

---

## CLI Reference

LocalStrm provides a unified command-line interface (`cli.py`) for offline model inspection, memory preflight validation, continuous batching inference, OpenAI-compatible HTTP serving, and automated subsystem benchmarking.

```bash
python cli.py <subcommand> [flags...]
```

### 1. Command Overview

The CLI exposes 15 dedicated subcommands and 1 alias across four operational categories:

| Subcommand | Category | Description | Primary Engine / Backend |
| :--- | :--- | :--- | :--- |
| `engine` | Inference | Unified continuous batching inference with PagedAttention, chunked prefill, and chained block-hash prefix caching | `MiniVLLMEngine` |
| `chat` | Inference | Interactive multi-turn streaming terminal chat REPL | Hugging Face Pipeline |
| `serve` | HTTP Serving | OpenAI-compatible FastAPI streaming REST API server (`/v1/chat/completions`, `/v1/completions`) | `MiniVLLMEngine` or Hugging Face |
| `check` | Hardware & Preflight | Pre-download VRAM and system memory capacity check before weight allocation | `src.memory.preflight` |
| `inspect` | Hardware & Preflight | Deep model parameter, layer geometry, and VRAM budget breakdown | `benchmarks.inspect_model` |
| `list-models` | Hardware & Preflight | List locally downloaded and cached Hugging Face model repositories | Hugging Face Cache Hub |
| `models` | Hardware & Preflight | Alias for `list-models` | Hugging Face Cache Hub |
| `perf-benchmark` | Diagnostics & Benchmarks | Systematic multi-concurrency performance benchmark (Static Baseline vs LocalStrm) | `benchmarks.run_benchmark` |
| `benchmark` | Diagnostics & Benchmarks | Continuous batching throughput and latency benchmark across concurrent prompts | `ContinuousBatchingEngine` |
| `quantize` | Diagnostics & Benchmarks | Benchmark weight quantization (INT8, INT4, HYBRID_INT8) against unquantized FP16 | Quantized Engine |
| `prefix-cache` | Diagnostics & Benchmarks | Chained block-hash prefix cache deduplication efficiency and time-to-first-token benchmark | Prefix Cache Engine |
| `preempt` | Diagnostics & Benchmarks | Memory starvation preemption and recompute benchmark under extreme KV block constraints | `StarvationResilientEngine` |
| `chunked-prefill` | Diagnostics & Benchmarks | Interleaved chunked prefill vs monolithic prefill decode latency jitter benchmark | Chunked Prefill Scheduler |
| `metrics` | Diagnostics & Benchmarks | Execute inference workload with real-time Prometheus telemetry dashboard and HTTP exporter | `MiniVLLMEngine` + Prometheus |
| `stress-test` | Diagnostics & Benchmarks | Head-to-head stress benchmark comparing HF static tensor batching vs LocalStrm | Static vs Continuous Batching |
| `test` | Diagnostics & Benchmarks | Comprehensive automated regression test runner across all 12 engine subsystems | `tests/test_all_features.py` |

### 2. Common Options

All subcommands (except `test`, which is a standalone regression test runner) inherit the parent parser options:

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model MODEL` | String | Dynamic (see below) | Hugging Face model repository ID (e.g., `Qwen/Qwen2.5-0.5B`) or local directory path containing model weights and configuration. |
| `--device {cuda,cpu}` | String | Dynamic (see below) | Compute execution device. Auto-selects CUDA when an active GPU is detected; falls back to CPU otherwise. |
| `-v`, `--verbose` | Flag | `False` | Enables detailed debug logging, displaying block table allocations, scheduler state transitions, and step timing. |
| `--force` | Flag | `False` | Bypasses fail-closed preflight memory and hardware capacity checks, forcing execution even if estimated VRAM exceeds available capacity. |
| `-l`, `--list-models`, `--models` | Flag | `False` | Lists all locally cached Hugging Face models found in `~/.cache/huggingface/hub` and exits immediately without running the subcommand. |

#### Dynamic Default Resolution

- **`--model`**: Evaluates the environment variable `$MODEL_ID`. If unset, falls back to `Qwen/Qwen2.5-0.5B`.
- **`--device`**: Evaluates the environment variable `$ENGINE_DEVICE`. If unset, queries `torch.cuda.is_available()`; resolves to `"cuda"` if a compatible GPU is present, otherwise `"cpu"`.

### 3. Inference Commands

#### `cli.py engine`

Executes single or batched prompt inference using the full `MiniVLLMEngine` pipeline with PagedAttention, continuous batching, chunked prefill, and optional chained block-hash prefix caching.

```bash
python cli.py engine --prompt "Your prompt here" [flags...]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--prompt PROMPT` | String | `None` **(stdin allowed)** | Input prompt to generate from. Can be specified multiple times for batched concurrent evaluation (e.g., `--prompt "Q1" --prompt "Q2"`), or piped via stdin. |
| `--max-tokens INT` | Integer | `32` | Maximum number of new generation tokens per response sequence. |
| `--batch-size INT` | Integer | `4` | Maximum continuous batch capacity (concurrent sequences running in the scheduler). |
| `--total-blocks INT` | Integer | `None` **(auto-sized)** | Explicit total physical KV cache blocks allocated in the pool. Mutually exclusive with `--vram-budget`. |
| `--vram-budget STR` | String | `None` | Operational VRAM budget ceiling (e.g., `2GiB`, `1400MiB`). Sizer calculates maximum KV blocks after deducting model weights, activation workspace, and safety margins. Mutually exclusive with `--total-blocks`. |
| `--block-size INT` | Integer | `16` | Token capacity per physical KV block. Must align with PagedAttention kernel constraints. |
| `--quant {none,int8,int4,hybrid_int8}` | String | `none` | Weight quantization mode. `int8` (W8A16), `int4` (W4A16), or `hybrid_int8` (requires SM >= 8.0; uses `torch._int_mm` for MLP projections). |
| `--kv-cache-dtype {fp16,int8}` | String | `fp16` | KV cache storage datatype. Selecting `int8` activates symmetric per-token, per-head INT8 quantization (scales shaped `(layers, blocks, block_size, kv_heads, 1)`), reducing KV memory footprint by ~50%. |
| `--prefix-cache` / `--no-prefix-cache` | Flag | Enabled (`True`) | Enables or disables automatic chained block-hash prefix caching for common prompt prefixes and multi-turn conversations. |
| `--chunked-prefill` / `--no-chunked-prefill` | Flag | Enabled (`True`) | Enables or disables chunked prefill scheduling. When disabled, prompts prefill in a single monolithic step. |
| `--chunk-size INT` | Integer | `32` | Maximum token chunk budget allocated per prefill step to prevent decode latency spikes. |
| `--temperature FLOAT` | Float | `0.0` | Sampling temperature. `0.0` executes deterministic greedy argmax sampling; `> 0.0` enables stochastic sampling. |
| `--top-p FLOAT` | Float | `1.0` | Top-P nucleus sampling threshold ($0.0 < p \le 1.0$). Retains smallest token set whose cumulative probability exceeds $P$. |
| `--top-k INT` | Integer | `-1` | Top-K sampling threshold. `-1` disables Top-K filtering. |
| `--min-p FLOAT` | Float | `0.0` | Dynamic minimum probability threshold relative to the top candidate token ($0.0 \le \text{min\_p} \le 1.0$). |
| `--repetition-penalty FLOAT` | Float | `1.0` | Penalizes tokens that have already appeared in the generated sequence (`1.0` disables this penalty). |
| `--stop STR` | String | `[]` | Early termination stop strings. Can be specified multiple times (e.g., `--stop "###" --stop "\n\n"`). |
| `--metrics` / `--no-metrics` | Flag | Enabled (`True`) | Displays a formatted ASCII telemetry summary table (throughput, latency, KV cache utilization) upon workload completion. |
| `--export-prometheus PATH` | String | `None` | File path to export engine telemetry metrics in standard Prometheus plaintext format upon completion. |

#### `cli.py chat`

Launches an interactive, multi-turn conversational streaming session in your terminal.

```bash
python cli.py chat [flags...]
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--engine {hf,localstrm,mini-vllm}` | String | `hf` | Chat engine backend: `localstrm` (LocalStrm PagedAttention continuous batching engine; legacy alias: `mini-vllm`) or `hf` (default: Hugging Face serial engine). |
| `--quant {none,int8,int4,hybrid_int8}` | String | `none` | Weight quantization precision applied to the chat model. |
| `--max-tokens INT` | Integer | `128` | Maximum new tokens generated per conversational turn. |
| `--temperature FLOAT` | Float | `0.7` | Sampling temperature for creative conversational variety (`0.0` for greedy). |
| `--system STR` | String | `None` | Custom system prompt overriding the model's default conversation persona. |
| `--batch-size INT` | Integer | `4` | Maximum continuous batch capacity for LocalStrm engine. |
| `--total-blocks INT` | Integer | `None` | Total physical KV cache blocks for LocalStrm engine. |
| `--vram-budget STR` | String | `None` | Total engine VRAM operational budget ceiling for LocalStrm engine. |
| `--kv-cache-dtype {fp16,int8}` | String | `fp16` | KV cache storage dtype for LocalStrm engine (default: fp16, int8 cuts KV memory by 50%). |

> [!NOTE]
> **Execution Backends: `localstrm` vs. `hf`**<br>
>
> - `cli.py chat --engine localstrm` executes LocalStrm's core `MiniVLLMEngine` architecture in an interactive terminal session, streaming responses token-by-token with non-contiguous PagedAttention memory management, prefix caching, and live TTFT/TPOT latency metrics.
>
> - `cli.py chat --engine hf` (default) provides a serial REPL powered by Hugging Face's sequential decode pipeline.

### 4. Hardware & Memory Preflight

#### `cli.py check`

Runs fail-closed static memory preflight verification before any model weights or KV cache memory are allocated.

```bash
python cli.py check --model Qwen/Qwen2.5-0.5B --device cuda
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--quant {none,int8,int4,hybrid_int8}` | String | `none` | Target quantization mode to evaluate during preflight budget calculations. |

- **Exit Codes:** Returns exit code `0` on successful capacity verification; returns exit code `1` if estimated memory requirements exceed safe operational thresholds (5% safety margin).

#### `cli.py inspect`

Inspects model configuration metadata, architecture parameters, tensor shapes, and calculated VRAM requirements across various context lengths and batch sizes without loading full model weights.

```bash
python cli.py inspect --model Qwen/Qwen2.5-0.5B
```

**Inherits common parent options (`--model`, `--device`, `-v`, `--force`). Does not require additional subcommand flags.**

#### `cli.py list-models` (alias: `cli.py models`)

Inspects the local Hugging Face hub cache directory (`~/.cache/huggingface/hub`) and prints a tabular summary of all downloaded models, snapshot paths, and disk sizes.

```bash
python cli.py list-models
# Or:
python cli.py models
```

### 5. HTTP Serving

#### `cli.py serve`

Launches an OpenAI-compatible REST API server powered by FastAPI and Uvicorn.

```bash
python cli.py serve --engine localstrm --model Qwen/Qwen2.5-0.5B --host 127.0.0.1 --port 8000
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host STR` | String | `127.0.0.1` | Network interface address to bind the HTTP server. The server provides no built-in authentication; binding to 0.0.0.0 exposes the API to the local network or internet. |
| `--port INT` | Integer | `8000` | TCP port to listen for incoming HTTP connections. |
| `--quant {none,int8,int4,hybrid_int8}` | String | `none` | Optional model weight quantization. |
| `--engine {hf,localstrm,mini-vllm}` | String | `hf` | Serving backend engine. Choose `localstrm` (canonical; legacy alias: `mini-vllm`) for true continuous batching or `hf` for sequential Hugging Face serving. |
| `--batch-size INT` | Integer | `4` | Maximum continuous batch capacity. **Applies strictly to `--engine localstrm` (or legacy `mini-vllm`).** |
| `--total-blocks INT` | Integer | `None` | Explicit physical KV cache block count. Mutually exclusive with `--vram-budget`. **Applies strictly to `--engine localstrm`.** |
| `--vram-budget STR` | String | `None` | Total VRAM operational budget ceiling (e.g., `2GiB`, `1400MiB`). Mutually exclusive with `--total-blocks`. **Applies strictly to `--engine localstrm`.** |
| `--kv-cache-dtype {fp16,int8}` | String | `fp16` | KV cache storage datatype (`fp16` or `int8`). **Applies strictly to `--engine localstrm`.** |

> [!WARNING]

> **Engine Backend Compatibility in `serve`:**<br>

> When running `--engine hf` (the default), requests are served sequentially via Hugging Face pipelines. Options `--total-blocks`, `--vram-budget`, and `--kv-cache-dtype` belong strictly to LocalStrm's PagedAttention engine; specifying them under `--engine hf` produces a warning and they are ignored. To use LocalStrm continuous batching, pass `--engine localstrm` (or legacy alias `--engine mini-vllm`).

#### Served Endpoints

Once running, the server exposes the following HTTP endpoints:
- `POST /v1/chat/completions`: OpenAI-compatible chat completions (supports streaming `stream: true` via Server-Sent Events).
- `POST /v1/completions`: OpenAI-compatible text completions (supports streaming).
- `GET /v1/models`: Lists active model identifier and metadata.
- `GET /health`: Server health check returning engine status and device utilization.
- `GET /metrics`: Prometheus plaintext telemetry metrics.
- `GET /docs`: Interactive Swagger / OpenAPI documentation UI.

### 6. Benchmarking & Diagnostics

LocalStrm includes dedicated diagnostic runners to isolate, test, and benchmark individual architectural subsystems.

#### `cli.py perf-benchmark`

Executes an automated multi-concurrency performance benchmark matrix comparing static batching against continuous batching under controlled synthetic or mixed workloads.

```bash
python cli.py perf-benchmark --profile smoke
```

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--profile {smoke,full}` | String | `smoke` | Preconfigured test execution profile. `smoke` executes a quick ~15-second verification run; `full` executes the exhaustive multi-concurrency matrix. |
| `--concurrencies STR` | String | `None` | Comma-separated list of concurrency levels (e.g., `"1,2,4,8"`). Overrides profile preset. |
| `--engines STR` | String | `None` | Comma-separated list of engines to benchmark (e.g., `"baseline,minivllm-fp16,minivllm-int8"`). |
| `--workloads STR` | String | `None` | Comma-separated list of prompt workload distributions (e.g., `"fixed,mixed"`). |
| `--num-requests INT` | Integer | `None` | Total requests dispatched per concurrency level. |
| `--repetitions INT` | Integer | `1` | Benchmark measurement iterations per test cell to calculate stable statistical means. |
| `--output-dir PATH` | String | `None` | Custom directory to save benchmark summary reports and raw metric logs. |

#### Subsystem Microbenchmarks

| Subcommand | Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`benchmark`** | `--prompt STR` | String (appendable) | Default suite | Custom benchmark prompt (can be passed multiple times). |
| | `--max-tokens INT` | Integer | `16` | Maximum generation tokens per request. |
| | `--batch-size INT` | Integer | `4` | Maximum continuous batch capacity. |
| **`quantize`** | `--prompt STR` | String | Sample prompt | Input prompt to test quantized generation accuracy. |
| | `--quant CHOICE` | `none,int8,int4,hybrid_int8` | `int8` | Target quantization mode to evaluate. |
| **`prefix-cache`** | `--system-prompt STR` | String | Sample prompt | Shared system prompt prefix to evaluate prefix caching hit rate. |
| | `--query STR` | String (appendable) | Sample queries | User queries sharing the prefix (can be specified multiple times). |
| **`preempt`** | `--prompt STR` | String (appendable) | Sample workload | Request prompts to induce memory pressure. |
| | `--tokens INT` | Integer | `12` | Generation tokens per sequence. |
| | `--total-blocks INT` | Integer | `5` | Artificially constrained physical block pool to trigger preemption. |
| | `--block-size INT` | Integer | `4` | Tokens per physical KV block. |
| **`chunked-prefill`**| `--chunk-size INT` | Integer | `32` | Maximum tokens per prefill chunk. |
| | `--prompt-short STR` | String | Sample prompt | Active short decoding sequence to measure latency disruption. |
| | `--prompt-long STR` | String | Sample prompt | Long arrival prefill prompt inducing prefill computation. |
| **`metrics`** | `--prompt STR` | String (appendable) | Sample prompt | Workload prompt text to process. |
| | `--max-tokens INT` | Integer | `24` | Maximum tokens per response. |
| | `--batch-size INT` | Integer | `4` | Maximum continuous batch capacity. |
| | `--kv-cache-dtype` | `fp16,int8` | `fp16` | KV cache storage datatype. |
| | `--export PATH` | String | `None` | File path to export final Prometheus metrics. |
| | `--serve` | Flag | `False` | Starts a persistent background HTTP server exposing `/metrics`. |
| | `--port INT` | Integer | `9090` | Port for the background telemetry HTTP exporter. |
| **`stress-test`** | `--requests INT` | Integer | `8` | Number of concurrent requests in the stress workload. |
| | `--batch-size INT` | Integer | `4` | Maximum continuous batch capacity. |
| | `--kv-cache-dtype` | `fp16,int8` | `fp16` | KV cache storage datatype for LocalStrm. |
| **`test`** | **(No flags)** | N/A | N/A | Executes the automated test suite (`python tests/test_all_features.py`). |

### 7. Practical Recipes

#### Recipe 1: Preflight Memory Safety Check

Verify whether a model and its KV cache can safely fit in available GPU VRAM before downloading weights:

```bash
python cli.py check --model Qwen/Qwen2.5-0.5B --device cuda
```

#### Recipe 2: Single-Prompt Deterministic Inference

Run greedy decoding on a single prompt with custom token length:

```bash
python cli.py engine \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "Explain virtual memory paging in two sentences." \
  --max-tokens 48 \
  --temperature 0.0
```

#### Recipe 3: Batched Inference Under Strict VRAM Budget

Run concurrent requests with INT8 KV cache under an explicit 2.0 GiB operational VRAM ceiling:

```bash
python cli.py engine \
  --model Qwen/Qwen2.5-0.5B \
  --vram-budget 2GiB \
  --kv-cache-dtype int8 \
  --prompt "What is a page table?" \
  --prompt "Describe copy-on-write." \
  --prompt "Explain TLB misses." \
  --max-tokens 32
```

#### Recipe 4: Interactive Streaming Chat with Quantization

Start an interactive terminal conversation using W8A16 weight quantization to conserve memory:

```bash
python cli.py chat \
  --model Qwen/Qwen2.5-0.5B \
  --quant int8 \
  --temperature 0.7 \
  --max-tokens 128
```

#### Recipe 5: OpenAI-Compatible REST Server

Launch the HTTP server backed by the LocalStrm continuous batching engine:

```bash
python cli.py serve \
  --engine localstrm \
  --model Qwen/Qwen2.5-0.5B \
  --host 127.0.0.1 \
  --port 8000 \
  --batch-size 8 \
  --kv-cache-dtype int8
```

#### Recipe 6: Quick Performance Benchmark (Smoke Profile)

Run a quick ~15-second benchmark comparing Hugging Face static tensor batching vs LocalStrm:

```bash
python cli.py perf-benchmark --profile smoke --device cuda
```

#### Recipe 7: Live Prometheus Telemetry Server

Process a batch of requests while serving live metrics on port 9090 for Prometheus scraping:

```bash
python cli.py metrics \
  --model Qwen/Qwen2.5-0.5B \
  --serve \
  --port 9090 \
  --batch-size 4 \
  --max-tokens 32
```

#### Recipe 8: Chained Block-Hash Prefix Cache Deduplication Benchmark

Measure time-to-first-token latency reduction when multiple user queries share an identical system prompt:

```bash
python cli.py prefix-cache \
  --model Qwen/Qwen2.5-0.5B \
  --system-prompt "You are an expert Linux kernel systems engineer. Answer concisely." \
  --query "How does kswapd work?" \
  --query "Explain buddy allocator fragmentation."
```

### 8. Option Compatibility & Troubleshooting

- **KV Cache Budget Mutual Exclusion (`--total-blocks` vs. `--vram-budget`):**
  You can configure KV cache allocation via explicit block counts (`--total-blocks`) OR an operational memory ceiling (`--vram-budget`), but **never both**. Specifying both flags simultaneously in `cli.py engine` or `cli.py serve` raises an immediate configuration error:

```
  Error: Conflicting KV-cache configuration: both --total-blocks (N) and --vram-budget ('X') were provided. Specify --total-blocks OR --vram-budget, not both.
```

- **Serving Engine Backend Constraints:**
  Options `--total-blocks`, `--vram-budget`, and `--kv-cache-dtype` configure LocalStrm's PagedAttention allocator. If passed to `cli.py serve` when `--engine hf` is selected, they will be ignored with a warning. Use `--engine localstrm` to activate these controls.
- **Quantization Hardware Requirements:**
  - `none`: FP16/BF16 weights and compute. Works on all CUDA and CPU devices.
  - `int8` (W8A16) and `int4` (W4A16): Supported across all CUDA architectures.
  - `hybrid_int8`: Requires an NVIDIA GPU with compute capability (SM) $\ge 8.0$ (Ampere, Ada Lovelace, Hopper, or newer) and a PyTorch build that provides `torch._int_mm`. Tensor Core execution was not independently verified with hardware performance counters in the reported benchmark.
- **Fail-Closed Memory Preflight (`--force`):**
  If preflight memory estimation determines that model weights and minimum KV cache exceed available GPU memory (including the 5% safety margin), LocalStrm refuses to initialize to reduce the risk of CUDA out-of-memory (OOM) failures. In development environments where memory is swapped or shared, pass `--force` to override this safeguard.
- **Dynamic Defaults and Environment Variables:**
  - If `--model` is omitted, LocalStrm checks `$MODEL_ID`. If unset, it defaults to `"Qwen/Qwen2.5-0.5B"`.
  - If `--device` is omitted, LocalStrm checks `$ENGINE_DEVICE`. If unset, it auto-detects `"cuda"` if available, otherwise `"cpu"`.

---

## Benchmark Studies: Concurrent Serving, Scheduling Efficiency, and Quantized Inference

> **Reporting scope:** These results summarize distinct experiments recorded during development and validation on the reference RTX 4050 laptop GPU. The studies use different prompts, concurrency levels, measurement boundaries, and objectives; their results should not be compared as though they came from one shared protocol. Raw benchmark JSON files and one-off experiment runners are not bundled in this release snapshot. The general CLI benchmark runners remain available for supported workloads, but they do not automatically reproduce every experiment below.

### Qwen2.5-0.5B: Concurrent Serving Benchmark (120 Trials)

This earlier controlled serving benchmark compared a serial Hugging Face baseline with LocalStrm using FP16 and INT8 KV caches, under streaming and non-streaming HTTP request modes.

**Experimental setup**
- **Model:** `Qwen/Qwen2.5-0.5B` with FP16 model weights
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB physical VRAM)
- **Software:** Python 3.11.15, PyTorch 2.13.0+cu132, CUDA 13.2
- **Prompt and generation:** Fixed 16-token prompt and 32 output tokens per request
- **Concurrency:** 1, 2, 4, and 6
- **Request modes:** Streaming (SSE) and non-streaming HTTP
- **Configurations:** Hugging Face serial, LocalStrm FP16 KV cache, LocalStrm INT8 KV cache
- **Repetitions:** 5 measured trials per scenario, totaling 120 trials

**Median output-token throughput**

| Concurrency | Request Mode | HF Serial | LocalStrm FP16 KV | FP16 Speedup | LocalStrm INT8 KV | INT8 Speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Non-streaming | 63.21 tok/s | 68.20 tok/s | 1.08× | 50.10 tok/s | 0.79× |
| 1 | Streaming | 61.35 tok/s | 61.67 tok/s | 1.01× | 50.66 tok/s | 0.83× |
| 2 | Non-streaming | 66.69 tok/s | 121.66 tok/s | 1.82× | 96.30 tok/s | 1.44× |
| 2 | Streaming | 63.46 tok/s | 112.91 tok/s | 1.78× | 82.94 tok/s | 1.31× |
| 4 | Non-streaming | 51.43 tok/s | 228.33 tok/s | **4.44×** | 167.17 tok/s | 3.25× |
| 4 | Streaming | 51.87 tok/s | 224.08 tok/s | **4.32×** | 153.68 tok/s | 2.96× |
| 6 | Non-streaming | 62.76 tok/s | 178.76 tok/s | 2.85× | 133.32 tok/s | 2.12× |
| 6 | Streaming | 78.27 tok/s | 179.88 tok/s | 2.30× | 63.26 tok/s | 0.81× |

At concurrency 4, the reported streaming time-to-first-token (TTFT) was **1,081.3 ms** for the Hugging Face serial baseline and **103.1 ms** for LocalStrm FP16, approximately **10.49× lower latency** in this test.

**Interpretation and limitations:** LocalStrm FP16 reached its highest measured throughput in this matrix at concurrency 4. Throughput did not increase monotonically at concurrency 6, and INT8 KV-cache mode was slower than FP16 in the tested scenarios. For non-streaming requests, the report's TTFT field represented full-response latency; streaming TTFT measured arrival of the first SSE chunk. The original validation report was conditional because it also identified an EOS-related `finish_reason` API validation issue. These measurements should not be treated as universal guarantees.

---

### Qwen2.5-0.5B: Real-Prompt Scheduling-Waste Benchmark

We evaluated LocalStrm's continuous scheduling against Hugging Face static batching using 16 natural-language questions about LLM and AI systems. The prompt and generation limits were controlled to compare scheduling behavior, rather than to simulate naturally observed user traffic.

**Experimental setup**
- Model: `Qwen/Qwen2.5-0.5B` (FP16)
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM)
- Software: PyTorch 2.13.0+cu132, CUDA 13.2
- Dataset: 16 natural-language prompts, 5–17 tokens each (mean: 11.3 tokens)
- Topics include self-attention, tokenization, encoder/decoder models, embeddings, KV caching, continuous batching, PagedAttention, quantization, RAG, sampling, and scheduling waste.

<details>
<summary>View the 16 natural-language prompts</summary>

1. What is transformer architecture?
2. Explain self-attention in a transformer.
3. What is the difference between encoder-only and decoder-only language models?
4. How does tokenization work in large language models?
5. Why do transformers use positional encodings?
6. Explain the role of embeddings in an LLM.
7. What is the key-value cache during autoregressive generation?
8. How does continuous batching work in LLM inference serving?
9. What is paged attention, and why does it help manage memory?
10. Compare FP16 and INT8 quantization for inference.
11. Why can batching improve GPU throughput while increasing latency?
12. Explain retrieval-augmented generation and when it is useful.
13. What causes GPU out-of-memory errors when serving language models?
14. How do temperature and top-p sampling affect generated text?
15. How would you benchmark time to first token and inter-token latency?
16. How can variable prompt lengths and early-finished requests cause wasted work in static batches?

</details>
- Workloads: Uniform 20-token generation targets with simultaneous arrivals, and mixed 16–36-token targets with staggered 0.05-second arrival offsets
- Concurrency: 1, 2, 4, 8, and 16 requests
- Engines: Hugging Face static FP16, LocalStrm FP16 KV cache,
  and LocalStrm INT8 KV cache
- Repetitions: 3 per test cell, totaling 90 runs

#### Mixed-Workload Sequence-Slot Scheduling Waste

| Concurrency | HF Static Waste | HF Mean Wasted Sequence-Step Slots per Run | LocalStrm FP16 Waste | LocalStrm INT8 Waste |
|---:|---:|---:|---:|---:|
| 1 | 0.0% | 0 | 0.0% | 0.0% |
| 2 | 15.3% | 74 | 0.0% | 0.0% |
| 4 | 23.5% | 126 | 0.0% | 0.0% |
| 8 | 28.8% | 166 | 0.0% | 0.0% |
| 16 | 28.8% | 166 | 0.0% | 0.0% |

The waste rate is calculated as `wasted sequence-step slots / (useful slots + wasted sequence-step slots) × 100%`. It measures inactive sequence-slot scheduling waste, not hardware GPU-idle cycles or total GPU compute waste.

#### Mixed-Workload Output Throughput

| Concurrency | HF Static FP16 | LocalStrm FP16 KV | LocalStrm INT8 KV |
|---:|---:|---:|---:|
| 1 | 22.20 tok/s | 19.86 tok/s | 15.08 tok/s |
| 2 | 35.45 tok/s | 35.12 tok/s | 26.73 tok/s |
| 4 | 75.38 tok/s | 48.04 tok/s | 40.36 tok/s |
| 8 | 146.83 tok/s | 99.44 tok/s | 65.40 tok/s |
| 16 | 262.51 tok/s | 111.99 tok/s | 91.20 tok/s |

**Interpretation:**
LocalStrm recorded zero inactive sequence-slot waste under the measured metric. However, eliminating this scheduling waste did not guarantee higher throughput. Hugging Face static batching achieved higher raw throughput at most tested concurrency levels, while LocalStrm FP16 reached near-parity at concurrency 2.

### Qwen2.5-3B: Quantization and Decode Performance

We compared dynamic INT8 execution with LocalStrm's Hybrid INT8 execution using 16 real-prompt questions and repeated measurements on a consumer GPU.

**Experimental setup**
- Model: `Qwen/Qwen2.5-3B`
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- Software: Python 3.11.15, PyTorch 2.13.0+cu132, CUDA 13.2
- Dataset: 16 natural-language prompts
- Generation limit: 16 tokens per request
- Repetitions: 3 measured repetitions per prompt and execution mode
- Measurements: 96 measured requests in total, excluding one warm-up run per mode
- Reported execution failures and OOM errors: 0

#### Performance Comparison

| Metric | Dynamic INT8 | Hybrid INT8 |
|---|---:|---:|
| Decode throughput | 3.07 tok/s | 9.72 tok/s |
| End-to-end throughput | 3.06 tok/s | 9.60 tok/s |
| Mean of median TPOT measurements | 325.94 ms | 89.00 ms |
| Peak allocated VRAM | 3,486.7 MiB | 3,641.5 MiB |
| Peak reserved VRAM | 3,536.0 MiB | 3,793.3 MiB |

#### Key Findings

- Hybrid INT8 achieved approximately **3.17× decode throughput**.
- End-to-end throughput improved by approximately **3.13×**.
- The reported median TPOT measurement decreased by approximately **72.7%**.
- TTFT showed an approximately **2.70× speedup**.
- Peak allocated VRAM increased by approximately **154.8 MiB (4.44%)**.
- Peak reserved VRAM increased by approximately **257.3 MiB**.

**Output consistency:** Exact generated-token matches occurred in 9 of 48 paired requests (18.75%). The remaining 39 pairs differed, so these results must not be interpreted as token-for-token output parity between the two execution modes.

**Measurement limitations:** Tensor Core execution was not independently verified with hardware performance counters during this benchmark. Results reflect the tested model, hardware, prompts, generation limit, and software configuration, and may differ under other workloads.

---

## Tests, Supported Configurations & Known Limitations

### Test Suite

The repository includes an automated regression test suite covering the core engine and serving subsystems:
- **240 tests collected across 19 test modules** (`pytest tests/ --collect-only -q`).
- Test execution is split between **16 offline test modules** (executable without internet access or model downloads, 229 tests) and **3 model-dependent modules** (requiring local cached weights of `Qwen/Qwen2.5-0.5B`, 11 tests). See [`INSTALL.md`](INSTALL.md) for execution details. All **229 offline tests passed**; all **11 model-dependent integration tests passed** when cached weights are present (240/240 total passing).
- Tests validate: physical block allocation, block recycling, prefix cache chained block hashing, victim preemption under memory starvation, greedy sampling reproducibility, GQA dimensions, CPU-staged quantization loading, `torch._int_mm` fallbacks, preflight fail-closed guards, live HTTP FastAPI lifecycle events, multi-token stopping criteria vocabulary verification, INT4 slice dequantization numerical equivalence, and fail-closed architecture guards.

### Supported Model Architectures

- **Architecture pattern:** Standard decoder-only transformer architectures with dense linear projections (`model.layers[i]`, RMSNorm, RoPE, and Grouped Query Attention).
  - **Fully Supported & Verified:** `Qwen/Qwen2.5-0.5B` (Linux x86_64 CUDA, FP16 and INT8 KV-cache, continuous batching, prefix caching, multi-step generation, and HTTP serving verified across 11 integration tests and 90 benchmark runs).
  - **Smoke-Tested:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (Linux x86_64 CUDA, INT4 weight quantization via `cli.py chat` with multi-turn conversation and verified stop-token termination).
  - **Architecture Target (Not Universally Verified):** Structural compatibility targets LLaMA / Qwen2 / Mistral decoder-only families; however, individual checkpoints, alternative precision formats, and unverified backends must be tested individually and are not guaranteed to work without dedicated validation.
- **Incompatible with `--engine localstrm`:**
  - **Tri-hybrid architectures:** Models combining gated short convolutions, sliding-window attention, and global attention (such as LiquidAI LFM3 / `liquid_lfm`) do not use standard transformer attention blocks and fail closed with a clear diagnostic error.
  - **Other non-standard architectures:** Encoder-decoder models (T5), non-standard attention models (Bloom, Falcon, GPT-2), and mixture-of-experts architectures without dense projections. For non-supported models or custom architectures, use `--engine hf`.

### Known Limitations

1. **Single-Node Focus:** Tailored for single-GPU consumer/workstation environments. Does not implement multi-node InfiniBand tensor parallelism.
2. **Batched Numerical Reproducibility vs. Serial Consistency:** Parallel batched decoding (`batched_decode=True`) is the standard execution path. Different batch shapes can change floating-point reduction order, so greedy token output is not guaranteed to match isolated serial Hugging Face execution across every numerical configuration. An opt-in serial decode mode (`batched_decode=False`) remains available for numerical-contract verification.
3. **Server Concurrency:** The OpenAI REST server provides asynchronous queueing and SSE streaming. Thread-safe concurrent submission is an MVP implementation; concurrent engine shutdown or recovery from unrecoverable CUDA kernel panics is not guaranteed.

---

## Roadmap

- [ ] **Multi-Bucket CUDA Graphs:** Multi-batch CUDA Graph capture for continuous decode batches ($B \in \{1, 2, 4, 8\}$).
- [ ] **Custom Triton PagedAttention:** Dedicated Triton kernel for paged attention to unify non-contiguous gathering with FlashAttention-style online softmax.
- [ ] **AWQ / GPTQ 4-Bit Execution:** Native dequantization kernels for pre-packed 4-bit checkpoints.
- [ ] **Structured Speculative Decoding (Future / In Development):** Draft-target speculative verification with multi-candidate trees (not available in initial release; CLI command omitted pending full multi-candidate validation).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
