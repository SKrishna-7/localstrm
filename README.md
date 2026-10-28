# Mini-vLLM

> **A lightweight, from-scratch LLM inference engine built for memory-constrained consumer GPUs, featuring PagedAttention, continuous batching, prefix caching, and hybrid INT8 execution.**

[![Python: 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](INSTALL.md)
[![PyTorch: 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Test Suite: 193 Tests](https://img.shields.io/badge/Test%20Suite-193%20Tests-brightgreen.svg)](tests/)

---



## Pre-Download Model & VRAM Preflight

One of Mini-vLLM's core capabilities is its **Pre-Flight Memory & Hardware Guard** ([`src/memory/preflight.py`](src/memory/preflight.py)), accessible via the CLI:

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
1. **Lightweight Metadata Fetching:**<br>
   Before downloading multi-gigabyte tensor files, the preflight guard inspects the repository metadata via `huggingface_hub.HfApi.model_info()` or downloads only the lightweight `config.json` (~3 KB). For local directories, it scans file sizes on disk without loading weights into RAM or VRAM.
2. **Hardware Capacity Inspection:**<br>
   Queries `torch.cuda.mem_get_info()` for free VRAM, `psutil.virtual_memory()` for available host RAM, and `shutil.disk_usage()` for target cache storage.
3. **Multi-Stage Fit Calculations:**<br>
   - **Weight Memory:** Scaled by precision factor (1.0 for FP16, 0.50 for INT8, 0.55 for Hybrid INT8, 0.25 for INT4).
   - **KV-Cache Footprint:** Exact analytical formula: $2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes\_per\_elem} \times \text{max\_context} \times \text{batch\_size}$.
   - **Host RAM Staging Check:** When using quantized modes (`int8` or `hybrid_int8`), uncompressed FP16 weights are loaded into host RAM first for in-place quantization before transferring compressed weights to GPU. Preflight verifies host RAM can hold uncompressed weights before allowing download.
   - **Disk Space Check:** Verifies sufficient free disk space for uncompressed checkpoints plus a 10% safety buffer.
4. **Fail-Closed Safety:**<br>
   If model metadata cannot be resolved (e.g. invalid repository ID, network failure, or inaccessible model), preflight aborts with **exit code 1** and actionable guidance. It does not attempt unauthenticated blind downloads.
5. **Estimate vs. Guarantee:**<br>
   Preflight provides a **rigorous analytical estimate**, not an absolute operating guarantee. Dynamic activation memory varies with runtime batch composition, and CUDA allocator fragmentation can lead to allocation errors if background processes consume VRAM.

---

## Project Overview & Target Audience

Serving modern large language models on consumer hardware is constrained by GPU memory architecture. On an entry-level GPU (e.g. 6 GB VRAM with a 96-bit memory bus), running a 3-billion parameter model in FP16 exceeds the physical memory limit (~6.17 GB required for weights alone). Dynamic 8-bit quantization fits the weights (~3.48 GB), but repeated on-the-fly dequantization during single-token decoding collapses throughput.

**Mini-vLLM** is an educational, production-pattern LLM inference engine built from first principles in PyTorch. It bridges the gap between toy autoregressive scripts and massive enterprise C++ engines (such as vLLM or TensorRT-LLM).

### Target Audience
- **Systems & ML Engineers:** Understanding the mechanics of PagedAttention, continuous batching schedulers, Radix prefix caches, and quantization kernels.
- **Researchers & Students:** A readable, modular Python codebase implementing core modern serving invariants.
- **Edge & Consumer GPU Developers:** Running 0.5B to 3B parameter models locally on 6 GB consumer GPUs with realistic throughput.

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
│   │   ├── prefix_cache.py    # Content-addressed Radix tree block deduplication
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
1. **PagedAttention & Physical Block Manager:**<br>
   Treats KV-cache memory as fixed-size physical blocks (16 tokens/block), dynamically mapped via page tables. Completely eliminates external fragmentation and expensive contiguous tensor reallocation.
2. **Continuous (Iteration-Level) Batching:**<br>
   Evaluates decode steps iteration by iteration. Completed requests depart immediately, and arriving requests are scheduled into active iterations without waiting for entire batch completion, eliminating the inactive sequence-slot scheduling waste inherent in locked static batching (measured up to 28.8% on mixed-length workloads).
3. **Automatic Prefix Caching:**<br>
   Computes SHA-256 content hashes of token block sequences. Common system prompts or conversation prefixes are matched in a Radix cache, restoring KV state in zero time without re-running prefill compute.
4. **Hybrid INT8 Execution Mode:**<br>
   An opt-in precision mode (`--quant hybrid_int8`). Attention projections are cached in FP16 at initialization to remove dequantization overhead in the attention path, while MLP projections (`gate_proj`, `up_proj`, `down_proj`) execute integer matrix multiplications via `torch._int_mm`.
5. **CPU-Staged Quantized Loading:**<br>
   When loading quantized models on CUDA, uncompressed FP16 weights are loaded into host RAM first, quantized in place, and only compressed weights are transferred to GPU VRAM. Full uncompressed weights never touch GPU memory.
6. **OpenAI-Compatible REST Server:**<br>
   Production FastAPI server implementing `/v1/chat/completions` (supporting streaming Server-Sent Events and non-streaming responses), `/v1/models`, `/health`, and Prometheus `/metrics`.

---

## Installation & Quick Start

For detailed step-by-step instructions, see [`INSTALL.md`](INSTALL.md).

### 1. Requirements & Prerequisites
- Linux (Ubuntu 22.04 LTS or newer verified).
- Python 3.10 or 3.11 (tested on **Python 3.11.15**).
- NVIDIA GPU with CUDA Compute Capability $\ge 7.0$ for basic inference; **Compute Capability $\ge 8.0$ (Ampere / Ada Lovelace or newer)** is required for `hybrid_int8` mode.

### 2. Setup Virtual Environment & PyTorch
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch matching your NVIDIA driver (CUDA 12.1+ recommended):
pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Mini-vLLM Dependencies
```bash
# Core inference CLI:
pip install -r requirements.txt

# Or with OpenAI HTTP serving support:
pip install -r requirements-serving.txt

# Or full development, testing, and benchmark suite:
pip install -r requirements-dev.txt
```

### 4. Mandatory Environment Variable
```bash
export PYTHONPATH=.
```

### 5. Quick-Start Commands

```bash
# 1. Run preflight memory check on a model
python cli.py check --model Qwen/Qwen2.5-0.5B --device cuda

# 2. Run single-request inference via unified engine
python cli.py engine --model Qwen/Qwen2.5-0.5B --prompt "Explain operating systems in one sentence" --temperature 0.0

# 3. Interactive streaming chat session
python cli.py chat --model Qwen/Qwen2.5-0.5B

# 4. Launch OpenAI-compatible REST server (listening on 127.0.0.1:8000)
python cli.py serve --engine mini-vllm --model Qwen/Qwen2.5-0.5B --host 127.0.0.1 --port 8000
```

---

## Benchmark Study: Real-Prompt Scheduling Waste

All reported benchmarks were measured on the reference consumer testbed: **NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, Ada Lovelace SM 8.9, 96-bit memory bus, Linux x86_64, PyTorch 2.13.0+cu132, CUDA 13.2)**.

> [!NOTE]
> **Benchmark Reporting Scope:**<br>
> The figures below summarize measurements recorded during engine development and validation on the reference testbed. Raw benchmark JSON result files are not bundled in this initial public snapshot to focus the repository on the core inference codebase. Standard benchmark runners (`benchmarks/run_benchmark.py` / `cli.py perf-benchmark`) remain available for local execution.

### Real-Prompt Scheduling-Waste Benchmark: Static Batching vs. Continuous Scheduling

To evaluate sequence-slot scheduling waste under realistic inputs, a fresh, controlled 90-run benchmark was conducted comparing **Hugging Face static tensor batching (FP16)** against **Mini-vLLM continuous batching (FP16 and INT8 KV cache)**.

#### Experimental Context & Parameters
- **Model:** `Qwen/Qwen2.5-0.5B` (0.5B parameters, FP16)
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB physical VRAM, Ada Lovelace SM 8.9, CUDA 13.2)
- **Prompt Dataset:** 16 natural-language AI systems questions (e.g. self-attention, tokenization, paged attention, KV caching) without artificial padding or repetitive filler tokens. Prompt lengths ranged from **5 to 17 tokens** (reported mean: **11.3 tokens**).
- **Workloads:**
  - **Controlled Lengths (Workload A):** Common generation target of 20 tokens per request and simultaneous arrival ($t=0.0\text{s}$).
  - **Mixed Lengths (Workload B):** Genuinely variable generation targets ranging from 16 to 36 tokens, with staggered arrival offsets ($0.05\text{s}$ increments).
- **Concurrency Levels:** 1, 2, 4, 8, and 16 concurrent request slots.
- **Engines:** Hugging Face static batching (FP16), Mini-vLLM continuous batching (FP16 KV), and Mini-vLLM continuous batching (INT8 KV).
- **Repetitions & Scope:** 3 independent measured repetitions per test cell, totaling **90 executed runs** recorded in the benchmark manifest.
- **Methodology Note:** The prompts were genuine natural-language questions; output token limits and arrival intervals were controlled benchmark parameters to evaluate scheduling boundaries, not naturally observed user session traffic.

#### Measured Sequence-Slot Scheduling Waste

The metric measures **inactive sequence-slot scheduling waste** as defined by the benchmark implementation: the ratio of inactive/padded sequence-step slots executed inside batched decode steps to total sequence-step slots allocated ($\text{Wasted Slots} / [\text{Useful Slots} + \text{Wasted Slots}]$). It represents sequence-slot scheduling inefficiency, not a hardware GPU-idle-cycle counter or a direct measurement of total GPU compute.

For the mixed-length workload, the recorded sequence-slot waste rates and per-run wasted slot counts were:

| Concurrency | HF Static Sequence-Slot Waste | HF Wasted Sequence-Step Slots | Mini-vLLM FP16 Waste | Mini-vLLM INT8 Waste |
|---:|---:|---:|---:|---:|
| **C = 1** | 0.0% | 0 | 0.0% | 0.0% |
| **C = 2** | 15.3% | 74 | 0.0% | 0.0% |
| **C = 4** | 23.5% | 126 | 0.0% | 0.0% |
| **C = 8** | 28.8% | 166 | 0.0% | 0.0% |
| **C = 16** | 28.8% | 166 | 0.0% | 0.0% |

> [!NOTE]
> **Controlled vs. Mixed Workload Waste:**
> - **Controlled Workload Reference:** In Workload A (where all 16 questions shared a uniform 20-token target limit and none terminated prematurely on EOS), both Hugging Face static batching and Mini-vLLM recorded **0.0% sequence-slot waste** (0 wasted slots).
> - **Mixed Workload Inefficiency:** Workload B creates the informative real-world difference: because sequences finish at differing lengths (16 to 36 tokens), locked static batches must keep computing forward passes for completed sequence slots until the slowest sequence finishes. On the mixed-length Qwen2.5-0.5B workload, Hugging Face static batching recorded up to **28.8% inactive sequence-slot scheduling waste** (wasting exactly 166 sequence-step slots per run at $C \ge 8$), while Mini-vLLM recorded **0% waste** under the benchmark's sequence-slot metric by dynamically evicting finished sequences.

#### Mixed-Workload Output Throughput & Architectural Trade-offs

The corresponding mean output token throughput across the 3 independent repetitions was:

| Concurrency | HF Static FP16 | Mini-vLLM FP16 KV | Mini-vLLM INT8 KV |
|---:|---:|---:|---:|
| **C = 1** | 22.20 tok/s | 19.86 tok/s | 15.08 tok/s |
| **C = 2** | 35.45 tok/s | 35.12 tok/s | 26.73 tok/s |
| **C = 4** | 75.38 tok/s | 48.04 tok/s | 40.36 tok/s |
| **C = 8** | 146.83 tok/s | 99.44 tok/s | 65.40 tok/s |
| **C = 16** | 262.51 tok/s | 111.99 tok/s | 91.20 tok/s |

> [!IMPORTANT]
> **Interpreting Throughput vs. Scheduling Waste:**
> Eliminating measured inactive sequence slots does not automatically guarantee higher raw token throughput on a small 0.5B model. PyTorch's native static batching executes monolithic C++/CUDA GEMMs with minimal Python framework overhead. In contrast, Mini-vLLM executes dynamic iteration-level scheduling, ragged block gathering, and per-step bookkeeping. While Mini-vLLM achieves near-parity at $C=2$ ($35.12\text{ tok/s}$ vs. $35.45\text{ tok/s}$) and scales internally up to $5.64\times$ ($C=1 \to C=16$), static batching achieves higher raw token rates on this compact model when sequences are batched together.

### Methodological & Hardware Disclosures
1. **Measurement Boundary:** All latency timings are bounded by explicit `torch.cuda.synchronize()` calls before and after execution steps.
2. **Timing Definitions:**
   - **TTFT (Time to First Token):** Latency from request submission to the emission of the first token (includes prompt prefill).
   - **TPOT (Time Per Output Token):** Inter-token latency across decode steps.
   - **Request Throughput:** Calculated as generated output tokens divided by request duration.
3. **Tensor Core Attribution:** `hybrid_int8` MLP projections execute `torch._int_mm`, which calls cuBLASLt `cublasLtMatmul` with `CUDA_R_8I` input and `CUDA_R_32I` compute. While this library API configuration is architecturally routed to Tensor Cores on Ada Lovelace, independent hardware performance counter collection (such as NVIDIA Nsight Compute `ncu`) was not conducted.
4. **Scope Limitations:** Scheduling waste was evaluated across 16 real natural-language questions on `Qwen2.5-0.5B` under controlled output limits and arrival intervals. Results on other hardware, model architectures, or sequence lengths will vary.
5. **Benchmark Artifacts:** Raw benchmark measurement JSON files are not bundled in this release snapshot. Reported figures represent direct summaries of verified execution runs on the reference consumer testbed. Standard benchmark runners (`benchmarks/run_benchmark.py` / `cli.py perf-benchmark`) remain available for local execution.

---

## Tests, Supported Configurations & Known Limitations

### Test Suite
The repository includes an automated regression test suite covering all subsystems:
- **193 tests collected across 19 test modules** (`pytest tests/ --collect-only -q`).
- Test execution is split between **16 offline test modules** (executable without internet access or model downloads) and **3 model-dependent modules** (requiring local cached weights of `Qwen/Qwen2.5-0.5B`). See [`INSTALL.md`](INSTALL.md) for execution details.
- Tests validate: physical block allocation, block recycling, prefix cache radix trees, victim preemption under memory starvation, greedy sampling reproducibility, GQA dimensions, CPU-staged quantization loading, `torch._int_mm` fallbacks, preflight fail-closed guards, and live HTTP FastAPI lifecycle events.

### Supported Model Architectures
- **Supported:** Decoder-only transformer architectures following the LLaMA / Qwen2 design pattern (`model.layers[i]`, RMSNorm, RoPE, Grouped Query Attention).
- **Incompatible with `--engine mini-vllm`:** Encoder-decoder models (T5), non-standard architectures (Bloom, Falcon, GPT-2), and mixture-of-experts architectures without dense projections. For non-supported models, use `--engine hf`.

### Known Limitations
1. **Single-Node Focus:** Tailored for single-GPU consumer/workstation environments. Does not implement multi-node InfiniBand tensor parallelism.
2. **Serial Correctness Decode Path:** In multi-request engine execution, decode rows are processed independently to ensure token identity with serial Hugging Face reference execution. Experimental batched FP16 fast paths were rejected because batch-shape-dependent rounding caused greedy token divergence.
3. **Server Concurrency:** The OpenAI REST server provides asynchronous queueing and SSE streaming. Thread-safe concurrent submission is an MVP implementation; concurrent engine shutdown or recovery from unrecoverable CUDA kernel panics is not guaranteed.

---

## Roadmap

- [ ] **Multi-Bucket CUDA Graphs:** Multi-batch CUDA Graph capture for continuous decode batches ($B \in \{1, 2, 4, 8\}$).
- [ ] **Custom Triton PagedAttention:** Dedicated Triton kernel for paged attention to unify non-contiguous gathering with FlashAttention-style online softmax.
- [ ] **AWQ / GPTQ 4-Bit Execution:** Native dequantization kernels for pre-packed 4-bit checkpoints.
- [ ] **Structured Speculative Decoding (Future / In Development):** Draft-target speculative verification with multi-candidate trees (not available in initial release; CLI command omitted pending full multi-candidate validation).

---

## License

No license is currently provided. The repository is publicly viewable, but this does not grant general permission to use, modify, redistribute, or incorporate the code into other projects. Applicable copyright law and platform terms still apply.
