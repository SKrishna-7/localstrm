# LocalStrm Installation and Environment Setup Guide

This guide describes how to configure a reproducible Python environment for LocalStrm on Linux with NVIDIA GPU acceleration.

---

## 1. System Requirements & Hardware Targets

- **Operating System:** Linux verified (Ubuntu 22.04 LTS or newer), Windows untested/best-effort.
- **Python:** Tested on **Python 3.11.15** (compatible with Python 3.10+).
- **GPU Target:** NVIDIA GPU with CUDA Compute Capability $\ge 7.0$ for basic engine inference.
  - *Note on INT8 GEMM (`hybrid_int8`):* Requires **CUDA Compute Capability $\ge 8.0$ (Ampere / Ada Lovelace / Hopper)**. The engine's runtime check (`src/kernels/quantization.py:can_use_int8_gemm`) enforces `capability >= (8, 0)` and will automatically fall back to cached FP16 dequantization if run on older architectures (e.g. Turing SM 7.5 or Volta SM 7.0).
- **GPU Driver:** NVIDIA driver supporting CUDA 12.1+ or CUDA 13.x.


---

## 2. Environment Setup (Step-by-Step)

### Step 1: Create a Clean Virtual Environment

```bash
# Create a dedicated Python 3.11 virtual environment
python3.11 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

### Step 2: Install PyTorch with CUDA Acceleration

PyTorch must be installed with CUDA support before installing the remaining requirements. Generic `pip install torch` may install CPU-only binaries on some systems.

Select the wheel channel corresponding to your installed NVIDIA driver:

```bash
# For CUDA 12.1 (Recommended for most modern drivers):
pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4:
pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu124
```

> **Note:** Reported benchmarks used PyTorch 2.13.0+cu132 and Python 3.11.15 on CUDA 13.2.

Verify that CUDA is accessible in PyTorch:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 3. Install LocalStrm Dependencies

Choose the dependency profile suited to your intended workflow:

### Option A: Core Runtime Inference Only
If you only plan to run local inference via the CLI (`cli.py engine`, `cli.py chat`):

```bash
pip install -r requirements.txt
```

### Option B: Runtime + OpenAI-Compatible HTTP Serving
If you want to run the REST API server (`cli.py serve`):

```bash
pip install -r requirements.txt
pip install -r requirements-serving.txt
```

### Option C: Full Development, Testing & Benchmarking
If you plan to run automated unit tests (`pytest tests/`) or benchmark suites:

```bash
pip install -r requirements-dev.txt
```

---

## 4. Environment Variables Configuration

LocalStrm requires the repository root to be in the Python module search path. Set the following environment variables in your shell or `.env`:

```bash
# Mandatory: enable root module resolution (e.g. import src.engine.llm_engine)
export PYTHONPATH=.

# Optional: Default model ID and device
export MODEL_ID="Qwen/Qwen2.5-0.5B"
export ENGINE_DEVICE="cuda"

# Recommended for offline / cached-model execution (prevents network timeouts):
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 5. Post-Install Verification (Offline vs. Model-Dependent Tests)

The test suite contains 19 test modules collecting 240 total test cases. Tests are categorized into two tiers based on external model dependencies:

### Tier 1: Verified Offline Test Suite (16 Modules, 229 Tests)
These 16 test modules execute purely on synthetic tensors, mathematical mocks, or simulated GPU state, requiring **zero internet connectivity and no downloaded model weights**:

```bash
pytest tests/test_all_features.py \
       tests/test_api_server.py \
       tests/test_cli_reliability.py \
       tests/test_cli_serving.py \
       tests/test_gqa_generalization.py \
       tests/test_http_serving.py \
       tests/test_hybrid_int8.py \
       tests/test_memory_constrained.py \
       tests/test_metrics_lifecycle.py \
       tests/test_preflight.py \
       tests/test_quantized_loader.py \
       tests/test_reliability.py \
       tests/test_sampling_parity.py \
       tests/test_scheduler_transitions.py \
       tests/test_serving_adapter.py \
       tests/test_vram_budget.py -v
```

### Tier 2: Model-Dependent Integration Suite (3 Modules)
The following 3 test modules invoke `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")` to validate multi-step generation, long context, and batched execution on real model weights. They require either an active internet connection to Hugging Face or pre-cached local weights in `~/.cache/huggingface/hub/`:

- `tests/test_batched_runner.py`
- `tests/test_int8_long_context.py`
- `tests/test_int8_multi_step.py`

To execute these tests with pre-cached weights:
```bash
pytest tests/test_batched_runner.py tests/test_int8_long_context.py tests/test_int8_multi_step.py -v
```

### Preflight Diagnostics on a Local Checkpoint
To test hardware capacity estimation without loading model weights:
```bash
python cli.py check --model /path/to/local/Qwen2.5-0.5B/snapshot
```

---

## 6. Dependency Troubleshooting

1. **`ModuleNotFoundError: No module named 'src'`**:
   - Cause: `PYTHONPATH` is not set.
   - Solution: Run `export PYTHONPATH=.` from the repository root.
2. **`ModuleNotFoundError: No module named 'openai'`**:
   - Cause: Running `test_api_server.py` without testing dependencies.
   - Solution: Run `pip install -r requirements-dev.txt`.
3. **`pynvml` Deprecation Warning**:
   - Cause: `pynvml` emits a FutureWarning.
   - Solution: Install `nvidia-ml-py` (`pip install nvidia-ml-py`), which is pinned in `requirements-dev.txt`.
