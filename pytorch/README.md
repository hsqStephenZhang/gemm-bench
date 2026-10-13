# PyTorch: the CPU baseline for every operator

[English](README.md) | [简体中文](README.zh-CN.md)

PyTorch always runs. TVM, IREE, and OpenBLAS are measured alternately with it. The `eager-native` mode in `config/pytorch.toml` supports GEMM, GEMV, and Conv2D. It uses the shared uv environment, a fixed CPU, and one intra-op and inter-op thread.

```bash
cd /workspaces/ml-compiler/gemm-bench
uv sync --locked
uv run --frozen python run.py --backends none --check
uv run --frozen python run.py --backends none \
  --cases pack,mv_pack,conv_small --dump --output results/torch-baseline
# The individual entry point uses --output directly as its backend directory
uv run --frozen python pytorch/run.py --ops conv2d --cases conv_small \
  --dump --output results/torch-conv-direct
```

The native C++ wrapper references generated input arrays through ATen `from_blob`. GEMM calls `at::mm_out` and GEMV calls `at::mv_out`, both with preallocated outputs. Conv2D calls `at::convolution` without bias, with groups=1 and dilation=1; its timing includes output allocation and internal packing. Every native timing loop uses InferenceMode. Copying the result into the NumPy buffer happens after timing. GEMM/GEMV use the wheel's BLAS; Conv2D uses the CPU implementation PyTorch selects for that case. The project does not assume one implementation handles every convolution.

The wrapper first checks its output against the corresponding Python API, then against the shared FP64 reference. After warmup, native iteration counts are calibrated to meet both the case's reps floor and the minimum timing duration. The result is the median per-call average across nine batches. Standalone baseline rows have speedup 1. Compilation, Python, input loading, and export are excluded from timing.

The unified entry point's `--pytorch-config` controls the baseline for every backend. Configuration and build information are recorded in `baseline_config` and `baseline_metadata`, preventing incompatible PyTorch settings from being mixed. The CPU wheel version and dependencies are locked in `uv.lock`.

The unified entry point exports to `<output>/pytorch/artifacts/<operator_dims>/`. Files include `native_timer.cpp`, `pytorch_timer.so`, compile commands/logs, `operator.txt`, `dispatch_table.txt`, `timer_wrapper.s`, PyTorch build/dependency information, exact inputs, results, and a SHA256 manifest. `operator.txt` describes mm.out, mv.out, or convolution according to the selected operator. The assembly is wrapper disassembly; actual ATen kernels reside in precompiled libraries.

See the [project README](../README.md) for shared configuration, case formats, and selection options.

## PyTorch modes

`[pytorch].mode` supports `"eager"` (default) and `"inductor"`. CLI `--pytorch-mode` overrides the configuration; `config/pytorch-inductor.toml` is also provided. Paired baselines always use eager, even when the measured PyTorch runner uses Inductor.

```bash
uv run --frozen python run.py --backends none --pytorch-mode inductor \
  --ops all --cpu 0 --dump --output results/pytorch-inductor
uv run --frozen python pytorch/run.py --pytorch-mode eager --cases pack
```

Inductor uses `torch.export.export` → `torch._inductor.aoti_compile_and_package` → native C++ `AOTIModelPackageLoader::run`. It measures CPU Inductor compiled graphs with no Python calls in the timing loop, using the [official AOTInductor deployment path](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html). This is AOTInductor native invocation rather than timing a Python `torch.compile` callable. Inputs are runtime arguments, not constant weights. Output allocation, graph invocation and internal packing are included; compilation, graph export and correctness copies are excluded. Default `max_autotune=false`, `cpp.threads=1`; compilation errors stop the run rather than falling back to eager. `--dump` retains the exported program, FX graph, measured `.pt2` package, generated C++/shared libraries and native timer wrapper. Inductor and eager results need separate output directories.
