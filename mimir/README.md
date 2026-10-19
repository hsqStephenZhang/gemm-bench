# MimIR CPU GEMM / GEMV / Conv2D

[English](README.md) | [简体中文](README.zh-CN.md)

MimIR is an optional backend. PyTorch remains the required baseline. The adapter follows the tensor extern ABI and operator generation in [MimIR PR #1](https://github.com/hsqStephenZhang/MimIR/pull/1), while using this project's shared inputs, FP64 correctness checks, pinned single-thread CPU execution, native timing batches, and artifact manifests.

## Remote, ref, and persistent build directory

```toml
# config/common.toml
[build]
directory = "build"

# config/mimir.toml
[mimir]
remote = "mimir/mimir"
ref = "master"
jobs = 4
```

`remote` accepts a GitHub `owner/repo`, an HTTPS/SSH URL, or a local Git remote. `ref` accepts a branch, tag, commit, or pull ref. The default is the upstream `mimir/mimir` repository's `master`.

On the first run, the adapter clones into `<build>/mimir/source`. Later runs reuse that checkout: they fetch the requested ref, switch to its resolved commit, update submodules, and incrementally rebuild in `<build>/mimir/compiler`. Switching remotes updates origin in the same checkout. It does not create a clone per branch or per benchmark output directory. Local checkout edits are preserved; switching stops if the checkout is dirty.

Build prerequisites: Git, CMake, Ninja, a C++23-capable compiler for MimIR, and clang++ for the emitted LLVM IR. CMake defaults to Release with Python bindings and upstream tests disabled. Build logs and commands are saved under `<build>/mimir/`; `checkout-state.json` records the remote, requested ref, resolved commit, and paths. The build directory is ignored by Git. `--check` prepares the toolchain but does not time operators or export per-case artifacts.

```bash
cd /workspaces/ml-compiler/gemm-bench
uv sync --locked
uv run --frozen python run.py --backends mimir --check

# Only upstream MimIR master versus PyTorch; four representative cases
uv run --frozen python run.py --backends mimir --mimir-ref master \
  --cases pack,mv_pack,conv_small,conv_1x1 --reps 1 \
  --dump --output results/mimir-master-vs-pytorch

# Use a user's fork and PR ref in the same build checkout
uv run --frozen python run.py --backends mimir \
  --mimir-remote hsqStephenZhang/MimIR --mimir-ref refs/pull/1/head \
  --cases pack --output results/mimir-pr1

# Select another build root or backend configuration
uv run --frozen python run.py --backends mimir --build-dir build \
  --mimir-config config/mimir.toml --cases mv_pack --output results/mimir-gemv

# Individual entry point: --remote / --ref / --config
uv run --frozen python mimir/run.py --remote mimir/mimir --ref master \
  --ops conv2d --cases conv_small --dump --output results/mimir-direct
```

CLI options override the config. Use a fresh results directory when changing the ref, remote, or compiler settings; the build checkout and compiler directory are still reused. For reproducibility, use the recorded commit as `--mimir-ref` instead of a moving branch.

## Operators and timing

GEMM uses `tensor.product_2d`. GEMV uses `tensor.dot_product` with a genuine one-dimensional vector input and output. Conv2D uses `tensor.conv`: NCHW inputs, OIHW weights, no bias, groups=1, dilation=1, and the case's stride/padding. Convolution extents are passed explicitly to avoid singleton-axis inference ambiguity.

Each case produces MimIR source, runs the `opt` and `ll` plugins, and compiles the emitted LLVM IR plus a native C++ wrapper into a shared library. The default clang flags are `-O3` and `-mcpu=<benchmark.mcpu>` on ARM / `-march=<benchmark.mcpu>` on x86, matching the shared CPU target. With `mcpu=generic`, no CPU-specific template flag is added. Override `target_flags` for other architectures or targets.

The tensor extern returns a malloc-owned result. Native timing includes the operator, output/internal allocation, and freeing the result on every call. Correctness copies into a NumPy buffer happen outside timing. PyTorch preallocates GEMM/GEMV outputs, while its Conv2D timing includes framework allocation; both timing scopes are recorded. No Python-overhead subtraction is applied.

After shared warmup and correctness checks, MimIR alternates with PyTorch for nine batches, each at least 100 ms by default and at least the case's reps count. Results use the median per-call average, with `speedup = PyTorch / MimIR`. Compilation and export are excluded. The exact compiler commit, binary/plugin/import SHA256 hashes, clang version, flags, and remote/ref are recorded.

## Dump

The unified entry point exports to `<output>/mimir/artifacts/<operator_dims>/`; the individual entry point uses `<output>/artifacts/<operator_dims>/`.

| File | Contents |
|---|---|
| kernel.mim | Generated source for the selected operator |
| final.mim | MimIR's final optimized program dump |
| kernel.ll | LLVM IR emitted by MimIR |
| optimized_kernel.ll / kernel.s | clang-optimized LLVM IR / assembly with the measured target flags |
| kernel.so / native_timer.cpp | The measured shared library and native timer source |
| mimir_command.json / clang_command.json | Compilation commands |
| mimir_compile.log / clang_compile.log | Compiler logs |
| baseline/ | Measured PyTorch wrapper and operator dispatch information |
| A.npy / B.npy / benchmark_result.json / manifest.json | Exact inputs, results, environment, and SHA256 manifest |

See the [project README](../README.md) for shared operator cases and baseline configuration. PR #1 also contains network benchmarks; this adapter covers the current GEMM/GEMV/Conv2D suite only.
