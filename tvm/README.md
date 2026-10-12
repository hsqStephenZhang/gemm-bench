# TVM CPU GEMM / GEMV / Conv2D: optional auto tuning

[English](README.md) | [简体中文](README.zh-CN.md)

The unified entry point `run.py` runs the PyTorch baseline first, followed by the selected optional backends. Its `--output` is a root directory, with TVM results under the `tvm/` subdirectory. The individual entry point uses `--output` directly as the TVM directory and still measures PyTorch alternately as the reference.

## Environment

Use the `apache-tvm` wheel in the project's uv environment. The current locked version is 0.27.0.post1. uv installs the compiler, runtime, and matching FFI; no local TVM source tree or build/lib is required. See the [official TVM installation guide](https://tvm.apache.org/docs/install/).

```bash
cd /workspaces/ml-compiler/gemm-bench
uv add apache-tvm
# Use uv sync --locked to reproduce the locked environment
uv run --frozen python tvm/run.py --check
```

The launcher loads TVM and its bundled libraries directly from `.venv`, without copying or replacing the FFI. `--check` reports the package version, installation path, LLVM support, and thread settings. Result metadata records the wheel version, package path, and compiler build information.

## Auto tuning controls

Backend configuration in `config/tvm.toml`:

```toml
[tvm.auto_tune]
enabled = false
trials = 64
```

| Mode | Execution path | Result markers |
|---|---|---|
| Default / --no-auto-tune | Original operator TIR → default LLVM lowering/optimization | auto_tune=false, variant=untuned |
| --auto-tune | MetaSchedule-selected schedule → LLVM | auto_tune=true, variant=auto_tuned |

Untuned mode applies no manual tiling, schedule search, or database reuse. It is the original TIR reference implementation and should not be interpreted as a high-performance default GEMM implementation. Tuned mode uses EvolutionarySearch + RandomModel, with 64 candidates per case by default.

```bash
# Disable tuning
uv run --frozen python run.py --backends tvm --no-auto-tune \
  --cases pack,k49 --dump --output results/tvm-untuned

# Enable tuning; a fresh directory starts a new search
uv run --frozen python run.py --backends tvm --auto-tune --trials 64 \
  --cases pack,k49 --dump --output results/tvm-auto-tuned

# The individual entry point supports the same switches
uv run --frozen python tvm/run.py --auto-tune --trials 64 \
  --cases pack --output results/tvm-direct-tuned

# Use --tvm-config at the root, --config at the individual entry point
uv run --frozen python run.py --backends tvm --tvm-config config/tvm.toml
uv run --frozen python tvm/run.py --config config/tvm.toml --no-auto-tune
```

CLI switches override `[tvm.auto_tune].enabled`. `--trials` alone does not enable tuning. When multiple backends are selected, tuning switches are forwarded only to TVM. Disabled mode creates no builder/runner, reads or writes no tuning database, and records `tuning_s=0` and `valid_records=0`.

Use separate output directories for different modes; configuration checks prevent tuned and untuned results from being mixed. In enabled mode, a compatible directory whose record count reaches the budget can reuse its database. Use a fresh directory for a new search.

## Shared cases

Dimensions, CPU, device, seed, reps, tolerances, and shared baseline settings come from `config/common.toml`. PyTorch-specific settings come from `config/pytorch.toml`, or the unified entry point's `--pytorch-config`. The default `cases_files` contain 32 cases: 16 GEMM, 10 GEMV, and 6 Conv2D, in config/gemm_cases.txt, gemv_cases.txt, and conv2d_cases.txt. GEMM case files use M K N order; legacy `--shapes` retains M,N,K order.

Entry points support `--ops gemm,gemv,conv2d`, `--cases-file`, `--cases skinny,pack,k49`, and `--reps 200`. Reps is the minimum native calls per timing batch; minimum-duration calibration may increase it.

GEMV uses a one-dimensional output and reduction. Conv2D uses NCHW/OIHW reduction with stride/padding boundary checks. All three operators share the same `[tvm.auto_tune]` configuration and CLI switches. Output shapes come from the shared operator definitions.

```bash
uv run --frozen python run.py --backends tvm --ops gemv,conv2d \
  --cases mv_pack,conv_small --auto-tune --trials 64 --dump \
  --output results/tvm-multiop-tuned
```

## Benchmark method and timing boundaries

1. Shared code generates FP32 row-major A/B and an FP64 reference, pins the process to the shared CPU, and sets TVM execution threads to one.
2. When tuning is enabled, the MetaSchedule runner defaults to `number=1, repeat=3, min_repeat_ms=20` per candidate for schedule selection. Tuning time is recorded separately. Four builders can compile on CPUs 2/4/6/8; candidate measurements remain on shared CPU 0 with one execution thread. Adjust the builder CPU list on other machines.
3. Compile the final module, copy A/B into TVM tensors, and preallocate C. Shared code runs five warmups and checks correctness.
4. Both modes use the same native timing:

```python
evaluator = lib.time_evaluator(
    "main", tvm.cpu(), number=case_reps, repeat=1, min_repeat_ms=100
)
seconds_per_call = evaluator(a, b, c).results[0]
```

The native evaluator adaptively increases calls to meet the minimum duration and returns average seconds per call. Shared code measures TVM alternately with PyTorch for nine batches and takes the median batch average. Timing includes native TVM calls and kernel execution, and excludes Python, input copies, output preallocation, tuning, and compilation. The API does not expose the final adaptive count; `backend_iterations` is null.

`compile_s` is recorded separately. Summary variants distinguish the two modes. Inputs are reused without an explicit cache flush.

## Dump final artifacts

```bash
uv run --frozen python run.py --backends tvm --no-auto-tune \
  --cases pack --dump --output results/tvm-untuned-dump

uv run --frozen python tvm/run.py --auto-tune --trials 64 \
  --cases pack --output results/tvm-tuned-dump --dump-dir results/tvm-artifacts
```

The unified entry point exports to `<output>/tvm/artifacts/<operator_dims>/`; the individual entry point uses `<output>/artifacts/<operator_dims>/`. With `--dump-dir`, the unified entry point adds `tvm/`, while the individual entry point adds only the operator/dimension directory.

| File | Contents |
|---|---|
| final_tir.py | The final TIR actually compiled; the original operator TIR when tuning is disabled |
| final_trace.txt | The selected schedule trace, or an explicit statement that no schedule transformations were applied |
| module.so | The shared library measured in this run |
| kernel.ll / kernel.s | Optimized LLVM IR / CPU assembly from the measured build |
| A.npy / B.npy | Exact inputs |
| benchmark_result.json / manifest.json | Raw timings, errors, mode, parameters, and file SHA256 hashes |

Enabled mode also retains compatibility files best_schedule.py, best_trace.txt, best.so, and the tuning database. Disabled mode does not produce these best-schedule files. Export happens outside timing.

## Historical auto tuning results

Earlier local experiments stored six original cases in `tvm/reference-results/` and a later 16-case run, with 64 valid tuning records per case, in `results/expanded-cpu-20261012/`. These historical local artifacts are not included in this checkout. They used the previous local source build with auto_tune=true and do not represent the current wheel's performance. Use a fresh output directory for the wheel.

If you have the historical artifacts, inspect the original schedule with:

```bash
cat tvm/reference-results/1024_1024_1024/best_schedule.py
cat tvm/reference-results/1024_1024_1024/best_trace.txt
```

Reuse databases only with matching TVM versions, builds, and CPU targets. After switching to the wheel, search again in a fresh directory to avoid reusing records from the old source build.

## VS Code / ty import resolution

The `[tool.ty.environment]` section of `pyproject.toml` points to `.venv`. ty resolves `tvm.script` and `tvm.s_tir` from the installed `apache-tvm` package; local source extra-paths are unnecessary. If old diagnostics persist after saving configuration, run `ty: Restart server` or `Developer: Reload Window` from the command palette.

All new results use the PyTorch baseline; historical records retain their OpenBLAS baseline. The `baseline/` dump contains the measured PyTorch wrapper. See the [project README](../README.md) for Conv2D output allocation boundaries.
