# IREE CPU GEMM / GEMV / Conv2D

[English](README.md) | [简体中文](README.zh-CN.md)

Use the shared uv environment, the `llvm-cpu` backend, and the `local-sync` driver. CPU affinity, thread settings, and input configuration match the PyTorch baseline.

```bash
cd /workspaces/ml-compiler/gemm-bench
uv sync --locked
uv run --frozen python run.py --backends iree --check
uv run --frozen python run.py --backends iree \
  --cases pack,mv_pack,conv_small --dump --output results/iree-multiop
# The individual entry point uses the output directory directly
uv run --frozen python iree/run.py --ops conv2d --cases conv_small \
  --dump --output results/iree-conv-direct
```

GEMM uses `linalg.matmul`. GEMV uses `linalg.matvec` with a one-dimensional output. Conv2D uses [linalg.conv_2d_nchw_fchw](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nchw_fchw-linalgconv2dnchwfchwop), with `tensor.pad` when needed. Inputs are NCHW and weights are OIHW, without bias, with groups=1 and dilation=1. Stride and padding come from the named case.

The compiler lowers static MLIR to VMFB. The Python runtime is used only for warmup and correctness checks. Timing uses native `iree-benchmark-module`, the exact A.npy/B.npy inputs, and the main function. Each batch runs for the minimum duration. If the native iteration count is below reps, the batch is repeated with reps iterations. The selected native JSON is saved. Nine batches alternate with PyTorch; the result is the median per-call `real_time`.

IREE timing includes VM calls, dispatch, operator execution, and output allocation. It excludes compilation, Python, input loading, and export. PyTorch preallocates GEMM/GEMV outputs; its Conv2D timing includes framework output allocation. Results record both scopes, and speedup is PyTorch / IREE.

Backend configurations support the default pipeline, data tiling, and delayed materialization:

```bash
uv run --frozen python run.py --backends iree --ops conv2d \
  --iree-config config/iree-dt.toml --output results/iree-conv-dt
uv run --frozen python run.py --backends iree --ops conv2d \
  --iree-config config/iree-dt-late.toml --output results/iree-conv-dt-late
```

DT enables `--iree-opt-data-tiling=true` and `--iree-llvmcpu-enable-ukernels=all`. The late variant also disables early materialization. Measure on your machine to determine whether these options improve performance; historical model-level timings do not estimate individual operator performance. The default CPU target comes from the shared `mcpu` setting.

The unified entry point exports to `<output>/iree/artifacts/<operator_dims>/`. The retained filenames `matmul.mlir` and `matmul.vmfb` contain the selected operator. Dumps include compiler commands/logs, intermediate IR, CPU executables, and VM bytecode disassembly. The `baseline/` directory contains the measured PyTorch wrapper and operator dispatch information. Inputs, results, and a SHA256 manifest are included.

See the [project README](../README.md) for shared configuration, case formats, and selection options.

Default CPU optimization: TVM uses `[tvm].schedule = "llvm"`: one fixed-seed generation with `ms.ScheduleRule.create("llvm")`, `mutator_probs={}`, no search strategy or cost model, and direct selection of candidate 0. Target-selected postprocessing finalizes annotations, matching the default-rule experiment. Rejected candidate 0 fails; it never silently switches to a different candidate. AutoTune overrides this mode when enabled. DLight and naive remain explicit alternatives. IREE defaults to DT + `ukernels=all`; `config/iree-default.toml` preserves the unmodified compiler pipeline.
