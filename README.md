# gemm-bench

[English](README.md) | [简体中文](README.zh-CN.md)

A single uv project for comparing CPU FP32 GEMM, GEMV, and Conv2D. **PyTorch is the required baseline for every operator**; TVM, IREE, OpenBLAS, and MimIR are optional comparison backends. Each backend is measured alternately with PyTorch in the same run. Speedup is `PyTorch median / backend median`.

```bash
cd /workspaces/ml-compiler/gemm-bench
uv sync --locked
uv run --frozen python run.py --backends all --check

# PyTorch + TVM + IREE: one small case per operator
uv run --frozen python run.py --backends tvm,iree \
  --cases pack,mv_pack,conv_small --dump --output results/multiop

# Only the PyTorch baseline, using all GEMV cases
uv run --frozen python run.py --backends none --ops gemv --output results/gemv-baseline

# Conv2D comparison with optional TVM auto tuning
uv run --frozen python run.py --ops conv2d --backends tvm,iree,openblas \
  --cases conv_small,conv_1x1 --auto-tune --trials 64 \
  --output results/conv-tuned --dump
```

TVM uses the wheel installed with `uv add apache-tvm`, with its version locked in `uv.lock`. No local source build is required. The current locked version is 0.27.0.post1. Use `uv sync --locked` to reproduce the environment.

## Shared configuration and selection

All configuration lives in `config/`. The `[benchmark]` section of `config/common.toml` controls the device, CPU affinity, single-thread execution, seed, dtype, case files, warmup, measurement batches, and error tolerances. CPU is currently the only implemented device; `--device gpu` reports an error.

```toml
[benchmark]
ops = ["gemm", "gemv", "conv2d"]
cases_files = ["config/gemm_cases.txt", "config/gemv_cases.txt", "config/conv2d_cases.txt"]

[backends]
tvm = true
iree = true
openblas = true
mimir = true
```

`--backends` (alias: `--backend`) overrides optional backend selection. It accepts any combination, `all`, or `none`. PyTorch runs automatically and cannot be disabled; omit it from the optional list. `--ops gemm,gemv` overrides `benchmark.ops`; `--cases name1,name2` selects named cases from the chosen operators; `--cases-file` replaces the configured file list. The legacy `cases_file` setting is still supported.

`--pytorch-config` controls both the standalone baseline run and the baseline interleaved with each backend. `--tvm-config`, `--iree-config`, and `--mimir-config` select backend configurations; `--common-config` selects the shared configuration. The individual entry points also accept shared options. Their `--config` selects the backend configuration, and their `--output` is used directly as that backend's directory:

```bash
uv run --frozen python tvm/run.py --ops gemv --cases mv_pack --auto-tune --trials 16
uv run --frozen python iree/run.py --ops conv2d --cases conv_small
uv run --frozen python pytorch/run.py --ops all --cases pack,mv_pack,conv_small
```

CPU selection uses the process's allowed affinity mask. With no global or per-runner CPU specified, every backend uses the lowest allowed CPU; CPU 0 is not required. An explicit CPU outside that mask fails with the allowed CPU list. The unified entry point validates the selection before starting a backend and forwards the same CPU to all measurements. Results record both the selected CPU and the original allowed mask. For example, `taskset -c 2,4 uv run --frozen python run.py --backends none --cases mv_pack` selects CPU 2, provided those CPUs are available.

## Runner CPU assignments and parallel execution

Configure `[runner_cpus]` in `config/common.toml`, or pass `--runner-cpus pytorch=0,tvm=2,iree=4,openblas=6,mimir=8`. CPU precedence is: a CLI runner entry, global `--cpu`, a configured runner entry, `benchmark.cpu`, then the lowest allowed CPU. Individual entry points honor their configured runner CPU too. All selected assignments are validated before launching any runner.

Each backend and its interleaved PyTorch baseline use the same assigned core. The standalone PyTorch runner may use another core. JSON/CSV record `cpu` and `baseline_cpu`; speedup uses the baseline measured on that backend's core.

`--parallel` runs selected backends concurrently, with a separate `<output>/<backend>.log`; each runner must have a distinct logical CPU. Prefer CPUs from different physical cores, of the same core type. Concurrent runners still share cache and memory bandwidth. `--no-parallel` forces sequential execution. The default is `benchmark.parallel = false`.

```bash
uv run --frozen python run.py --backends all --parallel \
  --runner-cpus pytorch=0,tvm=2,iree=4,openblas=6,mimir=8 \
  --dump --output results/full-suite
```

TVM automatic thread binding is disabled with `TVM_BIND_THREADS=0`, so it cannot reset the main thread to CPU 0. The harness verifies actual affinity after warmup and every timing batch; drift stops the run instead of silently recording the requested CPU. Each result records `observed_affinity`.

## MimIR remote and build cache

`config/mimir.toml` defaults to `remote = "mimir/mimir"` and `ref = "master"`. Override these with `--mimir-remote` and `--mimir-ref`. The first run clones into `<build>/mimir/source`; later runs fetch/switch in the same checkout and incrementally rebuild in `<build>/mimir/compiler`. Set `[build].directory` in `config/common.toml` or use `--build-dir` (default: `build`). Changing experimental outputs does not create another clone. MimIR's `--check` also prepares the compiler when needed.

```bash
# Only upstream MimIR master versus the required PyTorch baseline
uv run --frozen python run.py --backends mimir --mimir-ref master \
  --cases pack,mv_pack,conv_small,conv_1x1 --reps 1 --dump \
  --output results/mimir-master-vs-pytorch
```

## Case format

```text
# GEMM: A[M,K] @ B[K,N] -> C[M,N]
gemm    name M K N reps
# GEMV: A[M,K] @ x[K] -> y[M], with a vector output
gemv    name M K reps
# Conv2D: NCHW input, OIHW weight
conv2d  name N Cin H W Cout KH KW stride pad reps
```

`conv` is an alias for `conv2d`. Conv2D performs cross-correlation without bias, with groups=1 and dilation=1. The same stride and padding apply to H and W; padding may be zero. The output shape is `[N,Cout,(H+2*pad-KH)//stride+1,(W+2*pad-KW)//stride+1]`. Rectangular kernels and dimensions that are not powers of two are supported. Dimensions and reps must be positive, outputs must be nonempty, and case names must be unique across files.

The default suite has 32 cases: the original 16 GEMM cases, 10 GEMV cases, and 6 Conv2D cases. GEMV includes 1000/2048 sizes, 513×2047, 8×2304, and 64×130. Conv2D includes `conv_small` (rectangular kernel, stride=2, pad=1), `conv_1x1`, `r18_c1`, `r18_1x1`, `r18_s2`, and `vgg_c1`.

The legacy `--shapes '64,512,130'` option overrides GEMM only and retains **M,N,K** order, unlike the M K N order in case files. `--reps` overrides the minimum native calls per timing batch; `repeats` is the number of batches. The minimum timing duration still applies, so actual calls may exceed reps. TVM's evaluator does not expose its final adaptive count; that field is null.

## Benchmark methods

All backends use FP32 inputs generated with the same seed, the same CPU affinity, and one execution thread. A shared NumPy FP64 implementation independently checks correctness; the Conv2D reference contracts sliding windows with the weights. After warmup, each backend is measured alternately with PyTorch for nine native batches, each lasting at least 100 ms by default. Results use the median of per-call batch averages. Python calls, input loading, compilation, TVM tuning, and artifact export are excluded from timing.

| Backend | GEMM | GEMV | Conv2D | Timing scope |
|---|---|---|---|---|
| PyTorch baseline | ATen `mm.out` | ATen `mv.out` | ATen `convolution` | Native C++ loop; preallocated GEMM/GEMV output; Conv2D includes output allocation and internal packing |
| TVM | GEMM TIR | Vector reduction TIR | NCHW reduction TIR with padding checks | Native `time_evaluator`; preallocated output; optional MetaSchedule |
| IREE | `linalg.matmul` | `linalg.matvec` | `linalg.conv_2d_nchw_fchw` with `tensor.pad` when needed | Native `iree-benchmark-module`; includes VM, dispatch, and output allocation |
| MimIR | `tensor.product_2d` | `tensor.dot_product` | `tensor.conv` | Native C++ extern loop; includes output/internal allocation and free |
| OpenBLAS | `cblas_sgemm` | `cblas_sgemv` | im2col + SGEMM on every call | Native C loop; preallocated output/scratch; Conv2D includes packing and zero padding |

The OpenBLAS Conv2D path is this project's im2col wrapper; OpenBLAS has no direct Conv2D interface. Backend timing boundaries differ. Results explicitly record `timing_scope` and `baseline_timing_scope`.

GFLOPS uses `2*M*N*K` for GEMM, `2*M*K` for GEMV, and `2*N*Cout*OH*OW*Cin*KH*KW` for Conv2D. The Conv2D count covers the full kernel, including padding positions, and excludes bias and packing operations.

## TVM auto tuning

```toml
# config/tvm.toml
[tvm.auto_tune]
enabled = false
trials = 64
```

`--auto-tune` and `--no-auto-tune` override the subconfiguration for all three operators. `--trials` alone does not enable tuning; TVM must be selected to use its tuning options. When enabled, MetaSchedule EvolutionarySearch + RandomModel searches the actual operator TIR. When disabled, CPU DLight rules are applied. Compilation and tuning times are recorded separately.

Adjust the builder CPU list for your machine. Candidate measurements and final benchmarks remain on the shared CPU with one execution thread. Use a fresh output directory when changing modes, baselines, or compiler versions; configuration checks prevent incompatible results from being mixed.

## Reports and artifact dumps

For the unified entry point, `--output` always specifies a root directory: results are written to `<output>/<backend>/results.json` and `<output>/summary.csv`. The summary includes `op`, `shape`, `output_shape`, `baseline_backend`, `baseline_ms`, GFLOPS, speedup, and both timing scopes. Raw JSON records batch times, actual iteration counts, errors, and environment information.

`--dump` exports to `<output>/<backend>/artifacts/<operator_dims>/`. `--dump-dir` specifies another root, still with backend subdirectories. GEMM retains the legacy `<M>_<N>_<K>` directory naming; GEMV and Conv2D use operator prefixes to avoid collisions. Individual entry points do not add a backend subdirectory. `--dump-dir` implies export; `--no-dump` overrides shared configuration. `--check` does not generate artifacts.

Each dump includes exact A/B inputs, result JSON, environment/operator information, and a SHA256 manifest. The `baseline/` directory contains the measured PyTorch wrapper and operator dispatch information, also covered by the manifest. TVM exports final TIR, traces, LLVM IR, assembly, and shared libraries. IREE exports MLIR, VMFB, VM disassembly, and compiler artifacts. MimIR exports source/final MimIR IR, LLVM IR, assembly, and the measured shared library. OpenBLAS and PyTorch export their wrappers and information about precompiled dependencies.

Backend reproduction details: [TVM](tvm/README.md), [IREE](iree/README.md), [PyTorch](pytorch/README.md), [MimIR](mimir/README.md).

## Historical results

Earlier experiments used a local TVM build and an OpenBLAS baseline, with artifacts stored in `tvm/reference-results/` and `results/expanded-cpu-20261012/`. These paths refer to historical local artifacts, which are not included in this checkout. The 16-case, four-backend experiment used 64 valid tuning records per case and IREE DT + ukernels. Its JSON/CSV retain their original meaning. When reporting old results, the tool keeps `baseline_backend=openblas` and preserves their speedups; new results use `baseline_backend=pytorch`.

```bash
uv run --frozen python -m common.report \
  results/multiop/pytorch/results.json results/multiop/tvm/results.json \
  --output results/multiop/selected-summary.csv
uv run --frozen python -m unittest discover -s tests -v
```

Default CPU optimization: TVM uses DLight (`[tvm].schedule = "dlight"`); AutoTune overrides it. The wheel provides upstream GEMV/Reduction rules; this project adds `SpatialReduction` for GEMM/direct Conv2D (8-column tiles, vectorization, separate reduction initialization). Every result records the matched rule; unmatched operators fail instead of falling back to naive. Set `schedule = "naive"` explicitly for the original TIR. IREE defaults to DT + `ukernels=all`; `config/iree-default.toml` preserves the unmodified compiler pipeline.
