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

All configuration lives in `config/`. The `[benchmark]` section of `config/common.toml` controls the device, CPU affinity, thread count, seed, dtype, case files, warmup, measurement batches, and error tolerances. CPU is currently the only implemented device; `--device gpu` reports an error.

```toml
[benchmark]
ops = ["gemm", "gemv", "conv2d"]
cases_files = ["config/gemm_cases.txt", "config/gemv_cases.txt", "config/conv2d_cases.txt", "config/linear_cases.txt"]

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

On Linux, CPU selection uses the process's allowed affinity mask. With no global or per-runner CPU specified, every backend uses the lowest allowed CPU; CPU 0 is not required. An explicit CPU outside that mask fails with the allowed CPU list. The unified entry point validates the selection before starting a backend and forwards the same CPU to all measurements. Results record both the selected CPU and the original allowed mask. For example, `taskset -c 2,4 uv run --frozen python run.py --backends none --cases mv_pack` selects CPU 2, provided those CPUs are available.

## Runner CPU assignments and parallel execution

Configure `[runner_cpus]` in `config/common.toml`, or pass `--runner-cpus pytorch=0,tvm=2,iree=4,openblas=6,mimir=8`. CPU precedence is: a CLI runner entry, global `--cpu`, a configured runner entry, `benchmark.cpu`, then the lowest allowed CPU. Individual entry points honor their configured runner CPU too. All selected assignments are validated before launching any runner.

Each backend and its interleaved PyTorch baseline use the same assigned core. The standalone PyTorch runner may use another core. JSON/CSV record `cpu` and `baseline_cpu`; speedup uses the baseline measured on that backend's core.

`--parallel` runs selected backends concurrently, with a separate `<output>/<backend>.log`; each runner must have a distinct logical CPU. Prefer CPUs from different physical cores, of the same core type. Concurrent runners still share cache and memory bandwidth. `--no-parallel` forces sequential execution. The default is `benchmark.parallel = false`.

```bash
uv run --frozen python run.py --backends all --parallel \
  --runner-cpus pytorch=0,tvm=2,iree=4,openblas=6,mimir=8 \
  --dump --output results/full-suite
```

TVM automatic thread binding is disabled with `TVM_BIND_THREADS=0`, so it cannot reset the main thread to CPU 0. On Linux, the harness verifies actual affinity after warmup and every timing batch; drift stops the run instead of silently recording the requested CPU. Each result records `observed_affinity`.

## macOS Apple Silicon

The default `mcpu = "native"` resolves to an LLVM CPU name such as `apple-m2` on Apple Silicon. Override it with `--mcpu`. MimIR uses `-mcpu` on ARM and `-march` on x86. Linux retains strict CPU affinity and drift checks.

macOS has no public API for pinning a thread to a particular CPU, and Apple Silicon does not support Mach thread affinity policy. Darwin runners request `QOS_CLASS_USER_INTERACTIVE` with `pthread_set_qos_class_self_np` and check QoS after warmup and timing batches. The OS selects cores dynamically; performance cores and separate physical cores are not guaranteed. On macOS, `--cpu` and `--runner-cpus` identify runner slots; `--parallel` assigns distinct slots automatically if none are configured. Results record `scheduling.mode = "macos-qos"`, `strict = false`, and null actual CPU/affinity fields. CSV includes the slot and scheduling mode separately. Native IREE timing subprocesses inherit scheduling settings; QoS validation covers the Python runner thread.

```bash
brew install openblas llvm cmake ninja
uv sync --locked
uv run --frozen python run.py --backends all --ops all --parallel \
  --no-auto-tune --output results/macos-arm-full
```

OpenBLAS automatically locates Homebrew headers and libraries. PyTorch uses a PyPI ARM wheel on macOS and the CPU index on Linux. `VECLIB_MAXIMUM_THREADS=1` limits Accelerate threads. Native wrapper builds use Homebrew LLVM and the matching Command Line Tools SDK/linker when available. MimIR uses Homebrew LLVM for C++23 on macOS. IREE uses Darwin Mach-O system libraries with data tiling and ukernels enabled; this avoids the observed ARM microkernel crash in embedded ELF mode. OpenBLAS copies its library into the output directory and relinks its OpenMP dependency to PyTorch’s runtime, keeping installed libraries intact. Native timing libraries and dumps use `.dylib`; MimIR MODULE plugins retain upstream `.so` names.

## MimIR remote and build cache

`config/mimir.toml` defaults to `remote = "mimir/mimir"` and `ref = "master"`. Override these with `--mimir-remote` and `--mimir-ref`. The first run clones into `<build>/mimir/source`; later runs fetch/switch in the same checkout and incrementally rebuild in `<build>/mimir/compiler`. Set `[build].directory` in `config/common.toml` or use `--build-dir` (default: `build`). Changing experimental outputs does not create another clone. MimIR's `--check` also prepares the compiler when needed.

Any fork or branch is just another remote/ref, e.g. `--mimir-remote hsqStephenZhang/MimIR --mimir-ref feature/par`; use a separate `--build-dir` to keep several MimIR builds side by side.

```bash
# Only upstream MimIR master versus the required PyTorch baseline
uv run --frozen python run.py --backends mimir --mimir-ref master \
  --cases pack,mv_pack,conv_small,conv_1x1 --reps 1 --dump \
  --output results/mimir-master-vs-pytorch
```

## Case format

```text
# GEMM: A[M,K] @ B[K,N] -> C[M,N]
gemm    name M K N reps   [tags=family,...]
# GEMV: A[M,K] @ x[K] -> y[M], with a vector output
gemv    name M K reps     [tags=family,...]
# Conv2D: NCHW input, OIHW weight
conv2d  name N Cin H W Cout KH KW stride pad reps [tags=family,...]
# Frozen-weight GEMM (config/linear_cases.txt)
linear  name M K N reps   [tags=family,...]
```

`conv` is an alias for `conv2d`. Conv2D performs cross-correlation without bias, with groups=1 and dilation=1. The same stride and padding apply to H and W; padding may be zero. The output shape is `[N,Cout,(H+2*pad-KH)//stride+1,(W+2*pad-KW)//stride+1]`. Rectangular kernels and dimensions that are not powers of two are supported. Dimensions and reps must be positive, outputs must be nonempty, and case names must be unique across files.

The suite has 63 cases (22 GEMM, 13 GEMV, 11 Conv2D, 17 linear; the default `ops` leave out `linear`). Every case is tagged with its op, a **size** class from its FP32 working set (A+B+C): `small` ≤ 1 MiB, `medium` ≤ 32 MiB, `large` above; and a **regularity** class: `regular` when every tiled dimension is a multiple of 16 (GEMM/linear M, N, K; GEMV M, K; Conv2D output channels, output pixels and Cin·KH·KW), else `irregular`, plus `vector` for M = 1. Case files add family tags (`square`, `skinny`, `short-k`, `prime`, `odd-m`, `resnet`, `vgg`, `llm`, `llama2-7b`, `tinyllama`, ...). Every op × size × regularity combination has at least one case, including odd M such as the 49-pixel ResNet-50 stage-5 GEMMs and 7×7 convolutions.

```bash
uv run --frozen python run.py --list-cases --ops all --tags irregular,small      # inspect a selection
uv run --frozen python run.py --backends all --ops all --tags gemm,irregular --tags linear,!large
uv run --frozen python run.py --backends mimir --shape gemm:49,4608,512 --shape conv2d:1,512,7,7,512,3,3,1,1
```

`--tags a,b` keeps cases carrying all of the tags, `!tag` excludes one, and repeated `--tags` take the union. `--shape OP:dims` (repeatable) runs ad-hoc cases instead of the case files, with dims in case-file order (`x` also separates them); they are tagged `custom` plus their automatic tags, and `--ops` does not filter them. `--cases`, `--cases-file`, `--tags` and `--shape` reach every backend runner, and results record each case's tags.

The legacy `--shapes '64,512,130'` option overrides GEMM only and retains **M,N,K** order, unlike the M K N order in case files. `--reps` overrides the minimum native calls per timing batch; `repeats` is the number of batches. The minimum timing duration still applies, so actual calls may exceed reps. TVM's evaluator does not expose its final adaptive count; that field is null.

## Threads

`[benchmark].threads` (default 1) or `run.py --threads N` sets the thread count for every backend at once: PyTorch intra-op threads, OpenBLAS, TVM (`num-cores` and `TVM_NUM_THREADS`), IREE (`local-task` with N worker groups instead of `local-sync`) and MimIR. MimIR runs in parallel through its `omp` plugin (`mim -p omp`, kernel linked with `-fopenmp`) when the build has one, as on `mimir/mimir` `feature/omp`; other builds run as built, and results record `mimir_parallel`. All `*_NUM_THREADS` variables (`OMP`, `OPENBLAS`, `MKL`, `VECLIB`, `TVM`, `MIM`) follow the same count. On Linux a multi-threaded runner is pinned to `[benchmark].cpus` / `--cpus`, or to N allowed CPUs from its runner CPU, one per physical core first; on hybrid CPUs list performance cores explicitly. On macOS MimIR links PyTorch's `libomp`, so the process holds one OpenMP runtime. `--parallel` requires one thread per runner.

## Frozen-weight GEMM (`linear`)

`linear` models an LLM projection: `Y[M,N] = X[M,K] @ W[K,N]` where W is known when the kernel is prepared. Cases live in `config/linear_cases.txt` (Llama-2-7B attention-out/MLP-up/MLP-down and TinyLlama QKV/MLP shapes, M = 1 for decode up to 128 for prefill) and run with `--ops linear` (or `--ops all`). Each backend receives W once in `prepare` and may transform it there; only calls with the activation X are timed. The one-time transform is recorded as `prepack_s` and its result as `weight_layout` in JSON/CSV.

| Backend | Frozen-weight handling |
|---|---|
| PyTorch (baseline) | Eager `aten::mm.out` on W as-is; without oneDNN, the BLAS packs W inside every call |
| OpenBLAS | Same `cblas_sgemm`; no packed-GEMM API is exported, so W is packed per call |
| TVM | `layout_free_buffers=[1]` lets MetaSchedule's `RewriteLayout` choose a packed W layout; `SplitLayoutRewritePreproc` splits the transform into `pack`, run once (first call, includes lazy load) |
| IREE | W is an immutable `#stream.parameter.named<"model"::"w">` global read from `weights.irpa`; const-expr hoisting moves the data-tiling pack into the module initializer. `prepack_s` is module load including the initializer, which `iree-benchmark-module` runs before timing |
| MimIR | No weight prepack. `linear_weight_layout = "kn"` passes W as-is; `"nk"` transposes it once to `[N,K]` and contracts the last axes. M = 1 uses the flat matrix-vector form because MimIR shape inference treats a `«1, K»` operand as a vector |

`config/pytorch.toml` `macos_blas = "openblas"` routes PyTorch's Accelerate BLAS imports (`sgemm_`, `sgemv_`, ...) to OpenBLAS through a dyld interposer, so the baseline uses only NEON CPU cores instead of the AMX coprocessor; `"accelerate"` restores the wheel's default. Results record `pytorch_blas` and the routed call count per case.

## Evaluating a MimIR revision

`eval/mimir.toml` names a `[candidate]` and a `[baseline]` MimIR (remote + ref) and the ops/cases to run. `mimir_eval.py` runs the `mimir` backend once per side with that remote/ref and its own persistent checkout (`<build>/mimir-eval/{baseline,candidate}`), back to back on the same CPUs with `--keep-going` and `[run].threads` (see [Threads](#threads)), and writes `report.md` and `comparison.csv`: per case, candidate GFLOPS / baseline GFLOPS (> 1 = candidate faster) with each side's speedup over its interleaved PyTorch run, plus per-op geometric means. A failing case is reported as FAIL instead of aborting; candidate failures make the command exit 1.

```bash
uv run --frozen python mimir_eval.py                                  # spec as committed
uv run --frozen python mimir_eval.py --candidate-ref <sha> --ops gemm,linear --output results/eval-x
uv run --frozen python mimir_eval.py --output results/eval-x --report-only   # rebuild the report
```

CI (`.github/workflows/mimir-eval.yml`) runs the same command when a push changes `eval/mimir.toml`, or manually (`gh workflow run "MimIR eval" -f candidate_ref=<sha>`); the report becomes the job summary and results are uploaded as an artifact. It uses GitHub-hosted `ubuntu-26.04` by default, which is noisy; for stable numbers register a self-hosted runner and set the repository variables `GEMM_BENCH_RUNS_ON` (JSON labels, e.g. `["self-hosted","gemm-bench"]`) and `GEMM_BENCH_CPU`. The workflow has no `pull_request` trigger, so forks never run code on that machine.

## Benchmark methods

All backends use FP32 inputs generated with the same seed and the same thread count (one by default; strict CPU affinity on Linux, QoS scheduling on macOS). A shared NumPy FP64 implementation independently checks correctness; the Conv2D reference contracts sliding windows with the weights. After warmup, each backend is measured alternately with PyTorch for nine native batches, each lasting at least 100 ms by default. Results use the median of per-call batch averages. Python calls, input loading, compilation, TVM tuning, and artifact export are excluded from timing.

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

`--auto-tune` and `--no-auto-tune` override the subconfiguration for all three operators. `--trials` alone does not enable tuning; TVM must be selected to use its tuning options. When enabled, MetaSchedule EvolutionarySearch + RandomModel searches the actual operator TIR. When disabled, LLVM rules generate candidates once and candidate 0 is used without search or mutation. Compilation and tuning times are recorded separately.

Adjust the builder CPU list for your machine. Candidate measurements and final benchmarks run on the runner's CPUs with the benchmark thread count. Use a fresh output directory when changing modes, baselines, or compiler versions; configuration checks prevent incompatible results from being mixed.

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

Default CPU optimization: TVM uses `[tvm].schedule = "llvm"`: one fixed-seed generation with `ms.ScheduleRule.create("llvm")`, `mutator_probs={}`, no search strategy or cost model, and direct selection of candidate 0. Target-selected postprocessing finalizes annotations, matching the default-rule experiment. Rejected candidate 0 fails; it never silently switches to a different candidate. AutoTune overrides this mode when enabled. DLight and naive remain explicit alternatives. IREE defaults to DT + `ukernels=all`; `config/iree-default.toml` preserves the unmodified compiler pipeline.

## Visualize benchmark results

`plot.py` uses matplotlib from the shared uv environment. Pass the output directory or its long-form `summary.csv`:

```bash
uv sync --locked
uv run --frozen python plot.py results/full-cpu-llvm-default-dt-20261012
# Absolute latency instead of throughput
uv run --frozen python plot.py results/full-cpu-llvm-default-dt-20261012 --absolute latency
# Filter operators/backends and export PDF with a linear axis
uv run --frozen python plot.py results/full-cpu-llvm-default-dt-20261012 \
  --ops gemm,gemv,conv --backends pytorch,tvm,iree --scale linear \
  --formats png,pdf --output results/selected-plots
```

Default outputs are a six-panel overview plus individual GEMM/GEMV/Conv2D figures in `<input directory>/plots/`, as PNG and SVG. Left panels show GFLOP/s (higher is better), or latency in ms (lower is better); right panels show paired-baseline speedup (higher is better), with a dashed 1× reference. Logarithmic axes are the default; `--scale linear` changes both columns. Speedups use the recorded same-core baseline rather than the standalone PyTorch column. Legends retain backend variants and CPU IDs; axes retain case dimensions (`--no-shapes` hides them). Missing results leave gaps, and duplicate or inconsistent rows fail validation. Plot metadata records the source CSV hash. Compilation, packing, allocation and invocation scope differences remain those recorded in the benchmark results.

## PyTorch modes

`[pytorch].mode` supports `"eager"` (default) and `"inductor"`. CLI `--pytorch-mode` overrides the configuration; `config/pytorch-inductor.toml` is also provided. Paired baselines always use eager, even when the measured PyTorch runner uses Inductor.

```bash
uv run --frozen python run.py --backends none --pytorch-mode inductor \
  --ops all --cpu 0 --dump --output results/pytorch-inductor
uv run --frozen python pytorch/run.py --pytorch-mode eager --cases pack
```

Inductor uses `torch.export.export` → `torch._inductor.aoti_compile_and_package` → native C++ `AOTIModelPackageLoader::run`. It measures CPU Inductor compiled graphs with no Python calls in the timing loop, using the [official AOTInductor deployment path](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html). This is AOTInductor native invocation rather than timing a Python `torch.compile` callable. Inputs are runtime arguments, not constant weights. Output allocation, graph invocation and internal packing are included; compilation, graph export and correctness copies are excluded. Default `max_autotune=false`, `cpp.threads=1`; compilation errors stop the run rather than falling back to eager. `--dump` retains the exported program, FX graph, measured `.pt2` package, generated C++/shared libraries and native timer wrapper. Inductor and eager results need separate output directories.
