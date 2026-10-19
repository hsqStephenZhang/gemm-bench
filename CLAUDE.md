# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All Python runs through uv with the locked environment (Python 3.11; TVM, IREE and PyTorch come from `uv.lock`, no source builds):

```bash
uv sync --locked
uv run --frozen python run.py --backends all --check            # validate toolchains, no timing
uv run --frozen python run.py --backends all --ops all --output results/<name>   # default ops omit linear; --ops all includes it
uv run --frozen python run.py --backends none --cases sq512     # PyTorch baseline only
uv run --frozen python run.py --backends tvm,mimir-par --ops linear --cases tl_qkv_m32 --output <dir>

uv run --frozen python mimir_eval.py --cases sq64,mv64           # MimIR candidate vs baseline from eval/mimir.toml -> report.md
uv run --frozen python -m unittest discover -s tests             # tests (pytest is not installed)
uv run --frozen python -m unittest discover -s tests -k test_mandatory_torch   # single test
```

macOS needs `brew install openblas llvm cmake ninja`. MimIR backends clone and build the compiler on first use under `build/<name>/` (slow); later runs fetch and rebuild incrementally.

Each `--output` directory is bound to one configuration: `common/runner.py` refuses to append rows when execution/baseline/backend metadata differ ("incompatible …; choose a fresh --output directory"). Use a new directory whenever a config, compiler ref or BLAS mode changes. Every `run.py` call also reruns the standalone PyTorch baseline into `<output>/pytorch`, so rerunning a subset of backends into an existing root fails the same check; give each invocation its own root and merge afterwards with `uv run --frozen python -m common.report <root>/*/<backend>/results.json --output <dir>/summary.csv`.

## Architecture

**Unified entry (`run.py`)** resolves backends, runner CPUs and env, then launches one subprocess per backend: `<backend>/run.py` (OpenBLAS: `common/benchmark_openblas.py`; `mimir-par` reuses `mimir/run.py` with `config/mimir-par.toml`). Each `<backend>/run.py` only builds the environment (thread vars from `[threads]`, `GEMM_*` config paths) and execs `<backend>/benchmark.py`, which calls `common.runner.run(name, AdapterClass)`. Afterwards `common/report.summarize` merges every `<output>/<backend>/results.json` into `<output>/summary.csv`.

**Runner (`common/runner.py`)** is the single measurement loop shared by all backends. Per case it generates seeded inputs and an FP64 NumPy reference (`common/operators.py`), prepares the backend kernel and an interleaved PyTorch baseline kernel, warms up, checks both against the reference, then alternates 9 timed batches (≥ `min_repeat_ms`) between backend and baseline. Speedup = PyTorch median / backend median, so every backend run also re-measures PyTorch on the same core.

**Adapter contract** (implemented by each backend class): `__init__(config, settings)` sets `metadata` and `identity_keys`; `prepare(case, a, b, work, previous_row)` compiles and returns a kernel with `run()` (returns the output for correctness), `measure(min_repeat_ms, min_reps)` (native timing, seconds per call), `scope` (timing-scope string recorded in results), optional `details` (merged into the row, e.g. `compile_s`, `prepack_s`, `weight_layout`) and `dump(destination)` for `--dump`. All timing loops are native (C/C++ wrappers, TVM `time_evaluator`, `iree-benchmark-module`); Python is never inside the timed region.

**Operators and cases.** Case files in `config/*_cases.txt`: `gemm|linear name M K N reps`, `gemv name M K reps`, `conv2d name N Cin H W Cout KH KW stride pad reps`. Internally GEMM/linear shapes are stored as `[M, N, K]` (note the reorder from the file's M K N). `linear` is GEMM with a frozen right operand W: backends may transform W once in `prepare` (outside timing) and report it as `prepack_s` — TVM via `layout_free_buffers` + `SplitLayoutRewritePreproc` (`common/tvm_ops.py`), IREE via a `#stream.parameter.named` global loaded from `weights.irpa` (`common/iree_ops.py`), MimIR via `linear_weight_layout`. Per-backend operator text lives in `common/tvm_ops.py`, `common/iree_ops.py`, `mimir/operators.py`.

**MimIR evaluation.** `mimir_eval.py` reads `eval/mimir.toml`, writes candidate/baseline configs from `config/mimir-par.toml`/`config/mimir.toml` (same backend names and build dirs, only remote/ref replaced), runs `run.py --backends mimir,mimir-par --keep-going` and builds `report.md` from the two `results.json`. `--keep-going` (env `GEMM_KEEP_GOING=1`) records failing cases as `{"error": ...}` rows, which `common/report.py` skips. `.github/workflows/mimir-eval.yml` runs it on pushes touching the spec or manually.

**Configuration.** `config/common.toml` holds shared benchmark settings, `[backends]` defaults, `[runner_cpus]`, `[threads]` (all backends single-threaded: `OMP/OPENBLAS/VECLIB/TVM/MIM_NUM_THREADS = 1`) and `[build].directory`. Each backend has its own `config/<backend>.toml`; alternates such as `iree-dt.toml` or `tvm-long.toml` are selected with `--<backend>-config`. A MimIR config's `name` sets both the backend name and its checkout directory.

**Host differences (`common/host.py`).** Linux pins each runner to a CPU and checks affinity drift during timing. macOS cannot pin: runner "CPUs" are slots, threads request `QOS_CLASS_USER_INTERACTIVE`, and results record `scheduling.mode = "macos-qos"`. `mcpu = "native"` is resolved once (via `clang -### -march/-mcpu=native`) to the host LLVM CPU name (`apple-m2`, `znver3`, `raptorlake`) and passed to every backend, because TVM's bundled LLVM silently falls back to `generic` (no AVX2/FMA) when given `native` on x86.

## Pitfalls specific to this repo

- **AMX on Apple Silicon.** The macOS PyTorch wheel uses Accelerate, which runs FP32 GEMM on the AMX coprocessor (~1.2–1.4 TFLOPS single-thread vs ~100 GFLOPS NEON). `pytorch/blas.py` re-execs the runner with a dyld interposer (`pytorch/blas_interpose.c`, built content-keyed under `build/pytorch-blas/`) that routes PyTorch's `sgemm_`/`sgemv_`/… to OpenBLAS; `macos_blas = "openblas"` in `config/pytorch.toml` enables it, `"accelerate"` restores AMX. Results record `pytorch_blas` and `blas_calls_routed` per case.
- **TVM 0.27 renames:** `tvm.script.tirx`, `tvm.s_tir.meta_schedule`, `tvm.tirx.build`, `T.sblock`, attr `tirx.noalias`; `R.call_tir(..., out_ty=...)`. TVMScript cannot see a function's local variables from inside a nested `@I.ir_module` class, so modules with runtime shapes are built from text with `tvm.script.from_source` (see `split_weight_prepack`). Without `TVM_NUM_THREADS=1`, generated `T.parallel` loops go multi-threaded.
- **MimIR:** upstream master can break operators (e.g. GEMV `tensor.dot_product` type error); pin with `--mimir-ref <commit>`, which switches the shared `build/mimir` checkout (it refuses to switch a checkout with local changes). A `«1, K»` operand is mistaken for a vector, so M = 1 contractions use flat vector forms. `mimir-par`'s runtime thread pool defaults to all performance cores unless `MIM_NUM_THREADS` is set.
- **OpenBLAS (Homebrew) is an OpenMP build:** `OPENBLAS_NUM_THREADS=1` alone is ignored; `OMP_NUM_THREADS=1` / `openblas_set_num_threads(1)` are required. On macOS its library is copied and relinked to PyTorch's `libomp` to avoid two OpenMP runtimes in one process.
- **Long or remote runs:** run each backend as its own `run.py --backends <b>` call inside `tmux` so one failure or hang does not lose the rest; pick a measurement CPU whose SMT sibling stays idle (and a P-core on hybrid Intel); set `[tvm.auto_tune].builder_cpus` away from that core. Do not copy `build/` to another path: CMake caches absolute source/build paths, so MimIR must be rebuilt (the `source/` checkout can be reused).
- **IREE** native Mach-O (not embedded ELF) is required on ARM macOS; ukernels + data tiling are on by default (`config/iree.toml`). Module initializers (e.g. hoisted weight packs) run before `iree-benchmark-module` timing.

## Code style

The Python is deliberately dense: short lines joined with `;`, minimal blank lines, compact names, comments only for non-obvious constraints. Match it. README changes are kept in both `README.md` and `README.zh-CN.md`.

Always run ``ruff format before committing.``
