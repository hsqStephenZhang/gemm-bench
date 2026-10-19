# gemm-bench

[English](README.md) | [简体中文](README.zh-CN.md)

使用一个 uv 项目对比 CPU FP32 GEMM、GEMV 和 Conv2D。**PyTorch 是所有算子的必选 baseline**；TVM、IREE、OpenBLAS、MimIR 是可选对比后端。每个后端与同次运行的 PyTorch 交替测量，加速比为 `PyTorch median / backend median`。

```bash
cd /workspaces/ml-compiler/gemm-bench
uv sync --locked
uv run --frozen python run.py --backends all --check

# PyTorch + TVM + IREE：三个算子，各取一个小用例
uv run --frozen python run.py --backends tvm,iree \
  --cases pack,mv_pack,conv_small --dump --output results/multiop

# 只运行 PyTorch baseline；所有 GEMV 用例
uv run --frozen python run.py --backends none --ops gemv --output results/gemv-baseline

# Conv2D 对比；可选 TVM 自动调优
uv run --frozen python run.py --ops conv2d --backends tvm,iree,openblas \
  --cases conv_small,conv_1x1 --auto-tune --trials 64 \
  --output results/conv-tuned --dump
```

TVM 使用 `uv add apache-tvm` 安装的 wheel，版本由 `uv.lock` 锁定；无需本地源码构建。当前为 0.27.0.post1。复现使用 `uv sync --locked`。

## 公共配置和选择规则

所有配置集中在 `config/`。`config/common.toml` 的 `[benchmark]` 设置 `device`、CPU affinity、单线程、seed、dtype、用例文件、预热、测量轮数和误差容限。当前只实现 CPU；`--device gpu` 明确报错。

```toml
[benchmark]
ops = ["gemm", "gemv", "conv2d"]
cases_files = ["config/gemm_cases.txt", "config/gemv_cases.txt", "config/conv2d_cases.txt", "config/linear_cases.txt"]

[backends]
tvm = true
iree = true
openblas = true
mimir = true
mimir-par = true
```

`--backends`（同义参数 `--backend`）覆盖可选后端配置，支持任意组合、`all`、`none`。PyTorch 自动运行，不能关闭，不加入可选列表。`--ops gemm,gemv` 覆盖 `benchmark.ops`；`--cases name1,name2` 从选中算子的用例中选择；`--cases-file` 替换整套文件；旧公共配置 `cases_file` 仍可读取。

`--pytorch-config` 同时用于独立 baseline 和各后端的交替 baseline，确保设置一致。`--tvm-config` / `--iree-config` / `--mimir-config` 选择对应专属配置；`--common-config` 选择公共配置。各实现独立入口仍支持公共参数，`--config` 指定该实现配置，`--output` 直接作为本实现目录：

```bash
uv run --frozen python tvm/run.py --ops gemv --cases mv_pack --auto-tune --trials 16
uv run --frozen python iree/run.py --ops conv2d --cases conv_small
uv run --frozen python pytorch/run.py --ops all --cases pack,mv_pack,conv_small
```

Linux CPU 选择依据进程当前允许的 affinity 集合。未指定全局或 runner CPU 时，所有后端使用最小可用 CPU，不要求 CPU 0 可用。显式指定集合外的 CPU 会报错并列出允许的 CPU。统一入口在启动后端前校验，并将同一 CPU 转发给所有测量；结果同时记录实际 CPU 和原始允许集合。例如，在 CPU 2/4 可用的机器上，`taskset -c 2,4 uv run --frozen python run.py --backends none --cases mv_pack` 自动选择 CPU 2。

## Runner CPU 分配与并行执行

通过 `config/common.toml` 的 `[runner_cpus]` 或 CLI `--runner-cpus pytorch=0,tvm=2,iree=4,openblas=6,mimir=8` 分配 CPU。优先级为：CLI runner 条目、全局 `--cpu`、配置中的 runner 条目、`benchmark.cpu`、最小允许 CPU。独立入口也读取对应 runner 的配置。所有选中 runner 的 CPU 在启动前统一校验。

每个后端及其交替测量的 PyTorch baseline 使用同一分配核心；独立 PyTorch runner 可以使用另一核心。JSON/CSV 保存 `cpu` 和 `baseline_cpu`，加速比使用该后端同核心测得的 baseline。

`--parallel` 并行运行选中的后端，每个保存独立 `<output>/<backend>.log`，要求各 runner 的逻辑 CPU 不同。建议使用同类型、不同物理核心的 CPU；并行 runner 仍共享 cache 和内存带宽。`--no-parallel` 强制串行，默认 `benchmark.parallel = false`。

```bash
uv run --frozen python run.py --backends all --parallel \
  --runner-cpus pytorch=0,tvm=2,iree=4,openblas=6,mimir=8 \
  --dump --output results/full-suite
```

使用 `TVM_BIND_THREADS=0` 关闭 TVM 自动线程绑核，避免主线程被重新绑定到 CPU 0。Linux 框架在预热后和每个计时批次后校验实际 affinity；发生偏移时停止运行，不会静默记录错误的请求 CPU。各条结果保存 `observed_affinity`。

## macOS Apple Silicon

默认 `mcpu = "native"` 在 Apple Silicon 上解析为 `apple-m1` / `apple-m2` 等 LLVM CPU 名称；可用 `--mcpu` 覆盖。MimIR 在 ARM 上使用 `-mcpu`，在 x86 上使用 `-march`。Linux 保留严格 affinity 绑定和漂移检查。

macOS 没有公开的逐核心绑核接口，Apple Silicon 不支持 Mach thread affinity policy。macOS runner 使用 `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`，并在预热和计时批次后检查 QoS。系统动态选择核心，不能保证使用某个性能核心，也不能保证不同 runner 位于不同物理核心。`--cpu` / `--runner-cpus` 在 macOS 上表示 runner slot；`--parallel` 未指定编号时自动分配不同 slot。结果保存 `scheduling.mode = "macos-qos"`、`strict = false`，实际 `cpu`、`baseline_cpu` 和 `observed_affinity` 为 null，CSV 单独保存 slot 和调度方式。IREE 原生计时子进程继承调度设置；QoS 检查针对 Python runner 线程。

```bash
brew install openblas llvm cmake ninja
uv sync --locked
uv run --frozen python run.py --backends all --ops all --parallel \
  --no-auto-tune --output results/macos-arm-full
```

OpenBLAS 自动查找 Homebrew 的头文件和动态库；PyTorch 在 macOS 从 PyPI 安装 ARM wheel，Linux 继续使用 CPU wheel。设置 `VECLIB_MAXIMUM_THREADS=1` 限制 Accelerate 线程。原生包装器构建优先使用 Homebrew LLVM，并匹配 Command Line Tools 的 SDK/linker；MimIR 使用 Homebrew LLVM 的 C++23 编译器。IREE 默认使用 Darwin Mach-O 系统动态库，保留 data tiling 和 ukernels，避免本机 embedded ELF 模式的 ARM microkernel 崩溃。OpenBLAS 将库复制到输出目录并将 OpenMP 依赖改为 PyTorch runtime，不修改已安装的库。独立计时动态库和 dump 使用 `.dylib`，MimIR MODULE plugin 保持上游 `.so` 命名。

## MimIR remote 和 build 缓存

`config/mimir.toml` 默认 `remote = "mimir/mimir"`、`ref = "master"`，可用 `--mimir-remote` / `--mimir-ref` 覆盖。首次 clone 到 `<build>/mimir/source`，之后在同一 checkout 中 fetch/切换，并在 `<build>/mimir/compiler` 增量构建。公共配置 `[build].directory` 或 CLI `--build-dir` 指定构建根目录，默认 `build`。切换实验输出目录不会产生新的 clone。MimIR 的 `--check` 也会按需准备编译器。

`mimir-par` 是同一 adapter 的第二个实例，由 `config/mimir-par.toml` 配置（`hsqStephenZhang/MimIR` 的 `feature/par` 分支，带分块与打包的收缩）。配置中的 `name` 同时决定 backend 名称和 `<build>/mimir-par/` 下的 checkout，因此不会影响上游 checkout；`--mimir-remote`/`--mimir-ref` 只作用于 `mimir`，`--mimir-par-config` 可选择其他 fork 配置。该分支的运行时线程池读取 `MIM_NUM_THREADS`，`[threads]` 中设为 1。

```bash
# 只跑上游 MimIR master 与必选 PyTorch baseline
uv run --frozen python run.py --backends mimir --mimir-ref master \
  --cases pack,mv_pack,conv_small,conv_1x1 --reps 1 --dump \
  --output results/mimir-master-vs-pytorch
```

## 用例格式

```text
# GEMM: A[M,K] @ B[K,N] -> C[M,N]
gemm    name M K N reps
# GEMV: A[M,K] @ x[K] -> y[M]，真实向量输出
gemv    name M K reps
# Conv2D: NCHW input, OIHW weight
conv2d  name N Cin H W Cout KH KW stride pad reps
```

`conv` 是 `conv2d` 的别名。Conv2D 为 cross-correlation，无 bias，groups=1、dilation=1；stride/pad 同时应用于 H/W，pad 可以为 0。输出为 `[N,Cout,(H+2*pad-KH)//stride+1,(W+2*pad-KW)//stride+1]`。非方形 kernel、非 2 的幂尺寸都支持。尺寸/reps 必须为正，输出不能为空，用例名称跨文件唯一。

默认套件共 32 组：原 16 组 GEMM、10 组 GEMV、6 组 Conv2D。GEMV 包括 1000/2048、513×2047、8×2304、64×130；Conv2D 包括 `conv_small`（非方形 kernel、stride=2、pad=1）、`conv_1x1`、`r18_c1`、`r18_1x1`、`r18_s2`、`vgg_c1`。

旧 `--shapes '64,512,130'` 仅覆盖 GEMM，顺序仍为 **M,N,K**，与文件的 M K N 不同。`--reps` 覆盖每批次最少原生调用次数，`repeats` 是批次数。最低计时窗口仍有效，实际调用次数可以超过 reps；TVM evaluator 不暴露最终自适应次数，结果为 null。

## 冻结权重 GEMM（`linear`）

`linear` 对应 LLM 的投影层：`Y[M,N] = X[M,K] @ W[K,N]`，W 在 prepare 时已知。用例在 `config/linear_cases.txt`（Llama-2-7B 的 attention-out/MLP-up/MLP-down 与 TinyLlama 的 QKV/MLP 形状，M = 1 对应 decode，最大 128 对应 prefill），通过 `--ops linear`（或 `--ops all`）运行。每个后端在 `prepare` 中拿到一次 W 并可以变换它；只计时输入激活 X 的调用。一次性变换耗时记为 `prepack_s`，结果布局记为 `weight_layout`。

| 后端 | 冻结权重的处理方式 |
|---|---|
| PyTorch（基线） | eager `aten::mm.out`，W 原样传入；没有 oneDNN 时 BLAS 每次调用都会打包 W |
| OpenBLAS | 同样调用 `cblas_sgemm`；未导出 packed-GEMM 接口，每次调用打包 W |
| TVM | `layout_free_buffers=[1]` 让 MetaSchedule 的 `RewriteLayout` 选择打包布局；`SplitLayoutRewritePreproc` 把变换拆为 `pack`，只运行一次（首次调用，含延迟加载） |
| IREE | W 是从 `weights.irpa` 读取的不可变 `#stream.parameter.named<"model"::"w">` 全局量；const-expr hoisting 把 data-tiling 的 pack 移到模块初始化器中。`prepack_s` 为包含初始化器的模块加载时间，`iree-benchmark-module` 在计时前完成它 |
| MimIR、mimir-par | 没有权重预打包。`linear_weight_layout = "kn"` 原样传入 W；`"nk"` 预先转置为 `[N,K]` 并收缩最后一维。M = 1 使用扁平的矩阵-向量形式，因为 MimIR 的形状推断会把 `«1, K»` 当作向量 |

`config/pytorch.toml` 中 `macos_blas = "openblas"` 通过 dyld interposer 把 PyTorch 从 Accelerate 导入的 BLAS（`sgemm_`、`sgemv_` 等）转到 OpenBLAS，使基线只使用 NEON CPU 核心而不是 AMX 协处理器；设为 `"accelerate"` 恢复 wheel 默认行为。结果中记录 `pytorch_blas` 以及每个用例被转发的调用次数。

## 评估 MimIR 版本

`eval/mimir.toml` 指定 `[candidate]` 与 `[baseline]` 两个 MimIR（remote + ref）以及要运行的算子/用例。`mimir_eval.py` 在各自的持久 checkout（候选为 `build/mimir-par`，基线为 `build/mimir`）中构建两者，以 `--keep-going` 在同一核心上先后运行，并生成 `report.md` 与 `comparison.csv`：每个用例给出候选 GFLOPS / 基线 GFLOPS（> 1 表示候选更快）、两侧相对各自交错测得的 PyTorch 的加速比，以及每个算子的几何平均。失败的用例记为 FAIL 而不中止运行；候选有失败时命令以 1 退出。

```bash
uv run --frozen python mimir_eval.py                                  # 使用已提交的配置
uv run --frozen python mimir_eval.py --candidate-ref <sha> --ops gemm,linear --output results/eval-x
uv run --frozen python mimir_eval.py --output results/eval-x --report-only   # 只重新生成报告
```

CI（`.github/workflows/mimir-eval.yml`）在 push 修改了 `eval/mimir.toml` 时，或手动触发时（`gh workflow run "MimIR eval" -f candidate_ref=<sha>`）运行同一命令；报告作为 job summary，结果作为 artifact 上传。默认使用 GitHub 托管的 `ubuntu-26.04`，噪声较大；需要稳定数据时注册 self-hosted runner，并设置仓库变量 `GEMM_BENCH_RUNS_ON`（JSON 标签，如 `["self-hosted","gemm-bench"]`）与 `GEMM_BENCH_CPU`。工作流没有 `pull_request` 触发器，fork 的代码不会在该机器上运行。

## Benchmark 方式

各后端使用相同 seed 的 FP32 输入和一个运行线程。Linux 严格绑定 CPU；macOS 使用下述 QoS 调度。公共 NumPy FP64 实现独立校验结果；Conv2D 参考通过滑动窗口 contraction 计算。预热后与 PyTorch 交替计时 9 个原生批次，每批默认至少 100 ms，取每调用均值的中位数。Python 调用、输入加载、编译、TVM 调优和 dump 不计入耗时。

| 后端 | GEMM | GEMV | Conv2D | 计时范围 |
|---|---|---|---|---|
| PyTorch baseline | ATen `mm.out` | ATen `mv.out` | ATen `convolution` | 原生 C++ 循环；GEMM/GEMV 输出预分配；Conv2D 包含输出分配和框架内部 packing |
| TVM | GEMM TIR | 向量 reduction TIR | NCHW reduction TIR，包含 padding 判断 | 原生 `time_evaluator`；输出预分配，可选 MetaSchedule |
| IREE | `linalg.matmul` | `linalg.matvec` | `linalg.conv_2d_nchw_fchw` + 必要的 `tensor.pad` | 原生 `iree-benchmark-module`；包含 VM、dispatch、输出分配 |
| MimIR | `tensor.product_2d` | `tensor.dot_product` | `tensor.conv` | 原生 C++ extern 循环；包含输出/内部内存分配和 free |
| OpenBLAS | `cblas_sgemm` | `cblas_sgemv` | 每次执行 im2col + SGEMM | 原生 C 循环；输出/scratch 预分配；Conv2D packing 和补零计入耗时 |

Conv2D 的 OpenBLAS 行是本项目的 im2col 包装实现，OpenBLAS 没有直接 Conv2D 接口。各后端计时边界不同，`timing_scope` 和 `baseline_timing_scope` 明确记录差异。

GFLOPS：GEMM `2*M*N*K`；GEMV `2*M*K`；Conv2D `2*N*Cout*OH*OW*Cin*KH*KW`（按完整 kernel 计数，包含 padding 位置，不计 bias/packing 操作）。

## TVM 自动调优

```toml
# config/tvm.toml
[tvm.auto_tune]
enabled = false
trials = 64
```

`--auto-tune` / `--no-auto-tune` 优先于子配置，对三种算子生效。`--trials` 单独使用不启用调优；没有选择 TVM 时不能指定其调优参数。开启时 MetaSchedule EvolutionarySearch + RandomModel 搜索实际算子 TIR；关闭时 LLVM 规则生成一次候选，直接使用第 0 个候选，不搜索、不变异。编译/调优时间单独记录。

调优 builder 的 CPU 列表需要适配机器；候选测量和正式 benchmark 保持公共 CPU 单线程。不同模式、baseline 或编译器版本请使用新输出目录；已有结果的配置校验会阻止混用。

## 汇总和 dump

统一入口的 `--output` 始终是根目录：`<output>/<backend>/results.json` 和 `<output>/summary.csv`。汇总包含 `op`、`shape`、`output_shape`、`baseline_backend`、`baseline_ms`、GFLOPS、加速比和双方计时范围；原始 JSON 记录每批时间、实际循环次数、误差和环境。

`--dump` 导出到 `<output>/<backend>/artifacts/<operator_dims>/`；`--dump-dir` 另指定根目录，仍添加后端子目录。GEMM 保留旧的 `<M>_<N>_<K>` 目录；GEMV/Conv2D 添加算子前缀，避免同尺寸冲突。独立入口不再添加后端子目录。`--dump-dir` 隐含开启，`--no-dump` 覆盖公共配置；`--check` 不生成产物。

每个 dump 包含精确 A/B 输入、结果 JSON、环境/算子信息和 SHA256 清单；`baseline/` 保存本次 PyTorch 被测包装器和对应算子的分派信息，也纳入清单。TVM 保存最终 TIR、trace、LLVM IR、汇编和动态库；IREE 保存 MLIR、VMFB、VM 反汇编及编译产物；MimIR 保存源码/最终 MimIR IR、LLVM IR、汇编和被测动态库；OpenBLAS/PyTorch 保存包装器和预编译依赖信息。

更多复现细节：[TVM](tvm/README.zh-CN.md)、[IREE](iree/README.zh-CN.md)、[PyTorch](pytorch/README.zh-CN.md)、[MimIR](mimir/README.zh-CN.md)。

## 历史结果

旧本地实验将 TVM 源码构建、OpenBLAS baseline 的 GEMM 结果保存在 `tvm/reference-results/` 和 `results/expanded-cpu-20261012/`。这些历史本地产物未包含在当前 checkout 中。旧 16 组 × 四后端实验使用每组 64 条有效调优记录及 IREE DT + ukernels；对应 JSON/CSV 保留其原始含义。报告工具读取旧结果时仍标记 `baseline_backend=openblas`，不重新解释其加速比；当前新结果使用 `baseline_backend=pytorch`。

```bash
uv run --frozen python -m common.report \
  results/multiop/pytorch/results.json results/multiop/tvm/results.json \
  --output results/multiop/selected-summary.csv
uv run --frozen python -m unittest discover -s tests -v
```

默认 CPU 优化：TVM 使用 `[tvm].schedule = "llvm"`：固定 seed，以 `ms.ScheduleRule.create("llvm")` 生成一次候选，`mutator_probs={}`，不创建搜索策略或 cost model，直接使用候选 0。target 后处理仅落实标注，与默认规则实验一致。候选 0 被拒绝则报错，不静默切换其他候选。AutoTune 开启时优先；DLight 和 naive 保留为显式可选项。IREE 默认 DT + `ukernels=all`；`config/iree-default.toml` 保留默认编译 pipeline。
