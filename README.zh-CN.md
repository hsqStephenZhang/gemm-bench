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
cases_files = ["config/gemm_cases.txt", "config/gemv_cases.txt", "config/conv2d_cases.txt"]

[backends]
tvm = true
iree = true
openblas = true
mimir = true
```

`--backends`（同义参数 `--backend`）覆盖可选后端配置，支持任意组合、`all`、`none`。PyTorch 自动运行，不能关闭，不加入可选列表。`--ops gemm,gemv` 覆盖 `benchmark.ops`；`--cases name1,name2` 从选中算子的用例中选择；`--cases-file` 替换整套文件；旧公共配置 `cases_file` 仍可读取。

`--pytorch-config` 同时用于独立 baseline 和各后端的交替 baseline，确保设置一致。`--tvm-config` / `--iree-config` / `--mimir-config` 选择对应专属配置；`--common-config` 选择公共配置。各实现独立入口仍支持公共参数，`--config` 指定该实现配置，`--output` 直接作为本实现目录：

```bash
uv run --frozen python tvm/run.py --ops gemv --cases mv_pack --auto-tune --trials 16
uv run --frozen python iree/run.py --ops conv2d --cases conv_small
uv run --frozen python pytorch/run.py --ops all --cases pack,mv_pack,conv_small
```

## MimIR remote 和 build 缓存

`config/mimir.toml` 默认 `remote = "mimir/mimir"`、`ref = "master"`，可用 `--mimir-remote` / `--mimir-ref` 覆盖。首次 clone 到 `<build>/mimir/source`，之后在同一 checkout 中 fetch/切换，并在 `<build>/mimir/compiler` 增量构建。公共配置 `[build].directory` 或 CLI `--build-dir` 指定构建根目录，默认 `build`。切换实验输出目录不会产生新的 clone。MimIR 的 `--check` 也会按需准备编译器。

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

## Benchmark 方式

各后端使用相同 seed 的 FP32 输入、CPU affinity 和一个运行线程。公共 NumPy FP64 实现独立校验结果；Conv2D 参考通过滑动窗口 contraction 计算。预热后与 PyTorch 交替计时 9 个原生批次，每批默认至少 100 ms，取每调用均值的中位数。Python 调用、输入加载、编译、TVM 调优和 dump 不计入耗时。

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

`--auto-tune` / `--no-auto-tune` 优先于子配置，对三种算子生效。`--trials` 单独使用不启用调优；没有选择 TVM 时不能指定其调优参数。开启时 MetaSchedule EvolutionarySearch + RandomModel 搜索实际算子 TIR；关闭时原始 TIR 直接编译。编译/调优时间单独记录。

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
