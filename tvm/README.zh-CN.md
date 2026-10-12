# TVM CPU GEMM / GEMV / Conv2D：可选自动调优

[English](README.md) | [简体中文](README.zh-CN.md)

统一入口 `run.py` 每次先运行 PyTorch 基准，再运行选中的可选后端；`--output` 是根目录，本后端结果位于其 `tvm/` 子目录。独立入口的 `--output` 直接作为本后端目录，仍交替测量 PyTorch 参照。


## 环境

使用项目根目录 uv 环境的 `apache-tvm` wheel，当前锁定版本为 0.27.0.post1。编译器、运行时和匹配的 FFI 由 uv 安装，无需本地 TVM 源码或 build/lib。安装方式见 [TVM 官方指南](https://tvm.apache.org/docs/install/)。

```bash
cd /workspaces/ml-compiler/gemm-bench
uv add apache-tvm
# 复现锁定环境时使用 uv sync --locked
uv run --frozen python tvm/run.py --check
```

启动器直接加载 `.venv` 的 TVM 包及其随附动态库，不再复制或替换 FFI。`--check` 输出包版本、安装路径、LLVM 支持和单线程设置；结果元数据保存 wheel 版本、包路径及编译构建信息。

## 自动调优开关

专属配置 `config/tvm.toml`：

```toml
[tvm.auto_tune]
enabled = false
trials = 64
```

| 模式 | 执行路径 | 结果标记 |
|---|---|---|
| 默认 / --no-auto-tune | CPU DLight 规则 → LLVM | auto_tune=false, variant=dlight |
| --auto-tune | MetaSchedule 选择调度 → LLVM | auto_tune=true, variant=auto_tuned |

默认模式应用确定性 CPU DLight 调度，不搜索或复用数据库。GEMM/Conv2D 使用项目规则，GEMV 使用上游规则。调优模式仍使用 EvolutionarySearch + RandomModel，每组默认 64 候选。

```bash
# 关闭调优
uv run --frozen python run.py --backends tvm --no-auto-tune \
  --cases pack,k49 --dump --output results/tvm-dlight

# 开启调优；使用新目录实施全新搜索
uv run --frozen python run.py --backends tvm --auto-tune --trials 64 \
  --cases pack,k49 --dump --output results/tvm-auto-tuned

# 独立入口支持相同开关
uv run --frozen python tvm/run.py --auto-tune --trials 64 \
  --cases pack --output results/tvm-direct-tuned

# 根入口用 --tvm-config，独立入口用 --config
uv run --frozen python run.py --backends tvm --tvm-config config/tvm.toml
uv run --frozen python tvm/run.py --config config/tvm.toml --no-auto-tune
```

CLI 优先于 `[tvm.auto_tune].enabled`。`--trials` 单独使用不启用调优。选择多个后端时，调优开关只转发给 TVM。关闭模式不创建 builder/runner，不读取或写入调优数据库，`tuning_s=0`、`valid_records=0`。

不同模式请使用不同输出目录；结果配置校验阻止调优与未调优结果混合。启用时，记录数达到预算的相同目录可以复用数据库；严格重新搜索请用新目录。

## 公共用例

尺寸、CPU、设备、seed、reps、误差及 baseline 公共设置读取 `config/common.toml`；PyTorch 专属设置读取 `config/pytorch.toml` 或统一入口的 `--pytorch-config`。默认 32 组来自公共 cases_files：16 GEMM、10 GEMV、6 Conv2D，分别位于 config/gemm_cases.txt、gemv_cases.txt、conv2d_cases.txt。用例文件使用 M K N；旧 `--shapes` 接口仍为 M,N,K。

入口支持 `--ops gemm,gemv,conv2d`、`--cases-file`、`--cases skinny,pack,k49`、`--reps 200`。reps 是每批次最少调用次数，最低时间校准可能增加次数。

GEMV 使用独立的一维向量输出/reduction；Conv2D 使用 NCHW/OIHW reduction，包含 stride/pad 的边界判断。三种算子均支持同一 `[tvm.auto_tune]` 和 CLI 开关，最终输出形状来自公共算子定义。

```bash
uv run --frozen python run.py --backends tvm --ops gemv,conv2d \
  --cases mv_pack,conv_small --auto-tune --trials 64 --dump \
  --output results/tvm-multiop-tuned
```

## Benchmark 方式与计时边界

1. 公共模块生成 FP32 行主序 A/B 和 FP64 参考结果，将进程固定到公共 CPU，TVM 运行线程数为 1。
2. 若启用调优，MetaSchedule runner 默认每候选 `number=1, repeat=3, min_repeat_ms=20`，用于选择调度；调优时间单独记录。最多四个 builder 在原始允许的 CPU 集合内编译。builder_cpus 为空时自动选择测量 CPU 之外的允许 CPU；只有一个 CPU 时回退到该 CPU。显式列表会校验是否属于原始允许集合。候选测量仍在所选公共 CPU 单线程执行。
3. 编译最终模块，复制 A/B 到 TVM tensor 并预分配 C；公共模块预热 5 次并校验。
4. 两种模式使用相同的原生计时：

```python
evaluator = lib.time_evaluator(
    "main", tvm.cpu(), number=case_reps, repeat=1, min_repeat_ms=100
)
seconds_per_call = evaluator(a, b, c).results[0]
```

原生 evaluator 自适应增加调用次数以达到最低时间，返回每调用平均秒数。公共模块与 PyTorch 交替测 9 个批次，取批次均值中位数。计时包含 TVM 原生调用和实际内核，排除 Python、输入复制、输出预分配、调优及编译。实际自适应次数未通过 API 暴露，`backend_iterations` 为 null。

`compile_s` 单独记录；汇总 variant 区分两种模式。相同输入重复执行，无主动 cache flush。

## Dump 最终产物

```bash
uv run --frozen python run.py --backends tvm --no-auto-tune \
  --cases pack --dump --output results/tvm-dlight-dump

uv run --frozen python tvm/run.py --auto-tune --trials 64 \
  --cases pack --output results/tvm-tuned-dump --dump-dir results/tvm-artifacts
```

统一入口导出到 `<output>/tvm/artifacts/<operator_dims>/`；独立入口为 `<output>/artifacts/<operator_dims>/`。`--dump-dir` 在统一入口下添加 `tvm/` 子目录，独立入口直接添加尺寸子目录：

| 文件 | 内容 |
|---|---|
| final_tir.py | 实际编译的最终 TIR；默认是 DLight 调度后的 TIR |
| final_trace.txt | AutoTune 或 DLight 的实际调度 trace；naive 模式记录未变换 |
| module.so | 本次实际测量模块的动态库 |
| kernel.ll / kernel.s | 被测构建的优化后 LLVM IR / CPU 汇编 |
| A.npy / B.npy | 精确输入 |
| benchmark_result.json / manifest.json | 原始计时、误差、模式、参数及文件 SHA256 |

开启模式额外保留兼容文件 best_schedule.py / best_trace.txt / best.so 及调优数据库。关闭模式不生成这些“最佳调度”文件。导出在计时之外进行。

## 历史自动调优结果

旧本地实验中，`tvm/reference-results/` 保存原六组；`results/expanded-cpu-20261012/` 保存后来 16 组、每组 64 条有效调优记录的实验。这些历史本地产物未包含在当前 checkout 中。它们使用旧本地源码构建，模式均为 auto_tune=true；历史数据保留不变，不能当作当前 wheel 的测量结果。新 wheel 请使用新输出目录。

如果保留了历史产物，可查看原始调度：

```bash
cat tvm/reference-results/1024_1024_1024/best_schedule.py
cat tvm/reference-results/1024_1024_1024/best_trace.txt
```

只有 TVM 版本、构建和 CPU target 一致的数据库才适合复用。切换到 wheel 后应重新搜索；使用新输出目录，避免复用旧源码构建的调优记录。

## VS Code / ty 导入解析

项目 `pyproject.toml` 的 `[tool.ty.environment]` 指定 `.venv`，ty 直接从已安装的 `apache-tvm` 解析 `tvm.script` 和 `tvm.s_tir`，无需本地源码 extra-paths。保存配置后若旧提示仍存在，在命令面板执行 `ty: Restart server`；也可执行 `Developer: Reload Window`。

更多配置、用例和选择说明见[项目 README](../README.zh-CN.md)。

所有新结果的 baseline 为 PyTorch；旧历史记录仍使用原 OpenBLAS baseline。`baseline/` dump 保存实际测量的 PyTorch 包装器，Conv2D 的框架输出分配范围见公共 README。

默认 CPU 优化：TVM 使用 DLight（`[tvm].schedule = "dlight"`），AutoTune 开启时优先。wheel 提供上游 GEMV/Reduction 规则；项目为 GEMM/直接 Conv2D 补充 `SpatialReduction`（8 列分块、向量化、分离 reduction 初始化）。每条结果记录命中规则，未匹配则报错，不退回 naive。显式设置 `schedule = "naive"` 可测试原始 TIR。IREE 默认 DT + `ukernels=all`；`config/iree-default.toml` 保留默认编译 pipeline。
