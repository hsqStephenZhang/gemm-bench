# MimIR CPU GEMM / GEMV / Conv2D

[English](README.md) | [简体中文](README.zh-CN.md)

MimIR 是可选后端，PyTorch 仍为必选 baseline。适配器参考 [MimIR PR #1](https://github.com/hsqStephenZhang/MimIR/pull/1) 的 tensor extern ABI 和算子生成方式，使用本项目的统一输入、FP64 校验、固定单线程 CPU、原生计时批次和导出清单。

## Remote、ref 与持久构建目录

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

`remote` 支持 GitHub `owner/repo`、HTTPS/SSH URL 或本地 Git remote；`ref` 支持分支、tag、commit 或 pull ref。默认使用上游 `mimir/mimir` 仓库的 `master`。

首次运行 clone 到 `<build>/mimir/source`。之后复用这个 checkout，fetch 指定 ref、切换到解析出的提交、更新 submodule，并在 `<build>/mimir/compiler` 增量编译。切换 remote 时更新同一 checkout 的 origin，不按分支或实验输出目录重新 clone。checkout 中的本地修改不会被丢弃；存在修改时停止切换。

构建需要 Git、CMake、Ninja、支持 C++23 的编译器，以及编译生成 LLVM IR 的 clang++。默认 CMake Release 构建，关闭 Python 绑定和上游测试。构建日志和命令位于 `<build>/mimir/`；`checkout-state.json` 保存 remote、指定 ref、最终提交和目录。build 目录已加入 Git 忽略。`--check` 会准备工具链，但不计时算子或导出用例产物。

```bash
cd /workspaces/ml-compiler/gemm-bench
uv sync --locked
uv run --frozen python run.py --backends mimir --check

# 只跑上游 MimIR master 与 PyTorch，选择四个代表用例
uv run --frozen python run.py --backends mimir --mimir-ref master \
  --cases pack,mv_pack,conv_small,conv_1x1 --reps 1 \
  --dump --output results/mimir-master-vs-pytorch

# 在同一 build checkout 中使用用户 fork 和 PR ref
uv run --frozen python run.py --backends mimir \
  --mimir-remote hsqStephenZhang/MimIR --mimir-ref refs/pull/1/head \
  --cases pack --output results/mimir-pr1

# 指定其他 build 根目录或后端配置
uv run --frozen python run.py --backends mimir --build-dir build \
  --mimir-config config/mimir.toml --cases mv_pack --output results/mimir-gemv

# 独立入口：--remote / --ref / --config
uv run --frozen python mimir/run.py --remote mimir/mimir --ref master \
  --ops conv2d --cases conv_small --dump --output results/mimir-direct
```

CLI 优先于配置。切换 ref、remote 或编译选项时使用新的结果目录；源码 checkout 和编译目录继续复用。严格复现时用记录的提交号作为 `--mimir-ref`，替代可能移动的分支名。

## 算子与计时

GEMM 使用 `tensor.product_2d`；GEMV 使用 `tensor.dot_product`，输入和输出都是真正的一维向量；Conv2D 使用 `tensor.conv`，输入 NCHW、权重 OIHW，无 bias，groups=1、dilation=1，stride/pad 来自用例。卷积尺寸显式传参，避免 singleton axis 的隐式推导歧义。

各用例生成 MimIR 源码，经 `opt` 和 `ll` plugin 输出 LLVM IR，再与原生 C++ 计时包装器编译为动态库。默认 clang 使用 `-O3` 和 `-march=<benchmark.mcpu>`，与公共 CPU target 一致。`mcpu=generic` 时不添加 CPU 专属模板标志；其他架构可修改 `target_flags`。

tensor extern 返回 malloc 分配的结果。原生计时每次包含算子、输出/内部内存分配及 free；校验结果复制到 NumPy 缓冲区发生在计时之外。PyTorch GEMM/GEMV 输出预分配，Conv2D 包含框架分配；结果明确记录双方范围。不进行 Python 开销扣减。

公共预热和正确性检查之后，与 PyTorch 交替测九批，默认每批至少 100 ms 且不少于 reps 次。取每调用均值的中位数，加速比为 `PyTorch / MimIR`。编译和 dump 不计时。元数据记录实际编译器提交、二进制/plugin/import 的 SHA256、clang 版本、参数及 remote/ref。

## Dump

统一入口导出到 `<output>/mimir/artifacts/<operator_dims>/`；独立入口使用 `<output>/artifacts/<operator_dims>/`。

| 文件 | 内容 |
|---|---|
| kernel.mim | 当前算子的生成源码 |
| final.mim | MimIR 最终优化后程序 dump |
| kernel.ll | MimIR 输出的 LLVM IR |
| optimized_kernel.ll / kernel.s | 使用被测 target 参数的 clang 优化 LLVM IR / 汇编 |
| kernel.so / native_timer.cpp | 实际计时的动态库和原生计时源码 |
| mimir_command.json / clang_command.json | 编译命令 |
| mimir_compile.log / clang_compile.log | 编译日志 |
| baseline/ | 被测 PyTorch 包装器和算子分派信息 |
| A.npy / B.npy / benchmark_result.json / manifest.json | 精确输入、结果、环境及 SHA256 清单 |

公共用例和 baseline 配置见[项目 README](../README.zh-CN.md)。PR #1 还包含网络 benchmark；本适配器仅覆盖当前 GEMM/GEMV/Conv2D 套件。
