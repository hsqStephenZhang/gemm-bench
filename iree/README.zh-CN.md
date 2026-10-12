# IREE CPU GEMM / GEMV / Conv2D

[English](README.md) | [简体中文](README.zh-CN.md)

使用统一 uv 环境、`llvm-cpu` 后端和 `local-sync` 驱动，公共 CPU affinity/线程/输入配置与 PyTorch baseline 一致。

```bash
cd /workspaces/ml-compiler/gemm-bench
uv sync --locked
uv run --frozen python run.py --backends iree --check
uv run --frozen python run.py --backends iree \
  --cases pack,mv_pack,conv_small --dump --output results/iree-multiop
# 独立入口直接使用指定输出目录
uv run --frozen python iree/run.py --ops conv2d --cases conv_small \
  --dump --output results/iree-conv-direct
```

GEMM 使用 linalg.matmul，GEMV 使用真实一维输出的 linalg.matvec，Conv2D 使用 [linalg.conv_2d_nchw_fchw](https://mlir.llvm.org/docs/Dialects/Linalg/#linalgconv_2d_nchw_fchw-linalgconv2dnchwfchwop)，必要时插入 tensor.pad。输入 NCHW，权重 OIHW，无 bias，groups=1、dilation=1，stride/padding 读取具名用例。

编译器把静态 MLIR 降低到 VMFB。Python runtime 仅用于预热和正确性；正式计时调用原生 iree-benchmark-module，使用精确 A.npy/B.npy 输入和 main 函数。每批按最低时间运行，原生次数不足 reps 时改按 reps 次重测，保存实际采用的原生 JSON。与 PyTorch 交替测 9 批，取每调用 real_time 的中位数。

IREE 耗时包含 VM 调用、dispatch、算子及输出分配；排除编译、Python、输入加载和导出。PyTorch GEMM/GEMV 输出预分配，Conv2D 包含框架输出分配；双方范围记录在结果中，加速比统一为 PyTorch / IREE。

专属配置仍可选择默认 pipeline、数据平铺、延后物化：

```bash
uv run --frozen python run.py --backends iree --ops conv2d \
  --iree-config config/iree-dt.toml --output results/iree-conv-dt
uv run --frozen python run.py --backends iree --ops conv2d \
  --iree-config config/iree-dt-late.toml --output results/iree-conv-dt-late
```

DT 启用 `--iree-opt-data-tiling=true`、`--iree-llvmcpu-enable-ukernels=all`，late 额外关闭 early materialization；更快与否由本机测试决定，不用模型端到端历史数据估算单算子性能。默认 target CPU 读取公共 mcpu。

Dump 位于 `<output>/iree/artifacts/<operator_dims>/`：沿用 `matmul.mlir` / `matmul.vmfb` 文件名，内容对应实际选中的算子；包含编译命令/日志、阶段 IR、CPU executable 产物和 VM 字节码反汇编。baseline/ 保存实际测得的 PyTorch 包装器/算子分派信息，输入、结果及 SHA256 清单一起保留。

公共配置、用例格式和选择参数见[项目 README](../README.zh-CN.md)。
