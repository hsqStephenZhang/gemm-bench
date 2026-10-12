# PyTorch：所有算子的 CPU baseline

[English](README.md) | [简体中文](README.zh-CN.md)

PyTorch 每次固定运行，TVM/IREE/OpenBLAS 与它交替测量。专属配置 `config/pytorch.toml` 的 `eager-native` 模式支持 GEMM、GEMV、Conv2D。使用统一 uv 环境，固定 CPU 和单个 intra-op/inter-op 线程。

```bash
cd /workspaces/ml-compiler/gemm-bench
uv sync --locked
uv run --frozen python run.py --backends none --check
uv run --frozen python run.py --backends none \
  --cases pack,mv_pack,conv_small --dump --output results/torch-baseline
# 独立入口 --output 直接作为本后端目录
uv run --frozen python pytorch/run.py --ops conv2d --cases conv_small \
  --dump --output results/torch-conv-direct
```

原生 C++ 包装器通过 ATen from_blob 引用已生成的输入：GEMM 调用 `at::mm_out`，GEMV 调用 `at::mv_out`，两者输出预分配；Conv2D 调用 `at::convolution`，无 bias、groups=1、dilation=1，计时包含输出分配与框架内部 packing。每个原生计时循环使用 InferenceMode；结果提取到 NumPy 缓冲区发生在计时后。GEMM/GEMV 使用当前 wheel 的 BLAS，Conv2D 使用 PyTorch 对该用例选择的 CPU 实现，不假定所有卷积都由同一种实现处理。

包装器先和相应 Python API 的结果核对，再通过公共 FP64 参考校验。预热后校准原生循环次数，满足用例 reps 下限与最低时间窗口。取 9 批每调用平均耗时的中位数；baseline 独立行的加速比为 1。编译、Python、输入加载和 dump 不计时。

统一入口 `--pytorch-config` 同时控制所有后端的 baseline。配置/构建信息记录在 `baseline_config` / `baseline_metadata`，避免不同 PyTorch 设置混用。当前 CPU wheel 版本和依赖由 `uv.lock` 锁定。

Dump 位于统一入口 `<output>/pytorch/artifacts/<operator_dims>/`。文件包括 `native_timer.cpp`、`pytorch_timer.so`、编译命令/日志、`operator.txt`、`dispatch_table.txt`、`timer_wrapper.s`、PyTorch 构建/依赖库信息、精确输入、结果和 SHA256 清单。`operator.txt` 随算子选择 mm.out / mv.out / convolution。汇编是包装器反汇编，实际 ATen 内核位于预编译库中。

公共配置、用例格式和选择参数见[项目 README](../README.zh-CN.md)。
