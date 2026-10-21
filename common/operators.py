"""Operator shapes, deterministic inputs and independent FP64 references.

linear is GEMM with a frozen right operand: B is the weight W[K,N], known when the kernel is
prepared, so backends may repack it once outside timing (reported as prepack_s).
"""

import numpy as np


def tensor_shapes(case):
    dims = case["shape"]
    if case["op"] in ("gemm", "linear"):
        m, n, k = dims
        return (m, k), (k, n), (m, n)
    if case["op"] == "gemv":
        m, k = dims
        return (m, k), (k,), (m,)
    n, ci, h, w, co, kh, kw, stride, pad = dims
    oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    return (n, ci, h, w), (co, ci, kh, kw), (n, co, oh, ow)


def flops(case):
    a, b, c = tensor_shapes(case)
    reduction = a[-1] if case["op"] != "conv2d" else b[1] * b[2] * b[3]
    return 2 * int(np.prod(c)) * reduction


def work_name(case):
    dims = "_".join(map(str, case["shape"]))
    return dims if case["op"] == "gemm" else case["op"] + "_" + dims


def inputs(case, seed):
    a_shape, b_shape, _ = tensor_shapes(case)
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, a_shape).astype("float32")
    b = rng.uniform(-1, 1, b_shape).astype("float32")
    x, y = a.astype("float64"), b.astype("float64")
    if case["op"] != "conv2d":
        reference = x @ y
    else:
        *_, kh, kw, stride, pad = case["shape"]
        padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, (kh, kw), axis=(2, 3)
        )[:, :, ::stride, ::stride]
        reference = np.einsum("nchwij,ocij->nohw", windows, y, optimize=True)
    return a, b, reference
