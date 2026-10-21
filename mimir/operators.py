"""MimIR tensor externs, following MimIR PR #1 with true vector GEMV support."""

from common.operators import tensor_shapes

PRELUDE = """plugin tensor;
plugin core;
plugin math;
let F32 = math.F32;
let R: tensor.Ring = (F32, 0:F32, math.arith.add math.mode.contract, math.arith.mul math.mode.contract);
"""


def module(case, weight_layout="kn"):
    a, b, c = tensor_shapes(case)

    def tensor(shape):
        return "«" + ", ".join(map(str, shape)) + "; F32»"

    if case["op"] == "linear":
        if weight_layout == "nk":
            b = b[::-1]  # frozen W transposed once to [N,K] (nn.Linear layout)
        if a[0] == 1:
            # MimIR shape inference mistakes a «1, K» operand for a vector; use the same flat
            # x[K] -> y[N] buffers as a matrix-vector contraction (identical memory layout).
            a, c = a[1:], c[1:]
            expression = (
                "tensor.dot_product R (1₂, 0₁, (), ()) (b, a)"
                if weight_layout == "nk"
                else "tensor.dot_product R (0₁, 0₂, (), ()) (a, b)"
            )
        else:
            expression = (
                "tensor.dot_product R (1₂, 1₂, (), ()) (a, b)"
                if weight_layout == "nk"
                else "tensor.product_2d R (a, b)"
            )
    elif case["op"] == "gemm":
        expression = "tensor.product_2d R (a, b)"
    elif case["op"] == "gemv":
        expression = "tensor.dot_product R (1₂, 0₁, (), ()) (a, b)"
    elif case["op"] == "conv2d":
        n, ci, h, w, co, kh, kw, stride, pad = case["shape"]
        # Explicit extents prevent singleton axes from confusing implicit shape inference.
        expression = f"tensor.conv R @ ({n}, {ci}, {co}, {h}, {w}, {kh}, {kw}) (({stride}, {stride}), (1, 1), ({pad}, {pad})) (a, b)"
    else:
        raise ValueError(f"Unsupported MimIR operator: {case['op']}")
    return (
        PRELUDE
        + f"extern fun mim_kernel (a: {tensor(a)}, b: {tensor(b)}): {tensor(c)} = return ({expression});\n"
    )
