"""Static linalg GEMM, matvec and NCHW Conv2D modules."""

from common.operators import tensor_shapes


def module(case):
    a, b, c = tensor_shapes(case)

    def ty(shape):
        return "tensor<" + "x".join(map(str, shape)) + "xf32>"

    at, bt, ct = ty(a), ty(b), ty(c)
    prefix = f"""module {{
  func.func @main(%a: {at}, %b: {bt}) -> {ct} {{
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : {ct}
    %init = linalg.fill ins(%zero : f32) outs(%empty : {ct}) -> {ct}
"""
    lhs = "%a"
    if case["op"] == "conv2d":
        n, ci, h, w, co, kh, kw, stride, pad = case["shape"]
        if pad:
            pat = ty((n, ci, h + 2 * pad, w + 2 * pad))
            prefix += f"""    %padded = tensor.pad %a low[0, 0, {pad}, {pad}] high[0, 0, {pad}, {pad}] {{
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %zero : f32
    }} : {at} to {pat}
"""
            lhs = "%padded"
            at = pat
        op = f"linalg.conv_2d_nchw_fchw {{strides = dense<{stride}> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>}}"
    else:
        op = "linalg.matmul" if case["op"] == "gemm" else "linalg.matvec"
    return (
        prefix
        + f"""    %result = {op} ins({lhs}, %b : {at}, {bt}) outs(%init : {ct}) -> {ct}
    return %result : {ct}
  }}
}}
"""
    )
