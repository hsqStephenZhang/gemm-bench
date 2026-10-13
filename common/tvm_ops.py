"""TVMScript operators consumed by the same optional MetaSchedule pipeline."""

import tvm
from tvm.script import tirx as T
from common.operators import tensor_shapes


def module(case):
    if case["op"] == "gemm":
        m, n, k = case["shape"]

        @T.prim_func(s_tir=True)
        def main(
            A: T.Buffer((m, k), "float32"),
            B: T.Buffer((k, n), "float32"),
            C: T.Buffer((m, n), "float32"),
        ):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            for i, j, r in T.grid(m, n, k):
                with T.sblock("C"):
                    vi, vj, vr = T.axis.remap("SSR", [i, j, r])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vr] * B[vr, vj]
    elif case["op"] == "gemv":
        m, k = case["shape"]

        @T.prim_func(s_tir=True)
        def main(
            A: T.Buffer((m, k), "float32"),
            B: T.Buffer((k,), "float32"),
            C: T.Buffer((m,), "float32"),
        ):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            for i, r in T.grid(m, k):
                with T.sblock("C"):
                    vi, vr = T.axis.remap("SR", [i, r])
                    with T.init():
                        C[vi] = T.float32(0)
                    C[vi] = C[vi] + A[vi, vr] * B[vr]
    else:
        n, ci, h, w, co, kh, kw, stride, pad = case["shape"]
        _, _, out = tensor_shapes(case)
        oh, ow = out[2:]

        @T.prim_func(s_tir=True)
        def main(
            A: T.Buffer((n, ci, h, w), "float32"),
            B: T.Buffer((co, ci, kh, kw), "float32"),
            C: T.Buffer((n, co, oh, ow), "float32"),
        ):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            for bn, oc, y, x, rc, ry, rx in T.grid(n, co, oh, ow, ci, kh, kw):
                with T.sblock("C"):
                    vn, vo, vy, vx, vc, vry, vrx = T.axis.remap(
                        "SSSSRRR", [bn, oc, y, x, rc, ry, rx]
                    )
                    with T.init():
                        C[vn, vo, vy, vx] = T.float32(0)
                    C[vn, vo, vy, vx] = (
                        C[vn, vo, vy, vx]
                        + T.if_then_else(
                            (vy * stride + vry >= pad)
                            and (vy * stride + vry < h + pad)
                            and (vx * stride + vrx >= pad)
                            and (vx * stride + vrx < w + pad),
                            A[vn, vc, vy * stride + vry - pad, vx * stride + vrx - pad],
                            T.float32(0),
                        )
                        * B[vo, vc, vry, vrx]
                    )

    return tvm.IRModule({"main": main})
