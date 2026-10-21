"""Verify CPU DLight schedules on tail dimensions and padded convolutions."""

import unittest
import numpy as np
import tvm
from common.host import resolve_mcpu
from common.tvm_ops import module
from common.tvm_dlight import schedule
from common.operators import tensor_shapes, inputs


class DLightTests(unittest.TestCase):
    def test_tail_gemm_and_gemv(self):
        rng = np.random.default_rng(5)
        for op, shape in [("gemm", [7, 13, 17]), ("gemv", [9, 17])]:
            case = dict(op=op, shape=shape)
            a_shape, b_shape, c_shape = tensor_shapes(case)
            a = rng.uniform(-1, 1, a_shape).astype("float32")
            b = rng.uniform(-1, 1, b_shape).astype("float32")
            target = tvm.target.Target(
                {"kind": "llvm", "mcpu": resolve_mcpu("native"), "num-cores": 1}
            )
            mod, trace, rule = schedule(module(case), target, case)
            self.assertIn("vectorize", trace)
            self.assertEqual(
                rule,
                "upstream.cpu.GEMV" if op == "gemv" else "project.cpu.SpatialReduction",
            )
            lib = tvm.tirx.build(mod, target=target)
            c = tvm.runtime.empty(c_shape, "float32")
            lib(tvm.runtime.tensor(a), tvm.runtime.tensor(b), c)
            np.testing.assert_allclose(
                c.numpy(),
                a.astype("float64") @ b.astype("float64"),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_padded_convolution_with_vector_tail(self):
        case = dict(op="conv2d", shape=[1, 3, 9, 11, 5, 3, 2, 2, 1])
        a, b, reference = inputs(case, 5)
        target = tvm.target.Target(
            {"kind": "llvm", "mcpu": resolve_mcpu("native"), "num-cores": 1}
        )
        mod, trace, rule = schedule(module(case), target, case)
        self.assertEqual(rule, "project.cpu.SpatialReduction")
        self.assertIn("vectorize", trace)
        lib = tvm.tirx.build(mod, target=target)
        c = tvm.runtime.empty(tensor_shapes(case)[2], "float32")
        lib(tvm.runtime.tensor(a), tvm.runtime.tensor(b), c)
        np.testing.assert_allclose(c.numpy(), reference, rtol=1e-3, atol=1e-3)
