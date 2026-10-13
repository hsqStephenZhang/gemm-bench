"""Default-rule candidates are generated without search or mutation."""

import unittest
import tvm
from tvm.s_tir import meta_schedule as ms
from common.tvm_ops import module


class DefaultRules(unittest.TestCase):
    def test_seeded_generation_has_no_search_or_mutators(self):
        mod = module(dict(op="gemm", shape=[7, 13, 17]))
        target = tvm.target.Target(
            {"kind": "llvm", "mcpu": "alderlake", "num-cores": 1}
        )
        for kind in ["llvm", "from-target"]:
            with self.subTest(kind=kind):
                space = ms.space_generator.PostOrderApply(
                    sch_rules=kind, postprocs="from-target", mutator_probs={}
                )
                ctx = ms.TuneContext(
                    mod=mod,
                    target=target,
                    space_generator=space,
                    rand_state=42,
                    num_threads=1,
                )
                self.assertIsNone(ctx.search_strategy)
                generator = ctx.space_generator
                assert generator is not None
                mutators = generator.mutator_probs
                postprocs = generator.postprocs
                assert mutators is not None and postprocs is not None
                self.assertEqual(len(mutators), 0)
                schedules = generator.generate_design_space(mod)
                self.assertGreater(len(schedules), 0)
                for schedule in schedules:
                    schedule.enter_postproc()
                    self.assertTrue(all(p.apply(schedule) for p in postprocs))

    def test_default_selects_candidate_zero_and_compiles_all_ops_without_tuning(self):
        import numpy as np
        from unittest.mock import patch
        from common.operators import inputs, tensor_shapes
        from common.tvm_schedule_rules import schedule as default_schedule

        target = tvm.target.Target(
            {"kind": "llvm", "mcpu": "alderlake", "num-cores": 1}
        )
        cases = [
            dict(op="gemm", shape=[7, 13, 17]),
            dict(op="gemv", shape=[9, 17]),
            dict(op="conv2d", shape=[1, 3, 9, 11, 5, 3, 2, 2, 1]),
        ]
        for case in cases:
            with (
                self.subTest(op=case["op"]),
                patch.object(
                    ms.tir_integration,
                    "tune_tir",
                    side_effect=AssertionError("Default mode must not tune"),
                ),
            ):
                final, trace, details = default_schedule(module(case), target, 42)
                self.assertEqual(details["selected_candidate"], 0)
                self.assertIsNone(details["search_strategy"])
                self.assertEqual(details["mutators"], 0)
                self.assertGreater(details["generated_candidates"], 0)
                a, b, reference = inputs(case, 42)
                lib = tvm.tirx.build(final, target=target)
                c = tvm.runtime.empty(tensor_shapes(case)[2], "float32")
                lib(tvm.runtime.tensor(a), tvm.runtime.tensor(b), c)
                np.testing.assert_allclose(c.numpy(), reference, rtol=1e-3, atol=1e-3)
