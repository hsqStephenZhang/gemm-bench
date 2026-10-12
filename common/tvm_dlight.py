"""CPU DLight dispatch; project rules cover GEMM and direct NCHW convolution.

The installed upstream CPU rules cover GEMV and multi-block reductions only.
These deterministic rules are project extensions, not upstream CPU Matmul rules.
"""
from typing import Any, cast

import tvm
from tvm import s_tir
from tvm.s_tir import dlight as dl
from tvm.s_tir.dlight.cpu.base import CPUScheduleRule


class SpatialReduction(CPUScheduleRule):
    """Tile contiguous output columns, vectorize, and retain serial reduction."""
    def apply(self, func, target, tunable):
        sch = s_tir.Schedule(func)
        block = sch.get_sblock('C')
        stmt = sch.get(block)
        assert isinstance(stmt, s_tir.SBlock)
        kinds = [axis.iter_type for axis in stmt.iter_vars]
        spatial = kinds.count(0)
        if spatial < 2 or kinds != [0]*spatial + [2]*(len(kinds)-spatial):
            return None
        loops = list(sch.get_loops(block))
        outer, inner = sch.split(loops[spatial-1], factors=[None, 8])
        # TVM annotates variadic reorder arguments as lists, but accepts LoopRVs.
        cast(Any, sch.reorder)(*loops[:spatial-1], outer, *loops[spatial:], inner)
        sch.vectorize(inner)
        init = sch.decompose_reduction(block, loops[spatial])
        sch.vectorize(sch.get_loops(init)[-1])
        return sch


def schedule(mod, target, case):
    candidates = [("upstream.cpu.GEMV", dl.cpu.GEMV()),
                  ("upstream.cpu.Reduction", dl.cpu.Reduction()),
                  ("project.cpu.SpatialReduction", SpatialReduction())]
    for name, rule in candidates:
        sch = rule.apply(mod['main'], target, False)
        if sch is None:
            continue
        if isinstance(sch, list):
            if len(sch) != 1:
                raise RuntimeError('DLight returned multiple schedules')
            sch = sch[0]
        final = tvm.IRModule({'main': sch.mod['main'].with_attr('tirx.is_scheduled', True)})
        return final, str(sch.trace), name
    raise RuntimeError(f'No CPU DLight rule matched {case}; refusing naive fallback')
