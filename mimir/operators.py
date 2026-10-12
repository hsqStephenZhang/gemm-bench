"""MimIR tensor externs, following MimIR PR #1 with true vector GEMV support."""
from common.operators import tensor_shapes

PRELUDE='''plugin tensor;
plugin core;
plugin math;
let F32 = math.F32;
let R: tensor.Ring = (F32, 0:F32, math.arith.add math.mode.contract, math.arith.mul math.mode.contract);
'''


def module(case):
    a,b,c=tensor_shapes(case)
    def tensor(shape):return '«'+', '.join(map(str,shape))+'; F32»'
    if case['op']=='gemm':
        expression='tensor.product_2d R (a, b)'
    elif case['op']=='gemv':
        expression='tensor.dot_product R (1₂, 0₁, (), ()) (a, b)'
    elif case['op']=='conv2d':
        n,ci,h,w,co,kh,kw,stride,pad=case['shape']
        # Explicit extents prevent singleton axes from confusing implicit shape inference.
        expression=f'tensor.conv R @ ({n}, {ci}, {co}, {h}, {w}, {kh}, {kw}) (({stride}, {stride}), (1, 1), ({pad}, {pad})) (a, b)'
    else:raise ValueError(f"Unsupported MimIR operator: {case['op']}")
    return PRELUDE+f'extern fun mim_kernel (a: {tensor(a)}, b: {tensor(b)}): {tensor(c)} = return ({expression});\n'
