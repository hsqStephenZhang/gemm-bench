"""TVMScript operators consumed by the same optional MetaSchedule pipeline."""
import tvm
from tvm import relax
from tvm.script import tirx as T
from common.operators import tensor_shapes


def module(case):
    if case['op'] in ('gemm','linear'):
        m,n,k=case['shape']
        @T.prim_func(s_tir=True)
        def main(A:T.Buffer((m,k),'float32'),B:T.Buffer((k,n),'float32'),C:T.Buffer((m,n),'float32')):
            T.func_attr({'global_symbol':'main','tirx.noalias':True})
            for i,j,r in T.grid(m,n,k):
                with T.sblock('C'):
                    vi,vj,vr=T.axis.remap('SSR',[i,j,r])
                    with T.init():C[vi,vj]=T.float32(0)
                    C[vi,vj]=C[vi,vj]+A[vi,vr]*B[vr,vj]
    elif case['op']=='gemv':
        m,k=case['shape']
        @T.prim_func(s_tir=True)
        def main(A:T.Buffer((m,k),'float32'),B:T.Buffer((k,),'float32'),C:T.Buffer((m,),'float32')):
            T.func_attr({'global_symbol':'main','tirx.noalias':True})
            for i,r in T.grid(m,k):
                with T.sblock('C'):
                    vi,vr=T.axis.remap('SR',[i,r])
                    with T.init():C[vi]=T.float32(0)
                    C[vi]=C[vi]+A[vi,vr]*B[vr]
    else:
        n,ci,h,w,co,kh,kw,stride,pad=case['shape'];_,_,out=tensor_shapes(case);oh,ow=out[2:]
        @T.prim_func(s_tir=True)
        def main(A:T.Buffer((n,ci,h,w),'float32'),B:T.Buffer((co,ci,kh,kw),'float32'),C:T.Buffer((n,co,oh,ow),'float32')):
            T.func_attr({'global_symbol':'main','tirx.noalias':True})
            for bn,oc,y,x,rc,ry,rx in T.grid(n,co,oh,ow,ci,kh,kw):
                with T.sblock('C'):
                    vn,vo,vy,vx,vc,vry,vrx=T.axis.remap('SSSSRRR',[bn,oc,y,x,rc,ry,rx])
                    with T.init():C[vn,vo,vy,vx]=T.float32(0)
                    C[vn,vo,vy,vx]=C[vn,vo,vy,vx]+T.if_then_else(
                        (vy*stride+vry>=pad) and (vy*stride+vry<h+pad) and (vx*stride+vrx>=pad) and (vx*stride+vrx<w+pad),
                        A[vn,vc,vy*stride+vry-pad,vx*stride+vrx-pad],T.float32(0))*B[vo,vc,vry,vrx]
    if case['op']=='linear':
        # Frozen weight: MetaSchedule's RewriteLayout may then pick a packed layout for B.
        main=main.with_attr('layout_free_buffers',[1])
    return tvm.IRModule({'main':main})


def split_weight_prepack(mod, case):
    """Split a layout-rewritten GEMM into pack(B, Bp) and main(A, Bp, C).

    Returns (module, packed_shape), or (None, None) when the schedule kept B's layout.
    Uses the same Relax passes as static_shape_tuning_pipeline(cpu_weight_prepack=True).
    """
    if 'meta_schedule.layout_rewrite_preproc' not in mod.script():
        return None,None
    m,n,k=case['shape']
    # Parsed from text: TVMScript class bodies cannot see this function's local shapes.
    wrap=tvm.script.from_source(f'''
@I.ir_module
class Wrap:
    @T.prim_func(s_tir=True)
    def gemm(A: T.Buffer(({m}, {k}), "float32"), B: T.Buffer(({k}, {n}), "float32"), C: T.Buffer(({m}, {n}), "float32")):
        T.evaluate(0)
    @R.function
    def main(x: R.Tensor(({m}, {k}), "float32"), w: R.Tensor(({k}, {n}), "float32")) -> R.Tensor(({m}, {n}), "float32"):
        R.func_attr({{"num_input": 1}})
        with R.dataflow():
            y = R.call_tir(Wrap.gemm, (x, w), out_ty=R.Tensor(({m}, {n}), "float32"))
            R.output(y)
        return y
''')
    wrap['gemm']=mod['main'].with_attr('global_symbol','gemm').with_attr('tirx.is_scheduled',True)
    split=relax.transform.SplitLayoutRewritePreproc()(wrap)
    pack=split.get_global_var('gemm_weight_prepack')
    binding=next(b for b in split['main'].body.blocks[0].bindings if b.value.args[0].same_as(pack))
    shape=tuple(int(d) for d in binding.var.ty.shape)
    return tvm.IRModule({'main':split['gemm_prepacked'].with_attr('global_symbol','main'),
                         'pack':split['gemm_weight_prepack'].with_attr('global_symbol','pack')}),shape
