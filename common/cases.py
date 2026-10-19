"""Named CPU cases: GEMM/linear M K N, GEMV M K, Conv2D N Cin H W Cout KH KW stride pad."""
from pathlib import Path
from .operators import tensor_shapes

OPS = ('gemm','gemv','conv2d','linear')


def load_cases(path):
    cases,names = [],set()
    for number,line in enumerate(Path(path).read_text().splitlines(),1):
        fields = line.split('#',1)[0].split()
        if not fields:
            continue
        op = 'conv2d' if fields[0]=='conv' else fields[0]
        count = {'gemm':6,'linear':6,'gemv':5,'conv2d':12}.get(op)
        if count is None or len(fields)!=count:
            raise ValueError(f'{path}:{number}: expected gemm/linear name M K N reps, gemv name M K reps, or conv2d name N Cin H W Cout KH KW stride pad reps')
        name = fields[1]
        try:
            values = list(map(int,fields[2:]))
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: dimensions/reps must be integers') from exc
        dims,reps = values[:-1],values[-1]
        if reps<1 or min(dims[:-1] if op=='conv2d' else dims)<1 or (op=='conv2d' and dims[-1]<0) or name in names:
            raise ValueError(f'{path}:{number}: positive dimensions/reps, nonnegative pad and unique names required')
        if op in ('gemm','linear'):
            m,k,n=dims;dims=[m,n,k]
        case=dict(op=op,name=name,shape=dims,reps=reps)
        if min(tensor_shapes(case)[2])<1:
            raise ValueError(f'{path}:{number}: convolution output must be positive')
        cases.append(case);names.add(name)
    if not cases:
        raise ValueError(f'No cases in {path}')
    return cases


def select_cases(bench, shapes=None, case_file=None, names=None, reps=None, root=Path('.'), ops=None):
    requested_ops = ops.split(',') if ops else bench.get('ops',list(OPS))
    if requested_ops==['all']:
        requested_ops=list(OPS)
    if not requested_ops or len(set(requested_ops))!=len(requested_ops) or any(op not in OPS for op in requested_ops):
        raise ValueError('--ops accepts gemm,gemv,conv2d,linear (comma-separated) or all')
    if shapes and (names or case_file):
        raise ValueError('--shapes cannot be combined with --cases or --cases-file')
    if shapes:
        tuples = [tuple(map(int,s.split(','))) for s in shapes.split(';')]
        if 'gemm' not in requested_ops or any(len(s)!=3 or min(s)<1 for s in tuples):
            raise ValueError('--shapes requires GEMM and positive M,N,K triples separated by semicolons')
        cases=[dict(op='gemm',name=f'm{m}_k{k}_n{n}',shape=[m,n,k],reps=bench.get('default_reps',1)) for m,n,k in tuples]
    else:
        paths = [case_file] if case_file else bench.get('cases_files',[bench.get('cases_file','config/gemm_cases.txt')])
        cases=[]
        for path in paths:
            path=Path(path);cases.extend(load_cases(path if path.is_absolute() else root/path))
        if len({c['name'] for c in cases})!=len(cases):
            raise ValueError('Case names must be unique across case files')
        cases=[c for c in cases if c['op'] in requested_ops]
    if names:
        requested=names.split(',');available={c['name']:c for c in cases}
        if len(set(requested))!=len(requested) or any(name not in available for name in requested):
            raise ValueError('--cases requires unique names present in the selected operators/case files')
        cases=[available[name].copy() for name in requested]
    if not cases:
        raise ValueError('No cases match the selected operators')
    if reps is not None:
        if reps<1:
            raise ValueError('--reps must be positive')
        cases=[dict(case,reps=reps) for case in cases]
    return cases
