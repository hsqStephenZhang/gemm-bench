import argparse
import os
from pathlib import Path
import tomllib
from .cases import select_cases

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return tomllib.loads(Path(path).resolve().read_text())


def environment(config):
    env = os.environ.copy()
    env.update(config['threads'])
    return env


def parse_args(backend):
    config = load(os.environ.get('GEMM_COMMON_CONFIG', ROOT/'config/common.toml'))
    settings = load(os.environ.get('GEMM_BACKEND_CONFIG', ROOT/f'config/{backend}.toml')) if backend != 'openblas' else {'openblas':config['openblas'].copy()}
    if backend=='mimir':
        for key in ['remote','ref']:
            if 'GEMM_MIMIR_'+key.upper() in os.environ:settings['mimir'][key]=os.environ['GEMM_MIMIR_'+key.upper()]
    bench = config['benchmark'].copy()
    parser = argparse.ArgumentParser(description=f'{backend} FP32 CPU operator benchmark')
    parser.add_argument('--build-dir',type=Path,default=Path(os.environ.get('GEMM_BUILD_DIR',config.get('build',{}).get('directory','build'))))
    parser.add_argument('--device', choices=['cpu','gpu'], default=bench['device'])
    parser.add_argument('--cpu', type=int, default=bench['cpu'])
    parser.add_argument('--mcpu', default=bench['mcpu'])
    parser.add_argument('--shapes',help='Override with M,N,K triples (legacy CLI order)')
    parser.add_argument('--cases-file',type=Path,help='Named GEMM/GEMV/Conv2D case table')
    parser.add_argument('--ops',help='gemm,gemv,conv2d or all; overrides benchmark.ops')
    parser.add_argument('--cases',help='Comma-separated case names')
    parser.add_argument('--reps',type=int,help='Override minimum native calls per timing batch')
    parser.add_argument('--output', type=Path, default=ROOT/bench['output_root']/backend)
    parser.add_argument('--dump', action=argparse.BooleanOptionalAction, default=None, help='Export final artifacts outside the timed region')
    parser.add_argument('--dump-dir', type=Path, help='Artifact directory; implies --dump')
    if backend == 'tvm':
        parser.add_argument('--auto-tune',action=argparse.BooleanOptionalAction,default=None,help='Enable or disable TVM MetaSchedule auto tuning')
        parser.add_argument('--trials', type=int, default=settings['tvm']['auto_tune']['trials'])
    args = parser.parse_args()
    config['build']={'directory':str(args.build_dir.resolve())}
    if args.device != 'cpu':
        parser.error('This benchmark currently implements CPU only; GPU requires GPU kernels and a GPU BLAS baseline')
    if bench['dtype'] != 'float32':
        parser.error('Only float32 operators are implemented')
    try:
        cases = select_cases(bench,args.shapes,args.cases_file,args.cases,args.reps,ROOT,args.ops)
    except ValueError as exc:
        parser.error(str(exc))
    bench.update(device=args.device,cpu=args.cpu,mcpu=args.mcpu,
                 shapes=[case['shape'] for case in cases],cases=cases,
                 reps_policy='minimum native calls per batch; adaptive timing may run more')
    if min(bench['repeats'],bench['min_repeat_ms']) < 1 or bench['warmup'] < 0:
        parser.error('Invalid measurement configuration')
    if backend == 'tvm':
        if args.trials < 1:
            parser.error('--trials must be positive')
        settings['tvm']['auto_tune']['trials'] = args.trials
        settings['tvm']['auto_tune']['enabled'] = args.auto_tune if args.auto_tune is not None else settings['tvm']['auto_tune'].get('enabled',False)
    artifacts = config.get('artifacts', {}).copy()
    if args.dump_dir is not None and args.dump is False:
        parser.error('--no-dump cannot be combined with --dump-dir')
    artifacts['enabled'] = args.dump if args.dump is not None else bool(args.dump_dir or artifacts.get('enabled', False))
    artifacts['directory'] = str(args.dump_dir.resolve() if args.dump_dir else args.output.resolve()/artifacts.get('subdirectory', 'artifacts'))
    config['artifacts'] = artifacts
    config['benchmark'] = bench
    os.sched_setaffinity(0,{args.cpu})
    return args,config,settings,cases


def reject_dump_check(parser, args, forwarded):
    if args.check and any(arg in ['--dump','--dump-dir'] or arg.startswith('--dump-dir=') for arg in forwarded):
        parser.error('--check does not generate artifacts; omit --check when dumping')
