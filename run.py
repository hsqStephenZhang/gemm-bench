"""Project entry point for CPU operator comparisons against PyTorch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from common.config import ROOT,environment,load
from common.report import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backends','--backend',dest='backends',help='Optional backends: comma-separated tvm,iree,openblas; all or none. Overrides [backends]. PyTorch always runs.')
    parser.add_argument('--common-config',type=Path,default=ROOT/'config/common.toml')
    parser.add_argument('--tvm-config',type=Path,default=ROOT/'config/tvm.toml')
    parser.add_argument('--auto-tune',action=argparse.BooleanOptionalAction,default=None,help='Enable or disable tuning for the TVM backend')
    parser.add_argument('--iree-config',type=Path,default=ROOT/'config/iree.toml')
    parser.add_argument('--pytorch-config',type=Path,default=ROOT/'config/pytorch.toml')
    parser.add_argument('--device',choices=['cpu','gpu'])
    parser.add_argument('--shapes')
    parser.add_argument('--ops',help='gemm,gemv,conv2d or all')
    parser.add_argument('--cases-file',type=Path)
    parser.add_argument('--cases',help='Comma-separated named GEMM cases')
    parser.add_argument('--reps',type=int,help='Minimum calls per timing batch')
    parser.add_argument('--cpu',type=int)
    parser.add_argument('--mcpu')
    parser.add_argument('--trials',type=int)
    parser.add_argument('--output',type=Path,help='Output root; each backend uses a subdirectory')
    parser.add_argument('--dump',action=argparse.BooleanOptionalAction,default=None,help='Export final backend artifacts')
    parser.add_argument('--dump-dir',type=Path,help='Artifact root; each backend uses a subdirectory')
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    if args.dump_dir is not None and args.dump is False:
        parser.error('--no-dump cannot be combined with --dump-dir')
    if args.check and (args.dump or args.dump_dir):
        parser.error('--check does not generate artifacts; omit --check when dumping')
    config_path = args.common_config.resolve()
    config = load(config_path)
    device = args.device or config['benchmark']['device']
    if device!='cpu':
        parser.error('Only CPU benchmarking is implemented; GPU needs a GPU BLAS baseline')
    optional = ['tvm','iree','openblas']
    selection = config.get('backends',dict.fromkeys(optional,True))
    if set(selection)-set(optional) or any(type(value) is not bool for value in selection.values()):
        parser.error('[backends] accepts only boolean tvm, iree and openblas settings; PyTorch is mandatory')
    if args.backends is None:
        selected = [name for name in optional if selection.get(name,False)]
    elif args.backends == 'all':
        selected = optional
    elif args.backends == 'none':
        selected = []
    else:
        requested = args.backends.split(',')
        if any(name not in optional for name in requested) or len(set(requested)) != len(requested):
            parser.error('--backends accepts tvm,iree,openblas (comma-separated), all or none; PyTorch is mandatory')
        selected = [name for name in optional if name in requested]
    if 'tvm' not in selected and (args.auto_tune is not None or args.trials is not None):
        parser.error('--auto-tune/--no-auto-tune/--trials require TVM to be selected')
    backends = ['pytorch',*selected]
    output_root = args.output.resolve() if args.output else ROOT/config['benchmark']['output_root']
    env = environment(config)
    env.update(PYTHONPATH=str(ROOT),GEMM_COMMON_CONFIG=str(config_path),GEMM_PYTORCH_CONFIG=str(args.pytorch_config.resolve()))
    result_files = []
    for backend in backends:
        output = output_root/backend
        command = [sys.executable,str(ROOT/(f'{backend}/run.py' if backend!='openblas' else 'common/benchmark_openblas.py'))]
        if backend=='tvm':
            command.extend(['--config',str(args.tvm_config.resolve())])
        if backend=='pytorch':
            command.extend(['--config',str(args.pytorch_config.resolve())])
        if backend=='iree':
            command.extend(['--config',str(args.iree_config.resolve())])
        if backend!='openblas':
            command.extend(['--common-config',str(config_path)])
            if args.check:
                command.append('--check')
        elif args.check:
            code = 'from common.config import load; from common.openblas import OpenBLAS; import os,tempfile; from pathlib import Path; import json\nwith tempfile.TemporaryDirectory() as d:\n print(json.dumps(OpenBLAS(load(os.environ["GEMM_COMMON_CONFIG"]),Path(d)).metadata,indent=2))'
            command = [sys.executable,'-c',code]
        if not args.check:
            command.extend(['--device',device,'--output',str(output)])
            if args.dump is not None:
                command.append('--dump' if args.dump else '--no-dump')
            if args.dump_dir is not None:
                destination = args.dump_dir.resolve()/backend
                command.extend(['--dump-dir',str(destination)])
            for key in ['shapes','ops','cpu','mcpu','cases','reps']:
                value = getattr(args,key)
                if value is not None:
                    command.extend([f'--{key}',str(value)])
            if args.cases_file is not None:
                command.extend(['--cases-file',str(args.cases_file.resolve())])
            if backend=='tvm' and args.auto_tune is not None:
                command.append('--auto-tune' if args.auto_tune else '--no-auto-tune')
            if backend=='tvm' and args.trials is not None:
                command.extend(['--trials',str(args.trials)])
        print(f'Running {backend} ({device})',flush=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)
        result_files.append(output/'results.json')
    if not args.check:
        summary = output_root/'summary.csv'
        summarize(result_files,summary)


if __name__=='__main__':
    main()
