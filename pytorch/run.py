"""Launch PyTorch with the shared CPU and thread configuration."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.config import environment,load,reject_dump_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-config',type=Path,default=ROOT/'config/common.toml')
    parser.add_argument('--config',type=Path,default=ROOT/'config/pytorch.toml')
    parser.add_argument('--check',action='store_true')
    args,forwarded = parser.parse_known_args()
    reject_dump_check(parser,args,forwarded)
    common_path,config_path = args.common_config.resolve(),args.config.resolve()
    env = environment(load(common_path))
    env.update(PYTHONPATH=str(ROOT),GEMM_COMMON_CONFIG=str(common_path),GEMM_BACKEND_CONFIG=str(config_path))
    if args.check:
        code = '''import json,os,shutil,tempfile
from pathlib import Path
from common.config import load
from pytorch.benchmark import configure,metadata
settings=load(os.environ['GEMM_BACKEND_CONFIG'])['pytorch']
assert shutil.which(settings['compiler']), 'Missing C++ compiler'
configure(settings)
print(json.dumps(metadata(settings),indent=2))
'''
        command = [sys.executable,'-c',code]
    else:
        command = [sys.executable,str(ROOT/'pytorch/benchmark.py'),*forwarded]
    return subprocess.call(command,cwd=ROOT,env=env)


if __name__=='__main__':
    sys.exit(main())
