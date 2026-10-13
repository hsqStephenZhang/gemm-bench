"""Launch IREE with the project-wide configuration and thread environment."""

import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import environment, load, reject_dump_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-config", type=Path, default=ROOT / "config/common.toml"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/iree.toml")
    parser.add_argument("--check", action="store_true")
    args, forwarded = parser.parse_known_args()
    reject_dump_check(parser, args, forwarded)
    common_path, config_path = args.common_config.resolve(), args.config.resolve()
    common = load(common_path)
    env = environment(common)
    env.update(
        PYTHONPATH=str(ROOT),
        GEMM_COMMON_CONFIG=str(common_path),
        GEMM_BACKEND_CONFIG=str(config_path),
    )
    if args.check:
        code = """import importlib.metadata,json,os,iree.runtime as r
from common.config import ROOT,load
from pytorch.benchmark import configure,metadata
from iree.compiler.tools.binaries import find_tool
from iree.runtime.benchmark import benchmark_exe
settings=load(os.environ['GEMM_BACKEND_CONFIG'])['iree']
configure(load(os.environ.get('GEMM_PYTORCH_CONFIG',ROOT/'config/pytorch.toml'))['pytorch'])
r.Config(settings['driver'])
print(json.dumps(dict(compiler=importlib.metadata.version('iree-base-compiler'),runtime=importlib.metadata.version('iree-base-runtime'),compiler_tool=find_tool('iree-compile'),benchmark_tool=benchmark_exe(),driver=settings['driver'],baseline_backend='pytorch'),indent=2))
"""
        command = [sys.executable, "-c", code]
    else:
        command = [sys.executable, str(ROOT / "iree/benchmark.py"), *forwarded]
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())
