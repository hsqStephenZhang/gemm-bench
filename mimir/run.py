"""Launch MimIR with shared CPU/thread configuration and PyTorch baseline."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import environment, load, reject_dump_check
from common.host import available_cpus, exit_on_sigterm, run_child


def main():
    exit_on_sigterm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-config", type=Path, default=ROOT / "config/common.toml"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/mimir.toml")
    parser.add_argument("--remote", help="MimIR remote; overrides mimir.remote")
    parser.add_argument(
        "--ref", help="Branch, tag, commit, or pull ref; overrides mimir.ref"
    )
    parser.add_argument("--build-dir", type=Path, help="Shared persistent build root")
    parser.add_argument("--check", action="store_true")
    args, forwarded = parser.parse_known_args()
    reject_dump_check(parser, args, forwarded)
    common_path, config_path = args.common_config.resolve(), args.config.resolve()
    env = environment(load(common_path))
    # Capture allowed CPUs before the shared runner pins kernel measurements.
    env["GEMM_MIMIR_BUILD_CPUS"] = ",".join(map(str, available_cpus()))
    if args.build_dir is not None:
        env["GEMM_BUILD_DIR"] = str(args.build_dir.resolve())
    if args.remote is not None:
        env["GEMM_MIMIR_REMOTE"] = args.remote
    if args.ref is not None:
        env["GEMM_MIMIR_REF"] = args.ref
    env.update(
        PYTHONPATH=str(ROOT),
        GEMM_COMMON_CONFIG=str(common_path),
        GEMM_BACKEND_CONFIG=str(config_path),
    )
    if args.check:
        code = """import json,os
from common.config import ROOT,load
from mimir.benchmark import MimIR
from pytorch.benchmark import configure,metadata
common=load(os.environ['GEMM_COMMON_CONFIG'])
settings=load(os.environ.get('GEMM_PYTORCH_CONFIG',ROOT/'config/pytorch.toml'))['pytorch']
configure(settings,common['benchmark']['threads'])
backend=load(os.environ['GEMM_BACKEND_CONFIG'])
if 'GEMM_MIMIR_REMOTE' in os.environ:backend['mimir']['remote']=os.environ['GEMM_MIMIR_REMOTE']
if 'GEMM_MIMIR_REF' in os.environ:backend['mimir']['ref']=os.environ['GEMM_MIMIR_REF']
adapter=MimIR(common,backend)
print(json.dumps(dict(**adapter.metadata,baseline_backend='pytorch',baseline=metadata(settings)),indent=2))
"""
        command = [sys.executable, "-c", code]
    else:
        command = [sys.executable, str(ROOT / "mimir/benchmark.py"), *forwarded]
    return run_child(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
