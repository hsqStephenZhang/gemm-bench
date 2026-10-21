"""Launch the installed apache-tvm package in the project's uv environment."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import environment, load, reject_dump_check
from common.host import exit_on_sigterm, run_child


def main():
    exit_on_sigterm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--common-config", type=Path, default=ROOT / "config/common.toml"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/tvm.toml")
    parser.add_argument(
        "--check", action="store_true", help="Validate libraries without tuning"
    )
    args, forwarded = parser.parse_known_args()
    reject_dump_check(parser, args, forwarded)
    config_path = args.config.resolve()
    load(config_path)
    common_path = args.common_config.resolve()
    env = environment(load(common_path))
    # Resolve TVM and its FFI from site-packages, using the wheel's bundled libraries.
    env.pop("TVM_LIBRARY_PATH", None)
    env.pop("GEMM_TVM_SOURCE", None)
    env.update(
        PYTHONPATH=str(ROOT),
        GEMM_BACKEND_CONFIG=str(config_path),
        GEMM_COMMON_CONFIG=str(common_path),
    )
    if args.check:
        code = """import json,os,tvm,tvm_ffi
from importlib.metadata import version
from tvm.script import tirx
from tvm.s_tir import meta_schedule
from common.config import ROOT,load
from pytorch.benchmark import configure,metadata
threads=load(os.environ['GEMM_COMMON_CONFIG'])['benchmark']['threads']
settings=load(os.environ.get('GEMM_PYTORCH_CONFIG',ROOT/'config/pytorch.toml'))['pytorch']
configure(settings,threads)
assert tvm.runtime.enabled('llvm'), 'TVM compiler or LLVM failed to load'
assert tvm.runtime.num_threads()==threads, f'TVM uses {tvm.runtime.num_threads()} threads, expected {threads}'
print(json.dumps(dict(tvm=tvm.__version__,distribution=version('apache-tvm'),package_path=tvm.__file__,llvm=True,tvm_threads=tvm.runtime.num_threads(),baseline_backend='pytorch',baseline=metadata(settings),ffi_library=tvm_ffi.libinfo.find_libtvm_ffi()),indent=2))
"""
        command = [sys.executable, "-c", code]
    else:
        command = [sys.executable, str(ROOT / "tvm/benchmark.py"), *forwarded]
    return run_child(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
