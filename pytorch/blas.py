"""Optional macOS BLAS replacement for PyTorch: Accelerate (AMX) -> OpenBLAS (NEON).

The macOS wheel links Accelerate for sgemm_/sgemv_, which uses the AMX coprocessor.
For CPU-core comparisons the runner re-executes itself with a dyld interposer that
routes those imports to OpenBLAS; ATen dispatch and kernels are otherwise unchanged.
"""

import copy
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from common.host import configure_openblas, is_macos, native_compiler

SYMBOLS = ["sgemm_", "sgemv_", "saxpy_", "sscal_", "scopy_", "cblas_sdot"]
SOURCE = Path(__file__).with_name("blas_interpose.c")
REEXEC = "GEMM_PYTORCH_BLAS_REEXEC"


def mode(settings):
    value = settings.get("macos_blas", "accelerate")
    if value not in ("accelerate", "openblas"):
        raise ValueError("pytorch.macos_blas must be accelerate or openblas")
    return value if is_macos() else "bundled"


def interposer():
    try:
        library = ctypes.CDLL(None)
        library.gemm_bench_blas_routed.restype = ctypes.c_long
        library.gemm_bench_blas_library.restype = ctypes.c_char_p
        return library
    except AttributeError:
        return None


def build(config):
    from common.openblas import compatible_library

    settings = copy.deepcopy(config["openblas"])
    configure_openblas(settings)
    library = Path(settings["library"]).resolve()
    stat = library.stat()
    key = hashlib.sha256(
        SOURCE.read_bytes() + f"{library}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    output = Path(config["build"]["directory"]) / "pytorch-blas" / key
    target = output / "libgemm_bench_blas.dylib"
    # Content-keyed and published atomically: concurrent runners share one build.
    if target.is_file():
        return target
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=key + ".", dir=output.parent))
    # Same OpenMP runtime as PyTorch, as for the OpenBLAS backend; rename keeps the embedded path valid.
    blas, _ = compatible_library(str(library), staging)
    final_blas = output / Path(blas).name
    subprocess.run(
        ["install_name_tool", "-id", str(final_blas), blas],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["codesign", "--force", "--sign", "-", blas], check=True, capture_output=True
    )
    subprocess.run(
        [
            native_compiler("cc"),
            "-O2",
            "-dynamiclib",
            f'-DOPENBLAS_PATH="{final_blas}"',
            str(SOURCE),
            "-framework",
            "Accelerate",
            "-o",
            str(staging / target.name),
        ],
        check=True,
    )
    try:
        staging.rename(output)
    except OSError:
        shutil.rmtree(staging)
        if not target.is_file():
            raise
    return target


def activate(config, settings):
    """Re-exec once under the interposer when requested; returns metadata."""
    selected = mode(settings)
    if selected != "openblas":
        return dict(pytorch_blas=selected)
    library = interposer()
    if library is None:
        if os.environ.get(REEXEC):
            raise RuntimeError(
                "BLAS interposer did not load (DYLD_INSERT_LIBRARIES stripped?)"
            )
        env = dict(
            os.environ, DYLD_INSERT_LIBRARIES=str(build(config)), **{REEXEC: "1"}
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    # Children (compilers, TVM builders, IREE tools) must not inherit the interposer.
    for key in ["DYLD_INSERT_LIBRARIES", REEXEC]:
        os.environ.pop(key, None)
    path = library.gemm_bench_blas_library().decode()
    blas = ctypes.CDLL(path)
    blas.openblas_get_config.restype = ctypes.c_char_p
    blas.openblas_get_corename.restype = ctypes.c_char_p
    blas.openblas_set_num_threads(config["benchmark"]["threads"])
    return dict(
        pytorch_blas="openblas-interposed",
        pytorch_blas_library=path,
        pytorch_blas_symbols=SYMBOLS,
        pytorch_blas_config=blas.openblas_get_config().decode(),
        pytorch_blas_core=blas.openblas_get_corename().decode(),
    )


def routed_calls():
    library = interposer()
    return library.gemm_bench_blas_routed() if library else None
