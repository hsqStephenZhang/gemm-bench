import ctypes
from pathlib import Path
import subprocess
import shutil
import importlib.util

import numpy as np
from .operators import tensor_shapes
from .host import (
    shared_suffix,
    linker_flags,
    configure_openblas,
    native_compiler,
    is_macos,
)


def compatible_library(library, output):
    """Use the same OpenMP runtime as PyTorch, without modifying installed libs."""
    if not is_macos():
        return library, None
    dependencies = subprocess.check_output(["otool", "-L", library], text=True)
    omp = next(
        (
            line.strip().split(" (", 1)[0]
            for line in dependencies.splitlines()[1:]
            if "libomp.dylib" in line
        ),
        None,
    )
    if not omp:
        return library, None
    spec = importlib.util.find_spec("torch")
    runtime = Path(spec.origin).parent / "lib/libomp.dylib"
    if not runtime.is_file():
        raise ValueError("Cannot find the PyTorch OpenMP runtime for OpenBLAS")
    copied = Path(output).resolve() / "openblas-runtime.dylib"
    shutil.copy2(library, copied)
    subprocess.run(
        [
            "install_name_tool",
            "-change",
            omp,
            str(runtime),
            "-id",
            str(copied),
            str(copied),
        ],
        check=True,
    )
    subprocess.run(
        ["codesign", "--force", "--sign", "-", str(copied)],
        check=True,
        capture_output=True,
    )
    return str(copied), str(runtime)


class OpenBLAS:
    identity_keys = ["openblas", "openblas_core"]

    def __init__(self, config, output):
        settings = config["openblas"]
        configure_openblas(settings)
        library_path, omp_runtime = compatible_library(settings["library"], output)
        self.blas = ctypes.CDLL(library_path)
        self.threads = config["benchmark"]["threads"]
        self.blas.openblas_set_num_threads(self.threads)
        self.blas.openblas_get_config.restype = ctypes.c_char_p
        self.blas.openblas_get_corename.restype = ctypes.c_char_p
        assert self.blas.openblas_get_num_threads() == self.threads
        library = Path(output) / ("openblas_timer" + shared_suffix())
        # Link the timer to the same runtime library that was loaded above.
        link_flags = (
            [flag for flag in settings["link_flags"] if flag != "-lopenblas"]
            if omp_runtime
            else settings["link_flags"]
        )
        command = [
            native_compiler(settings["compiler"]),
            *settings["compile_flags"],
            *linker_flags(),
            str(Path(__file__).with_name("openblas_timer.c")),
            *link_flags,
            *([library_path] if omp_runtime else []),
            "-o",
            str(library),
        ]
        subprocess.run(command, check=True)
        self.library_path = library
        library_handle = ctypes.CDLL(str(library))
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.timers = {}
        for op, symbol, ints, pointers in [
            ("gemm", "measure", 3, 3),
            ("gemv", "measure_gemv", 2, 3),
            ("conv2d", "measure_conv2d", 9, 4),
        ]:
            timer = getattr(library_handle, symbol)
            timer.argtypes = [ctypes.c_int] * ints + [ptr] * pointers + [ctypes.c_int]
            timer.restype = ctypes.c_double
            self.timers[op] = timer
        # OpenBLAS exports no packed-GEMM API: frozen weights use the same sgemm, packing W per call.
        self.timers["linear"] = self.timers["gemm"]
        self.metadata = dict(
            openblas=self.blas.openblas_get_config().decode(),
            openblas_library=library_path,
            openblas_omp_runtime=omp_runtime,
            openblas_core=self.blas.openblas_get_corename().decode(),
            openblas_threads=self.blas.openblas_get_num_threads(),
        )

    def prepare(self, case, a, b, work=None, previous=None):
        return OpenBLASKernel(self.timers[case["op"]], case, a, b, self.library_path)


class OpenBLASKernel:
    scope = "native cblas_sgemm; preallocated output; no Python calls or input copies in timing"

    def __init__(self, timer, case, a, b, library_path):
        self.library_path = library_path
        self.timer, self.shape, self.a, self.b = timer, case["shape"], a, b
        self.output = np.empty(tensor_shapes(case)[2], dtype="float32")
        self.arguments = [*self.shape, a, b, self.output]
        if case["op"] == "conv2d":
            n, ci, h, w, co, kh, kw, stride, pad = self.shape
            self.scratch = np.empty(
                (ci * kh * kw, self.output.shape[2] * self.output.shape[3]),
                dtype="float32",
            )
            self.arguments.append(self.scratch)
            self.scope = "native im2col + cblas_sgemm; includes padding and packing per call; preallocated output/scratch; excludes Python"
        elif case["op"] == "gemv":
            self.scope = (
                "native cblas_sgemv; preallocated output; excludes Python/input copies"
            )
        elif case["op"] == "linear":
            self.scope += (
                "; frozen W [K,N] passed as-is (no packed-GEMM API), packed per call"
            )
            self.details = dict(prepack_s=0.0, weight_layout="KN (no prepack)")

    def dump(self, destination):
        shutil.copy2(self.library_path, destination / self.library_path.name)
        shutil.copy2(
            Path(__file__).with_name("openblas_timer.c"),
            destination / "openblas_timer.c",
        )
        return dict(
            notes="The wrapper invokes the configured prebuilt OpenBLAS library; its build information is in manifest metadata."
        )

    def run(self):
        self.timer(*self.arguments, 1)
        return self.output

    def measure(self, min_repeat_ms, min_reps=1):
        count = max(
            min_reps, int((min_repeat_ms / 1000 * 1.2) / self.timer(*self.arguments, 3))
        )
        seconds = self.timer(*self.arguments, count)
        while seconds * count < min_repeat_ms / 1000:
            count = max(count + 1, int(min_repeat_ms / 1000 * 1.2 / seconds))
            seconds = self.timer(*self.arguments, count)
        self.last_iterations = count
        return seconds
