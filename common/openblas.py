import ctypes
from pathlib import Path
import subprocess
import shutil

import numpy as np
from .operators import tensor_shapes


class OpenBLAS:
    identity_keys = ["openblas", "openblas_core"]

    def __init__(self, config, output):
        settings = config["openblas"]
        self.blas = ctypes.CDLL(settings["library"])
        self.blas.openblas_set_num_threads(1)
        self.blas.openblas_get_config.restype = ctypes.c_char_p
        self.blas.openblas_get_corename.restype = ctypes.c_char_p
        assert self.blas.openblas_get_num_threads() == 1
        library = Path(output) / "openblas_timer.so"
        command = [
            settings["compiler"],
            *settings["compile_flags"],
            str(Path(__file__).with_name("openblas_timer.c")),
            *settings["link_flags"],
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
        self.metadata = dict(
            openblas=self.blas.openblas_get_config().decode(),
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

    def dump(self, destination):
        shutil.copy2(self.library_path, destination / "openblas_timer.so")
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
