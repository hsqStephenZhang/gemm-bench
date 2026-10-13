"""PyTorch native eager GEMM/GEMV/Conv2D, the shared baseline for every backend."""

import ctypes
from pathlib import Path
import json
import subprocess
import shutil
import time
import weakref

import numpy as np
import torch

from common.runner import run
from common.operators import tensor_shapes


def configure(settings):
    if settings["intraop_threads"] != 1 or settings["interop_threads"] != 1:
        raise ValueError("The CPU comparison requires one intra-op and inter-op thread")
    torch.set_num_threads(settings["intraop_threads"])
    if torch.get_num_interop_threads() != settings["interop_threads"]:
        torch.set_num_interop_threads(settings["interop_threads"])
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1


def metadata(settings):
    return dict(
        pytorch=torch.__version__,
        torch_build=torch.__config__.show(),
        pytorch_threads=torch.get_num_threads(),
        pytorch_interop_threads=torch.get_num_interop_threads(),
        pytorch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
        variant="inductor-aoti"
        if settings.get("mode", "eager") == "inductor"
        else settings["variant"],
        mode=settings.get("mode", "eager"),
    )


class PyTorch:
    identity_keys = ["pytorch", "torch_build"]

    def __init__(self, config, settings):
        self.settings = settings["pytorch"]
        self.mode = self.settings.get("mode", "eager")
        if self.mode not in ["eager", "inductor"]:
            raise ValueError("pytorch.mode must be eager or inductor")
        if self.settings["variant"] not in ["eager-mm-out", "eager-native"]:
            raise ValueError("Only eager-native is implemented")
        configure(self.settings)
        self.metadata = metadata(self.settings)
        self.library = None
        self.artifacts = config.get("artifacts", {})

    def build_timer(self, output):
        torch_root = Path(torch.__file__).resolve().parent
        include, lib = torch_root / "include", torch_root / "lib"
        target = output / "pytorch_timer.so"
        command = [
            self.settings["compiler"],
            *self.settings["compile_flags"],
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
            "-I",
            str(include),
            "-I",
            str(include / "torch/csrc/api/include"),
            str(Path(__file__).with_name("native_timer.cpp")),
            "-L",
            str(lib),
            "-ltorch_cpu",
            "-lc10",
            f"-Wl,-rpath,{lib}",
            "-o",
            str(target),
        ]
        (output / "timer_compile_command.json").write_text(
            json.dumps(command, indent=2)
        )
        start = time.perf_counter()
        with (output / "timer_compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        self.build_s = time.perf_counter() - start
        self.timer_output = output
        self.library = ctypes.CDLL(str(target))
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.library.create_kernel.argtypes = [
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
        ] + [ptr] * 3
        self.library.create_kernel.restype = ctypes.c_void_p
        self.library.destroy_kernel.argtypes = [ctypes.c_void_p]
        self.library.destroy_kernel.restype = None
        self.library.measure.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.measure.restype = ctypes.c_double
        self.library.last_error.restype = ctypes.c_char_p

    def prepare(self, case, a, b, work, previous):
        if self.library is None and self.mode == "eager":
            self.build_timer(work.parent)
        if self.mode == "inductor":
            from pytorch.inductor import Kernel as InductorKernel

            return InductorKernel(self.settings, case, a, b, work, self.artifacts)
        kernel = Kernel(
            self.library, case, a, b, self.build_s, self.timer_output, self.artifacts
        )
        # Confirm that the native measured operator agrees with the Python API.
        with torch.inference_mode():
            x, y = torch.from_numpy(a), torch.from_numpy(b)
            if case["op"] == "gemm":
                expected = torch.mm(x, y).numpy()
            elif case["op"] == "gemv":
                expected = torch.mv(x, y).numpy()
            else:
                expected = torch.nn.functional.conv2d(
                    x, y, stride=case["shape"][7], padding=case["shape"][8]
                ).numpy()
        np.testing.assert_array_equal(kernel.run(), expected)
        return kernel


class Kernel:
    scope = "native ATen mm.out loop; preallocated output; includes dispatcher/BLAS; excludes Python/input copies"

    def __init__(self, library, case, a, b, build_s, output, artifacts):
        self.output, self.artifacts = output, artifacts
        self.library, self.a, self.b = library, a, b
        self.case = case
        self.operator = {
            "gemm": "aten::mm.out",
            "gemv": "aten::mv.out",
            "conv2d": "aten::convolution",
        }[case["op"]]
        self.scope = (
            "native "
            + self.operator
            + " loop; includes dispatcher and computation; "
            + (
                "includes output allocation and internal packing"
                if case["op"] == "conv2d"
                else "preallocated output"
            )
            + "; excludes Python/input copies"
        )
        self.c = np.empty(tensor_shapes(case)[2], dtype="float32")
        dims = np.array(case["shape"], dtype="int32")
        self.handle = library.create_kernel(
            {"gemm": 0, "gemv": 1, "conv2d": 2}[case["op"]], dims, a, b, self.c
        )
        if not self.handle:
            raise RuntimeError(library.last_error().decode())
        self.cleanup = weakref.finalize(self, library.destroy_kernel, self.handle)
        self.details = dict(timer_build_s=build_s)

    def dump(self, destination):
        for name in [
            "pytorch_timer.so",
            "timer_compile_command.json",
            "timer_compile.log",
        ]:
            shutil.copy2(self.output / name, destination / name)
        shutil.copy2(
            Path(__file__).with_name("native_timer.cpp"),
            destination / "native_timer.cpp",
        )
        (destination / "operator.txt").write_text(
            torch._C._dispatch_dump(self.operator)
        )
        (destination / "dispatch_table.txt").write_text(
            torch._C._dispatch_dump_table(self.operator)
        )
        (destination / "torch_build.txt").write_text(torch.__config__.show())
        libdir = Path(torch.__file__).resolve().parent / "lib"
        libraries = [
            dict(path=str(p), bytes=p.stat().st_size)
            for p in sorted(libdir.glob("*.so*"))
            if p.is_file()
        ]
        (destination / "torch_libraries.json").write_text(
            json.dumps(libraries, indent=2)
        )
        with (destination / "timer_wrapper.s").open("w") as out:
            subprocess.run(
                [
                    self.artifacts.get("objdump", "objdump"),
                    "-d",
                    "--disassemble=measure",
                    str(self.output / "pytorch_timer.so"),
                ],
                stdout=out,
                check=True,
            )
        return dict(
            notes="Eager ATen operators use prebuilt PyTorch libraries. timer_wrapper.s disassembles the measured wrapper, not a newly compiled GEMM kernel; torch_libraries.json references the actual runtime libraries."
        )

    def time(self, count):
        seconds = self.library.measure(self.handle, count)
        if seconds <= 0:
            raise RuntimeError(self.library.last_error().decode())
        return seconds

    def run(self):
        self.time(1)
        return self.c

    def measure(self, min_repeat_ms, min_reps=1):
        count = max(min_reps, int(min_repeat_ms / 1000 * 1.2 / self.time(3)))
        seconds = self.time(count)
        while seconds * count < min_repeat_ms / 1000:
            count = max(count + 1, int(min_repeat_ms / 1000 * 1.2 / seconds))
            seconds = self.time(count)
        self.last_iterations = count
        return seconds


if __name__ == "__main__":
    run("pytorch", PyTorch)
