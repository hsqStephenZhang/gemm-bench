"""AOTInductor CPU graphs measured through a native package-loader loop."""

import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import time
import weakref
import zipfile
import numpy as np
import torch
from common.host import linker_flags, native_compiler, shared_suffix
from common.operators import tensor_shapes
from pytorch.benchmark import Kernel as EagerKernel

_COMPILED_WRAPPERS = set()


class Operator(torch.nn.Module):
    def __init__(self, case):
        super().__init__()
        self.op = case["op"]
        self.stride = case["shape"][7] if self.op == "conv2d" else 1
        self.pad = case["shape"][8] if self.op == "conv2d" else 0

    def forward(self, a, b):
        # linear runs as GEMM with the weight passed per call (no freezing), as in eager mode.
        if self.op in ("gemm", "linear"):
            return torch.mm(a, b)
        if self.op == "gemv":
            return torch.mv(a, b)
        return torch.nn.functional.conv2d(a, b, stride=self.stride, padding=self.pad)


class Kernel(EagerKernel):
    scope = "native AOTInductor package-loader loop; includes graph invocation, output allocation and internal packing; excludes Python, compilation and input copies"

    def __init__(self, settings, case, a, b, work, artifacts, threads=1):
        self.output, self.artifacts, self.case = work, artifacts, case
        self.a, self.b = a, b
        self.c = np.empty(tensor_shapes(case)[2], dtype="float32")
        self.package = work / "model.pt2"
        options = settings.get("inductor", {})
        configs = {
            "compile_threads": options.get("compile_threads", 1),
            "cpp.threads": threads,  # benchmark.threads, as for every backend
            "max_autotune": options.get("max_autotune", False),
        }
        start = time.perf_counter()
        with torch.inference_mode():
            model = Operator(case).eval()
            x, y = torch.from_numpy(a), torch.from_numpy(b)
            exported = torch.export.export(model, (x, y))
            (work / "exported_graph.txt").write_text(str(exported.graph_module.code))
            torch.export.save(exported, work / "exported_program.pt2")
            torch._inductor.aoti_compile_and_package(
                exported, package_path=str(self.package), inductor_configs=configs
            )
        compile_s = time.perf_counter() - start
        (work / "inductor_config.json").write_text(json.dumps(configs, indent=2))
        torch_root = Path(torch.__file__).resolve().parent
        # Same toolchain resolution as the eager timer (macOS SDK, .dylib), via common.host.
        wrapper = work.parent / ("inductor_timer" + shared_suffix())
        command = [
            native_compiler(settings["compiler"]),
            *settings["compile_flags"],
            *linker_flags(),
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
            "-I",
            str(torch_root / "include"),
            "-I",
            str(torch_root / "include/torch/csrc/api/include"),
            str(Path(__file__).with_name("inductor_timer.cpp")),
            "-L",
            str(torch_root / "lib"),
            "-ltorch_cpu",
            "-lc10",
            f"-Wl,-rpath,{torch_root / 'lib'}",
            "-o",
            str(wrapper),
        ]
        if wrapper not in _COMPILED_WRAPPERS:
            (work.parent / "inductor_timer_command.json").write_text(
                json.dumps(command, indent=2)
            )
            with (work.parent / "inductor_timer_compile.log").open("w") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            _COMPILED_WRAPPERS.add(wrapper)
        self.wrapper = wrapper
        self.library = ctypes.CDLL(str(wrapper))
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.library.create_kernel.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            ptr,
            ptr,
            ptr,
        ]
        self.library.create_kernel.restype = ctypes.c_void_p
        self.library.destroy_kernel.argtypes = [ctypes.c_void_p]
        self.library.destroy_kernel.restype = None
        self.library.measure.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.measure.restype = ctypes.c_double
        self.library.last_error.restype = ctypes.c_char_p
        self.handle = self.library.create_kernel(
            str(self.package).encode(),
            {"gemm": 0, "linear": 0, "gemv": 1, "conv2d": 2}[case["op"]],
            np.array(case["shape"], dtype="int32"),
            a,
            b,
            self.c,
        )
        if not self.handle:
            raise RuntimeError(self.library.last_error().decode())
        self.cleanup = weakref.finalize(self, self.library.destroy_kernel, self.handle)
        self.details = dict(
            compile_s=compile_s,
            pytorch_mode="inductor",
            inductor_runtime="aoti-native",
            inductor_config=configs,
        )

    def dump(self, destination):
        for name in [
            "model.pt2",
            "exported_program.pt2",
            "exported_graph.txt",
            "inductor_config.json",
        ]:
            shutil.copy2(self.output / name, destination / name)
        shutil.copy2(self.wrapper, destination / "inductor_timer.so")
        shutil.copy2(
            Path(__file__).with_name("inductor_timer.cpp"),
            destination / "inductor_timer.cpp",
        )
        for name in ["inductor_timer_command.json", "inductor_timer_compile.log"]:
            shutil.copy2(self.output.parent / name, destination / name)
        # Retain exact generated sources and binaries from the measured package.
        with zipfile.ZipFile(self.package) as package:
            for item in package.infolist():
                path = Path(item.filename)
                if path.suffix not in [".cpp", ".h", ".so", ".json"]:
                    continue
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError("Unsafe package member")
                target = destination / "package" / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(package.read(item))
        with (destination / "timer_wrapper.s").open("w") as out:
            subprocess.run(
                [
                    self.artifacts.get("objdump", "objdump"),
                    "-d",
                    "--disassemble=measure",
                    str(self.wrapper),
                ],
                stdout=out,
                check=True,
            )
        (destination / "torch_build.txt").write_text(torch.__config__.show())
        return dict(
            notes="AOTInductor compiled .pt2 package, generated C++/shared libraries and exact measured native loader wrapper. Eager fallback is not used on compilation errors."
        )
