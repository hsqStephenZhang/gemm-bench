"""MimIR tensor → LLVM CPU operators with native timing and PyTorch baseline."""

import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from common.operators import tensor_shapes
from common.runner import run
from mimir.operators import module
from mimir.checkout import prepare
from common.host import shared_suffix, target_flags, is_macos, linker_flags


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class MimIR:
    identity_keys = [
        "mimir",
        "mimir_git",
        "mimir_build",
        "mimir_clang",
        "mimir_target_flags",
    ]

    def __init__(self, config, settings):
        self.settings = settings["mimir"]
        self.bench = config["benchmark"]
        if self.settings["variant"] != "tensor-llvm":
            raise ValueError("Only MimIR tensor-llvm is implemented")
        if self.settings["jobs"] < 1:
            raise ValueError("MimIR build jobs must be positive")
        self.source, self.build, self.checkout_state = prepare(config, self.settings)
        self.binary = self.build / "bin/mim"
        self.plugin_dir = self.build / "lib/mim"
        clangxx = self.settings["clangxx"]
        if (
            is_macos()
            and clangxx == "clang++"
            and Path("/opt/homebrew/opt/llvm/bin/clang++").is_file()
        ):
            clangxx = "/opt/homebrew/opt/llvm/bin/clang++"
        self.clang = shutil.which(clangxx)
        if not self.binary.is_file() or not self.clang:
            raise ValueError(
                "Missing built MimIR or clang++; configure config/mimir.toml"
            )
        required = ["tensor", "core", "math", *self.settings["plugins"]]
        paths = [self.binary]
        for name in dict.fromkeys(required):
            # MimIR MODULE plugins use .so on Darwin too (CMake's MODULE suffix).
            for suffix in [".mim", ".so"]:
                path = self.plugin_dir / (
                    name + suffix if suffix == ".mim" else "libmim_" + name + suffix
                )
                if not path.is_file():
                    raise ValueError(f"Missing MimIR plugin: {path}")
        # Include all built plugins/imports: transitive lowering dependencies affect generated code.
        paths.extend(sorted(self.plugin_dir.glob("*.mim")))
        paths.extend(sorted(self.plugin_dir.glob("libmim_*.so")))
        self.parallel, self.link_flags = self.openmp(self.bench["threads"])
        self.target_flags = []
        for flag in self.settings["target_flags"]:
            if flag == "auto":
                self.target_flags.extend(target_flags(self.bench["mcpu"]))
            elif self.bench["mcpu"] != "generic" or "{mcpu}" not in flag:
                self.target_flags.append(flag.format(mcpu=self.bench["mcpu"]))
        git = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        self.metadata = dict(
            variant=self.settings["variant"],
            mimir=subprocess.check_output(
                [str(self.binary), "--version"], text=True
            ).strip(),
            mimir_git=git.stdout.strip() if git.returncode == 0 else None,
            mimir_checkout=self.checkout_state,
            mimir_source=str(self.source),
            mimir_binary=str(self.binary),
            mimir_build={str(p.relative_to(self.build)): digest(p) for p in paths},
            mimir_clang=subprocess.check_output(
                [self.clang, "--version"], text=True
            ).strip(),
            mimir_target_flags=self.target_flags,
            mimir_parallel=self.parallel,
            reference_pr="https://github.com/hsqStephenZhang/MimIR/pull/1",
        )
        self.artifacts = config.get("artifacts", {})

    def openmp(self, threads):
        """(mode, link flags): MimIR's `omp` plugin runs loop nests on OMP_NUM_THREADS threads."""
        if threads == 1:
            return "none: single-threaded", []
        # Incremental builds keep outputs of plugins a later revision dropped, so ask the
        # checked-out source, not the plugin directory, whether this revision has omp.
        if not (self.source / "src/mim/plug/omp").is_dir():
            return "none: this MimIR revision has no omp plugin", []
        if not (self.plugin_dir / "omp.mim").is_file():
            raise ValueError(
                "This MimIR revision has the omp plugin, but the build did not produce it"
            )
        if not (self.plugin_dir / "rt/omp_rt.ll").is_file():
            raise ValueError(
                "MimIR has the omp plugin but no rt/omp_rt.ll: rebuild it with an OpenMP-capable clang and llvm-link"
            )
        if not is_macos():
            return "omp", ["-fopenmp"]
        # Link PyTorch's libomp so the process holds a single OpenMP runtime (as for OpenBLAS).
        torch_lib = Path(importlib.util.find_spec("torch").origin).parent / "lib"
        self.torch_omp = torch_lib / "libomp.dylib"
        return "omp", ["-fopenmp", f"-L{torch_lib}", f"-Wl,-rpath,{torch_lib}"]

    def relink_torch_omp(self, library):
        """Point the kernel at PyTorch's libomp through @rpath.

        The wheel's libomp carries a build-machine install name (/opt/llvm-openmp/...) that only
        resolves because PyTorch has already loaded it; @rpath makes the dependency explicit.
        """
        install_name = subprocess.check_output(
            ["otool", "-D", str(self.torch_omp)], text=True
        )
        install_name = install_name.strip().splitlines()[-1]
        if install_name != "@rpath/libomp.dylib":
            subprocess.run(
                [
                    "install_name_tool",
                    "-change",
                    install_name,
                    "@rpath/libomp.dylib",
                    str(library),
                ],
                check=True,
            )
            subprocess.run(
                ["codesign", "--force", "--sign", "-", str(library)],
                check=True,
                capture_output=True,
            )

    def prepare(self, case, a, b, work, previous):
        layout = (
            self.settings.get("linear_weight_layout", "kn")
            if case["op"] == "linear"
            else "kn"
        )
        if layout not in ("kn", "nk"):
            raise ValueError("mimir.linear_weight_layout must be kn or nk")
        prepack = (
            dict(prepack_s=0.0, weight_layout="KN (no prepack)")
            if case["op"] == "linear"
            else {}
        )
        if layout == "nk":
            start = time.perf_counter()
            b = np.ascontiguousarray(b.T)
            prepack = dict(
                prepack_s=time.perf_counter() - start,
                weight_layout="NK (transposed once)",
            )
        source = work / "kernel.mim"
        source.write_text(module(case, layout))
        raw = work / "kernel.ll"
        raw.unlink(missing_ok=True)
        omp = ["-p", "omp"] if self.parallel == "omp" else []
        command = [
            str(self.binary),
            "-P",
            str(self.plugin_dir),
            *self.settings["compiler_flags"],
            "-p",
            self.settings["plugins"][0],
            *omp,
            str(source),
        ]
        for plugin in self.settings["plugins"][1:]:
            command.extend(["-p", plugin])
        # Emit the final optimized MimIR program alongside LLVM lowering for inspection.
        command.extend(["--output-mim", str(work / "final.mim")])
        (work / "mimir_command.json").write_text(json.dumps(command, indent=2))
        start = time.perf_counter()
        with (work / "mimir_compile.log").open("w") as log:
            subprocess.run(
                command, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        if not raw.is_file():
            raise RuntimeError(f"MimIR did not emit {raw}; see mimir_compile.log")
        wrapper = Path(__file__).with_name("native_timer.cpp")
        library = work / ("kernel" + shared_suffix())
        command = [
            self.clang,
            *self.settings["compile_flags"],
            *self.target_flags,
            *self.link_flags,
            *linker_flags(),
            str(raw),
            str(wrapper),
            "-o",
            str(library),
        ]
        (work / "clang_command.json").write_text(json.dumps(command, indent=2))
        with (work / "clang_compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if self.parallel == "omp" and is_macos():
            self.relink_torch_omp(library)
        kernel = Kernel(
            library,
            a,
            b,
            tensor_shapes(case)[2],
            work,
            time.perf_counter() - start,
            self.clang,
            self.settings["compile_flags"],
            self.target_flags,
        )
        kernel.details.update(prepack)
        return kernel


class Kernel:
    scope = "native MimIR tensor extern loop; includes output/internal allocation and free; excludes Python/input copies and correctness-result copy"

    def __init__(
        self, library, a, b, shape, work, compile_s, clang, flags, target_flags
    ):
        self.work, self.clang, self.flags, self.target_flags = (
            work,
            clang,
            flags,
            target_flags,
        )
        self.a, self.b = a, b
        self.c = np.empty(shape, dtype="float32")
        self.library = ctypes.CDLL(str(library))
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.library.run_kernel.argtypes = [ptr] * 3 + [ctypes.c_size_t]
        self.library.run_kernel.restype = ctypes.c_int
        self.library.measure.argtypes = [ptr, ptr, ctypes.c_int]
        self.library.measure.restype = ctypes.c_double
        self.details = dict(compile_s=compile_s)

    def run(self):
        if self.library.run_kernel(self.a, self.b, self.c, self.c.size) != 0:
            raise RuntimeError("MimIR output allocation failed")
        return self.c

    def time(self, count):
        seconds = self.library.measure(self.a, self.b, count)
        if seconds <= 0:
            raise RuntimeError("MimIR native measurement failed")
        return seconds

    def measure(self, min_repeat_ms, min_reps=1):
        count = max(min_reps, int(min_repeat_ms / 1000 * 1.2 / self.time(3)))
        seconds = self.time(count)
        while seconds * count < min_repeat_ms / 1000:
            count = max(count + 1, int(min_repeat_ms / 1000 * 1.2 / seconds))
            seconds = self.time(count)
        self.last_iterations = count
        return seconds

    def dump(self, destination):
        for name in [
            "kernel.mim",
            "final.mim",
            "kernel.ll",
            "kernel" + shared_suffix(),
            "mimir_command.json",
            "clang_command.json",
            "mimir_compile.log",
            "clang_compile.log",
        ]:
            shutil.copy2(self.work / name, destination / name)
        shutil.copy2(
            Path(__file__).with_name("native_timer.cpp"),
            destination / "native_timer.cpp",
        )
        flags = [f for f in self.flags if f != "-shared"]
        commands = [
            [
                self.clang,
                *flags,
                *self.target_flags,
                "-S",
                "-emit-llvm",
                str(self.work / "kernel.ll"),
                "-o",
                str(destination / "optimized_kernel.ll"),
            ],
            [
                self.clang,
                *flags,
                *self.target_flags,
                "-S",
                str(self.work / "kernel.ll"),
                "-o",
                str(destination / "kernel.s"),
            ],
        ]
        (destination / "dump_commands.json").write_text(json.dumps(commands, indent=2))
        for command in commands:
            subprocess.run(command, capture_output=True, text=True, check=True)
        return dict(
            notes="MimIR source and final MimIR IR, lowered/Clang-optimized LLVM IR, assembly and the measured shared library. Native timing includes output/internal allocation and free, following the tensor extern ABI in PR #1."
        )


if __name__ == "__main__":
    run("mimir", MimIR)
