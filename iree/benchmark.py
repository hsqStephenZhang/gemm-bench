"""IREE llvm-cpu adapter using native module timing and shared OpenBLAS inputs."""

import importlib.metadata
import json
from pathlib import Path
import subprocess
import shutil
import time

import iree.runtime as runtime
from iree.compiler.tools.binaries import find_tool
from iree.runtime.benchmark import benchmark_exe

from common.runner import run
from common.host import iree_host_flags


from common.iree_ops import module as operator_module


class IREE:
    identity_keys = ["iree_compiler", "iree_runtime"]

    def __init__(self, config, settings):
        self.bench, self.settings = config["benchmark"], settings["iree"]
        self.settings["compiler_flags"] = [
            *iree_host_flags(self.settings["compiler_flags"]),
            *self.settings["compiler_flags"],
        ]
        self.artifacts = config.get("artifacts", {})
        if (
            self.settings["driver"] != "local-sync"
            or self.settings["backend"] != "llvm-cpu"
        ):
            raise ValueError("The CPU comparison requires llvm-cpu + local-sync")
        threads = self.bench["threads"]
        if threads > 1:
            # Multi-threaded: the task executor with one worker group per benchmark thread.
            self.settings["driver"] = "local-task"
            self.settings["benchmark_flags"] = [
                *self.settings["benchmark_flags"],
                f"--task_topology_group_count={threads}",
            ]
        self.metadata = dict(
            iree_compiler=importlib.metadata.version("iree-base-compiler"),
            iree_runtime=importlib.metadata.version("iree-base-runtime"),
            driver=self.settings["driver"],
            target=self.settings["backend"],
            variant=self.settings.get("variant", "default"),
        )

    def prepare(self, case, a, b, work, previous):
        source = work / "matmul.mlir"
        source.write_text(operator_module(case))
        module = work / "matmul.vmfb"
        command = [
            find_tool("iree-compile"),
            str(source),
            f"--iree-hal-target-backends={self.settings['backend']}",
            f"--iree-llvmcpu-target-cpu={'host' if self.bench['mcpu'] == 'native' else self.bench['mcpu']}",
            *self.settings["compiler_flags"],
            "-o",
            str(module),
        ]
        if self.settings["dump_ir"]:
            dump = work / "ir"
            dump.mkdir(exist_ok=True)
            command.extend(
                [
                    f"--iree-hal-dump-executable-configurations-to={dump}",
                    f"--iree-hal-dump-executable-sources-to={dump}",
                ]
            )
        if self.artifacts.get("enabled"):
            for directory in ["executables", "phases"]:
                (work / directory).mkdir(exist_ok=True)
            command.extend(
                [
                    f"--iree-hal-dump-executable-files-to={work / 'executables'}",
                    f"--dump-compilation-phases-to={work / 'phases'}",
                ]
            )
        (work / "compile_command.json").write_text(json.dumps(command, indent=2))
        start = time.perf_counter()
        with (work / "compile.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        compile_s = time.perf_counter() - start
        parameters = None
        if case["op"] == "linear":
            parameters = work / "weights.irpa"
            index = runtime.ParameterIndex()
            index.add_buffer("w", b.tobytes())
            index.create_archive_file(str(parameters))
        return Kernel(module, a, b, self.settings, work, compile_s, parameters)


class Kernel:
    scope = "native IREE module invocation; includes VM, dispatch and output allocation; excludes Python/input loading"

    def __init__(self, module, a, b, settings, work, compile_s, parameters=None):
        self.module, self.a, self.b, self.settings, self.work, self.parameters = (
            module,
            a,
            b,
            settings,
            work,
            parameters,
        )
        self.details = dict(compile_s=compile_s)
        self.batch = 0
        if parameters is None:
            self.bound = runtime.load_vm_flatbuffer(
                module.read_bytes(), driver=settings["driver"]
            )
            return
        # Loading runs the module initializer, which packs the frozen weight once.
        config = runtime.Config(settings["driver"])
        index = runtime.ParameterIndex()
        index.load(str(parameters))
        start = time.perf_counter()
        modules = runtime.load_vm_modules(
            runtime.create_io_parameters_module(
                config.vm_instance, index.create_provider(scope="model")
            ),
            runtime.create_hal_module(config.vm_instance, config.device),
            runtime.VmModule.mmap(config.vm_instance, str(module)),
            config=config,
        )
        self.bound = modules[-1]
        self.details.update(
            prepack_s=time.perf_counter() - start,
            weight_layout="parameter; data-tiling pack hoisted to initializer",
        )
        self.scope += "; frozen W loaded from a parameter archive, packed by the module initializer (prepack_s includes module load)"

    def dump(self, destination):
        for name in [
            "matmul.mlir",
            "matmul.vmfb",
            "compile_command.json",
            "compile.log",
        ]:
            shutil.copy2(self.work / name, destination / name)
        for name in ["ir", "executables", "phases"]:
            if (self.work / name).exists():
                shutil.copytree(
                    self.work / name, destination / name, dirs_exist_ok=True
                )
        dumper = Path(benchmark_exe()).with_name("iree-dump-module")
        with (destination / "vm_disassembly.txt").open("w") as out:
            subprocess.run(
                [str(dumper), "--output=all", str(self.module)], stdout=out, check=True
            )
        return dict(
            notes="VM disassembly is IREE bytecode; CPU executable binaries and compiler intermediates are in executables/."
        )

    def run(self):
        return (
            self.bound.main(self.a)
            if self.parameters
            else self.bound.main(self.a, self.b)
        ).to_host()

    def measure(self, min_repeat_ms, min_reps=1):
        seconds = self.measure_native(f"{min_repeat_ms / 1000}s")
        if self.last_iterations < min_reps:
            seconds = self.measure_native(f"{min_reps}x")
        if self.last_iterations < min_reps:
            raise RuntimeError(
                "IREE benchmark did not honor the requested minimum reps"
            )
        return seconds

    def measure_native(self, minimum_time):
        output = self.work / f"benchmark_{self.batch}.json"
        command = [
            benchmark_exe(),
            f"--module={self.module}",
            "--function=main",
            f"--device={self.settings['driver']}",
            f"--input=@{self.work / 'A.npy'}",
            *(
                [f"--input=@{self.work / 'B.npy'}"]
                if self.parameters is None
                else [f"--parameters=model={self.parameters}"]
            ),
            f"--benchmark_min_time={minimum_time}",
            "--benchmark_repetitions=1",
            f"--benchmark_out={output}",
            "--benchmark_out_format=json",
            *self.settings["benchmark_flags"],
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        (self.work / f"benchmark_{self.batch}.log").write_text(
            process.stdout + "\n" + process.stderr
        )
        (self.work / f"benchmark_{self.batch}_command.json").write_text(
            json.dumps(command, indent=2)
        )
        data = json.loads(output.read_text())
        rows = [
            r
            for r in data["benchmarks"]
            if r.get("run_type", "iteration") == "iteration"
        ]
        if len(rows) != 1 or rows[0].get("error_occurred"):
            raise RuntimeError(f"Unexpected IREE benchmark output: {rows}")
        self.last_iterations = rows[0]["iterations"]
        self.last_batch_file = str(output)
        self.batch += 1
        return (
            rows[0]["real_time"]
            * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[rows[0]["time_unit"]]
        )


if __name__ == "__main__":
    run("iree", IREE)
