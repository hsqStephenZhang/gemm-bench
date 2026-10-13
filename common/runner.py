from copy import deepcopy
import csv
import json
import os
from pathlib import Path

import numpy as np

from .artifacts import export_artifacts
from .config import ROOT, load, parse_args
from .operators import inputs, flops, tensor_shapes, work_name


def check_affinity(cpu):
    actual = os.sched_getaffinity(0)
    if actual != {cpu}:
        raise RuntimeError(
            f"Runner CPU affinity changed: expected {[cpu]}, observed {sorted(actual)}"
        )
    return sorted(actual)


def error(output, reference, config):
    np.testing.assert_allclose(
        output, reference, rtol=config["rtol"], atol=config["atol"]
    )
    return float(np.max(np.abs(output - reference)))


def write_results(output, metadata, rows, backend):
    (output / "results.json").write_text(
        json.dumps(dict(metadata=metadata, rows=rows), indent=2)
    )
    fields = [
        "op",
        "name",
        "shape",
        "output_shape",
        "m",
        "k",
        "n",
        "reps",
        "device",
        "cpu",
        "baseline_cpu",
        "auto_tune",
        f"{backend}_ms",
        "baseline_backend",
        "baseline_ms",
        f"{backend}_gflops",
        "baseline_gflops",
        "speedup",
        "tuning_s",
        "compile_s",
        "valid_records",
        "timing_scope",
        "baseline_timing_scope",
    ]
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(fields)), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def run(backend, factory=None):
    args, config, settings, cases = parse_args(backend)
    bench = config["benchmark"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Imported lazily so the PyTorch adapter can itself use this runner.
    from pytorch.benchmark import PyTorch

    baseline_settings = (
        settings
        if backend == "pytorch"
        else load(os.environ.get("GEMM_PYTORCH_CONFIG", ROOT / "config/pytorch.toml"))
    )
    baseline_settings = deepcopy(baseline_settings)
    baseline_settings["pytorch"]["mode"] = "eager"
    baseline = PyTorch(config, baseline_settings)
    compiled_pytorch = (
        backend == "pytorch" and settings["pytorch"].get("mode", "eager") == "inductor"
    )
    if backend == "openblas":
        from .openblas import OpenBLAS

        adapter = OpenBLAS(config, output)
    else:
        adapter = (
            factory(config, settings)
            if backend != "pytorch" or compiled_pytorch
            else None
        )
    cpu = next(
        (
            s.split(":", 1)[1].strip()
            for s in Path("/proc/cpuinfo").read_text().splitlines()
            if s.startswith("model name")
        ),
        "unknown",
    )
    metadata = dict(
        backend=backend,
        device=bench["device"],
        cpu=cpu,
        affinity=check_affinity(bench["cpu"]),
        mcpu=bench["mcpu"],
        dtype=bench["dtype"],
        seed=bench["seed"],
        config=config,
        execution=config["execution"],
        backend_config=settings,
        baseline_backend="pytorch",
        baseline_config=baseline_settings,
        baseline_metadata=baseline.metadata,
    )
    metadata.update(adapter.metadata if adapter else baseline.metadata)
    metadata["measurement"] = (
        f"hot cache; {bench['repeats']} interleaved batches >= {bench['min_repeat_ms']} ms; median"
    )
    metadata["speedup_definition"] = "PyTorch median seconds / backend median seconds"
    result_file = output / "results.json"
    previous = (
        json.loads(result_file.read_text()) if result_file.exists() else {"rows": []}
    )
    if previous["rows"]:
        for key in [
            "execution",
            "baseline_backend",
            "baseline_config",
            "baseline_metadata",
            "device",
            "affinity",
            "mcpu",
            "dtype",
            "seed",
            "backend_config",
            *getattr(adapter, "identity_keys", []),
        ]:
            if previous["metadata"].get(key) != metadata.get(key):
                raise RuntimeError(
                    f"Output contains incompatible {key}; choose a fresh --output directory"
                )
    rows = previous["rows"]
    print(json.dumps(metadata), flush=True)
    for case in cases:
        shape = case["shape"]
        print(
            f"CASE {case['op']} {case['name']} dims={shape} reps>={case['reps']}",
            flush=True,
        )
        work = output / work_name(case)
        work.mkdir(exist_ok=True)
        a, b, reference = inputs(case, bench["seed"])
        np.save(work / "A.npy", a)
        np.save(work / "B.npy", b)
        baseline_work = work / "baseline"
        baseline_work.mkdir(exist_ok=True)
        baseline_kernel = baseline.prepare(case, a, b, baseline_work, {})
        previous_row = next((r for r in rows if r.get("name") == case["name"]), {})
        if previous_row and (
            previous_row.get("op") != case["op"] or previous_row.get("shape") != shape
        ):
            raise RuntimeError(
                "Case name has changed operator or dimensions; choose a fresh --output"
            )
        kernel = (
            adapter.prepare(case, a, b, work, previous_row)
            if adapter
            else baseline_kernel
        )
        print(f"PREPARED {backend} {case['name']}", flush=True)
        for warmup in range(bench["warmup"]):
            kernel.run()
            if adapter:
                baseline_kernel.run()
            check_affinity(bench["cpu"])
            print(
                f"WARMUP {backend} {case['name']} {warmup + 1}/{bench['warmup']}",
                flush=True,
            )
        backend_error = error(kernel.run(), reference, bench)
        baseline_error = error(baseline_kernel.run(), reference, bench)
        check_affinity(bench["cpu"])
        (
            backend_times,
            baseline_times,
            backend_iterations,
            baseline_iterations,
            batch_files,
        ) = [], [], [], [], []
        for index in range(bench["repeats"]):
            if not adapter:
                value = kernel.measure(bench["min_repeat_ms"], case["reps"])
                backend_times.append(value)
                baseline_times.append(value)
            elif index % 2 == 0:
                backend_times.append(
                    kernel.measure(bench["min_repeat_ms"], case["reps"])
                )
                baseline_times.append(
                    baseline_kernel.measure(bench["min_repeat_ms"], case["reps"])
                )
            else:
                baseline_times.append(
                    baseline_kernel.measure(bench["min_repeat_ms"], case["reps"])
                )
                backend_times.append(
                    kernel.measure(bench["min_repeat_ms"], case["reps"])
                )
            check_affinity(bench["cpu"])
            backend_iterations.append(getattr(kernel, "last_iterations", None))
            baseline_iterations.append(baseline_kernel.last_iterations)
            batch_files.append(getattr(kernel, "last_batch_file", None))
            print(
                f"TIMING {backend} {case['name']} {index + 1}/{bench['repeats']}",
                flush=True,
            )
        seconds, baseline_seconds = (
            float(np.median(backend_times)),
            float(np.median(baseline_times)),
        )
        operations = flops(case)
        row = dict(
            op=case["op"],
            name=case["name"],
            shape=shape,
            output_shape=list(tensor_shapes(case)[2]),
            flops=operations,
            reps=case["reps"],
            device=bench["device"],
            cpu=bench["cpu"],
            baseline_cpu=bench["cpu"],
            observed_affinity=check_affinity(bench["cpu"]),
            baseline_backend="pytorch",
            backend_ms=seconds * 1000,
            baseline_ms=baseline_seconds * 1000,
            backend_gflops=operations / seconds / 1e9,
            baseline_gflops=operations / baseline_seconds / 1e9,
            speedup=baseline_seconds / seconds,
            backend_seconds=backend_times,
            baseline_seconds=baseline_times,
            backend_iterations=backend_iterations,
            baseline_iterations=baseline_iterations,
            backend_batch_files=batch_files,
            backend_max_abs_error=backend_error,
            baseline_max_abs_error=baseline_error,
            timing_scope=kernel.scope,
            baseline_timing_scope=baseline_kernel.scope,
            **getattr(kernel, "details", {}),
        )
        if case["op"] == "gemm":
            row.update(zip(["m", "n", "k"], shape))
        elif case["op"] == "gemv":
            row.update(m=shape[0], n=1, k=shape[1])
        row[f"{backend}_ms"], row[f"{backend}_gflops"] = (
            row["backend_ms"],
            row["backend_gflops"],
        )
        if config["artifacts"]["enabled"]:
            destination = Path(config["artifacts"]["directory"]) / work_name(case)
            row["artifact_manifest"] = str(destination / "manifest.json")
            export_artifacts(kernel, destination, work, metadata, row, baseline_kernel)
            print(f"ARTIFACTS {destination}", flush=True)
        rows = [r for r in rows if r.get("name") != case["name"]] + [row]
        write_results(output, metadata, rows, backend)
        print("RESULT " + json.dumps(row), flush=True)
