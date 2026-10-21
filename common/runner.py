import csv
import faulthandler
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import psutil

from .artifacts import export_artifacts
from .config import ROOT, load, parse_args
from .operators import inputs, flops, tensor_shapes, work_name
from .host import (
    check_affinity,
    cpu_model,
    exit_on_sigterm,
    is_macos,
    scheduling_metadata,
)


def error(output, reference, config):
    np.testing.assert_allclose(
        output, reference, rtol=config["rtol"], atol=config["atol"]
    )
    return float(np.max(np.abs(output - reference)))


def memory_limit_bytes(bench):
    """benchmark.memory_limit_mb, or 60% of physical memory when unset or 0."""
    limit_mb = bench.get("memory_limit_mb", 0)
    return limit_mb * 2**20 if limit_mb else int(0.6 * psutil.virtual_memory().total)


def start_memory_guard(limit, on_exceed):
    """Exit before the machine runs out of memory: kernels from some compiler revisions leak
    buffers on every call, and one runner process measures every case."""

    def watch():
        process = psutil.Process()
        while True:
            rss = process.memory_info().rss
            if rss > limit:
                on_exceed(rss)
                os._exit(3)
            time.sleep(0.2)

    threading.Thread(target=watch, name="memory-guard", daemon=True).start()


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
        "runner_slot",
        "auto_tune",
        f"{backend}_ms",
        "baseline_backend",
        "baseline_ms",
        f"{backend}_gflops",
        "baseline_gflops",
        "speedup",
        "prepack_s",
        "weight_layout",
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
    exit_on_sigterm()  # in-flight compiler subprocesses are killed with the runner
    args, config, settings, cases = parse_args(backend)
    bench = config["benchmark"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # One CPU (recorded as before) single-threaded; the runner's CPU set otherwise.
    pin = bench["cpu"] if bench["threads"] == 1 else bench["cpus"]
    # Imported lazily so the PyTorch adapter can itself use this runner.
    from pytorch.benchmark import PyTorch

    baseline_settings = (
        settings
        if backend == "pytorch"
        else load(os.environ.get("GEMM_PYTORCH_CONFIG", ROOT / "config/pytorch.toml"))
    )
    baseline = PyTorch(config, baseline_settings)
    if backend == "openblas":
        from .openblas import OpenBLAS

        adapter = OpenBLAS(config, output)
    else:
        adapter = factory(config, settings) if backend != "pytorch" else None
    cpu = cpu_model()
    metadata = dict(
        backend=backend,
        device=bench["device"],
        cpu=cpu,
        affinity=check_affinity(pin),
        scheduling=scheduling_metadata(pin),
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
            "scheduling",
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
    current = {}
    limit = memory_limit_bytes(bench)

    def memory_exceeded(rss):
        # Runs on the guard thread while the main thread may be stuck in native code.
        case = current.get("case")
        message = (
            f"MemoryLimit: runner RSS {rss / 2**20:.0f} MiB exceeded {limit / 2**20:.0f} MiB "
            "(benchmark.memory_limit_mb); a kernel probably leaks memory on every call"
        )
        print(
            f"FAILED {backend} {case['name'] if case else '-'}: {message}", flush=True
        )
        faulthandler.dump_traceback(all_threads=True)
        if case:
            saved = (
                json.loads(result_file.read_text())["rows"]
                if result_file.exists()
                else []
            )
            failed = dict(
                op=case["op"], name=case["name"], shape=case["shape"], error=message
            )
            saved = [r for r in saved if r.get("name") != case["name"]] + [failed]
            write_results(output, metadata, saved, backend)

    start_memory_guard(limit, memory_exceeded)

    def measure_case(case):
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
        # Memory growth across warmup and timing exposes kernels that leak on every call.
        rss_before = psutil.Process().memory_info().rss
        for warmup in range(bench["warmup"]):
            kernel.run()
            if adapter:
                baseline_kernel.run()
            check_affinity(pin)
            print(
                f"WARMUP {backend} {case['name']} {warmup + 1}/{bench['warmup']}",
                flush=True,
            )
        backend_error = error(kernel.run(), reference, bench)
        baseline_error = error(baseline_kernel.run(), reference, bench)
        check_affinity(pin)
        (
            backend_times,
            baseline_times,
            backend_iterations,
            baseline_iterations,
            batch_files,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )
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
            check_affinity(pin)
            backend_iterations.append(getattr(kernel, "last_iterations", None))
            baseline_iterations.append(baseline_kernel.last_iterations)
            batch_files.append(getattr(kernel, "last_batch_file", None))
            print(
                f"TIMING {backend} {case['name']} {index + 1}/{bench['repeats']}",
                flush=True,
            )
        rss_growth = psutil.Process().memory_info().rss - rss_before
        counted = [n for n in backend_iterations if n is not None]
        calls = (
            bench["warmup"] + 1 + sum(counted)
            if len(counted) == len(backend_iterations)
            else None
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
            cpu=None if is_macos() else bench["cpu"],
            baseline_cpu=None if is_macos() else bench["cpu"],
            runner_slot=bench["cpu"],
            scheduling=scheduling_metadata(pin),
            observed_affinity=check_affinity(pin),
            threads=bench["threads"],
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
        row["tags"] = case["tags"]
        row["rss_growth_mb"] = rss_growth / 2**20
        row["rss_growth_per_call_kib"] = rss_growth / calls / 1024 if calls else None
        if case["op"] in ("gemm", "linear"):
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
        return row

    keep_going = os.environ.get("GEMM_KEEP_GOING") == "1"
    case_timeout = bench.get("case_timeout_s", 3600)
    for case in cases:
        # Watchdog: a case stuck in native code (e.g. a runtime deadlock) dumps every thread's
        # Python stack and exits instead of spinning forever; run.py --keep-going moves on.
        faulthandler.dump_traceback_later(case_timeout, exit=True)
        current["case"] = case
        try:
            row = measure_case(case)
        except Exception as exc:
            # Comparisons (mimir_eval.py) record a failing case and continue with the rest.
            if not keep_going:
                raise
            row = dict(
                op=case["op"],
                name=case["name"],
                shape=case["shape"],
                error=f"{type(exc).__name__}: {exc}"[:2000],
            )
            print(f"FAILED {backend} {case['name']}: {row['error']}", flush=True)
        finally:
            faulthandler.cancel_dump_traceback_later()
        rows = [r for r in rows if r.get("name") != case["name"]] + [row]
        write_results(output, metadata, rows, backend)
        print("RESULT " + json.dumps(row), flush=True)
