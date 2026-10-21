import argparse
import json
import os
from pathlib import Path
import tomllib
from .cases import select_cases
from .host import (
    available_cpus,
    bind_cpu,
    configure_host,
    resolve_mcpu,
    is_macos,
    smt_siblings,
)

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    config = tomllib.loads(Path(path).resolve().read_text())
    if "benchmark" in config and "openblas" in config:
        configure_host(config)
        config["benchmark"]["threads"] = thread_count(config)
    return config


# Every runtime's worker-count variable; all follow the one benchmark thread count.
THREAD_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TVM_NUM_THREADS",
    "MIM_NUM_THREADS",
]


def thread_count(config):
    return int(
        os.environ.get("GEMM_THREADS", config.get("benchmark", {}).get("threads", 1))
    )


def environment(config):
    env = os.environ.copy()
    env.update(config["threads"])
    env.update({name: str(thread_count(config)) for name in THREAD_VARS})
    # TVM otherwise rebinds worker 0 (the main thread) to CPU 0 at pool initialization.
    env["TVM_BIND_THREADS"] = "0"
    return env


def resolve_cpu(parser, cpu=None, allowed_cpus=None):
    allowed_cpus = available_cpus() if allowed_cpus is None else allowed_cpus
    if cpu is None:
        return min(allowed_cpus)
    if cpu not in allowed_cpus:
        parser.error(f"CPU {cpu} is unavailable. Allowed CPUs: {sorted(allowed_cpus)}")
    return cpu


def resolve_runner_cpus(
    parser, config, backends, cpu=None, overrides=None, allowed_cpus=None
):
    allowed_cpus = available_cpus() if allowed_cpus is None else allowed_cpus
    names = {"pytorch", "tvm", "iree", "openblas", "mimir"}
    configured = config.get("runner_cpus", {})
    if (
        not isinstance(configured, dict)
        or set(configured) - names
        or any(type(c) is not int for c in configured.values())
    ):
        parser.error("[runner_cpus] requires known runner names and integer CPU IDs")
    explicit = {}
    if overrides:
        try:
            for item in overrides.split(","):
                name, value = item.split("=")
                if name not in names or name in explicit:
                    raise ValueError()
                explicit[name] = int(value)
        except ValueError:
            parser.error(
                "--runner-cpus requires unique runner=CPU pairs, e.g. pytorch=0,tvm=2"
            )
    result = {}
    for backend in backends:
        selected = explicit.get(
            backend,
            cpu
            if cpu is not None
            else configured.get(backend, config["benchmark"].get("cpu")),
        )
        result[backend] = resolve_cpu(parser, selected, allowed_cpus)
    return result


def parse_args(backend):
    config = load(os.environ.get("GEMM_COMMON_CONFIG", ROOT / "config/common.toml"))
    settings = (
        load(os.environ.get("GEMM_BACKEND_CONFIG", ROOT / f"config/{backend}.toml"))
        if backend != "openblas"
        else {"openblas": config["openblas"].copy()}
    )
    if backend == "mimir":
        for key in ["remote", "ref"]:
            if "GEMM_MIMIR_" + key.upper() in os.environ:
                settings["mimir"][key] = os.environ["GEMM_MIMIR_" + key.upper()]
    bench = config["benchmark"].copy()
    parser = argparse.ArgumentParser(
        description=f"{backend} FP32 CPU operator benchmark"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "GEMM_BUILD_DIR", config.get("build", {}).get("directory", "build")
            )
        ),
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default=bench["device"])
    parser.add_argument("--cpu", type=int, help="Override the CPU for this runner")
    parser.add_argument("--runner-cpus", help="Override runner=CPU pairs")
    parser.add_argument("--mcpu", default=bench["mcpu"])
    parser.add_argument(
        "--shapes", help="Override with M,N,K triples (legacy CLI order)"
    )
    parser.add_argument(
        "--cases-file", type=Path, help="Named GEMM/GEMV/Conv2D case table"
    )
    parser.add_argument(
        "--ops", help="gemm,gemv,conv2d or all; overrides benchmark.ops"
    )
    parser.add_argument("--cases", help="Comma-separated case names")
    parser.add_argument(
        "--tags",
        action="append",
        help="Select cases having all of these comma-separated tags (!tag excludes); repeat for a union",
    )
    parser.add_argument(
        "--shape",
        action="append",
        help="Ad-hoc case OP:dims in case-file order, e.g. gemm:49,4608,512; repeatable",
    )
    parser.add_argument(
        "--reps", type=int, help="Override minimum native calls per timing batch"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / bench["output_root"] / backend
    )
    parser.add_argument(
        "--dump",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export final artifacts outside the timed region",
    )
    parser.add_argument(
        "--dump-dir", type=Path, help="Artifact directory; implies --dump"
    )
    if backend == "tvm":
        parser.add_argument(
            "--auto-tune",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable or disable TVM MetaSchedule auto tuning",
        )
        parser.add_argument(
            "--trials", type=int, default=settings["tvm"]["auto_tune"]["trials"]
        )
    args = parser.parse_args()
    config["build"] = {"directory": str(args.build_dir.resolve())}
    allowed_cpus = available_cpus()
    args.cpu = resolve_runner_cpus(
        parser, config, [backend], args.cpu, args.runner_cpus, allowed_cpus
    )[backend]
    threads = thread_count(config)
    if threads < 1:
        parser.error("benchmark.threads must be positive")
    cpus = runner_cpu_list(parser, bench, args.cpu, threads, allowed_cpus)
    config["threads"]["TVM_BIND_THREADS"] = os.environ.get("TVM_BIND_THREADS", "0")
    config["execution"] = {
        "parallel": os.environ.get("GEMM_RUN_PARALLEL") == "1",
        "runner_cpus": json.loads(
            os.environ.get("GEMM_RUNNER_CPUS", json.dumps({backend: args.cpu}))
        ),
        "threads": threads,
        "cpus": cpus,
    }
    if args.device != "cpu":
        parser.error(
            "This benchmark currently implements CPU only; GPU requires GPU kernels and a GPU BLAS baseline"
        )
    if bench["dtype"] != "float32":
        parser.error("Only float32 operators are implemented")
    try:
        cases = select_cases(
            bench,
            args.shapes,
            args.cases_file,
            args.cases,
            args.reps,
            ROOT,
            args.ops,
            tags=args.tags,
            custom=args.shape,
        )
    except ValueError as exc:
        parser.error(str(exc))
    bench.update(
        device=args.device,
        cpu=args.cpu,
        cpus=cpus,
        threads=threads,
        allowed_cpus=sorted(allowed_cpus),
        mcpu=resolve_mcpu(args.mcpu),
        cpu_selection="runner slot; not a core pin"
        if is_macos()
        else "logical CPU affinity",
        shapes=[case["shape"] for case in cases],
        cases=cases,
        reps_policy="minimum native calls per batch; adaptive timing may run more",
    )
    if min(bench["repeats"], bench["min_repeat_ms"]) < 1 or bench["warmup"] < 0:
        parser.error("Invalid measurement configuration")
    if backend == "tvm":
        if args.trials < 1:
            parser.error("--trials must be positive")
        settings["tvm"]["auto_tune"]["trials"] = args.trials
        settings["tvm"]["auto_tune"]["enabled"] = (
            args.auto_tune
            if args.auto_tune is not None
            else settings["tvm"]["auto_tune"].get("enabled", False)
        )
        if settings["tvm"]["auto_tune"]["enabled"]:
            builder_cpus = settings["tvm"]["auto_tune"].get("builder_cpus", [])
            if builder_cpus:
                invalid = set(builder_cpus) - allowed_cpus
                if invalid:
                    parser.error(
                        f"TVM builder CPUs {sorted(invalid)} are unavailable. Allowed CPUs: {sorted(allowed_cpus)}"
                    )
            else:
                builder_cpus = sorted(allowed_cpus - set(cpus)) or [args.cpu]
            settings["tvm"]["auto_tune"]["builder_cpus"] = builder_cpus
    artifacts = config.get("artifacts", {}).copy()
    if args.dump_dir is not None and args.dump is False:
        parser.error("--no-dump cannot be combined with --dump-dir")
    artifacts["enabled"] = (
        args.dump
        if args.dump is not None
        else bool(args.dump_dir or artifacts.get("enabled", False))
    )
    artifacts["directory"] = str(
        args.dump_dir.resolve()
        if args.dump_dir
        else args.output.resolve() / artifacts.get("subdirectory", "artifacts")
    )
    config["artifacts"] = artifacts
    config["benchmark"] = bench
    bind_cpu(args.cpu if threads == 1 else cpus)
    return args, config, settings, cases


def runner_cpu_list(parser, bench, cpu, threads, allowed_cpus):
    """The runner's CPUs: [cpu] single-threaded; else benchmark.cpus / GEMM_CPUS, or `threads` allowed CPUs from `cpu` on."""
    if threads == 1:
        return [cpu]
    explicit = os.environ.get("GEMM_CPUS") or bench.get("cpus")
    if explicit:
        cpus = sorted(
            int(c)
            for c in (explicit.split(",") if isinstance(explicit, str) else explicit)
        )
        if (
            len(cpus) != threads
            or len(set(cpus)) != threads
            or set(cpus) - allowed_cpus
        ):
            parser.error(
                f"benchmark.cpus must list {threads} distinct allowed CPUs; allowed: {sorted(allowed_cpus)}"
            )
        return cpus
    if is_macos():
        return [cpu]  # runner slot only; macOS schedules the worker threads itself
    # One thread per physical core first (SMT siblings last), starting at the runner's CPU.
    ordered = sorted(allowed_cpus)
    ordered = ordered[ordered.index(cpu) :]
    first = [c for c in ordered if min(smt_siblings(c) & allowed_cpus) == c]
    cpus = (first + [c for c in ordered if c not in first])[:threads]
    if len(cpus) < threads:
        parser.error(
            f"{threads} threads need {threads} allowed CPUs from CPU {cpu}; allowed: {ordered}. Set benchmark.cpus or --cpu"
        )
    return cpus


def reject_dump_check(parser, args, forwarded):
    if args.check and any(
        arg in ["--dump", "--dump-dir"] or arg.startswith("--dump-dir=")
        for arg in forwarded
    ):
        parser.error("--check does not generate artifacts; omit --check when dumping")
