"""Project entry point for CPU operator comparisons against PyTorch."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from common.cases import describe, select_cases
from common.config import ROOT, environment, load, resolve_runner_cpus
from common.report import summarize
from common.host import is_macos, exit_on_sigterm, run_child


def recorded_cases(path):
    """Case names with a row (result or recorded failure) in a runner's results.json."""
    if not path.is_file():
        return set()
    return {row["name"] for row in json.loads(path.read_text())["rows"]}


def with_cases(command, names):
    """`command` restricted to `names` via --cases (replacing an earlier --cases)."""
    if "--cases" in command:
        index = command.index("--cases")
        command = command[:index] + command[index + 2 :]
    return [*command, "--cases", ",".join(names)]


def main():
    exit_on_sigterm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        "--backend",
        dest="backends",
        help="Optional backends: comma-separated tvm,iree,openblas,mimir; all or none. Overrides [backends]. PyTorch always runs.",
    )
    parser.add_argument(
        "--common-config", type=Path, default=ROOT / "config/common.toml"
    )
    parser.add_argument("--tvm-config", type=Path, default=ROOT / "config/tvm.toml")
    parser.add_argument(
        "--auto-tune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable tuning for the TVM backend",
    )
    parser.add_argument("--iree-config", type=Path, default=ROOT / "config/iree.toml")
    parser.add_argument("--mimir-remote", help="GitHub owner/repo or Git remote URL")
    parser.add_argument(
        "--mimir-ref", help="MimIR branch/tag/commit; default from config"
    )
    parser.add_argument("--build-dir", type=Path, help="Persistent project build root")
    parser.add_argument("--mimir-config", type=Path, default=ROOT / "config/mimir.toml")
    parser.add_argument(
        "--pytorch-config", type=Path, default=ROOT / "config/pytorch.toml"
    )
    parser.add_argument(
        "--pytorch-mode",
        choices=["eager", "inductor"],
        help="Measured PyTorch mode; paired baselines remain eager",
    )
    parser.add_argument("--device", choices=["cpu", "gpu"])
    parser.add_argument("--shapes")
    parser.add_argument("--ops", help="gemm,gemv,conv2d or all")
    parser.add_argument("--cases-file", type=Path)
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
        "--list-cases",
        action="store_true",
        help="Print the selected cases and their tags, then exit",
    )
    parser.add_argument("--reps", type=int, help="Minimum calls per timing batch")
    parser.add_argument(
        "--cpu",
        type=int,
        help="Override every runner CPU except explicit --runner-cpus entries",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Threads per runner for every backend (overrides benchmark.threads)",
    )
    parser.add_argument(
        "--cpus",
        help="Comma-separated CPUs for a multi-threaded runner (Linux affinity)",
    )
    parser.add_argument(
        "--runner-cpus",
        help="Comma-separated runner=CPU assignments, e.g. pytorch=0,tvm=2,iree=4",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run backends concurrently; requires distinct CPU assignments",
    )
    parser.add_argument("--mcpu")
    parser.add_argument("--trials", type=int)
    parser.add_argument(
        "--output", type=Path, help="Output root; each backend uses a subdirectory"
    )
    parser.add_argument(
        "--dump",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Export final backend artifacts",
    )
    parser.add_argument(
        "--dump-dir", type=Path, help="Artifact root; each backend uses a subdirectory"
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record a failing case and continue instead of aborting the backend",
    )
    args = parser.parse_args()
    if args.dump_dir is not None and args.dump is False:
        parser.error("--no-dump cannot be combined with --dump-dir")
    if args.check and (args.dump or args.dump_dir):
        parser.error("--check does not generate artifacts; omit --check when dumping")
    config_path = args.common_config.resolve()
    config = load(config_path)
    try:
        # Validate the selection once up front (every runner repeats it) and support --list-cases.
        selected_cases = select_cases(
            config["benchmark"],
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
    if args.list_cases:
        print(describe(selected_cases))
        print(f"{len(selected_cases)} cases")
        return
    device = args.device or config["benchmark"]["device"]
    if device != "cpu":
        parser.error(
            "Only CPU benchmarking is implemented; GPU needs a GPU BLAS baseline"
        )
    optional = ["tvm", "iree", "openblas", "mimir"]
    selection = config.get("backends", dict.fromkeys(optional, True))
    if set(selection) - set(optional) or any(
        type(value) is not bool for value in selection.values()
    ):
        parser.error(
            "[backends] accepts only boolean tvm, iree, openblas and mimir settings; PyTorch is mandatory"
        )
    if args.backends is None:
        selected = [name for name in optional if selection.get(name, False)]
    elif args.backends == "all":
        selected = optional
    elif args.backends == "none":
        selected = []
    else:
        requested = args.backends.split(",")
        if any(name not in optional for name in requested) or len(
            set(requested)
        ) != len(requested):
            parser.error(
                "--backends accepts tvm,iree,openblas,mimir (comma-separated), all or none; PyTorch is mandatory"
            )
        selected = [name for name in optional if name in requested]
    if "mimir" not in selected and (
        args.mimir_remote is not None or args.mimir_ref is not None
    ):
        parser.error("--mimir-remote/--mimir-ref require MimIR to be selected")
    if "tvm" not in selected and (
        args.auto_tune is not None or args.trials is not None
    ):
        parser.error("--auto-tune/--no-auto-tune/--trials require TVM to be selected")
    backends = ["pytorch", *selected]
    parallel = (
        args.parallel
        if args.parallel is not None
        else config["benchmark"].get("parallel", False)
    )
    if (
        is_macos()
        and parallel
        and args.cpu is None
        and args.runner_cpus is None
        and not config.get("runner_cpus")
    ):
        config["runner_cpus"] = dict(zip(backends, range(len(backends))))
    cpu_map = resolve_runner_cpus(parser, config, backends, args.cpu, args.runner_cpus)
    threads = (
        args.threads
        if args.threads is not None
        else config["benchmark"].get("threads", 1)
    )
    if threads < 1:
        parser.error("--threads must be positive")
    if parallel and threads > 1:
        parser.error(
            "--parallel runs one CPU per runner; run multi-threaded backends sequentially"
        )
    if parallel and len(set(cpu_map.values())) != len(backends):
        parser.error(
            "--parallel requires a distinct CPU for each runner; use --runner-cpus"
        )
    output_root = (
        args.output.resolve()
        if args.output
        else ROOT / config["benchmark"]["output_root"]
    )
    os.environ["GEMM_THREADS"] = str(
        threads
    )  # environment() derives every *_NUM_THREADS from it
    env = environment(config)
    env.update(
        GEMM_RUN_PARALLEL="1" if parallel else "0", GEMM_RUNNER_CPUS=json.dumps(cpu_map)
    )
    if args.keep_going:
        env["GEMM_KEEP_GOING"] = "1"
    if args.cpus:
        env["GEMM_CPUS"] = args.cpus
    if args.build_dir is not None:
        env["GEMM_BUILD_DIR"] = str(args.build_dir.resolve())
    env.update(
        PYTHONPATH=str(ROOT),
        GEMM_COMMON_CONFIG=str(config_path),
        GEMM_PYTORCH_CONFIG=str(args.pytorch_config.resolve()),
    )

    def execute(backend):
        output = output_root / backend
        script = (
            "common/benchmark_openblas.py"
            if backend == "openblas"
            else f"{backend}/run.py"
        )
        command = [sys.executable, str(ROOT / script)]
        if backend == "tvm":
            command.extend(["--config", str(args.tvm_config.resolve())])
        if backend == "mimir":
            command.extend(["--config", str(args.mimir_config.resolve())])
            if args.mimir_remote is not None:
                command.extend(["--remote", args.mimir_remote])
            if args.mimir_ref is not None:
                command.extend(["--ref", args.mimir_ref])
            if args.build_dir is not None:
                command.extend(["--build-dir", str(args.build_dir.resolve())])
        if backend == "pytorch":
            command.extend(["--config", str(args.pytorch_config.resolve())])
            if args.pytorch_mode is not None:
                command.extend(["--pytorch-mode", args.pytorch_mode])
        if backend == "iree":
            command.extend(["--config", str(args.iree_config.resolve())])
        if backend != "openblas":
            command.extend(["--common-config", str(config_path)])
            if args.check:
                command.append("--check")
        elif args.check:
            code = 'from common.config import load; from common.openblas import OpenBLAS; import os,tempfile; from pathlib import Path; import json\nwith tempfile.TemporaryDirectory() as d:\n print(json.dumps(OpenBLAS(load(os.environ["GEMM_COMMON_CONFIG"]),Path(d)).metadata,indent=2))'
            command = [sys.executable, "-c", code]
        if not args.check:
            command.extend(
                [
                    "--device",
                    device,
                    "--cpu",
                    str(cpu_map[backend]),
                    "--output",
                    str(output),
                ]
            )
            if args.dump is not None:
                command.append("--dump" if args.dump else "--no-dump")
            if args.dump_dir is not None:
                destination = args.dump_dir.resolve() / backend
                command.extend(["--dump-dir", str(destination)])
            for key in ["shapes", "ops", "mcpu", "cases", "reps"]:
                value = getattr(args, key)
                if value is not None:
                    command.extend([f"--{key}", str(value)])
            if args.cases_file is not None:
                command.extend(["--cases-file", str(args.cases_file.resolve())])
            for key in ["tags", "shape"]:
                for value in getattr(args, key) or []:
                    command.extend([f"--{key}", value])
            if backend == "tvm" and args.auto_tune is not None:
                command.append("--auto-tune" if args.auto_tune else "--no-auto-tune")
            if backend == "tvm" and args.trials is not None:
                command.extend(["--trials", str(args.trials)])
        placement = (
            f"macOS QoS, runner slot {cpu_map[backend]}"
            if is_macos()
            else f"CPU {cpu_map[backend]}"
        )
        print(f"Running {backend} ({device}, {placement})", flush=True)
        log_path = output_root / f"{backend}.log"
        if parallel and not args.check:
            output_root.mkdir(parents=True, exist_ok=True)
            log_path.unlink(missing_ok=True)
            print(f"{backend} log: {log_path}", flush=True)

        def launch(command):
            if not (parallel and not args.check):
                return run_child(
                    command, cwd=ROOT, env=env, check=not args.keep_going
                ).returncode
            with log_path.open("a") as log:
                return run_child(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=not args.keep_going,
                ).returncode

        returncode = launch(command)
        failed_once = bool(returncode)
        # --keep-going: a runner that died mid-suite (memory guard, watchdog, crash) loses every
        # later case, so relaunch it in a fresh process for the cases without a result. A case that
        # makes no progress across a relaunch is skipped.
        names = [case["name"] for case in selected_cases]
        skipped, previous = [], None
        while (
            args.keep_going
            and returncode
            and not args.check
            and not (args.shape or args.shapes)
        ):
            done = recorded_cases(output / "results.json")
            remaining = [n for n in names if n not in done and n not in skipped]
            if remaining and done == previous:
                skipped.append(remaining.pop(0))
                print(
                    f"SKIPPED {backend} {skipped[-1]}: no progress after a relaunch",
                    flush=True,
                )
            if not remaining:
                break
            previous = done
            print(
                f"RELAUNCH {backend} for {len(remaining)} cases without results",
                flush=True,
            )
            returncode = launch(with_cases(command, remaining))
        if args.keep_going and (failed_once or skipped):
            # --keep-going: a backend that fails must not stop the remaining ones.
            print(
                f"FAILED backend {backend} (exit {returncode}, skipped {skipped})",
                flush=True,
            )
            failed.append(backend)
        return output / "results.json"

    failed = []
    if parallel:
        with ThreadPoolExecutor(max_workers=len(backends)) as pool:
            result_files = list(pool.map(execute, backends))
    else:
        result_files = [execute(backend) for backend in backends]
    if not args.check:
        summary = output_root / "summary.csv"
        summarize(
            [
                path
                for path, backend in zip(result_files, backends)
                if backend not in failed
            ],
            summary,
        )
    if failed:
        sys.exit(f"Failed backends: {','.join(failed)}")


if __name__ == "__main__":
    main()
