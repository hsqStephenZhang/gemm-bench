"""Compare two MimIR revisions with the gemm-bench harness and write a Markdown report.

Each side is the ordinary `mimir` backend pointed at its remote/ref, run one after the other on
the same CPUs with its own persistent checkout (<build>/mimir-eval/<side>). The headline figure is
candidate GFLOPS / baseline GFLOPS; each side's speedup over its interleaved PyTorch run is
shown for context (dividing by PyTorch would add PyTorch's own noise on tiny cases).
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import tomllib

from common.host import exit_on_sigterm, run_child

ROOT = Path(__file__).resolve().parent
LEAK_KIB = 64  # per-call runner memory growth reported as a leak


def load(path):
    if not path.is_file():
        return {}, {}
    data = json.loads(path.read_text())
    return data["metadata"], {row["name"]: row for row in data["rows"]}


def geomean(values):
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def report(spec, output, elapsed):
    base_meta, base = load(output / "baseline/mimir/results.json")
    cand_meta, cand = load(output / "candidate/mimir/results.json")
    names = list(dict.fromkeys([*cand, *base]))
    rows, ratios, failures = [], {}, {"baseline": [], "candidate": []}
    for name in names:
        b, c = base.get(name, {}), cand.get(name, {})
        op = c.get("op", b.get("op"))
        for side, row in [("baseline", b), ("candidate", c)]:
            if "error" in row or not row:
                failures[side].append(name)
        ok = "backend_gflops" in b and "backend_gflops" in c
        ratio = c["backend_gflops"] / b["backend_gflops"] if ok else None
        if ok:
            ratios.setdefault(op, []).append(ratio)
        rows.append(
            dict(
                op=op,
                case=name,
                shape=json.dumps(c.get("shape", b.get("shape"))),
                pytorch_gflops=c.get("baseline_gflops", b.get("baseline_gflops")),
                baseline_gflops=b.get("backend_gflops"),
                candidate_gflops=c.get("backend_gflops"),
                baseline_vs_pytorch=b.get("speedup"),
                candidate_vs_pytorch=c.get("speedup"),
                candidate_vs_baseline=ratio,
                baseline_error=b.get("error", "" if b else "missing"),
                candidate_error=c.get("error", "" if c else "missing"),
            )
        )
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)
    num = lambda v, fmt=".2f": "FAIL" if v is None else format(v, fmt)
    bold = lambda v: (
        "FAIL" if v is None else (f"**{v:.2f}**" if v >= 1.0 else f"{v:.2f}")
    )
    revision = lambda meta, side: (
        f"`{spec[side]['remote']}` @ `{(meta.get('mimir_git') or spec[side]['ref'])[:12]}`"
    )
    host = cand_meta or base_meta
    lines = [
        f"# MimIR eval: {spec['candidate'].get('label', 'candidate')} vs {spec['baseline'].get('label', 'baseline')}",
        "",
        f"- Candidate: {revision(cand_meta, 'candidate')}",
        f"- Baseline: {revision(base_meta, 'baseline')}",
        f"- Host: {host.get('cpu', '?')}, mcpu `{host.get('mcpu', '?')}`, {host.get('execution', {}).get('threads', 1)} thread(s), MimIR parallel: {host.get('mimir_parallel', '?')}; {elapsed / 60:.0f} min",
        "- Ratio = candidate GFLOPS / baseline GFLOPS on the same core; > 1 means the candidate is faster. (×N) = speedup over PyTorch timed interleaved with that side.",
        "",
        "| Op | Cases | Geomean candidate/baseline |",
        "|---|---|---|",
    ]
    for op, values in ratios.items():
        lines.append(f"| {op} | {len(values)} | {bold(geomean(values))} |")
    every = [v for values in ratios.values() for v in values]
    lines += [f"| **all** | {len(every)} | {bold(geomean(every))} |", ""]
    for side in ["candidate", "baseline"]:
        if failures[side]:
            lines.append(f"**{side.capitalize()} failed:** {', '.join(failures[side])}")
    # Kernels whose runner memory keeps growing per call: a leak in the generated code.
    leaks = [
        (
            name,
            base.get(name, {}).get("rss_growth_per_call_kib"),
            cand.get(name, {}).get("rss_growth_per_call_kib"),
        )
        for name in names
    ]
    leaks = [(n, b, c) for n, b, c in leaks if max(b or 0, c or 0) > LEAK_KIB]
    if leaks:
        lines += [
            "",
            f"**Memory growth per call above {LEAK_KIB} KiB** (a leak in the generated kernel):",
            "",
            "| Case | Baseline KiB/call | Candidate KiB/call |",
            "|---|---|---|",
            *[f"| {n} | {num(b, '.0f')} | {num(c, '.0f')} |" for n, b, c in leaks],
        ]
    lines += [
        "",
        "| Op | Case | Shape | PyTorch GFLOPS | Baseline GFLOPS (×PyTorch) | Candidate GFLOPS (×PyTorch) | Candidate/baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    side = lambda g, x: "FAIL" if g is None else f"{g:.1f} (×{x:.2f})"
    for row in rows:
        lines.append(
            f"| {row['op']} | {row['case']} | {row['shape']} | {num(row['pytorch_gflops'], '.1f')} | {side(row['baseline_gflops'], row['baseline_vs_pytorch'])} | {side(row['candidate_gflops'], row['candidate_vs_pytorch'])} | {bold(row['candidate_vs_baseline'])} |"
        )
    text = "\n".join(lines) + "\n"
    (output / "report.md").write_text(text)
    return text, failures


def build_directory(args):
    """The persistent build root, with the harness's precedence: CLI, GEMM_BUILD_DIR, config."""
    if args.build_dir is not None:
        return args.build_dir.resolve()
    config = tomllib.loads(
        (args.common_config or ROOT / "config/common.toml").read_text()
    )
    path = Path(
        os.environ.get(
            "GEMM_BUILD_DIR", config.get("build", {}).get("directory", "build")
        )
    )
    return path if path.is_absolute() else (ROOT / path).resolve()


def main():
    exit_on_sigterm()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, default=ROOT / "eval/mimir.toml")
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh output root (default: results/mimir-eval-<time>)",
    )
    for side in ["candidate", "baseline"]:
        parser.add_argument(f"--{side}-remote")
        parser.add_argument(f"--{side}-ref")
    parser.add_argument("--ops")
    parser.add_argument("--cases")
    parser.add_argument(
        "--tags", action="append", help="Case tag selection, as for run.py"
    )
    parser.add_argument(
        "--shape", action="append", help="Ad-hoc OP:dims case, as for run.py"
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cpus")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--common-config", type=Path)
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build root (default: GEMM_BUILD_DIR, else [build].directory of the common config)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild report.md from an existing --output",
    )
    args = parser.parse_args()
    spec = tomllib.loads(args.spec.read_text())
    for side in ["candidate", "baseline"]:
        for key in ["remote", "ref"]:
            if getattr(args, f"{side}_{key}"):
                spec[side][key] = getattr(args, f"{side}_{key}")
    run = spec.get("run", {})
    output = (
        args.output or ROOT / "results" / time.strftime("mimir-eval-%Y%m%d-%H%M%S")
    ).resolve()
    start, status = time.time(), 0
    if not args.report_only:
        output.mkdir(parents=True, exist_ok=True)
        (output / "spec.json").write_text(json.dumps(spec, indent=2))
        common = ["--ops", args.ops or run.get("ops", "all"), "--keep-going"]
        cases = args.cases or run.get("cases", "")
        if cases:
            common += ["--cases", cases]
        for key in ["tags", "shape"]:
            for value in getattr(args, key) or run.get(key, []):
                common += [f"--{key}", value]
        threads = args.threads or run.get("threads")
        for flag, value in [
            ("--threads", threads),
            ("--cpus", args.cpus),
            ("--cpu", args.cpu),
            ("--common-config", args.common_config),
        ]:
            if value is not None:
                common += [flag, str(value)]
        build_root = build_directory(args)
        for side in ["baseline", "candidate"]:
            command = [
                sys.executable,
                str(ROOT / "run.py"),
                "--backends",
                "mimir",
                "--mimir-remote",
                spec[side]["remote"],
                "--mimir-ref",
                spec[side]["ref"],
                "--build-dir",
                str(build_root / "mimir-eval" / side),
                "--output",
                str(output / side),
                *common,
            ]
            print("+ " + " ".join(command), flush=True)
            status = run_child(command, cwd=ROOT).returncode or status
    text, failures = report(spec, output, time.time() - start)
    print(text)
    print(f"Report: {output / 'report.md'}")
    # Candidate failures fail the job; baseline failures are only reported.
    return status or (1 if failures["candidate"] else 0)


if __name__ == "__main__":
    sys.exit(main())
