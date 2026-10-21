import csv
import json
from pathlib import Path


def summarize(paths, output):
    fields = [
        "backend",
        "variant",
        "device",
        "cpu",
        "baseline_cpu",
        "runner_slot",
        "scheduling_mode",
        "strict_affinity",
        "op",
        "name",
        "shape",
        "output_shape",
        "m",
        "k",
        "n",
        "reps",
        "median_ms",
        "baseline_backend",
        "baseline_ms",
        "gflops",
        "baseline_gflops",
        "speedup",
        "tags",
        "prepack_s",
        "weight_layout",
        "timing_scope",
        "baseline_timing_scope",
    ]
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        backend = data["metadata"]["backend"]
        for row in data["rows"]:
            if "error" in row:
                continue  # failed case under GEMM_KEEP_GOING; reported by mimir_eval.py
            # Old results retain their original OpenBLAS baseline when reported.
            affinity = data["metadata"].get("affinity") or []
            scheduling = row.get("scheduling", {})
            rows.append(
                dict(
                    backend=backend,
                    variant=data["metadata"].get("variant", "default"),
                    device=row["device"],
                    cpu=row.get("cpu", next(iter(affinity), "")),
                    baseline_cpu=row.get("baseline_cpu", next(iter(affinity), "")),
                    runner_slot=row.get("runner_slot", ""),
                    scheduling_mode=scheduling.get("mode", ""),
                    strict_affinity=scheduling.get("strict", ""),
                    op=row.get("op", "gemm"),
                    name=row.get("name", str(row.get("shape", "legacy"))),
                    shape=json.dumps(
                        row.get("shape", [row.get("m"), row.get("n"), row.get("k")])
                    ),
                    output_shape=json.dumps(
                        row.get("output_shape", [row.get("m"), row.get("n")])
                    ),
                    m=row.get("m", ""),
                    n=row.get("n", ""),
                    k=row.get("k", ""),
                    reps=row.get("reps", 1),
                    median_ms=row["backend_ms"],
                    baseline_backend=row.get("baseline_backend", "openblas"),
                    baseline_ms=row.get("baseline_ms", row.get("openblas_ms")),
                    gflops=row["backend_gflops"],
                    baseline_gflops=row.get(
                        "baseline_gflops", row.get("openblas_gflops")
                    ),
                    speedup=row["speedup"],
                    tags=",".join(row.get("tags", [])),
                    prepack_s=row.get("prepack_s", ""),
                    weight_layout=row.get("weight_layout", ""),
                    timing_scope=row["timing_scope"],
                    baseline_timing_scope=row.get(
                        "baseline_timing_scope", "legacy OpenBLAS cblas_sgemm"
                    ),
                )
            )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Combined summary: {output}")
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge operator results, retaining baseline identity"
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()
    summarize(args.inputs, args.output)
