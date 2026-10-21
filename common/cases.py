"""Named CPU cases: GEMM/linear M K N, GEMV M K, Conv2D N Cin H W Cout KH KW stride pad.

Every case carries tags: its op, an automatic size class (working set of A, B and C in FP32:
small <= 1 MiB, medium <= 32 MiB, large above), an automatic regularity class (regular when all
tiled dimensions are multiples of 16, irregular otherwise, vector when M is 1) and any family
tags from the case file (`tags=llm,resnet`). Selection by tags, names, ops or ad-hoc shapes
(`--shape op:dims`) happens in select_cases.
"""

from pathlib import Path

import numpy as np

from .operators import tensor_shapes

OPS = ("gemm", "gemv", "conv2d", "linear")
# Dimensions per operator, in case-file order (also the order of --shape).
DIMS = {
    "gemm": "M K N",
    "linear": "M K N",
    "gemv": "M K",
    "conv2d": "N Cin H W Cout KH KW stride pad",
}
SMALL_BYTES, MEDIUM_BYTES = 1 << 20, 32 << 20
TILE = 16


def make_case(op, name, values, reps, tags=()):
    """Validate case-file-order dimensions and build a case (GEMM shapes are stored as M,N,K)."""
    op = "conv2d" if op == "conv" else op
    if op not in DIMS:
        raise ValueError(f"unknown operator {op!r}; expected one of {', '.join(OPS)}")
    if len(values) != len(DIMS[op].split()):
        raise ValueError(f"{op} takes {DIMS[op]}")
    if reps < 1 or min(values[:-1] if op == "conv2d" else values) < 1:
        raise ValueError("dimensions and reps must be positive")
    if op == "conv2d" and values[-1] < 0:
        raise ValueError("padding must be nonnegative")
    dims = list(values)
    if op in ("gemm", "linear"):
        m, k, n = dims
        dims = [m, n, k]
    case = dict(op=op, name=name, shape=dims, reps=reps)
    if min(tensor_shapes(case)[2]) < 1:
        raise ValueError("convolution output must be positive")
    case["tags"] = sorted({*auto_tags(case), *tags})
    return case


def auto_tags(case):
    a, b, c = tensor_shapes(case)
    working_set = 4 * sum(int(np.prod(shape)) for shape in (a, b, c))
    size = (
        "small"
        if working_set <= SMALL_BYTES
        else "medium"
        if working_set <= MEDIUM_BYTES
        else "large"
    )
    if case["op"] in ("gemm", "linear"):
        m, n, k = case["shape"]
        tiled = [n, k] if m == 1 else [m, n, k]
    elif case["op"] == "gemv":
        tiled = list(case["shape"])
    else:
        _, ci, _, _, co, kh, kw, _, _ = case["shape"]
        tiled = [
            co,
            c[2] * c[3],
            ci * kh * kw,
        ]  # output channels, output pixels, reduction
    regularity = "regular" if all(d % TILE == 0 for d in tiled) else "irregular"
    tags = [case["op"], size, regularity]
    if case["op"] in ("gemm", "linear") and case["shape"][0] == 1:
        tags.append("vector")
    return tags


def load_cases(path):
    cases, names = [], set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        tags = []
        if fields[-1].startswith("tags="):
            tags = [tag for tag in fields.pop()[len("tags=") :].split(",") if tag]
        if len(fields) < 3:
            raise ValueError(
                f"{path}:{number}: expected op name dims... reps [tags=a,b]"
            )
        op, name = fields[0], fields[1]
        try:
            values = list(map(int, fields[2:]))
        except ValueError as exc:
            raise ValueError(
                f"{path}:{number}: dimensions/reps must be integers"
            ) from exc
        if name in names:
            raise ValueError(f"{path}:{number}: duplicate case name {name}")
        try:
            case = make_case(op, name, values[:-1], values[-1], tags)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        cases.append(case)
        names.add(name)
    if not cases:
        raise ValueError(f"No cases in {path}")
    return cases


def parse_shape(spec, reps):
    """--shape OP:D1,D2,... (case-file order; `x` also separates), e.g. gemm:49,4608,512."""
    op, sep, dims = spec.partition(":")
    if not sep:
        raise ValueError(f"--shape {spec!r}: expected OP:dims, e.g. gemm:49,4608,512")
    try:
        values = [int(v) for v in dims.replace("x", ",").split(",")]
    except ValueError as exc:
        raise ValueError(f"--shape {spec!r}: dimensions must be integers") from exc
    op = "conv2d" if op == "conv" else op
    try:
        return make_case(
            op, f"{op}_{'x'.join(map(str, values))}", values, reps, ["custom"]
        )
    except ValueError as exc:
        raise ValueError(f"--shape {spec!r}: {exc}") from exc


def matches(case, groups):
    """`groups`: --tags values; a case matches if every tag of any one group matches (!tag negates)."""
    if not groups:
        return True
    tags = set(case["tags"])
    return any(
        all(
            (tag[1:] not in tags) if tag.startswith("!") else (tag in tags)
            for tag in group.split(",")
        )
        for group in groups
    )


def select_cases(
    bench,
    shapes=None,
    case_file=None,
    names=None,
    reps=None,
    root=Path("."),
    ops=None,
    tags=None,
    custom=None,
):
    requested_ops = ops.split(",") if ops else bench.get("ops", list(OPS))
    if requested_ops == ["all"]:
        requested_ops = list(OPS)
    if (
        not requested_ops
        or len(set(requested_ops)) != len(requested_ops)
        or any(op not in OPS for op in requested_ops)
    ):
        raise ValueError(
            "--ops accepts gemm,gemv,conv2d,linear (comma-separated) or all"
        )
    if (shapes or custom) and (names or case_file):
        raise ValueError(
            "--shape/--shapes cannot be combined with --cases or --cases-file"
        )
    if custom:
        # Ad-hoc shapes are explicit, so --ops does not filter them.
        cases = [parse_shape(spec, bench.get("default_reps", 1)) for spec in custom]
        if len({c["name"] for c in cases}) != len(cases):
            raise ValueError("--shape values must be distinct")
    elif shapes:
        tuples = [tuple(map(int, s.split(","))) for s in shapes.split(";")]
        if "gemm" not in requested_ops or any(
            len(s) != 3 or min(s) < 1 for s in tuples
        ):
            raise ValueError(
                "--shapes requires GEMM and positive M,N,K triples separated by semicolons"
            )
        # Legacy option: M,N,K order (case files and --shape use M,K,N).
        cases = [
            make_case(
                "gemm", f"m{m}_k{k}_n{n}", [m, k, n], bench.get("default_reps", 1)
            )
            for m, n, k in tuples
        ]
    else:
        paths = (
            [case_file]
            if case_file
            else bench.get(
                "cases_files", [bench.get("cases_file", "config/gemm_cases.txt")]
            )
        )
        cases = []
        for path in paths:
            path = Path(path)
            cases.extend(load_cases(path if path.is_absolute() else root / path))
        if len({c["name"] for c in cases}) != len(cases):
            raise ValueError("Case names must be unique across case files")
        cases = [c for c in cases if c["op"] in requested_ops]
    if names:
        requested = names.split(",")
        available = {c["name"]: c for c in cases}
        if len(set(requested)) != len(requested) or any(
            name not in available for name in requested
        ):
            raise ValueError(
                "--cases requires unique names present in the selected operators/case files"
            )
        cases = [available[name].copy() for name in requested]
    cases = [c for c in cases if matches(c, tags)]
    if not cases:
        raise ValueError("No cases match the selected operators/tags")
    if reps is not None:
        if reps < 1:
            raise ValueError("--reps must be positive")
        cases = [dict(case, reps=reps) for case in cases]
    return cases


def describe(cases):
    """One line per case for --list-cases."""
    lines = [f"{'op':7} {'name':16} {'dims (case-file order)':34} tags"]
    for case in cases:
        dims = list(case["shape"])
        if case["op"] in ("gemm", "linear"):
            m, n, k = dims
            dims = [m, k, n]
        text = " ".join(f"{k}={v}" for k, v in zip(DIMS[case["op"]].split(), dims))
        lines.append(
            f"{case['op']:7} {case['name']:16} {text:34} {','.join(case['tags'])}"
        )
    return "\n".join(lines)
