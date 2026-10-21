import json
from pathlib import Path
import tempfile
import unittest

from types import SimpleNamespace
from unittest.mock import Mock, patch

from common.config import runner_cpu_list
from mimir.benchmark import MimIR
from mimir_eval import build_directory, report

SPEC = dict(
    candidate=dict(remote="me/MimIR", ref="abc", label="cand"),
    baseline=dict(remote="mimir/mimir", ref="def", label="base"),
)


def results(path, backend, rows, sha):
    path.mkdir(parents=True)
    (path / "results.json").write_text(
        json.dumps(
            dict(
                metadata=dict(backend=backend, mimir_git=sha, cpu="test", mcpu="x"),
                rows=rows,
            )
        )
    )


def row(name, op, gflops, speedup):
    return dict(
        name=name,
        op=op,
        shape=[2, 2, 2],
        backend_gflops=gflops,
        baseline_gflops=gflops / speedup,
        speedup=speedup,
    )


class MimIREval(unittest.TestCase):
    def test_report_ratios_geomean_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            results(
                out / "baseline/mimir",
                "mimir",
                [
                    row("a", "gemm", 10, 1.0),
                    row("b", "gemm", 40, 1.0),
                    row("c", "gemv", 5, 1.0),
                ],
                "d" * 40,
            )
            results(
                out / "candidate/mimir",
                "mimir",
                [
                    row("a", "gemm", 20, 2.0),
                    row("b", "gemm", 20, 0.5),
                    dict(name="c", op="gemv", shape=[2, 2], error="boom"),
                ],
                "a" * 40,
            )
            text, failures = report(SPEC, out, 60)
            self.assertEqual(failures, {"baseline": [], "candidate": ["c"]})
            self.assertIn("| gemm | 2 | **1.00** |", text)  # geomean of 2.0 and 0.5
            self.assertIn(
                "| gemv | c | [2, 2] | 5.0 | 5.0 (×1.00) | FAIL | FAIL |", text
            )
            self.assertIn("`aaaaaaaaaaaa`", text)
            self.assertTrue((out / "comparison.csv").is_file())

    def test_multithreaded_runner_prefers_one_cpu_per_core(self):
        siblings = {0: {0, 1}, 1: {0, 1}, 2: {2, 3}, 3: {2, 3}, 4: {4, 5}, 5: {4, 5}}
        with (
            patch("common.config.is_macos", return_value=False),
            patch("common.config.smt_siblings", side_effect=lambda c: siblings[c]),
        ):
            self.assertEqual(runner_cpu_list(None, {}, 2, 2, set(range(6))), [2, 4])
            self.assertEqual(
                runner_cpu_list(None, {}, 0, 4, set(range(6))), [0, 2, 4, 1]
            )
            self.assertEqual(
                runner_cpu_list(None, {"cpus": [5, 1]}, 0, 2, set(range(6))), [1, 5]
            )
            self.assertEqual(runner_cpu_list(None, {}, 3, 1, set(range(6))), [3])

    def test_report_lists_leaking_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            leak = dict(row("a", "conv2d", 10, 1.0), rss_growth_per_call_kib=3260.0)
            clean = dict(row("a", "conv2d", 20, 1.0), rss_growth_per_call_kib=0.0)
            results(out / "baseline/mimir", "mimir", [leak], "d" * 40)
            results(out / "candidate/mimir", "mimir", [clean], "a" * 40)
            text, _ = report(SPEC, out, 60)
            self.assertIn("Memory growth per call above 64 KiB", text)
            self.assertIn("| a | 3260 | 0 |", text)

    def test_explicit_cpus_must_be_distinct(self):
        parser = Mock(error=Mock(side_effect=SystemExit))
        with patch("common.config.is_macos", return_value=False):
            with self.assertRaises(SystemExit):
                runner_cpu_list(parser, {"cpus": [0, 0, 0, 0]}, 0, 4, set(range(8)))

    def test_omp_requires_the_plugin_in_the_checked_out_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, plugins = Path(tmp) / "source", Path(tmp) / "lib/mim"
            (plugins / "rt").mkdir(parents=True)
            source.mkdir()
            # Stale outputs from an earlier omp revision must not enable -p omp.
            (plugins / "omp.mim").write_text("")
            (plugins / "rt/omp_rt.ll").write_text("")
            mimir = SimpleNamespace(source=source, plugin_dir=plugins)
            self.assertEqual(
                MimIR.openmp(mimir, 4)[0], "none: this MimIR revision has no omp plugin"
            )
            (source / "src/mim/plug/omp").mkdir(parents=True)
            with patch("mimir.benchmark.is_macos", return_value=False):
                self.assertEqual(MimIR.openmp(mimir, 4), ("omp", ["-fopenmp"]))
            self.assertEqual(MimIR.openmp(mimir, 1)[0], "none: single-threaded")

    def test_build_directory_follows_cli_env_then_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            common = Path(tmp) / "common.toml"
            common.write_text('[build]\ndirectory = "/srv/cache"\n')
            args = SimpleNamespace(build_dir=None, common_config=common)
            with patch.dict("os.environ", {}, clear=False) as env:
                env.pop("GEMM_BUILD_DIR", None)
                self.assertEqual(build_directory(args), Path("/srv/cache"))
                env["GEMM_BUILD_DIR"] = "/srv/env"
                self.assertEqual(build_directory(args), Path("/srv/env"))
            args.build_dir = Path(tmp) / "cli"
            self.assertEqual(build_directory(args), (Path(tmp) / "cli").resolve())


if __name__ == "__main__":
    unittest.main()
