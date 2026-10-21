import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from common.config import ROOT, resolve_runner_cpus, environment
from common.runner import check_affinity


class RunnerCPUs(unittest.TestCase):
    def test_selection_precedence(self):
        parser = argparse.ArgumentParser()
        cfg = dict(benchmark=dict(cpu=2), runner_cpus=dict(pytorch=4, tvm=6))
        allowed = {2, 4, 6}
        backends = ["pytorch", "tvm", "iree"]
        self.assertEqual(
            resolve_runner_cpus(parser, cfg, backends, allowed_cpus=allowed),
            dict(pytorch=4, tvm=6, iree=2),
        )
        self.assertEqual(
            resolve_runner_cpus(
                parser,
                cfg,
                backends,
                cpu=2,
                overrides="pytorch=6",
                allowed_cpus=allowed,
            ),
            dict(pytorch=6, tvm=2, iree=2),
        )
        self.assertEqual(
            resolve_runner_cpus(
                parser, dict(benchmark={}), backends, allowed_cpus={4, 6}
            ),
            dict.fromkeys(backends, 4),
        )

    def test_mapping_validation(self):
        parser = argparse.ArgumentParser()
        for cfg, override in [
            (dict(benchmark={}, runner_cpus=dict(typo=2)), None),
            (dict(benchmark={}, runner_cpus=dict(tvm=True)), None),
            (dict(benchmark={}), "tvm=2,tvm=4"),
            (dict(benchmark={}), "tvm=no"),
            (dict(benchmark={}), "tvm=0"),
        ]:
            with (
                self.subTest(cfg=cfg, override=override),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                resolve_runner_cpus(
                    parser, cfg, ["tvm"], overrides=override, allowed_cpus={2, 4}
                )

    def test_tvm_automatic_binding_is_disabled_and_affinity_drift_is_rejected(self):
        self.assertEqual(
            environment({"threads": {"TVM_BIND_THREADS": "1"}})["TVM_BIND_THREADS"], "0"
        )
        with (
            patch("common.host.is_macos", return_value=False),
            patch("common.host.os.sched_getaffinity", return_value={2}, create=True),
        ):
            self.assertEqual(check_affinity(2), [2])
        with (
            patch("common.host.is_macos", return_value=False),
            patch("common.host.os.sched_getaffinity", return_value={0}, create=True),
            self.assertRaisesRegex(RuntimeError, r"expected \[2\], observed \[0\]"),
        ):
            check_affinity(2)

    def entry(self):
        spec = importlib.util.spec_from_file_location("entry", ROOT / "run.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        return entry

    def test_parallel_dispatch_forwards_per_runner_cpu_and_separate_logs(self):
        entry = self.entry()
        with tempfile.TemporaryDirectory() as tmp:
            flags = [
                "run.py",
                "--backends",
                "openblas",
                "--parallel",
                "--runner-cpus",
                "pytorch=2,openblas=4",
                "--output",
                tmp,
            ]
            with (
                patch.object(sys, "argv", flags),
                patch("common.host.cpu_model", return_value="Apple M2 Pro"),
                patch("common.config.available_cpus", return_value={2, 4}),
                patch.object(entry, "run_child") as run,
                patch.object(entry, "summarize") as summary,
            ):
                entry.main()
            cpus = {}
            for call in run.call_args_list:
                cmd = call.args[0]
                cpus[Path(cmd[cmd.index("--output") + 1]).name] = int(
                    cmd[cmd.index("--cpu") + 1]
                )
                self.assertIn("stdout", call.kwargs)
            self.assertEqual(cpus, dict(pytorch=2, openblas=4))
            for backend in cpus:
                self.assertTrue((Path(tmp) / (backend + ".log")).is_file())
            self.assertEqual(
                summary.call_args.args[0],
                [
                    Path(tmp).resolve() / b / "results.json"
                    for b in ["pytorch", "openblas"]
                ],
            )

    def test_all_cpu_assignments_validated_before_any_launch(self):
        entry = self.entry()
        for flags in [
            ["--backends", "tvm", "--runner-cpus", "pytorch=2,tvm=0"],
            ["--backends", "openblas", "--parallel", "--cpu", "2"],
        ]:
            with (
                self.subTest(flags=flags),
                patch.object(sys, "argv", ["run.py", *flags]),
                patch("common.host.cpu_model", return_value="Apple M2 Pro"),
                patch("common.config.available_cpus", return_value={2, 4}),
                patch.object(entry, "run_child") as run,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                entry.main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
