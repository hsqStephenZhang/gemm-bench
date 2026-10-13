import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from common.config import ROOT
from mimir.checkout import checkout, git, remote_url
from mimir.operators import module


def call(*args):
    return subprocess.check_output(
        list(args), text=True, stderr=subprocess.DEVNULL
    ).strip()


def repository(path):
    call("git", "init", "-b", "master", str(path))
    call("git", "-C", str(path), "config", "user.email", "test@example.invalid")
    call("git", "-C", str(path), "config", "user.name", "Benchmark test")
    (path / "value.txt").write_text("master\n")
    call("git", "-C", str(path), "add", "value.txt")
    call("git", "-C", str(path), "commit", "-m", "master")
    master = git(path, "rev-parse", "HEAD")
    call("git", "-C", str(path), "switch", "-c", "feature")
    (path / "value.txt").write_text("feature\n")
    call("git", "-C", str(path), "commit", "-am", "feature")
    return master, git(path, "rev-parse", "HEAD")


class MimIR(unittest.TestCase):
    def test_remote_defaults_and_urls(self):
        self.assertEqual(
            remote_url("mimir/mimir"), "https://github.com/mimir/mimir.git"
        )
        self.assertEqual(
            remote_url("git@github.com:owner/repo.git"), "git@github.com:owner/repo.git"
        )

    def test_checkout_reuses_clone_for_refs_and_remotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote"
            master, feature = repository(remote)
            source = root / "build/mimir/source"
            with patch("mimir.checkout.subprocess.run", wraps=subprocess.run) as run:
                self.assertEqual(checkout(source, str(remote), "master"), master)
                inode = (source / ".git").stat().st_ino
                self.assertEqual(checkout(source, str(remote), "feature"), feature)
                self.assertEqual((source / ".git").stat().st_ino, inode)
                other = root / "other"
                other_master, _ = repository(other)
                self.assertEqual(checkout(source, str(other), "master"), other_master)
                self.assertEqual((source / ".git").stat().st_ino, inode)
                self.assertEqual(
                    sum(c.args[0][:2] == ["git", "clone"] for c in run.call_args_list),
                    1,
                )
            self.assertEqual((source / "value.txt").read_text(), "master\n")
            (source / "value.txt").write_text("local edit\n")
            with self.assertRaises(ValueError):
                checkout(source, str(other), "feature")
            self.assertEqual((source / "value.txt").read_text(), "local edit\n")

    def test_operator_generation_handles_singleton_conv_and_vector(self):
        code = module(dict(op="conv2d", shape=[1, 3, 9, 11, 5, 3, 2, 2, 1]))
        self.assertIn("tensor.conv R @ (1, 3, 5, 9, 11, 3, 2)", code)
        self.assertIn("«1, 5, 5, 6; F32»", code)
        vector = module(dict(op="gemv", shape=[64, 130]))
        self.assertIn("tensor.dot_product R (1₂, 0₁, (), ())", vector)
        self.assertIn("b: «130; F32»", vector)
        self.assertIn(": «64; F32»", vector)

    def test_remote_ref_and_build_overrides_reach_check_entrypoint(self):
        spec = importlib.util.spec_from_file_location("entry", ROOT / "run.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        flags = [
            "run.py",
            "--backends",
            "mimir",
            "--mimir-remote",
            "owner/repo",
            "--mimir-ref",
            "feature",
            "--build-dir",
            "/tmp/mimir-test-build",
            "--check",
        ]
        with (
            patch.object(sys, "argv", flags),
            patch.object(entry.subprocess, "run") as run,
        ):
            entry.main()
        commands = [c.args[0] for c in run.call_args_list]
        self.assertEqual(len(commands), 2)
        mimir = commands[-1]
        for option, value in [
            ("--remote", "owner/repo"),
            ("--ref", "feature"),
            ("--build-dir", "/tmp/mimir-test-build"),
        ]:
            self.assertEqual(mimir[mimir.index(option) + 1], value)
        self.assertIn("--check", mimir)


if __name__ == "__main__":
    unittest.main()
