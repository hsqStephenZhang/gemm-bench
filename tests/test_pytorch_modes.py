import os
import sys
import unittest
from unittest.mock import patch
from common.config import ROOT, parse_args


class PyTorchModes(unittest.TestCase):
    def test_default_eager_and_inductor_override(self):
        for flags, expected in [
            ([], "eager"),
            (["--pytorch-mode", "inductor"], "inductor"),
        ]:
            with (
                self.subTest(mode=expected),
                patch.dict(
                    os.environ,
                    dict(
                        GEMM_COMMON_CONFIG=str(ROOT / "config/common.toml"),
                        GEMM_BACKEND_CONFIG=str(ROOT / "config/pytorch.toml"),
                    ),
                ),
                patch.object(sys, "argv", ["benchmark.py", "--cases", "pack", *flags]),
                patch("common.config.os.sched_getaffinity", return_value={0}),
                patch("common.config.os.sched_setaffinity"),
            ):
                _, _, settings, _ = parse_args("pytorch")
                self.assertEqual(settings["pytorch"]["mode"], expected)
