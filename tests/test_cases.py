import collections
from pathlib import Path
import tempfile
import unittest

from common.cases import OPS, describe, load_cases, matches, parse_shape, select_cases
from common.config import ROOT, load


def bench():
    return load(ROOT / "config/common.toml")["benchmark"]


class Cases(unittest.TestCase):
    def test_automatic_size_and_regularity_tags(self):
        self.assertEqual(
            parse_shape("gemm:64,64,64", 1)["tags"],
            ["custom", "gemm", "regular", "small"],
        )
        self.assertIn("irregular", parse_shape("gemm:49,4608,512", 1)["tags"])
        self.assertIn("medium", parse_shape("gemm:49,4608,512", 1)["tags"])
        self.assertIn("large", parse_shape("gemv:4096,4096", 1)["tags"])
        decode = parse_shape("linear:1,4096,4096", 1)["tags"]
        self.assertTrue({"vector", "regular", "large"} <= set(decode))
        # Conv2D regularity looks at output channels, output pixels and the reduction.
        self.assertIn(
            "irregular", parse_shape("conv2d:1,512,7,7,512,3,3,1,1", 1)["tags"]
        )
        self.assertIn("regular", parse_shape("conv2d:1,16,16,16,16,3,3,1,1", 1)["tags"])

    def test_shape_uses_case_file_order(self):
        case = parse_shape("gemm:49x4608x512", 3)
        self.assertEqual(
            (case["name"], case["shape"], case["reps"]),
            ("gemm_49x4608x512", [49, 512, 4608], 3),
        )
        self.assertEqual(parse_shape("conv:1,3,8,8,4,3,3,1,1", 1)["op"], "conv2d")
        for bad in [
            "gemm",
            "gemm:1,2",
            "gemv:0,4",
            "conv2d:1,1,2,2,1,3,3,1,0",
            "mm:1,2,3",
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_shape(bad, 1)

    def test_tag_groups_and_negation(self):
        case = dict(tags=["gemm", "irregular", "medium", "resnet"])
        self.assertTrue(matches(case, None))
        self.assertTrue(matches(case, ["gemm,irregular"]))
        self.assertFalse(matches(case, ["gemm,small"]))
        self.assertTrue(matches(case, ["gemm,small", "resnet"]))  # union of groups
        self.assertFalse(matches(case, ["gemm,!resnet"]))

    def test_selection_by_tags_shapes_and_files(self):
        small_irregular = select_cases(
            bench(), root=ROOT, ops="all", tags=["small,irregular"]
        )
        self.assertTrue(small_irregular)
        self.assertTrue(
            all({"small", "irregular"} <= set(c["tags"]) for c in small_irregular)
        )
        custom = select_cases(
            bench(), root=ROOT, custom=["linear:7,64,96", "gemm:3,5,7"]
        )
        self.assertEqual(
            [c["op"] for c in custom], ["linear", "gemm"]
        )  # --ops does not filter --shape
        with self.assertRaises(ValueError):
            select_cases(bench(), root=ROOT, custom=["gemm:3,5,7"], names="sq64")
        with self.assertRaises(ValueError):
            select_cases(bench(), root=ROOT, ops="all", tags=["no-such-tag"])
        self.assertIn("M=3 K=5 N=7", describe(custom))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.txt"
            path.write_text("gemm a 4 5 6 1 tags=x,y\ngemv b 3 4 1\n")
            loaded = load_cases(path)
            self.assertEqual(
                [set(c["tags"]) & {"x", "y"} for c in loaded], [{"x", "y"}, set()]
            )

    def test_default_suite_covers_every_size_and_regularity(self):
        grid = collections.Counter()
        for case in select_cases(bench(), root=ROOT, ops="all"):
            tags = set(case["tags"])
            grid[
                (
                    case["op"],
                    *(tags & {"small", "medium", "large"}),
                    *(tags & {"regular", "irregular"}),
                )
            ] += 1
        for op in OPS:
            for size in ["small", "medium", "large"]:
                for regularity in ["regular", "irregular"]:
                    with self.subTest(op=op, size=size, regularity=regularity):
                        self.assertGreater(grid[(op, size, regularity)], 0)


if __name__ == "__main__":
    unittest.main()
