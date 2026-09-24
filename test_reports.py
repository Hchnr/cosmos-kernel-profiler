"""CPU-only invariants; unittest avoids repository pytest GPU fixtures."""

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from profiler_reports import export_kernel_reports
from validate_reports import validate


class ReportsTest(unittest.TestCase):
    def build(self, root):
        trace = root / "rank-0.json.gz"
        events = []
        for size, count in ((16, 2), (32, 1)):
            for index in range(count):
                events.append(
                    {
                        "cat": "cpu_op",
                        "ph": "X",
                        "name": "aten::mm",
                        "pid": 1,
                        "tid": 1,
                        "ts": len(events) * 100,
                        "dur": 20,
                        "args": {"Input Dims": [[size, size]], "Input type": ["float"]},
                    }
                )
                events.append({"cat": "kernel", "name": "gemm", "dur": 5 + index})
        with gzip.open(trace, "wt") as stream:
            json.dump({"traceEvents": events}, stream)
        prof_events = [
            NS(
                key="aten::mm",
                input_shapes=[[16, 16]],
                kernels=[NS(name="gemm", duration=5)],
            ),
            NS(
                key="aten::mm",
                input_shapes=[[16, 16]],
                kernels=[NS(name="gemm", duration=6)],
            ),
            NS(
                key="aten::mm",
                input_shapes=[[32, 32]],
                kernels=[NS(name="gemm", duration=5)],
            ),
            NS(key="c10d::allreduce", kernels=[NS(name="ncclAllReduce", duration=7)]),
            NS(
                key="annotation",
                is_user_annotation=True,
                kernels=[NS(name="gemm", duration=100)],
            ),
        ]
        export_kernel_reports(prof_events, root, 0, trace)
        return validate(root)

    def test_variants_filtering_and_trace_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            quality = self.build(Path(temp))
            self.assertEqual(quality["detail_variants"], 2)
            self.assertEqual(quality["operator_groups"], 1)
            self.assertEqual(quality["report_kernel_calls"], 3)
            self.assertEqual(quality["report_kernel_time_us"], 16)
            self.assertEqual(quality["kernel_name_count_mismatches"], [])
            self.assertEqual(
                quality["missing_metadata"]["input_dtypes"]["variant_rows"], 0
            )

    def test_count_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.build(root)
            path = root / "rank-0_kernel_summary.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[0]["kernel_call_count"] = "4"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Call count mismatch"):
                validate(root)

    def test_missing_cuda_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with gzip.open(root / "rank-0.json.gz", "wt") as stream:
                json.dump({"traceEvents": []}, stream)
            with self.assertRaisesRegex(ValueError, "CUDA kernels"):
                validate(root)


if __name__ == "__main__":
    unittest.main()
