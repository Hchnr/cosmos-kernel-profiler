"""Reject incomplete or non-VLM captures before archiving them as deliverables."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from validate_vlm_run import validate_vlm_run


class VLMValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "batches").mkdir()
        self.write("metadata.json", {"returncode": 0, "run_id": "test"})
        self.write(
            "profile_window.json",
            {"rank": 0, "active_completed_steps": [1], "grad_accum_iter": 3},
        )
        config = {
            "model": {
                "config": {
                    "vision_gen": False,
                    "action_gen": False,
                    "load_vision_tokenizer": False,
                    "load_vlm_visual": True,
                    "train_vlm_visual": True,
                    "predict_text_tokens": True,
                    "und_loss_reduction": "sample_mean",
                    "und_loss_normalization": "accumulation_window",
                    "parallelism": {
                        "data_parallel_shard_degree": 8,
                        "data_parallel_replicate_degree": 1,
                        "context_parallel_shard_degree": 1,
                    },
                    "compile": {"enabled": False},
                    "activation_checkpointing": {"mode": "full"},
                    "vlm_config": {"pretrained_weights": {"enabled": True}},
                }
            },
            "checkpoint": {"load_training_state": False},
        }
        (self.run / "config.yaml").write_text(yaml.safe_dump(config))
        self.rows = [{"record_type": "trace_start", "iteration": 0, "world_size": 8}]
        self.rows += [
            {
                "record_type": "microbatch",
                "iteration": 0,
                "data": {
                    "packed_sequence": {
                        "has_generation_vision": False,
                        "has_generation_action": False,
                        "sequence_length": 36000,
                        "sample_lens": [36000],
                    },
                    "vlm_inputs": [{"pixel_values": {"shape": [1024, 1536]}}],
                },
                "output": {"cross_entropy_loss": {}},
            }
            for _ in range(3)
        ]
        self.rows += [
            {
                "record_type": "optimizer_step",
                "iteration": 1,
                "loss": 1.0,
                "loss_window": 1.0,
                "loss_window_weight": 3.0,
            },
            {"record_type": "trace_end", "iteration": 1},
        ]
        for rank in range(8):
            self.write_rows(rank)
            self.write(
                f"batches/rank_{rank:05d}_parameters.json",
                [{"name": "net.visual.weight", "requires_grad": True}],
            )
        self.quality = {
            "kernel_name_count_mismatches": [],
            "excess_report_kernel_names": [],
            "compile_timed_ranges": {},
            "phase_annotations": {
                "kernel_list.forward": 3,
                "kernel_list.backward": 3,
                "kernel_list.optimizer": 1,
            },
            "report_kernel_calls": 100,
            "unique_report_kernels": 10,
        }
        self.mock = patch("validate_vlm_run.validate", return_value=self.quality)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value))

    def write_rows(self, rank):
        (self.run / "batches" / f"rank_{rank:05d}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in self.rows)
        )

    def test_complete_capture(self):
        result = validate_vlm_run(self.run)
        self.assertEqual(result["captured_sample_count"], 3)
        self.assertEqual(len(result["ranks"]), 8)

    def test_generation_payload_rejected(self):
        self.rows[1]["data"]["packed_sequence"]["has_generation_vision"] = True
        self.write_rows(3)
        with self.assertRaisesRegex(ValueError, "Generation payload"):
            validate_vlm_run(self.run)

    def test_unsupervised_update_rejected(self):
        self.rows[-2]["loss_window_weight"] = 0
        self.write_rows(7)
        with self.assertRaisesRegex(ValueError, "unsupervised optimizer"):
            validate_vlm_run(self.run)

    def test_incomplete_rank_rejected(self):
        self.rows.pop()
        self.write_rows(6)
        with self.assertRaisesRegex(ValueError, "Rank 6 incomplete"):
            validate_vlm_run(self.run)

    def test_kernel_coverage_mismatch_rejected(self):
        self.quality["kernel_name_count_mismatches"] = [{"kernel_name": "missing"}]
        with self.assertRaisesRegex(ValueError, "kernel counts differ"):
            validate_vlm_run(self.run)

    def test_diagnostic_mode_preserves_coverage_failure(self):
        gaps = [{"kernel_name": "missing", "trace_count": 10, "report_count": 9}]
        self.quality["kernel_name_count_mismatches"] = gaps
        result = validate_vlm_run(self.run, report_coverage_gaps=True)
        self.assertEqual(result["distributed_training"], "passed")
        self.assertEqual(result["kernel_count_coverage"], "incomplete")
        self.assertEqual(result["kernel_name_count_mismatches"], gaps)

    def test_diagnostic_mode_still_rejects_incomplete_training(self):
        self.rows.pop()
        self.write_rows(6)
        with self.assertRaisesRegex(ValueError, "Rank 6 incomplete"):
            validate_vlm_run(self.run, report_coverage_gaps=True)


if __name__ == "__main__":
    unittest.main()
