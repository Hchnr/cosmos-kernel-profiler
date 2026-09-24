"""Validate the real iter_speed log and nested LoadBalanceTrace data contracts."""

import json
import tempfile
import unittest
from pathlib import Path

from validate_run import validate_run


class RunValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        (self.run / "batches").mkdir()
        documents = {
            "metadata.json": {"returncode": 0, "run_id": "fixture"},
            "profile_window.json": {
                "active_completed_steps": [12601, 12602],
                "grad_accum_iter": 2,
                "global_start_step": 12600,
                "rank": 0,
            },
            "report_quality.json": {
                "phase_annotations": {
                    "kernel_list.forward": 4,
                    "kernel_list.backward": 4,
                    "kernel_list.optimizer": 2,
                }
            },
        }
        for name, value in documents.items():
            (self.run / name).write_text(json.dumps(value))
        for rank in range(8):
            rows = [{"record_type": "trace_start", "iteration": 12600}]
            for step in (12600, 12601):
                for micro in range(2):
                    rows.append(
                        {
                            "record_type": "microbatch",
                            "iteration": step,
                            "microbatch_in_iteration": micro,
                            "world_size": 8,
                            "dp_shard": 8,
                            "dp_replicate": 1,
                            "cp_size": 1,
                            "output": {
                                "physical_und_token_length": 3072,
                                "physical_gen_token_length": 98304,
                            },
                            "data": {
                                "dataset_names": ["usr:1"],
                                "source_dataset_names": [],
                                "vision_input_mode": ["offline"],
                                "sample_count_from_ids": 1,
                                "local_sample_ids": ["usr:sample"],
                                "sequence_plans": [{"vision_conditioning": "t2v"}],
                                "video_latent_shapes": [[[48, 150, 40, 40]]],
                            },
                        }
                    )
            rows.append({"record_type": "trace_end", "iteration": 12602})
            (self.run / "batches" / f"rank_{rank:05d}.jsonl").write_text(
                "\n".join(map(json.dumps, rows))
            )
        self.log = "\n".join(
            f"[INFO] [RANK {rank}] Iteration {step}: Loss: 0.1356 | Time: 40.60s | Warmup: 2/50"
            for rank in range(8)
            for step in (12601, 12602)
        )
        (self.run / "train.log").write_text(self.log)

    def test_nested_data_and_all_rank_losses(self):
        result = validate_run(self.run)
        self.assertEqual(result["captured_samples"], 4)
        self.assertEqual(result["captured_dataset_label_occurrences"], {"usr:1": 4})
        self.assertEqual(result["captured_conditions"], {"t2v": 4})
        self.assertEqual(result["captured_sample_id_prefix_counts"], {"usr": 4})
        self.assertEqual(len(result["active_losses_by_rank"]), 8)

    def test_nonfinite_other_rank_rejected(self):
        (self.run / "train.log").write_text(
            self.log.replace(
                "[RANK 7] Iteration 12602: Loss: 0.1356",
                "[RANK 7] Iteration 12602: Loss: nan",
            )
        )
        with self.assertRaisesRegex(ValueError, "rank 7"):
            validate_run(self.run)

    def test_missing_rank_loss_rejected(self):
        (self.run / "train.log").write_text("\n".join(self.log.splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, "rank 7"):
            validate_run(self.run)


if __name__ == "__main__":
    unittest.main()
