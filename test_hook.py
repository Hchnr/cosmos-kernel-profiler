"""CPU-only schedule and rank-selection checks."""

import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from profile_hook import maybe_enable_profiling, schedule_parameters


class HookTest(unittest.TestCase):
    def settings(self):
        return NS(
            enable_profiling=True,
            target_ranks=[0],
            profile_freq=25,
            profile_warmup=2,
            profile_active=3,
        )

    def test_single_local_window(self):
        schedule = torch.profiler.schedule(**schedule_parameters(self.settings()))
        action = torch.profiler.ProfilerAction
        self.assertEqual(
            [schedule(i) for i in (0, 19, 20, 21, 22, 23, 24, 25, 50)],
            [
                action.NONE,
                action.NONE,
                action.WARMUP,
                action.WARMUP,
                action.RECORD,
                action.RECORD,
                action.RECORD_AND_SAVE,
                action.NONE,
                action.NONE,
            ],
        )

    def test_non_target_rank_does_not_create_profiler(self):
        config = NS(trainer=NS(profiling=self.settings()))
        with (
            patch("profile_hook.dist.is_initialized", return_value=True),
            patch("profile_hook.dist.get_rank", return_value=1),
            patch("profile_hook.torch.profiler.profile") as create,
        ):
            with maybe_enable_profiling(config, global_step=12600) as profiler:
                self.assertIsNone(profiler)
            create.assert_not_called()

    def test_unexpected_resume_is_rejected(self):
        config = NS(trainer=NS(profiling=self.settings()))
        with (
            patch.dict(os.environ, {"COSMOS_KERNEL_START_STEP": "12600"}),
            patch("profile_hook.dist.is_initialized", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "Unexpected resume"):
                with maybe_enable_profiling(config, global_step=0):
                    self.fail("entered incorrect resume")

    def test_invalid_schedule(self):
        settings = self.settings()
        settings.profile_freq = 4
        with self.assertRaises(ValueError):
            schedule_parameters(settings)


if __name__ == "__main__":
    unittest.main()
