"""Profiler annotations through existing callback hooks; no extra CUDA synchronization."""

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


class PhaseAudit(Callback):
    def __init__(self):
        super().__init__()
        self.active = None

    def _begin(self, name):
        if dist.get_rank() == 0:
            if self.active is not None:
                raise RuntimeError("Overlapping phase annotations")
            self.active = torch.profiler.record_function("kernel_list." + name)
            self.active.__enter__()

    def _end(self):
        if self.active is not None:
            self.active.__exit__(None, None, None)
            self.active = None

    def on_before_forward(self, iteration=0):
        self._begin("forward")

    def on_after_forward(self, iteration=0):
        self._end()

    def on_before_backward(self, model, loss, iteration=0):
        self._begin("backward")

    def on_after_backward(self, model, iteration=0):
        self._end()

    def on_before_optimizer_step(
        self, model, optimizer, scheduler, grad_scaler, iteration=0
    ):
        self._begin("optimizer")

    def on_before_zero_grad(self, model, optimizer, scheduler, iteration=0):
        self._end()
