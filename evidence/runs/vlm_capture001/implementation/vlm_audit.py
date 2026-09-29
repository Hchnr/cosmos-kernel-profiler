"""Record pure VLM workload metadata through existing training callbacks."""

import json
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


def describe(value):
    """Only inspect tensor metadata; never copy tensor contents off the GPU."""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "sample_lens"):
        return {
            "sample_lens": list(value.sample_lens),
            "sequence_length": value.sequence_length,
            "label_ids": describe(value.label_ids),
            "has_generation_vision": value.vision is not None,
            "has_generation_action": value.action is not None,
        }
    if hasattr(value, "has_vision") and hasattr(value, "has_action"):
        return {
            "has_text": value.has_text,
            "has_vision": value.has_vision,
            "has_action": value.has_action,
        }
    return {"type": type(value).__name__}


class VLMAudit(Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.stream = None

    def write(self, record_type, iteration, **fields):
        self.stream.write(
            json.dumps(
                {
                    "record_type": record_type,
                    "iteration": iteration,
                    "rank": dist.get_rank(),
                    **fields,
                }
            )
            + "\n"
        )
        self.stream.flush()

    def on_train_start(self, model, iteration=0):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream = (self.output_dir / f"rank_{dist.get_rank():05d}.jsonl").open("x")
        parameters = [
            {
                "name": name,
                "shape": list(param.shape),
                "requires_grad": param.requires_grad,
            }
            for name, param in model.named_parameters()
        ]
        (self.output_dir / f"rank_{dist.get_rank():05d}_parameters.json").write_text(
            json.dumps(parameters, indent=2) + "\n"
        )
        self.write("trace_start", iteration, world_size=dist.get_world_size())

    def on_training_step_batch_end(
        self, model, data_batch, output_batch, loss, iteration=0
    ):
        self.write(
            "microbatch",
            iteration,
            data=describe(data_batch),
            output=describe(output_batch),
        )

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        self.write(
            "optimizer_step",
            iteration,
            loss=float(loss.detach()),
            loss_window=float(output_batch["loss_window"]),
            loss_window_weight=float(output_batch["loss_window_weight"]),
        )

    def on_train_end(self, model, iteration=0):
        self.write("trace_end", iteration)
        self.stream.close()
