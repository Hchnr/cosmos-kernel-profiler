"""Check the actual compiled ragged attention backend before loading the checkpoint."""

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--shared-numerical-check", action="store_true")
    args = parser.parse_args()
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if busy and not args.shared_numerical_check:
        raise RuntimeError("Attention probe requires idle GPUs")
    if args.shared_numerical_check:
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if device not in {str(i) for i in range(8)}:
            raise ValueError(
                "Shared numerical check requires one explicitly selected GPU"
            )
        free = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    device,
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if free < 32768:
            raise RuntimeError("Shared numerical check requires at least 32 GiB free")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.output / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(args.output / "triton")
    import torch

    from cosmos_framework.model.attention import attention
    from cosmos_framework.model.attention.masks import CausalType

    # This tiny correctness test does not collect timings. Limit tensor allocation
    # to 2.5% of one GPU when run alongside unrelated workloads.
    if args.shared_numerical_check:
        torch.cuda.set_per_process_memory_fraction(0.025)
    torch.manual_seed(42)
    results = []
    for causal in (True, False):
        q_offsets = [0, 64, 96, 128] if causal else [0, 48, 80, 112]
        kv_offsets = [0, 64, 96, 128]
        cq = torch.tensor(q_offsets, device="cuda", dtype=torch.int32)
        ck = (
            cq if causal else torch.tensor(kv_offsets, device="cuda", dtype=torch.int32)
        )
        inputs = [
            torch.randn(
                1,
                length,
                heads,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for length, heads in ((q_offsets[-1], 4), (128, 2), (128, 2))
        ]

        def operation(q, k, v):
            return attention(
                q,
                k,
                v,
                is_causal=causal,
                causal_type=CausalType.DontCare if causal else None,
                cumulative_seqlen_Q=cq,
                cumulative_seqlen_KV=ck,
                max_seqlen_Q=max(b - a for a, b in zip(q_offsets, q_offsets[1:])),
                max_seqlen_KV=64,
            )

        compiled = torch.compile(operation, fullgraph=True, dynamic=False)
        output = compiled(*inputs)
        reference_inputs = [x.detach().float().requires_grad_() for x in inputs]
        pieces = []
        for qa, qb, ka, kb in zip(q_offsets, q_offsets[1:], kv_offsets, kv_offsets[1:]):
            q, k, v = reference_inputs
            q = q[:, qa:qb].transpose(1, 2)
            k = k[:, ka:kb].repeat_interleave(2, dim=2).transpose(1, 2)
            v = v[:, ka:kb].repeat_interleave(2, dim=2).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) * (128**-0.5)
            if causal:
                mask = torch.ones(
                    qb - qa, kb - ka, dtype=torch.bool, device="cuda"
                ).triu(1)
                scores = scores.masked_fill(mask, float("-inf"))
            pieces.append((scores.softmax(-1) @ v).transpose(1, 2))
        reference = torch.cat(pieces, dim=1)
        gradient = torch.randn_like(output)
        output.backward(gradient)
        reference.backward(gradient.float())
        torch.testing.assert_close(output.float(), reference, atol=0.025, rtol=0.025)
        for actual, expected in zip(inputs, reference_inputs):
            torch.testing.assert_close(
                actual.grad.float(), expected.grad, atol=0.04, rtol=0.04
            )
        # A second call checks the cached compiled path, including backward.
        compiled(*inputs).float().square().mean().backward()
        torch.cuda.synchronize()
        results.append(
            {
                "causal": causal,
                "q_offsets": q_offsets,
                "kv_offsets": kv_offsets,
                "forward_max_abs_error": (output.float() - reference)
                .abs()
                .max()
                .item(),
                "compiled_forward_backward_passed": True,
            }
        )
    versions = {}
    for name in ("natten", "flash-attn-3-nv"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    summary = {
        "shared_numerical_check": args.shared_numerical_check,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "attention_packages": versions,
        "cases": results,
    }
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
