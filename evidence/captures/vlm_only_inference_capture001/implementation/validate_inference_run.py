"""Validate completed inference captures and their real generated outputs (CPU only)."""

import argparse
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text())


def validate_run(root):
    completion = read(root / "completion.json")
    assert completion["status"] == "complete"
    quality = read(root / "report_quality.json")
    assert quality["table_validation"] == "passed"
    assert not quality["kernel_name_count_mismatches"]
    assert not quality["excess_report_kernel_names"]
    assert quality["report_kernel_calls"] == quality["trace_compute_kernel_events"]
    assert not quality["compile_timed_ranges"], (
        "Compilation occurred inside active window"
    )
    assert quality["phase_annotations"]["kernel_list.inference_request"] == 1
    result = {
        "mode": completion["mode"],
        "report_validation": "passed",
        "kernel_names": quality["unique_report_kernels"],
        "compute_kernel_calls": quality["report_kernel_calls"],
    }
    if completion["mode"] == "vlm_only":
        generation = read(root / "generation.json")
        assert generation["warmup_matches_capture"]
        assert generation["phase_calls"] == {
            "prefill": 1,
            "decode": generation["generated_tokens"] - 1,
            "vision": 1,
        }
        assert 0 < generation["generated_tokens"] <= 128
        result["generation"] = generation
    else:
        videos = []
        batches = []
        for batch_file in sorted(
            (root / "batches").glob("rank-*/resolved_samples.json")
        ):
            samples = read(batch_file)
            assert len(samples) == 4
            batch = {"rank": batch_file.parent.name, "samples": []}
            for sample in samples:
                assert sample["model_mode"] == "text2video"
                assert (
                    not sample["enable_sound"]
                    and not sample["native_prompt_upsampling"]
                )
                assert sample["resolution"] == "480" and sample["aspect_ratio"] == "4,3"
                assert (
                    sample["num_steps"] == 50
                    and sample["guidance"] == 5
                    and sample["shift"] == 10
                )
                assert sample["seed"] == 2026090900 and sample["fps"] == 30
                assert 93 <= sample["num_frames"] <= 245
                out = Path(sample["output_dir"])
                metadata = read(out / "sample_outputs.json")
                assert metadata.get("status", "success") == "success", metadata.get(
                    "status"
                )
                video = out / "vision.mp4"
                info = json.loads(
                    subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_streams",
                            "-of",
                            "json",
                            str(video),
                        ],
                        text=True,
                    )
                )
                streams = info["streams"]
                assert len(streams) == 1 and streams[0]["codec_type"] == "video"
                stream = streams[0]
                assert stream["width"] == 736 and stream["height"] == 544
                assert int(stream["nb_frames"]) == sample["num_frames"]
                assert stream["r_frame_rate"] == "30/1"
                item = {
                    "name": sample["name"],
                    "num_frames": sample["num_frames"],
                    "width": stream["width"],
                    "height": stream["height"],
                    "vision_tokens": ((sample["num_frames"] - 1) // 4 + 1) * 23 * 17,
                }
                batch["samples"].append(item)
                videos.append(str(video.relative_to(root)))
            batch["packed_vision_tokens_per_cfg_branch"] = sum(
                x["vision_tokens"] for x in batch["samples"]
            )
            batches.append(batch)
        assert len(batches) == 4 and len(videos) == 16
        result["batches"] = batches
        result["videos"] = videos
        result["token_formula"] = (
            "((frames-1)//4+1) * (736//32) * (544//32); VAE 4x16x16 and patch2; excludes text tokens"
        )
    result["status"] = "passed"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate_run(args.directory.resolve())
    (args.directory / "inference_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"videos", "batches", "generation"}
            }
        )
    )
