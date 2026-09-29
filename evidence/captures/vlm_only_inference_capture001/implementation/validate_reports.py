"""Validate FlagScale tables and quantify their coverage of a Kineto trace."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from profiler_reports import (
    DETAIL_FIELDS,
    OPERATOR_FIELDS,
    SUMMARY_FIELDS,
    _is_communication,
)


def read_csv(path, fields):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"Invalid header in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty report: {path}")
    return rows


def key(row):
    return tuple(
        row[name] for name in ("custom_operator", "execution_operator", "kernel_name")
    )


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def validate(directory, rank=0):
    directory = Path(directory)
    trace_path = directory / f"rank-{rank}.json.gz"
    with gzip.open(trace_path, "rt") as stream:
        trace = json.load(stream)
    events = trace.get("traceEvents", [])
    cpu = [event for event in events if event.get("cat") == "cpu_op"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    if not cpu or not kernels:
        raise ValueError("Trace must contain both CPU operators and CUDA kernels")
    detail_path = directory / f"rank-{rank}_kernel_details_report.csv"
    summary_path = directory / f"rank-{rank}_kernel_summary.csv"
    operator_path = directory / f"rank-{rank}_operator_list.csv"
    details = read_csv(detail_path, DETAIL_FIELDS)
    summary = read_csv(summary_path, SUMMARY_FIELDS)
    operators = read_csv(operator_path, OPERATOR_FIELDS)
    grouped = defaultdict(lambda: [0, 0.0, 0])
    for row in details:
        count, duration = int(row["kernel_event_count"]), float(row["kernel_time_us"])
        if count <= 0 or duration < 0 or not math.isfinite(duration):
            raise ValueError(f"Invalid detail count/duration: {row}")
        json.loads(row["input_shapes"])
        json.loads(row["input_dtypes"])
        bucket = grouped[key(row)]
        bucket[0] += count
        bucket[1] += duration
        bucket[2] += 1
    summary_keys = [key(row) for row in summary]
    operator_keys = [key(row) for row in operators]
    if len(set(summary_keys)) != len(summary_keys) or len(set(operator_keys)) != len(
        operator_keys
    ):
        raise ValueError("Duplicate summary/operator mappings")
    if set(grouped) != set(summary_keys) or set(summary_keys) != set(operator_keys):
        raise ValueError("Detail, summary and operator mapping sets differ")
    total_time = sum(float(row["kernel_time_us"]) for row in summary)
    if total_time <= 0:
        raise ValueError("No positive kernel time")
    ids, inverse_ids = {}, {}
    for row in operators:
        pair = key(row)[:2]
        identifier = int(row["operator_id"])
        if (
            ids.setdefault(pair, identifier) != identifier
            or inverse_ids.setdefault(identifier, pair) != pair
        ):
            raise ValueError("Inconsistent operator IDs")
    for row in summary:
        count, duration, variants = grouped[key(row)]
        if int(row["kernel_call_count"]) != count:
            raise ValueError(f"Call count mismatch: {key(row)}")
        if abs(float(row["kernel_time_us"]) - duration) > (variants + 1) * 0.00051:
            raise ValueError(f"Time mismatch: {key(row)}")
        percent = 100 * float(row["kernel_time_us"]) / total_time
        if row["percent"] == "<0.001%":
            if not 0 <= percent < 0.00101:
                raise ValueError("Invalid small percentage")
        elif abs(float(row["percent"].removesuffix("%")) - percent) > 0.00052:
            raise ValueError(f"Percentage mismatch: {key(row)}")

    raw_counts, raw_times = Counter(), Counter()
    communication_count, communication_time = 0, 0.0
    for event in kernels:
        name, duration = str(event["name"]), float(event.get("dur", 0))
        if _is_communication("", name):
            communication_count += 1
            communication_time += duration
        else:
            raw_counts[name] += 1
            raw_times[name] += duration
    report_counts, report_times = Counter(), Counter()
    for row in summary:
        report_counts[row["kernel_name"]] += int(row["kernel_call_count"])
        report_times[row["kernel_name"]] += float(row["kernel_time_us"])
    unmatched = [
        {
            "kernel_name": name,
            "trace_count": count,
            "report_count": report_counts[name],
            "trace_time_us": raw_times[name],
            "report_time_us": report_times[name],
        }
        for name, count in raw_counts.items()
        if count != report_counts[name]
    ]
    excess = [name for name, count in report_counts.items() if count > raw_counts[name]]
    missing_metadata = {}
    for column in ("input_shapes", "input_dtypes"):
        missing = [row for row in details if not json.loads(row[column])]
        missing_metadata[column] = {
            "variant_rows": len(missing),
            "kernel_calls": sum(int(row["kernel_event_count"]) for row in missing),
            "kernel_time_us": sum(float(row["kernel_time_us"]) for row in missing),
        }
    annotations = Counter(
        event["name"]
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("kernel_list.")
    )
    compile_ranges = Counter(
        event["name"]
        for event in events
        if event.get("ph") == "X"
        and "dynamo_timed" in str(event.get("name", ""))
        and any(
            marker in str(event.get("name", ""))
            for marker in (
                "compile_inner",
                "compile_fx",
                "GraphLowering",
                "Scheduler.codegen",
            )
        )
    )
    result = {
        "table_validation": "passed",
        "rank": rank,
        "detail_variants": len(details),
        "operator_kernel_mappings": len(summary),
        "operator_groups": len(ids),
        "unique_report_kernels": len(report_counts),
        "trace_cpu_ops": len(cpu),
        "trace_kernel_events": len(kernels),
        "trace_compute_kernel_events": sum(raw_counts.values()),
        "trace_compute_kernel_time_us": sum(raw_times.values()),
        "trace_communication_kernel_events": communication_count,
        "trace_communication_kernel_time_us": communication_time,
        "trace_memory_events": sum(
            event.get("cat") in {"gpu_memcpy", "gpu_memset"} for event in events
        ),
        "report_kernel_calls": sum(report_counts.values()),
        "report_kernel_time_us": total_time,
        "missing_metadata": missing_metadata,
        "kernel_name_count_mismatches": sorted(
            unmatched, key=lambda row: -row["trace_time_us"]
        ),
        "excess_report_kernel_names": excess,
        "phase_annotations": dict(annotations),
        "compile_timed_ranges": dict(compile_ranges),
        "top_kernel_mappings": summary[:20],
        "files": [
            fingerprint(path)
            for path in (trace_path, detail_path, summary_path, operator_path)
        ],
        "coverage_note": "GPU classification by kernel name; CPU communication-parent filtering can also exclude kernels. "
        "Count agreement is a coverage check, not independent proof of exact event attribution.",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quality = validate(args.directory, args.rank)
    args.output.write_text(json.dumps(quality, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                k: quality[k]
                for k in (
                    "table_validation",
                    "detail_variants",
                    "operator_groups",
                    "report_kernel_calls",
                )
            }
        )
    )
