#!/usr/bin/env python3
"""Compare two full-validation diagnostic reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "composite_score",
    "detection_f1",
    "detection_precision",
    "detection_recall",
    "classification_accuracy",
    "region_cer",
    "page_cer",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def fmt(value):
    return "n/a" if value is None else f"{value:.6f}"


def main():
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    lines = [
        "# Validation Run Comparison",
        "",
        f"Baseline: `{args.baseline}`",
        f"Candidate: `{args.candidate}`",
        "",
    ]
    for stage in ("gt_ocr", "e2e"):
        old = (baseline.get(stage) or {}).get("official_score") or {}
        new = (candidate.get(stage) or {}).get("official_score") or {}
        if not old or not new:
            continue
        lines.extend(
            [
                f"## {stage}",
                "",
                "| Metric | Baseline | Candidate | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric in METRICS:
            old_value = old.get(metric)
            new_value = new.get(metric)
            delta = (
                new_value - old_value
                if old_value is not None and new_value is not None
                else None
            )
            lines.append(
                f"| {metric} | {fmt(old_value)} | {fmt(new_value)} | "
                f"{fmt(delta)} |"
            )
        lines.extend(
            [
                "",
                "### Region CER by type",
                "",
                "| Type | Baseline | Candidate | Delta |",
                "|---|---:|---:|---:|",
            ]
        )
        old_types = baseline[stage]["analysis"]["by_type"]
        new_types = candidate[stage]["analysis"]["by_type"]
        for region_type in sorted(set(old_types) | set(new_types)):
            old_value = (old_types.get(region_type) or {}).get("region_cer")
            new_value = (new_types.get(region_type) or {}).get("region_cer")
            delta = (
                new_value - old_value
                if old_value is not None and new_value is not None
                else None
            )
            lines.append(
                f"| {region_type} | {fmt(old_value)} | {fmt(new_value)} | "
                f"{fmt(delta)} |"
            )
        lines.append("")
    text = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
