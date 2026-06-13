#!/usr/bin/env python3
"""Build a local visual viewer for RUKOPYS predictions.

The generated viewer overlays predicted bounding boxes on the real test
images and exposes OCR text, class, source, geometry, suspicious flags, and
differences between the raw submission and postprocessed candidates.

No ML dependencies or GPU are required.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_SUBMISSIONS = {
    "raw": Path(
        "rukopys_full_063/input/datasets/ngovietan/"
        "htr-output-kaggle-full-ver1-061/submission.csv"
    ),
    "v2": Path("analysis_outputs/postprocess_v2/submission_postprocess_v2.csv"),
    "v3_safe": Path("analysis_outputs/postprocess_v3/submission_v3_safe.csv"),
    "v3_balanced": Path("analysis_outputs/postprocess_v3/submission_v3_balanced.csv"),
    "v3_text_strong": Path(
        "analysis_outputs/postprocess_v3/submission_v3_text_strong.csv"
    ),
    "v3_dedup_strong": Path(
        "analysis_outputs/postprocess_v3/submission_v3_dedup_strong.csv"
    ),
}

LONG_CHAR_RUN = re.compile(r"(.)\1{11,}", re.DOTALL)
LONG_TOKEN_RUN = re.compile(r"\b(\w{2,})(?:\s+\1){4,}\b", re.IGNORECASE)
DASH_RUN = re.compile(r"(?:\s*[—–_-]\s*){8,}")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_submission(path: Path) -> dict[str, list[dict]]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                regions = json.loads(row.get("regions", "[]"))
            except json.JSONDecodeError:
                regions = []
            result[row["image"]] = regions if isinstance(regions, list) else []
    return result


def area(box: list) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: list, b: list) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def iou(a: list, b: list) -> float:
    inter = intersection(a, b)
    return inter / max(area(a) + area(b) - inter, 1e-9)


def ioa(a: list, b: list) -> float:
    return intersection(a, b) / max(min(area(a), area(b)), 1e-9)


def region_key(region: dict) -> str:
    box = region.get("bbox", [0, 0, 0, 0])
    return f"{','.join(str(int(x)) for x in box)}|{region.get('type', 'handwritten')}"


def text_flags(text: str) -> list[str]:
    flags = []
    if len(text) >= 180:
        flags.append("long_text")
    if LONG_CHAR_RUN.search(text):
        flags.append("character_loop")
    if LONG_TOKEN_RUN.search(text):
        flags.append("token_loop")
    if DASH_RUN.search(text):
        flags.append("dash_loop")
    return flags


def annotate_regions(
    regions: list[dict],
    width: int,
    height: int,
    raw_lookup: dict[str, dict],
) -> list[dict]:
    annotated = []
    overlaps = [set() for _ in regions]
    contained = [set() for _ in regions]

    for i, first in enumerate(regions):
        for j, second in enumerate(regions[i + 1:], i + 1):
            first_box = first.get("bbox", [0, 0, 0, 0])
            second_box = second.get("bbox", [0, 0, 0, 0])
            if iou(first_box, second_box) >= 0.80:
                overlaps[i].add(j)
                overlaps[j].add(i)
            if ioa(first_box, second_box) >= 0.95:
                contained[i].add(j)
                contained[j].add(i)

    for index, region in enumerate(regions):
        box = [int(x) for x in region.get("bbox", [0, 0, 0, 0])]
        text = str(region.get("text", "") or "")
        key = region_key(region)
        raw = raw_lookup.get(key)
        flags = text_flags(text)
        if overlaps[index]:
            flags.append("overlap_iou80")
        if contained[index]:
            flags.append("containment_ioa95")
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            flags.append("out_of_bounds")
        if raw and str(raw.get("text", "") or "") != text:
            flags.append("text_changed")

        annotated.append({
            "id": index,
            "bbox": box,
            "type": region.get("type", "handwritten"),
            "text": text,
            "raw_text": str(raw.get("text", "") or "") if raw else None,
            "flags": flags,
            "overlap_with": sorted(overlaps[index]),
            "contained_with": sorted(contained[index]),
        })
    return annotated


def build_dataset(
    metadata_path: Path,
    submissions: dict[str, Path],
) -> dict:
    metadata = read_jsonl(metadata_path)
    predictions = {
        name: read_submission(path)
        for name, path in submissions.items()
        if path.exists()
    }
    if "raw" not in predictions:
        raise FileNotFoundError("The raw submission is required.")

    candidates = list(predictions)
    images = []
    totals = Counter()

    for item in metadata:
        image = Path(item["file_name"]).name
        width = int(item["image_width"])
        height = int(item["image_height"])
        raw_regions = predictions["raw"].get(image, [])
        raw_lookup = {region_key(region): region for region in raw_regions}
        per_candidate = {}

        for name in candidates:
            regions = predictions[name].get(image, [])
            annotated = annotate_regions(regions, width, height, raw_lookup)
            present_keys = {region_key(region) for region in regions}
            removed = [
                {
                    "bbox": [int(x) for x in region.get("bbox", [0, 0, 0, 0])],
                    "type": region.get("type", "handwritten"),
                    "text": str(region.get("text", "") or ""),
                }
                for region in raw_regions
                if region_key(region) not in present_keys
            ]
            flag_count = sum(bool(region["flags"]) for region in annotated)
            per_candidate[name] = {
                "regions": annotated,
                "removed_from_raw": removed,
                "flag_count": flag_count,
            }
            totals[f"{name}_regions"] += len(annotated)
            totals[f"{name}_flags"] += flag_count
            totals[f"{name}_removed"] += len(removed)

        images.append({
            "image": image,
            "width": width,
            "height": height,
            "source": item.get("source", "unknown"),
            "predictions": per_candidate,
        })

    return {
        "candidates": candidates,
        "images": images,
        "totals": dict(totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("rukopys_raw/test/metadata.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs/visual_diagnostic"),
    )
    args = parser.parse_args()

    submissions = {
        name: path for name, path in DEFAULT_SUBMISSIONS.items() if path.exists()
    }
    dataset = build_dataset(args.metadata, submissions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.out_dir / "viewer_data.json"
    data_path.write_text(
        json.dumps(dataset, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(json.dumps({
        "metadata": str(args.metadata),
        "data": str(data_path),
        "images": len(dataset["images"]),
        "candidates": dataset["candidates"],
        "totals": dataset["totals"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
