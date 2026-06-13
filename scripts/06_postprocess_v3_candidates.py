#!/usr/bin/env python3
"""
Generate several auditable Postprocess V3 submission candidates.

The V2 submission improved the public leaderboard from about 0.61 to 0.66,
but its generic token repetition cleaner can also shorten valid formulas and
tables. V3 starts from the original full submission and separates independent
postprocessing ideas so they can be submitted and compared safely:

- safe: V2-style high-IoU deduplication plus natural-text-only loop cleanup
- balanced: safe plus containment/near-duplicate suppression
- text_strong: balanced geometry with stronger natural-text loop truncation
- dedup_strong: balanced text cleanup with more aggressive duplicate removal

No GPU or model weights are required.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


VALID_TYPES = {
    "handwritten", "printed", "formula", "table",
    "annotation", "image", "graph",
}
NON_SCORABLE_TYPES = {"image", "graph"}
NATURAL_TEXT_TYPES = {"handwritten", "printed", "annotation"}
MATH_HINT_RE = re.compile(r"[0-9=+\-−–—×*/^√∫π%≈≤≥<>()[\]{}]|[A-Za-z]{1,4}\d")
DASH_ONLY_RE = re.compile(r"^[\s\-−–—_.,·|]+$")
TOKEN_EDGE_RE = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LONG_CHAR_RUN_RE = re.compile(r"(.)\1{23,}", re.DOTALL)


@dataclass(frozen=True)
class Profile:
    name: str
    loop_keep: int
    allow_two_repeat_loops: bool
    containment_dedup: bool
    strong_dedup: bool


PROFILES = (
    Profile("safe", loop_keep=2, allow_two_repeat_loops=False,
            containment_dedup=False, strong_dedup=False),
    Profile("balanced", loop_keep=2, allow_two_repeat_loops=False,
            containment_dedup=True, strong_dedup=False),
    Profile("text_strong", loop_keep=1, allow_two_repeat_loops=True,
            containment_dedup=True, strong_dedup=False),
    Profile("dedup_strong", loop_keep=2, allow_two_repeat_loops=False,
            containment_dedup=True, strong_dedup=True),
)


def parse_regions(value: str) -> list[dict]:
    try:
        parsed = json.loads(value) if value else []
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "regions"])
        writer.writeheader()
        writer.writerows(rows)


def dump_regions(regions: list[dict]) -> str:
    result = []
    for region in regions:
        bbox = region.get("bbox", [0, 0, 0, 0])
        result.append({
            "bbox": [int(round(float(v))) for v in bbox],
            "type": str(region.get("type", "handwritten")),
            "text": str(region.get("text", "") or "").strip(),
        })
    return json.dumps(result, ensure_ascii=False)


def box_area(box: list) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def intersection_area(a: list, b: list) -> float:
    width = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    height = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    return width * height


def box_iou(a: list, b: list) -> float:
    inter = intersection_area(a, b)
    return inter / max(box_area(a) + box_area(b) - inter, 1e-9)


def box_ioa_min(a: list, b: list) -> float:
    """Intersection divided by the smaller box area."""
    return intersection_area(a, b) / max(min(box_area(a), box_area(b)), 1e-9)


def normalized_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = ZERO_WIDTH_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.,;:!?\"'`ʼ’‘“”„«»()\[\]{}]", "", text)
    return text.strip()


def compact_text(text: str) -> str:
    return re.sub(r"[^\w]+", "", normalized_text(text), flags=re.UNICODE)


def text_similarity(a: str, b: str) -> float:
    a, b = compact_text(a), compact_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def looks_math(text: str) -> bool:
    return bool(MATH_HINT_RE.search(str(text or "")))


def region_keep_score(region: dict) -> tuple:
    typ = region.get("type", "handwritten")
    text = str(region.get("text", "") or "")
    length = len(text)
    is_math = looks_math(text)

    if typ == "formula" and is_math:
        class_score = 5
    elif typ == "handwritten" and not is_math:
        class_score = 4
    elif typ == "printed" and not is_math:
        class_score = 3
    elif typ == "annotation":
        class_score = 2
    elif typ == "formula":
        class_score = 1
    else:
        class_score = 0

    plausible_length = min(length, 140) - 2 * max(0, length - 180)
    return class_score, plausible_length, box_area(region["bbox"])


def sanitize_text(text: str) -> tuple[str, bool]:
    original = str(text or "")
    cleaned = unicodedata.normalize("NFKC", original)
    cleaned = ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = CONTROL_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n").strip()
    return cleaned, cleaned != original


def token_key(token: str) -> str:
    return TOKEN_EDGE_RE.sub("", token.lower())


def find_consecutive_loop(tokens: list[str], min_repeats: int) -> tuple[int, int, int] | None:
    """Return start, chunk length, repeat count for the strongest exact token loop."""
    keys = [token_key(token) for token in tokens]
    best = None
    best_coverage = 0
    max_chunk = min(14, len(tokens) // min_repeats)

    for chunk_len in range(1, max_chunk + 1):
        for start in range(0, len(tokens) - chunk_len * min_repeats + 1):
            chunk = keys[start:start + chunk_len]
            if not any(chunk):
                continue
            repeats = 1
            while start + (repeats + 1) * chunk_len <= len(tokens):
                candidate = keys[
                    start + repeats * chunk_len:start + (repeats + 1) * chunk_len
                ]
                if candidate != chunk:
                    break
                repeats += 1
            if repeats < min_repeats:
                continue
            coverage = repeats * chunk_len
            if coverage > best_coverage or (
                coverage == best_coverage and best and chunk_len > best[1]
            ):
                best = (start, chunk_len, repeats)
                best_coverage = coverage
    return best


def cleanup_natural_text(
    text: str,
    region_type: str,
    profile: Profile,
) -> tuple[str, str | None]:
    cleaned, sanitized = sanitize_text(text)
    action = "unicode_sanitized" if sanitized else None

    if region_type not in NATURAL_TEXT_TYPES:
        return cleaned, action

    if len(cleaned) >= 24 and DASH_ONLY_RE.fullmatch(cleaned):
        return "—", "dash_filler_collapsed"

    # Clear generation failure: one character repeated dozens of times.
    # Keep two copies so a legitimate doubled mark is not erased completely.
    collapsed_runs = LONG_CHAR_RUN_RE.sub(r"\1\1", cleaned)
    if collapsed_runs != cleaned:
        return collapsed_runs, "character_loop_collapsed"

    tokens = cleaned.split()
    if len(cleaned) < 90 or len(tokens) < 8:
        return cleaned, action

    min_repeats = 2 if profile.allow_two_repeat_loops and len(cleaned) >= 180 else 3
    loop = find_consecutive_loop(tokens, min_repeats=min_repeats)
    if not loop:
        return cleaned, action

    start, chunk_len, repeats = loop
    repeated_tokens = chunk_len * repeats
    # Avoid shortening ordinary prose due to a short repeated phrase.
    if chunk_len == 1 and repeats < 6:
        return cleaned, action
    if repeated_tokens < 6 or repeated_tokens / len(tokens) < 0.25:
        return cleaned, action

    keep_repeats = min(profile.loop_keep, repeats)
    end = start + chunk_len * keep_repeats
    # Generation loops almost always continue to the truncated end. Preserve a
    # meaningful suffix only when it is longer than the repeated block.
    suffix_start = start + chunk_len * repeats
    suffix = tokens[suffix_start:]
    if len(suffix) >= chunk_len + 3:
        result_tokens = tokens[:end] + suffix
    else:
        result_tokens = tokens[:end]
    result = " ".join(result_tokens).strip()
    return result, "token_loop_collapsed"


def choose_drop(a: dict, b: dict) -> int:
    """Return 0 to drop a or 1 to drop b."""
    ta, tb = compact_text(a.get("text", "")), compact_text(b.get("text", ""))
    if ta and tb and ta != tb:
        if ta in tb and len(tb) > len(ta):
            return 0
        if tb in ta and len(ta) > len(tb):
            return 1
    return 1 if region_keep_score(a) >= region_keep_score(b) else 0


def duplicate_decision(a: dict, b: dict, profile: Profile) -> tuple[int | None, str | None]:
    if a.get("type") in NON_SCORABLE_TYPES or b.get("type") in NON_SCORABLE_TYPES:
        return None, None

    iou = box_iou(a["bbox"], b["bbox"])
    ioa = box_ioa_min(a["bbox"], b["bbox"])
    similarity = text_similarity(a.get("text", ""), b.get("text", ""))
    ta, tb = compact_text(a.get("text", "")), compact_text(b.get("text", ""))
    pair = {a.get("type"), b.get("type")}
    risky_pair = pair in ({"formula", "handwritten"}, {"printed", "handwritten"})

    # Reproduce the high-value V2 behavior, but without its formula/table text cleaner.
    if iou >= 0.80 and (similarity >= 0.62 or risky_pair):
        return choose_drop(a, b), "high_iou_duplicate"

    if profile.containment_dedup and ioa >= 0.95:
        exact = bool(ta and ta == tb)
        substring = bool(ta and tb and (ta in tb or tb in ta))
        if exact or similarity >= 0.90 or (substring and similarity >= 0.58):
            return choose_drop(a, b), "contained_text_duplicate"

    if profile.containment_dedup and risky_pair and iou >= 0.62 and similarity >= 0.72:
        return choose_drop(a, b), "cross_class_duplicate"

    if profile.strong_dedup:
        if ioa >= 0.90 and (similarity >= 0.78 or (ta and tb and (ta in tb or tb in ta))):
            return choose_drop(a, b), "strong_containment_duplicate"
        if risky_pair and iou >= 0.52 and similarity >= 0.58:
            return choose_drop(a, b), "strong_cross_class_duplicate"

    return None, None


def load_test_metadata(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        image = Path(item["file_name"]).name
        result[image] = {
            "width": int(item["image_width"]),
            "height": int(item["image_height"]),
            "source": item.get("source", "unknown"),
        }
    return result


def sanitize_region(region: dict, image_info: dict | None, stats: Counter) -> dict | None:
    typ = str(region.get("type", "handwritten"))
    if typ not in VALID_TYPES:
        typ = "handwritten"
        stats["invalid_type_fixed"] += 1

    try:
        box = [int(round(float(v))) for v in region.get("bbox", [])]
    except (TypeError, ValueError):
        stats["invalid_bbox_removed"] += 1
        return None
    if len(box) != 4:
        stats["invalid_bbox_removed"] += 1
        return None

    if image_info:
        width, height = image_info["width"], image_info["height"]
        clamped = [
            min(max(box[0], 0), width),
            min(max(box[1], 0), height),
            min(max(box[2], 0), width),
            min(max(box[3], 0), height),
        ]
        if clamped != box:
            stats["bbox_clamped"] += 1
        box = clamped
    if box[2] <= box[0] or box[3] <= box[1]:
        stats["invalid_bbox_removed"] += 1
        return None

    return {
        "bbox": box,
        "type": typ,
        "text": str(region.get("text", "") or ""),
    }


def process_regions(
    regions: list[dict],
    profile: Profile,
    image_info: dict | None,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    work = []
    for region in regions:
        clean_region = sanitize_region(region, image_info, stats)
        if clean_region is None:
            continue
        new_text, action = cleanup_natural_text(
            clean_region["text"], clean_region["type"], profile
        )
        clean_region["text"] = new_text
        if action:
            stats[action] += 1
        work.append(clean_region)

    dropped: set[int] = set()
    for i in range(len(work)):
        if i in dropped:
            continue
        for j in range(i + 1, len(work)):
            if j in dropped:
                continue
            decision, reason = duplicate_decision(work[i], work[j], profile)
            if decision is None:
                continue
            drop_index = i if decision == 0 else j
            dropped.add(drop_index)
            stats[reason or "duplicate_removed"] += 1
            if drop_index == i:
                break

    return [region for idx, region in enumerate(work) if idx not in dropped], stats


def summarize_rows(rows: list[dict]) -> dict:
    stats = Counter()
    long_texts = []
    for row in rows:
        regions = parse_regions(row.get("regions", "[]"))
        stats["rows"] += 1
        stats["regions"] += len(regions)
        if not regions:
            stats["empty_rows"] += 1
        local_overlap = False
        for i, region in enumerate(regions):
            text = str(region.get("text", "") or "")
            if len(text) >= 180:
                stats["texts_ge_180"] += 1
                long_texts.append((len(text), row["image"], region.get("type"), text[:160]))
            for other in regions[i + 1:]:
                if box_iou(region["bbox"], other["bbox"]) >= 0.80:
                    stats["overlap_pairs_iou80"] += 1
                    local_overlap = True
                if box_ioa_min(region["bbox"], other["bbox"]) >= 0.95:
                    stats["containment_pairs_ioa95"] += 1
        if local_overlap:
            stats["images_with_overlap_iou80"] += 1
    return {
        **dict(stats),
        "long_examples": [
            {"length": n, "image": image, "type": typ, "text_head": text}
            for n, image, typ, text in sorted(long_texts, reverse=True)[:12]
        ],
    }


def audit_profile_on_train_gt(metadata_path: Path, profile: Profile) -> dict:
    stats = Counter()
    affected_pages = 0
    if not metadata_path.exists():
        return {"available": False}

    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        regions = [
            {
                "bbox": region["bbox"],
                "type": region.get("type", "handwritten"),
                "text": region.get("text", ""),
            }
            for region in page.get("regions", [])
        ]
        output, actions = process_regions(regions, profile, {
            "width": int(page["image_width"]),
            "height": int(page["image_height"]),
            "source": page.get("source", "unknown"),
        })
        stats["regions_before"] += len(regions)
        stats["regions_after"] += len(output)
        stats.update(actions)
        if actions or len(output) != len(regions):
            affected_pages += 1
    return {
        "available": True,
        "affected_pages": affected_pages,
        **dict(stats),
    }


def default_submission() -> Path:
    candidates = (
        Path("rukopys_full_063/input/datasets/ngovietan/htr-output-kaggle-full-ver1-061/submission.csv"),
        Path("output_kaggle/submission.csv"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No full submission found; pass --submission.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument(
        "--test-metadata",
        type=Path,
        default=Path("rukopys_raw/test/metadata.jsonl"),
    )
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=Path("rukopys_raw/train/metadata.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs/postprocess_v3"),
    )
    args = parser.parse_args()

    submission_path = args.submission or default_submission()
    original_rows = read_csv_rows(submission_path)
    image_info = load_test_metadata(args.test_metadata)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "input_submission": str(submission_path),
        "before": summarize_rows(original_rows),
        "profiles": {},
    }

    for profile in PROFILES:
        output_rows = []
        actions = Counter()
        for row in original_rows:
            regions = parse_regions(row.get("regions", "[]"))
            processed, local_actions = process_regions(
                regions, profile, image_info.get(row["image"])
            )
            actions.update(local_actions)
            output_rows.append({
                "image": row["image"],
                "regions": dump_regions(processed),
            })

        output_path = args.out_dir / f"submission_v3_{profile.name}.csv"
        write_csv_rows(output_path, output_rows)
        report["profiles"][profile.name] = {
            "output": str(output_path),
            "actions": dict(actions),
            "summary": summarize_rows(output_rows),
            "train_gt_audit": audit_profile_on_train_gt(args.train_metadata, profile),
        }

    json_path = args.out_dir / "postprocess_v3_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Postprocess V3 Candidate Report",
        "",
        f"Input: `{submission_path}`",
        "",
        "All candidates start from the original full submission, not from V2.",
        "",
        "## Candidate Summary",
        "",
        "| Profile | Regions | IoU>=.80 pairs | IoA>=.95 pairs | Texts >=180 | GT regions removed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        item = report["profiles"][profile.name]
        summary = item["summary"]
        audit = item["train_gt_audit"]
        removed = audit.get("regions_before", 0) - audit.get("regions_after", 0)
        md.append(
            f"| `{profile.name}` | {summary.get('regions', 0)} | "
            f"{summary.get('overlap_pairs_iou80', 0)} | "
            f"{summary.get('containment_pairs_ioa95', 0)} | "
            f"{summary.get('texts_ge_180', 0)} | {removed} |"
        )

    md.extend([
        "",
        "## Profiles",
        "",
        "- `safe`: corrected V2 behavior; does not clean formulas/tables.",
        "- `balanced`: adds high-confidence containment and cross-class deduplication.",
        "- `text_strong`: truncates stronger OCR generation loops; higher PageCER upside and risk.",
        "- `dedup_strong`: removes more overlapping/contained regions; higher detection risk.",
        "",
        "## Recommended Submission Order",
        "",
        "1. `safe` - lowest measured GT risk and fixes the V2 formula/table cleanup bug.",
        "2. `balanced` - removes additional high-confidence contained duplicates.",
        "3. `text_strong` - tests whether remaining PageCER is dominated by generation loops.",
        "4. `dedup_strong` - highest geometry cleanup upside but also the highest detection risk.",
        "",
        "Do not apply blind spelling correction, hard text-length caps, or broad type",
        "reclassification without validation predictions. Those operations can alter",
        "legitimate Ukrainian, archive-language, formula, and table content.",
        "",
        "## Actions",
        "",
    ])
    for profile in PROFILES:
        item = report["profiles"][profile.name]
        md.append(f"### `{profile.name}`")
        md.append("")
        md.append(f"- Output: `{item['output']}`")
        md.append(f"- Actions: `{item['actions']}`")
        md.append(f"- Train GT audit: `{item['train_gt_audit']}`")
        md.append("")

    md_path = args.out_dir / "postprocess_v3_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "input": str(submission_path),
        "outputs": {
            name: item["output"] for name, item in report["profiles"].items()
        },
        "report": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
