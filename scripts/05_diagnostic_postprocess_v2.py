#!/usr/bin/env python3
"""
Offline diagnostic + conservative postprocess for RUKOPYS submissions.

This does not need GPU. It reads a submission CSV, reports likely PageCER
hazards, and writes a postprocessed candidate CSV:

- suppress near-duplicate overlapping scorable regions
- clean obvious OCR repetition such as dash-only hallucinations
- preserve row order from sample_submission.csv when available
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


SCORABLE_SKIP_TYPES = {"image", "graph"}
MATH_HINT_RE = re.compile(r"[0-9=+\-−–—×*/^√∫π%≈≤≥<>()[\]{}]|[A-Za-z]{1,4}\d")
DASH_ONLY_RE = re.compile(r"^[\\s\\-−–—_.,·|]+$")


def parse_regions(value: str) -> list[dict]:
    try:
        parsed = json.loads(value) if value else []
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def dump_regions(regions: list[dict]) -> str:
    cleaned = []
    for r in regions:
        bbox = r.get("bbox") or [0, 0, 0, 0]
        cleaned.append({
            "bbox": [int(x) for x in bbox],
            "type": str(r.get("type", "handwritten")),
            "text": str(r.get("text", "") or "").strip(),
        })
    return json.dumps(cleaned, ensure_ascii=False)


def box_iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def norm_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"[.,;:!?\"'`ʼ’‘“”„«»()\\[\\]{}]", "", text)
    return text.strip()


def text_similarity(a: str, b: str) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return difflib.SequenceMatcher(None, a, b).ratio()


def looks_math(text: str) -> bool:
    return bool(MATH_HINT_RE.search(str(text or "")))


def region_keep_score(region: dict) -> tuple:
    """Higher is better for duplicate suppression."""
    typ = region.get("type", "handwritten")
    text = str(region.get("text", "") or "")
    length = len(text)
    has_math = looks_math(text)

    # Prefer plausible formula class for math-like content, otherwise prefer
    # natural text classes. This is deliberately conservative.
    if typ == "formula" and has_math:
        class_score = 5
    elif typ == "handwritten" and not has_math:
        class_score = 4
    elif typ == "printed" and not has_math:
        class_score = 3
    elif typ == "annotation":
        class_score = 2
    elif typ == "formula":
        class_score = 1
    else:
        class_score = 0

    # Penalize obvious overlong hallucinations when choosing between duplicates.
    length_score = min(length, 120) - max(0, length - 180)
    return (class_score, length_score)


def should_drop_duplicate(a: dict, b: dict, iou_threshold: float) -> int | None:
    """Return 0 to drop a, 1 to drop b, None to keep both."""
    if a.get("type") in SCORABLE_SKIP_TYPES or b.get("type") in SCORABLE_SKIP_TYPES:
        return None
    if box_iou(a.get("bbox", [0, 0, 0, 0]), b.get("bbox", [0, 0, 0, 0])) < iou_threshold:
        return None

    ta, tb = str(a.get("text", "") or ""), str(b.get("text", "") or "")
    sim = text_similarity(ta, tb)
    pair = {a.get("type"), b.get("type")}

    # Main known failure mode: same visual line appears as both formula and
    # handwritten. Even if OCR differs, a near-identical box is risky for PageCER.
    risky_type_overlap = pair in ({"formula", "handwritten"}, {"printed", "handwritten"})
    if sim < 0.62 and not risky_type_overlap:
        return None

    score_a, score_b = region_keep_score(a), region_keep_score(b)
    if score_a >= score_b:
        return 1
    return 0


def collapse_repeated_tokens(text: str) -> tuple[str, bool]:
    original = str(text or "").strip()
    if not original:
        return original, False

    if len(original) >= 24 and DASH_ONLY_RE.match(original):
        return "—", True

    tokens = original.split()
    if len(tokens) < 8:
        return original, False

    changed = False
    out: list[str] = []
    i = 0
    while i < len(tokens):
        collapsed = False
        for n in range(6, 1, -1):
            if i + 3 * n <= len(tokens):
                chunk = tokens[i:i + n]
                if tokens[i + n:i + 2 * n] == chunk and tokens[i + 2 * n:i + 3 * n] == chunk:
                    out.extend(chunk)
                    i += 3 * n
                    changed = True
                    collapsed = True
                    break
        if collapsed:
            continue
        out.append(tokens[i])
        i += 1

    cleaned = " ".join(out)
    # Collapse token repeated many times consecutively.
    cleaned2 = re.sub(r"\\b(\\w{3,})(?:\\s+\\1){3,}\\b", r"\\1 \\1", cleaned, flags=re.IGNORECASE)
    if cleaned2 != cleaned:
        changed = True
        cleaned = cleaned2
    return cleaned, changed


def postprocess_regions(regions: list[dict], iou_threshold: float) -> tuple[list[dict], dict]:
    stats = Counter()

    # Text cleanup first, so duplicate decisions see cleaner text.
    work = []
    for r in regions:
        nr = dict(r)
        cleaned, changed = collapse_repeated_tokens(str(nr.get("text", "") or ""))
        if changed:
            stats["text_cleaned"] += 1
            nr["text"] = cleaned
        work.append(nr)

    drop: set[int] = set()
    for i in range(len(work)):
        if i in drop:
            continue
        for j in range(i + 1, len(work)):
            if j in drop:
                continue
            decision = should_drop_duplicate(work[i], work[j], iou_threshold)
            if decision == 0:
                drop.add(i)
                stats["duplicates_removed"] += 1
                break
            if decision == 1:
                drop.add(j)
                stats["duplicates_removed"] += 1

    return [r for idx, r in enumerate(work) if idx not in drop], dict(stats)


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "regions"])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    counts = []
    type_counter = Counter()
    text_lens = []
    long_texts = []
    overlap_pairs = 0
    overlap_images = 0
    overlap_type_pairs = Counter()
    invalid_json = 0
    empty_rows = 0

    for row in rows:
        regions = parse_regions(row.get("regions", "[]"))
        if not isinstance(regions, list):
            invalid_json += 1
            regions = []
        counts.append(len(regions))
        if not regions:
            empty_rows += 1

        local_overlap = 0
        for i, r in enumerate(regions):
            typ = r.get("type", "missing")
            type_counter[typ] += 1
            text = str(r.get("text", "") or "")
            if text:
                text_lens.append(len(text))
                if len(text) >= 180:
                    long_texts.append({
                        "image": row.get("image"),
                        "type": typ,
                        "length": len(text),
                        "text_head": text[:180].replace("\\n", " | "),
                    })
            for r2 in regions[i + 1:]:
                if box_iou(r.get("bbox", [0, 0, 0, 0]), r2.get("bbox", [0, 0, 0, 0])) >= 0.80:
                    local_overlap += 1
                    overlap_type_pairs[(typ, r2.get("type", "missing"))] += 1
        if local_overlap:
            overlap_images += 1
            overlap_pairs += local_overlap

    counts_sorted = sorted(counts)
    text_lens_sorted = sorted(text_lens)

    def pct(vals: list[int], p: float) -> int:
        if not vals:
            return 0
        idx = min(len(vals) - 1, round((len(vals) - 1) * p))
        return vals[idx]

    return {
        "rows": len(rows),
        "empty_rows": empty_rows,
        "invalid_json_rows": invalid_json,
        "regions_total": sum(counts),
        "regions_mean": round(sum(counts) / max(len(counts), 1), 2),
        "regions_median": median(counts) if counts else 0,
        "regions_p90": pct(counts_sorted, 0.90),
        "regions_max": max(counts) if counts else 0,
        "type_counter": dict(type_counter),
        "text_len_mean": round(sum(text_lens) / max(len(text_lens), 1), 2),
        "text_len_p90": pct(text_lens_sorted, 0.90),
        "text_len_max": max(text_lens) if text_lens else 0,
        "texts_ge_180": len(long_texts),
        "long_examples": sorted(long_texts, key=lambda x: x["length"], reverse=True)[:10],
        "images_with_overlap_iou80": overlap_images,
        "overlap_pairs_iou80": overlap_pairs,
        "overlap_type_pairs_top": [
            {"pair": list(k), "count": v}
            for k, v in overlap_type_pairs.most_common(12)
        ],
    }


def reorder_like_sample(rows: list[dict], sample_csv: Path | None) -> list[dict]:
    if not sample_csv or not sample_csv.exists():
        return rows
    by_image = {r["image"]: r["regions"] for r in rows}
    ordered = []
    for r in read_csv_rows(sample_csv):
        img = r["image"]
        ordered.append({"image": img, "regions": by_image.get(img, "[]")})
    return ordered


def default_submission() -> Path:
    candidates = [
        Path("rukopys_full_063/input/datasets/ngovietan/htr-output-kaggle-full-ver1-061/submission.csv"),
        Path("output_kaggle/submission.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted(Path(".").glob("**/submission.csv"))
    if hits:
        return hits[0]
    raise FileNotFoundError("No submission.csv found. Pass --submission.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--sample", type=Path, default=Path("rukopys_raw/sample_submission.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/postprocess_v2"))
    parser.add_argument("--iou", type=float, default=0.80)
    args = parser.parse_args()

    sub_path = args.submission or default_submission()
    rows = read_csv_rows(sub_path)
    before = summarize(rows)

    new_rows = []
    post_stats = Counter()
    for row in rows:
        regions = parse_regions(row.get("regions", "[]"))
        new_regions, stats = postprocess_regions(regions, args.iou)
        post_stats.update(stats)
        new_rows.append({"image": row["image"], "regions": dump_regions(new_regions)})

    new_rows = reorder_like_sample(new_rows, args.sample)
    after = summarize(new_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "submission_postprocess_v2.csv"
    out_json = args.out_dir / "diagnostic_postprocess_v2_report.json"
    out_md = args.out_dir / "diagnostic_postprocess_v2_report.md"

    write_csv_rows(out_csv, new_rows)

    report = {
        "input_submission": str(sub_path),
        "output_submission": str(out_csv),
        "iou_duplicate_threshold": args.iou,
        "postprocess_actions": dict(post_stats),
        "before": before,
        "after": after,
        "delta": {
            "regions_total": after["regions_total"] - before["regions_total"],
            "overlap_pairs_iou80": after["overlap_pairs_iou80"] - before["overlap_pairs_iou80"],
            "images_with_overlap_iou80": after["images_with_overlap_iou80"] - before["images_with_overlap_iou80"],
            "texts_ge_180": after["texts_ge_180"] - before["texts_ge_180"],
        },
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Diagnostic + Postprocess V2 Report",
        "",
        f"Input: `{sub_path}`",
        f"Output candidate: `{out_csv}`",
        "",
        "## Summary",
        "",
        f"- Regions: {before['regions_total']} -> {after['regions_total']} ({report['delta']['regions_total']:+d})",
        f"- Images with IoU>=0.80 overlaps: {before['images_with_overlap_iou80']} -> {after['images_with_overlap_iou80']} ({report['delta']['images_with_overlap_iou80']:+d})",
        f"- IoU>=0.80 overlap pairs: {before['overlap_pairs_iou80']} -> {after['overlap_pairs_iou80']} ({report['delta']['overlap_pairs_iou80']:+d})",
        f"- Texts >=180 chars: {before['texts_ge_180']} -> {after['texts_ge_180']} ({report['delta']['texts_ge_180']:+d})",
        f"- Actions: `{dict(post_stats)}`",
        "",
        "## Top Overlap Type Pairs Before",
        "",
    ]
    for item in before["overlap_type_pairs_top"]:
        md.append(f"- {item['pair']}: {item['count']}")
    md.extend(["", "## Long OCR Examples Before", ""])
    for item in before["long_examples"][:6]:
        md.append(f"- `{item['image']}` {item['type']} len={item['length']}: {item['text_head']}")
    md.append("")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "input": str(sub_path),
        "output": str(out_csv),
        "regions": [before["regions_total"], after["regions_total"]],
        "overlap_pairs_iou80": [before["overlap_pairs_iou80"], after["overlap_pairs_iou80"]],
        "texts_ge_180": [before["texts_ge_180"], after["texts_ge_180"]],
        "actions": dict(post_stats),
        "report": str(out_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
