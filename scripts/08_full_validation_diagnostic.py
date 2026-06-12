#!/usr/bin/env python3
"""
Full validation diagnostic for the current RUKOPYS baseline.

This script does not train or modify model weights. It runs three resumable
measurements on the fixed validation split:

1. detector: detector-only predictions and detection/class diagnostics;
2. gt_ocr: OCR on ground-truth boxes to isolate recognizer quality;
3. e2e: detector boxes followed by OCR to measure the complete pipeline.

Each page is appended to JSONL immediately. Re-running the command resumes from
the existing progress files. CSV and Markdown/JSON reports are regenerated from
the latest completed records.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image


ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
ART = Path(
    os.getenv("HTR_ART_DIR", ROOT / "local_working" / "htr_artifacts")
).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_metric import (  # noqa: E402
    _build_page_text,
    _greedy_match,
    _is_scorable,
    _levenshtein,
    _normalize_text,
    score_detailed,
)
import inference_utils as iu  # noqa: E402


LOG = logging.getLogger("full_validation_diagnostic")
SCORABLE_TYPES = ("handwritten", "printed", "formula", "table", "annotation")
LENGTH_BINS = (
    ("0-15", 0, 15),
    ("16-40", 16, 40),
    ("41-96", 41, 96),
    ("97-180", 97, 180),
    ("181+", 181, 10**9),
)
HEIGHT_BINS = (
    ("0-32", 0, 32),
    ("33-64", 33, 64),
    ("65-128", 65, 128),
    ("129-256", 129, 256),
    ("257+", 257, 10**9),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--art", type=Path, default=ART)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/full_validation_diagnostic")
        if Path("/kaggle/working").exists()
        else ROOT / "analysis_outputs" / "full_validation_diagnostic",
    )
    parser.add_argument(
        "--modes",
        default="detector,gt_ocr,e2e,report",
        help="Comma-separated: detector,gt_ocr,e2e,report",
    )
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("YOLO_IMGSZ", "1024")))
    parser.add_argument("--ocr-batch", type=int, default=int(os.getenv("OCR_BATCH", "1")))
    parser.add_argument("--max-pages", type=int, default=0, help="0 means all 143 pages")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--time-budget-hours", type=float, default=10.5)
    parser.add_argument("--use-tta", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def image_name(row: dict[str, Any]) -> str:
    return str(
        row.get("image")
        or row.get("file_name")
        or row.get("filename")
        or row.get("id")
        or ""
    )


def latest_progress(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        name = str(row.get("image", ""))
        if name:
            result[name] = row
    return result


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_prediction_csv(
    path: Path,
    valid_rows: list[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "regions"])
        writer.writeheader()
        for row in valid_rows:
            name = image_name(row)
            prediction = progress.get(name, {}).get("regions", [])
            writer.writerow(
                {
                    "image": name,
                    "regions": json.dumps(prediction, ensure_ascii=False),
                }
            )
    tmp.replace(path)


def build_solution_df(valid_rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image": image_name(row),
                "regions": json.dumps(
                    row.get("regions") or row.get("annotations") or [],
                    ensure_ascii=False,
                ),
            }
            for row in valid_rows
        ]
    )


def build_prediction_df(
    valid_rows: list[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image": image_name(row),
                "regions": json.dumps(
                    progress.get(image_name(row), {}).get("regions", []),
                    ensure_ascii=False,
                ),
            }
            for row in valid_rows
        ]
    )


def resolve_data_root(args: argparse.Namespace, art: Path) -> Path:
    if args.data_root:
        return args.data_root.resolve()
    cfg_path = art / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in ("DATA_ROOT", "RUKOPYS_ROOT"):
        value = cfg.get(key)
        if value and Path(value).exists():
            return Path(value).resolve()
    train_img = cfg.get("TRAIN_IMG")
    if train_img and Path(train_img).exists():
        return Path(train_img).resolve().parent.parent
    raise FileNotFoundError("Cannot resolve raw data root from config.json")


def load_detector(art: Path):
    from ultralytics import YOLO

    best = art / "det_yolo" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {best}")
    LOG.info("[detector] loading %s", best)
    return YOLO(str(best))


def detect_page(
    path: Path,
    detector,
    conf: float,
    iou: float,
    imgsz: int,
    use_tta: bool,
) -> list[dict[str, Any]]:
    results = detector.predict(
        source=str(path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device="cuda:0",
        augment=use_tta,
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []
    regions = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        regions.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "type": iu.CLASSES[cls_id]
                if cls_id < len(iu.CLASSES)
                else "handwritten",
                "conf": float(box.conf[0].item()),
                "text": "",
            }
        )
    return iu.reading_order_sort(regions)


def load_qwen_qlora(art: Path):
    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )

    lora_dir = art / "qwen3_lora" / "best_checkpoint"
    if not lora_dir.exists():
        checkpoints = sorted(
            (art / "qwen3_lora").glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1])
            if path.name.split("-")[-1].isdigit()
            else -1,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No LoRA adapter under {art / 'qwen3_lora'}")
        lora_dir = checkpoints[-1]

    model_id = os.getenv("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
    min_pixels = int(os.getenv("MIN_PIXELS", str(128 * 28 * 28)))
    max_pixels = int(os.getenv("MAX_PIXELS", str(384 * 28 * 28)))
    dtype = torch.float16
    LOG.info("[ocr] base=%s adapter=%s", model_id, lora_dir)

    processor = AutoProcessor.from_pretrained(
        str(lora_dir),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    max_memory = {
        index: os.getenv("GPU_MAX_MEMORY", "13GiB")
        for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = os.getenv("CPU_MAX_MEMORY", "24GiB")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="sdpa",
        quantization_config=quantization,
        device_map="auto",
        max_memory=max_memory,
    )
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model.eval()
    if hasattr(model, "generation_config") and hasattr(
        model.generation_config, "enable_thinking"
    ):
        model.generation_config.enable_thinking = False
    return model, processor


def deadline_reached(deadline: float) -> bool:
    return time.monotonic() >= deadline


def run_detector_stage(
    valid_rows: list[dict[str, Any]],
    img_index: dict[str, Path],
    art: Path,
    output_dir: Path,
    args: argparse.Namespace,
    deadline: float,
) -> dict[str, dict[str, Any]]:
    progress_path = output_dir / "detector_progress.jsonl"
    progress = latest_progress(progress_path)
    pending = [row for row in valid_rows if image_name(row) not in progress]
    if not pending:
        LOG.info("[detector] cache complete: %d pages", len(progress))
        return progress

    detector = load_detector(art)
    for index, row in enumerate(pending, 1):
        if deadline_reached(deadline):
            LOG.warning("[detector] time budget reached")
            break
        name = image_name(row)
        path = iu.resolve_image_path(name, img_index)
        regions = (
            detect_page(
                path,
                detector,
                args.conf,
                args.iou,
                args.imgsz,
                args.use_tta,
            )
            if path
            else []
        )
        record = {
            "image": name,
            "source": row.get("source", "unknown"),
            "regions": regions,
            "conf": args.conf,
            "iou": args.iou,
        }
        append_jsonl(progress_path, record)
        progress[name] = record
        if index % args.save_every == 0:
            write_prediction_csv(
                output_dir / "valid_detector_only.csv", valid_rows, progress
            )
            LOG.info(
                "[detector] saved %d/%d pages",
                len(progress),
                len(valid_rows),
            )
    write_prediction_csv(output_dir / "valid_detector_only.csv", valid_rows, progress)
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    return progress


def gt_prediction_regions(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for region in row.get("regions") or row.get("annotations") or []:
        result.append(
            {
                "bbox": list(region.get("bbox") or region.get("bbox_2d") or []),
                "type": region.get("type", "handwritten"),
                "text": "",
            }
        )
    return iu.reading_order_sort(result)


def run_ocr_stage(
    stage: str,
    valid_rows: list[dict[str, Any]],
    img_index: dict[str, Path],
    detector_progress: dict[str, dict[str, Any]],
    model,
    processor,
    output_dir: Path,
    args: argparse.Namespace,
    deadline: float,
) -> dict[str, dict[str, Any]]:
    if stage not in {"gt_ocr", "e2e"}:
        raise ValueError(stage)
    progress_path = output_dir / f"{stage}_progress.jsonl"
    csv_path = output_dir / (
        "valid_gt_box_ocr.csv" if stage == "gt_ocr" else "valid_end_to_end.csv"
    )
    progress = latest_progress(progress_path)
    pending = [row for row in valid_rows if image_name(row) not in progress]
    batch_size = args.ocr_batch
    for index, row in enumerate(pending, 1):
        if deadline_reached(deadline):
            LOG.warning("[%s] time budget reached", stage)
            break
        name = image_name(row)
        path = iu.resolve_image_path(name, img_index)
        if stage == "gt_ocr":
            regions = gt_prediction_regions(row)
        else:
            regions = json.loads(
                json.dumps(
                    detector_progress.get(name, {}).get("regions", []),
                    ensure_ascii=False,
                )
            )
        if path and regions:
            regions, batch_size = iu.ocr_regions(
                str(path), regions, model, processor, batch_size
            )
            regions = iu.reading_order_sort(regions)
        record = {
            "image": name,
            "source": row.get("source", "unknown"),
            "regions": regions,
        }
        append_jsonl(progress_path, record)
        progress[name] = record
        if index % args.save_every == 0:
            write_prediction_csv(csv_path, valid_rows, progress)
            LOG.info("[%s] saved %d/%d pages", stage, len(progress), len(valid_rows))
    write_prediction_csv(csv_path, valid_rows, progress)
    return progress


def label_for(value: int, bins) -> str:
    for label, low, high in bins:
        if low <= value <= high:
            return label
    return bins[-1][0]


def empty_bucket() -> dict[str, Any]:
    return {
        "gt": 0,
        "pred": 0,
        "matched": 0,
        "class_correct": 0,
        "false_negative": 0,
        "false_positive": 0,
        "cer_values": [],
    }


def summarize_prediction(
    valid_rows: list[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_type = defaultdict(empty_bucket)
    by_source_pages: dict[str, list[tuple[list, list]]] = defaultdict(list)
    by_length = defaultdict(list)
    by_height = defaultdict(list)
    confusion = Counter()
    page_errors = []
    overlap_pairs = 0

    completed_rows = [row for row in valid_rows if image_name(row) in progress]
    for row in completed_rows:
        name = image_name(row)
        source = row.get("source", "unknown")
        gt = row.get("regions") or row.get("annotations") or []
        pred = progress.get(name, {}).get("regions", [])
        matched, unmatched_gt, unmatched_pred = _greedy_match(gt, pred, threshold=0.5)
        by_source_pages[source].append((gt, pred))

        for region in gt:
            by_type[region.get("type", "handwritten")]["gt"] += 1
        for region in pred:
            by_type[region.get("type", "handwritten")]["pred"] += 1
        for gi, pi in matched:
            gt_type = gt[gi].get("type", "handwritten")
            pred_type = pred[pi].get("type", "handwritten")
            bucket = by_type[gt_type]
            bucket["matched"] += 1
            bucket["class_correct"] += int(gt_type == pred_type)
            confusion[(gt_type, pred_type)] += 1
            if _is_scorable(gt[gi]):
                gt_text = _normalize_text(gt[gi].get("text", ""), gt_type)
                pred_text = _normalize_text(pred[pi].get("text", ""), gt_type)
                cer = _levenshtein(pred_text, gt_text) / max(len(gt_text), 1)
                bucket["cer_values"].append(cer)
                by_length[label_for(len(gt_text), LENGTH_BINS)].append(cer)
                bbox = gt[gi].get("bbox", [0, 0, 0, 0])
                height = max(0, int(bbox[3]) - int(bbox[1]))
                by_height[label_for(height, HEIGHT_BINS)].append(cer)
        for gi in unmatched_gt:
            by_type[gt[gi].get("type", "handwritten")]["false_negative"] += 1
        for pi in unmatched_pred:
            by_type[pred[pi].get("type", "handwritten")]["false_positive"] += 1

        gt_page = _build_page_text(gt, normalize=True)
        pred_drop = {pi for gi, pi in matched if not _is_scorable(gt[gi])}
        pred_page = _build_page_text(pred, normalize=True, drop_indices=pred_drop)
        page_cer = (
            _levenshtein(pred_page, gt_page) / len(gt_page) if gt_page else 0.0
        )
        page_errors.append(
            {
                "image": name,
                "source": source,
                "page_cer": page_cer,
                "gt_chars": len(gt_page),
                "pred_chars": len(pred_page),
                "gt_regions": len(gt),
                "pred_regions": len(pred),
            }
        )

        for i, first in enumerate(pred):
            for second in pred[i + 1 :]:
                a = first.get("bbox", [0, 0, 0, 0])
                b = second.get("bbox", [0, 0, 0, 0])
                ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
                area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                union = area_a + area_b - inter
                if union and inter / union >= 0.80:
                    overlap_pairs += 1

    type_rows = {}
    for region_type, bucket in sorted(by_type.items()):
        cer_values = bucket.pop("cer_values")
        type_rows[region_type] = {
            **bucket,
            "recall": bucket["matched"] / max(bucket["gt"], 1),
            "precision": bucket["matched"]
            / max(bucket["matched"] + bucket["false_positive"], 1),
            "class_accuracy": bucket["class_correct"] / max(bucket["matched"], 1),
            "region_cer": sum(cer_values) / len(cer_values)
            if cer_values
            else None,
            "cer_regions": len(cer_values),
        }

    source_rows = {}
    for source, pages in sorted(by_source_pages.items()):
        solution = pd.DataFrame(
            [
                {"image": str(index), "regions": json.dumps(gt, ensure_ascii=False)}
                for index, (gt, _) in enumerate(pages)
            ]
        )
        submission = pd.DataFrame(
            [
                {
                    "image": str(index),
                    "regions": json.dumps(pred, ensure_ascii=False),
                }
                for index, (_, pred) in enumerate(pages)
            ]
        )
        source_rows[source] = score_detailed(solution, submission, "image")

    total_tp = sum(item["matched"] for item in type_rows.values())
    total_fn = sum(item["false_negative"] for item in type_rows.values())
    total_fp = sum(item["false_positive"] for item in type_rows.values())
    total_class_correct = sum(item["class_correct"] for item in type_rows.values())
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    return {
        "completed_pages": len(completed_rows),
        "detection": {
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "class_accuracy": total_class_correct / max(total_tp, 1),
        },
        "by_type": type_rows,
        "by_source": source_rows,
        "by_text_length": {
            key: {
                "n": len(values),
                "region_cer": sum(values) / len(values) if values else None,
            }
            for key, values in by_length.items()
        },
        "by_crop_height": {
            key: {
                "n": len(values),
                "region_cer": sum(values) / len(values) if values else None,
            }
            for key, values in by_height.items()
        },
        "confusion": [
            {"gt": gt_type, "pred": pred_type, "n": count}
            for (gt_type, pred_type), count in confusion.most_common()
        ],
        "page_errors": sorted(
            page_errors, key=lambda item: item["page_cer"], reverse=True
        ),
        "iou80_overlap_pairs": overlap_pairs,
    }


def safe_score(
    valid_rows: list[dict[str, Any]],
    progress: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if len(progress) < len(valid_rows):
        return None
    return score_detailed(
        build_solution_df(valid_rows),
        build_prediction_df(valid_rows, progress),
        "image",
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full Validation Diagnostic",
        "",
        f"Completed pages: `{report['completed_pages']} / {report['target_pages']}`",
        "",
    ]
    detector = report.get("detector")
    if detector:
        analysis = detector["analysis"]
        metrics = analysis["detection"]
        lines.extend(
            [
                "## detector",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Completed pages | {analysis['completed_pages']} |",
                f"| Detector F1 | {metrics['f1']:.6f} |",
                f"| Detector precision | {metrics['precision']:.6f} |",
                f"| Detector recall | {metrics['recall']:.6f} |",
                f"| Class accuracy | {metrics['class_accuracy']:.6f} |",
                f"| IoU >= 0.80 overlap pairs | {analysis['iou80_overlap_pairs']} |",
                "",
                "### By type",
                "",
                "| Type | GT | Pred | Recall | Class acc | FN | FP |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for region_type, item in analysis["by_type"].items():
            lines.append(
                f"| {region_type} | {item['gt']} | {item['pred']} | "
                f"{item['recall']:.4f} | {item['class_accuracy']:.4f} | "
                f"{item['false_negative']} | {item['false_positive']} |"
            )
        lines.append("")

    for stage in ("gt_ocr", "e2e"):
        result = report.get(stage)
        if not result:
            continue
        lines.extend([f"## {stage}", ""])
        score = result.get("official_score")
        if score:
            lines.extend(
                [
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| Composite | {score['composite_score']:.6f} |",
                    f"| Detector F1 | {score['detection_f1']:.6f} |",
                    f"| Detector precision | {score['detection_precision']:.6f} |",
                    f"| Detector recall | {score['detection_recall']:.6f} |",
                    f"| Class accuracy | {score['classification_accuracy']:.6f} |",
                    f"| Region CER | {score['region_cer']:.6f} |",
                    f"| Page CER | {score['page_cer']:.6f} |",
                    "",
                ]
            )
        lines.extend(
            [
                "### By type",
                "",
                "| Type | GT | Pred | Recall | Class acc | Region CER | FN | FP |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for region_type, item in result["analysis"]["by_type"].items():
            cer = item["region_cer"]
            lines.append(
                f"| {region_type} | {item['gt']} | {item['pred']} | "
                f"{item['recall']:.4f} | {item['class_accuracy']:.4f} | "
                f"{cer:.4f} | {item['false_negative']} | {item['false_positive']} |"
                if cer is not None
                else f"| {region_type} | {item['gt']} | {item['pred']} | "
                f"{item['recall']:.4f} | {item['class_accuracy']:.4f} | n/a | "
                f"{item['false_negative']} | {item['false_positive']} |"
            )
        lines.extend(["", "### By source", ""])
        lines.extend(
            [
                "| Source | Composite | DetF1 | Class acc | Region CER | Page CER |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for source, item in result["analysis"]["by_source"].items():
            lines.append(
                f"| {source} | {item['composite_score']:.4f} | "
                f"{item['detection_f1']:.4f} | "
                f"{item['classification_accuracy']:.4f} | "
                f"{item['region_cer']:.4f} | {item['page_cer']:.4f} |"
            )
        lines.extend(["", "### Worst pages", ""])
        for item in result["analysis"]["page_errors"][:20]:
            lines.append(
                f"- `{item['image']}` source={item['source']} "
                f"PageCER={item['page_cer']:.4f}, "
                f"regions={item['pred_regions']}/{item['gt_regions']}"
            )
        lines.append("")
    return "\n".join(lines)


def make_report(
    valid_rows: list[dict[str, Any]],
    output_dir: Path,
    detector_progress: dict[str, dict[str, Any]],
    gt_progress: dict[str, dict[str, Any]],
    e2e_progress: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "target_pages": len(valid_rows),
        "completed_pages": {
            "detector": len(detector_progress),
            "gt_ocr": len(gt_progress),
            "e2e": len(e2e_progress),
        },
    }
    for name, progress in (("gt_ocr", gt_progress), ("e2e", e2e_progress)):
        if not progress:
            continue
        report[name] = {
            "official_score": safe_score(valid_rows, progress),
            "analysis": summarize_prediction(valid_rows, progress),
        }

    if detector_progress:
        report["detector"] = {
            "analysis": summarize_prediction(valid_rows, detector_progress)
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "full_validation_report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    args.art = args.art.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "diagnostic.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    modes = {item.strip() for item in args.modes.split(",") if item.strip()}
    data_root = resolve_data_root(args, args.art)
    valid_rows = read_jsonl(args.art / "valid_split.jsonl")
    if args.max_pages:
        valid_rows = valid_rows[: args.max_pages]
    train_img = data_root / "train" / "images"
    img_index = iu.build_image_index(train_img)
    deadline = time.monotonic() + args.time_budget_hours * 3600
    LOG.info(
        "pages=%d data=%s output=%s modes=%s",
        len(valid_rows),
        data_root,
        args.output_dir,
        sorted(modes),
    )

    detector_progress = latest_progress(args.output_dir / "detector_progress.jsonl")
    if "detector" in modes or "e2e" in modes:
        detector_progress = run_detector_stage(
            valid_rows,
            img_index,
            args.art,
            args.output_dir,
            args,
            deadline,
        )

    gt_progress = latest_progress(args.output_dir / "gt_ocr_progress.jsonl")
    e2e_progress = latest_progress(args.output_dir / "e2e_progress.jsonl")
    if ("gt_ocr" in modes or "e2e" in modes) and not deadline_reached(deadline):
        model, processor = load_qwen_qlora(args.art)
        iu.DEVICE = "cuda:0"
        iu.MAX_NEW_TOKENS = int(os.getenv("MAX_TOKENS", "96"))
        if "gt_ocr" in modes:
            gt_progress = run_ocr_stage(
                "gt_ocr",
                valid_rows,
                img_index,
                detector_progress,
                model,
                processor,
                args.output_dir,
                args,
                deadline,
            )
        if "e2e" in modes and not deadline_reached(deadline):
            e2e_progress = run_ocr_stage(
                "e2e",
                valid_rows,
                img_index,
                detector_progress,
                model,
                processor,
                args.output_dir,
                args,
                deadline,
            )
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    if "report" in modes:
        report = make_report(
            valid_rows,
            args.output_dir,
            detector_progress,
            gt_progress,
            e2e_progress,
        )
        print(json.dumps(report["completed_pages"], indent=2))
        print(f"Report: {args.output_dir / 'full_validation_report.md'}")


if __name__ == "__main__":
    main()
