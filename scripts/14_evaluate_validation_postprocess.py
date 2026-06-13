#!/usr/bin/env python3
"""Evaluate Postprocess V3 profiles on validation predictions to find the optimal settings."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_metric import score_detailed
import importlib
pp3 = importlib.import_module("06_postprocess_v3_candidates")


def find_file(name: str, search_dirs: list[Path]) -> Path:
    for d in search_dirs:
        path = d / name
        if path.exists():
            return path
        # Try recursive glob
        matches = list(d.glob(f"**/{name}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate {name}")


def main() -> None:
    print("=== Evaluating Postprocess V3 Profiles ===")

    # Search folders
    search_dirs = [
        ROOT,
        ROOT / "local_working" / "htr_artifacts",
        ROOT / "analysis_outputs" / "full_validation_v2_inference",
        Path("/kaggle/working/full_validation_v2_inference"),
        Path("/kaggle/working/rukopys/local_working/htr_artifacts"),
    ]

    # Find valid_split.jsonl
    try:
        gt_path = find_file("valid_split.jsonl", search_dirs)
    except FileNotFoundError:
        # Fallback to config.json lookup
        try:
            cfg_path = find_file("config.json", search_dirs)
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            gt_path = Path(cfg["ART"]) / "valid_split.jsonl"
            if not gt_path.exists():
                raise FileNotFoundError()
        except Exception:
            raise FileNotFoundError("Could not locate valid_split.jsonl in workspace or config")

    # Find candidate valid_end_to_end.csv
    pred_path = find_file("valid_end_to_end.csv", search_dirs)

    print(f"Ground-truth split: {gt_path}")
    print(f"Validation E2E predictions: {pred_path}")

    # Load GT (solution)
    gt_rows = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                file_name = item.get("file_name") or item.get("image")
                image_name = Path(file_name).name
                regions = item.get("regions") or item.get("annotations") or []
                gt_rows.append({
                    "image": image_name,
                    "regions": json.dumps(regions, ensure_ascii=False)
                })
    df_solution = pd.DataFrame(gt_rows)

    # Load Baseline predictions
    df_pred_baseline = pd.read_csv(pred_path)
    df_pred_baseline["image"] = df_pred_baseline["image"].apply(lambda x: Path(x).name)

    # Score baseline
    score_baseline = score_detailed(df_solution, df_pred_baseline, "image")

    # Print baseline
    print("\n--- Baseline V2 E2E Score ---")
    for k, v in score_baseline.items():
        print(f"  {k}: {v:.6f}")

    # Score each profile
    metrics = ["composite_score", "detection_f1", "classification_accuracy", "region_cer", "page_cer"]
    print("\n--- Evaluating Profiles ---")
    for profile in pp3.PROFILES:
        pp_rows = []
        for _, row in df_pred_baseline.iterrows():
            image = row["image"]
            regions = json.loads(row["regions"])
            processed, stats = pp3.process_regions(regions, profile, None)
            pp_rows.append({
                "image": image,
                "regions": json.dumps(processed, ensure_ascii=False)
            })
        df_pp = pd.DataFrame(pp_rows)
        score_pp = score_detailed(df_solution, df_pp, "image")
        
        print(f"\nProfile: {profile.name}")
        for k in metrics:
            val = score_pp[k]
            delta = val - score_baseline[k]
            print(f"  {k:25s}: {val:.6f} ({delta:+.6f})")


if __name__ == "__main__":
    main()
