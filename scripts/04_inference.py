#!/usr/bin/env python3
"""
scripts/04_inference.py — Phase 4: End-to-end test inference.

Pipeline:
YOLO detect -> reading order -> Qwen3-VL OCR -> optional PaddleOCR fallback
-> submission.csv
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
DATA_ROOT = Path(os.getenv('DATA_ROOT', ROOT / 'data' / 'kaggle_input' / 'rukopys-dataset' / 'rukopys_raw'))
OUTPUT_CSV = Path(os.getenv('OUTPUT_CSV', './submission.csv'))
OCR_BATCH = int(os.getenv('OCR_BATCH', '8'))
USE_TTA = os.getenv('USE_TTA', '0') == '1'

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from inference_utils import (
    build_image_index,
    ensemble_ocr_with_paddle,
    load_qwen3_model,
    load_yolo_detector,
    ocr_regions,
    reading_order_sort,
    regions_to_submission_json,
    resolve_image_path,
    try_load_paddle_ocr,
    yolo_detect_page,
)


def setup_logging():
    log_dir = ART / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'phase4_inference.log'),
            logging.StreamHandler(),
        ],
    )


def read_jsonl(path: Path) -> list:
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_paths() -> tuple:
    cfg_path = ART / 'config.json'
    if cfg_path.exists():
        cfg = json.load(open(cfg_path, encoding='utf-8'))
        test_meta = Path(cfg.get('TEST_META', DATA_ROOT / 'test' / 'metadata.jsonl'))
        test_img = Path(cfg.get('TEST_IMG', DATA_ROOT / 'test' / 'images'))
    else:
        test_meta = DATA_ROOT / 'test' / 'metadata.jsonl'
        test_img = DATA_ROOT / 'test' / 'images'

    if not test_meta.exists():
        raise FileNotFoundError(f'Test metadata not found: {test_meta}')
    if not test_img.exists():
        raise FileNotFoundError(f'Test image directory not found: {test_img}')
    return test_meta, test_img


def load_thresholds() -> tuple:
    best_path = ART / 'best_config.json'
    if best_path.exists():
        cfg = json.load(open(best_path, encoding='utf-8'))
        return float(cfg.get('conf', 0.25)), float(cfg.get('iou', 0.45))
    return float(os.getenv('YOLO_CONF', '0.25')), float(os.getenv('YOLO_IOU', '0.45'))


def metadata_image_name(row: dict) -> str:
    return str(row.get('image') or row.get('file_name') or row.get('id') or '')


def submission_image_name(image_name: str) -> str:
    return Path(image_name).name


def reorder_like_sample(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    sample_path = data_root / 'sample_submission.csv'
    if not sample_path.exists():
        return df

    sample = pd.read_csv(sample_path)
    if 'image' not in sample.columns:
        return df

    pred_by_image = {row.image: row.regions for row in df.itertuples(index=False)}
    ordered = []
    for image in sample['image'].astype(str):
        ordered.append({
            'image': image,
            'regions': pred_by_image.get(image, '[]'),
        })
    return pd.DataFrame(ordered)


def validate_submission(df: pd.DataFrame):
    required = {'image', 'regions'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'Submission missing columns: {sorted(missing)}')
    for i, value in enumerate(df['regions'].head(10)):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise RuntimeError(f'Row {i} regions is not a JSON list')


def main():
    setup_logging()
    log = logging.getLogger('phase4')

    log.info('PHASE 4 — FULL TEST INFERENCE')
    log.info('ART=%s', ART)
    log.info('DATA_ROOT=%s', DATA_ROOT)

    conf, iou = load_thresholds()
    log.info('YOLO thresholds: conf=%.3f iou=%.3f TTA=%s', conf, iou, USE_TTA)

    test_meta, test_img = load_paths()
    test_rows = read_jsonl(test_meta)
    img_index = build_image_index(test_img)
    log.info('Test rows=%d | indexed images=%d', len(test_rows), len(img_index))

    log.info('Loading YOLO detector...')
    detector = load_yolo_detector(ART)
    log.info('Loading Qwen3-VL recognizer...')
    ocr_model, ocr_processor = load_qwen3_model(ART)
    paddle_ocr = try_load_paddle_ocr()

    rows = []
    bs = OCR_BATCH
    t0 = time.time()

    for idx, meta in enumerate(test_rows, start=1):
        image_name = metadata_image_name(meta)
        submit_name = submission_image_name(image_name)
        img_path = resolve_image_path(image_name, img_index)

        if img_path is None:
            log.warning('Missing image for metadata row: %s', image_name)
            rows.append({'image': submit_name, 'regions': '[]'})
            continue

        regions = yolo_detect_page(str(img_path), detector, conf, iou, use_tta=USE_TTA)
        if regions:
            regions = reading_order_sort(regions)
            regions, bs = ocr_regions(str(img_path), regions, ocr_model, ocr_processor, bs)

            if paddle_ocr is not None:
                try:
                    from PIL import Image
                    page = Image.open(str(img_path)).convert('RGB')
                    width, height = page.size
                    for region in regions:
                        if region.get('type') in ('image', 'graph'):
                            continue
                        x1, y1, x2, y2 = region['bbox']
                        crop = page.crop((
                            max(0, x1 - 4), max(0, y1 - 4),
                            min(width, x2 + 4), min(height, y2 + 4),
                        ))
                        region['text'] = ensemble_ocr_with_paddle(
                            region.get('text', ''), crop, paddle_ocr)
                except Exception as e:
                    log.warning('Paddle fallback failed for %s: %s', submit_name, e)

        rows.append({
            'image': submit_name,
            'regions': regions_to_submission_json(regions),
        })

        if idx % 25 == 0 or idx == len(test_rows):
            elapsed = time.time() - t0
            log.info('[%d/%d] %.1fs | bs=%d | last=%s',
                     idx, len(test_rows), elapsed, bs, submit_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sub = pd.DataFrame(rows)
    sub = reorder_like_sample(sub, DATA_ROOT)
    validate_submission(sub)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUTPUT_CSV, index=False)
    log.info('Saved submission: %s (%d rows)', OUTPUT_CSV, len(sub))


if __name__ == '__main__':
    main()
