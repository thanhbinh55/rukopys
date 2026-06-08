#!/usr/bin/env python3
"""
scripts/03_validate.py — Phase 3: Validation & Threshold Grid Search.
Dùng official kaggle_metric.score_detailed() để tìm conf/iou tốt nhất.
Hardware: L40 48GB
"""
import os, gc, json, sys, time, itertools
from pathlib import Path
import torch
import pandas as pd

ROOT = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART  = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
LOGS = ART / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

# Grid search space
CONF_GRID = [float(x) for x in os.getenv(
    'CONF_GRID', '0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30').split(',')]
IOU_GRID  = [float(x) for x in os.getenv(
    'IOU_GRID',  '0.35,0.40,0.45,0.50,0.55,0.60').split(',')]
N_VALID   = int(os.getenv('N_VALID',  '200'))
OCR_BATCH = int(os.getenv('OCR_BATCH', '8'))
USE_TTA   = os.getenv('USE_TTA', '0') == '1'  # Test-Time Augmentation

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from kaggle_metric import score_detailed
from inference_utils import (
    load_yolo_detector, yolo_detect_page,
    reading_order_sort, load_qwen3_model,
    ocr_regions, regions_to_submission_json,
    try_load_paddle_ocr, ensemble_ocr_with_paddle,
    build_image_index, resolve_image_path,
)


def build_valid_gt_df(art: Path, n: int) -> pd.DataFrame:
    rows = []
    with open(art / 'valid_split.jsonl', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n or not line.strip():
                continue
            r = json.loads(line)
            img_name = (r.get('image') or r.get('file_name') or r.get('id') or '')
            regions  = r.get('regions') or r.get('annotations') or []
            rows.append({'image': img_name,
                         'regions': json.dumps(regions, ensure_ascii=False)})
    return pd.DataFrame(rows)


def run_pipeline(valid_rows: list, detector, ocr_model, ocr_processor,
                 conf: float, iou: float, img_index: dict,
                 paddle_ocr=None) -> pd.DataFrame:
    rows = []
    bs = OCR_BATCH

    for row in valid_rows:
        img_name = row.get('image') or row.get('file_name') or ''
        img_path = resolve_image_path(img_name, img_index)
        if img_path is None:
            rows.append({'image': img_name, 'regions': '[]'})
            continue

        regions = yolo_detect_page(str(img_path), detector, conf, iou, use_tta=USE_TTA)
        if not regions:
            rows.append({'image': img_name, 'regions': '[]'})
            continue

        regions = reading_order_sort(regions)
        regions, bs = ocr_regions(str(img_path), regions, ocr_model, ocr_processor, bs)

        # Optional ensemble
        if paddle_ocr is not None:
            try:
                from PIL import Image
                page = Image.open(str(img_path)).convert('RGB')
                W, H = page.size
                for r in regions:
                    if r.get('type') in ('image', 'graph'):
                        continue
                    x1, y1, x2, y2 = r['bbox']
                    crop = page.crop((max(0,x1-4), max(0,y1-4),
                                      min(W,x2+4), min(H,y2+4)))
                    r['text'] = ensemble_ocr_with_paddle(r.get('text',''), crop, paddle_ocr)
            except Exception:
                pass

        rows.append({'image': img_name, 'regions': regions_to_submission_json(regions)})

    return pd.DataFrame(rows)


def main():
    print('=' * 60)
    print('PHASE 3 — VALIDATION & THRESHOLD GRID SEARCH')
    print(f'Grid: conf={CONF_GRID}')
    print(f'      iou={IOU_GRID}')
    print(f'N_valid={N_VALID} | TTA={USE_TTA}')
    print('=' * 60)

    # Load models
    print('\n[models] Loading YOLO detector...')
    detector = load_yolo_detector(ART)

    print('[models] Loading Qwen3-VL...')
    ocr_model, ocr_processor = load_qwen3_model(ART)

    paddle_ocr = try_load_paddle_ocr()

    # Load valid data
    cfg       = json.load(open(ART / 'config.json'))
    img_root  = Path(cfg['TRAIN_IMG'])
    img_index = build_image_index(img_root)

    valid_rows = []
    with open(ART / 'valid_split.jsonl', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= N_VALID or not line.strip():
                continue
            valid_rows.append(json.loads(line))
    print(f'[data] Using {len(valid_rows):,} validation images')

    gt_df = build_valid_gt_df(ART, N_VALID)

    # ── GRID SEARCH ──────────────────────────────────────────────
    best_score = -1
    best_conf  = 0.25
    best_iou   = 0.45
    results    = []
    total      = len(CONF_GRID) * len(IOU_GRID)

    print(f'\n[grid] Testing {total} combinations...')

    for combo_i, (conf, iou) in enumerate(itertools.product(CONF_GRID, IOU_GRID)):
        t0 = time.time()

        pred_df = run_pipeline(valid_rows, detector, ocr_model, ocr_processor,
                               conf=conf, iou=iou, img_index=img_index,
                               paddle_ocr=paddle_ocr)

        try:
            r = score_detailed(gt_df, pred_df, 'image')
        except Exception as e:
            print(f'  [{combo_i+1}/{total}] conf={conf:.2f} iou={iou:.2f} ERROR: {e}')
            continue

        elapsed = time.time() - t0
        results.append({
            'conf': conf, 'iou': iou,
            'composite':  r['composite_score'],
            'det_f1':     r['detection_f1'],
            'det_prec':   r['detection_precision'],
            'det_recall': r['detection_recall'],
            'class_acc':  r['classification_accuracy'],
            'region_cer': r['region_cer'],
            'page_cer':   r['page_cer'],
        })

        flag = '★ NEW BEST' if r['composite_score'] > best_score else ''
        print(f"  [{combo_i+1}/{total}] conf={conf:.2f} iou={iou:.2f} "
              f"→ composite={r['composite_score']:.4f} "
              f"(DetF1={r['detection_f1']:.3f} "
              f"RegCER={r['region_cer']:.3f} "
              f"PageCER={r['page_cer']:.3f}) [{elapsed:.0f}s] {flag}")

        if r['composite_score'] > best_score:
            best_score = r['composite_score']
            best_conf  = conf
            best_iou   = iou

        torch.cuda.empty_cache()
        gc.collect()

    # ── SAVE RESULTS ─────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values('composite', ascending=False)
    results_df.to_csv(ART / 'grid_search_results.csv', index=False)

    best_config = {
        'conf': best_conf,
        'iou':  best_iou,
        'best_composite_score': best_score,
        'n_valid_images': N_VALID,
        'use_tta': USE_TTA,
    }
    json.dump(best_config, open(ART / 'best_config.json', 'w'), indent=2)

    print('\n' + '=' * 60)
    print(f'BEST: conf={best_conf:.2f}  iou={best_iou:.2f}  score={best_score:.4f}')
    print(f'Saved: {ART}/best_config.json')
    print('=' * 60)

    print('\nTop 5 configurations:')
    print(results_df.head(5).to_string(index=False))

    # ── DIAGNOSTIC ───────────────────────────────────────────────
    print('\n── Diagnosis ──')
    best_row = results_df.iloc[0]
    if best_row['page_cer'] > best_row['region_cer'] + 0.10:
        print('⚠ page_cer >> region_cer → READING ORDER có vấn đề')
    if best_row['det_recall'] < 0.70:
        print(f'⚠ det_recall={best_row["det_recall"]:.3f} thấp → Giảm conf threshold')
    if best_row['class_acc'] < 0.80:
        print(f'⚠ class_acc={best_row["class_acc"]:.3f} thấp → Tăng cls weight trong YOLO')
    if best_row['region_cer'] > 0.30:
        print(f'⚠ region_cer={best_row["region_cer"]:.3f} cao → Train Qwen3 thêm epochs')
    if best_row['composite'] >= 0.85:
        print(f'✓ Excellent! Score = {best_row["composite"]:.4f}')
    elif best_row['composite'] >= 0.75:
        print(f'~ Good. Score = {best_row["composite"]:.4f}. Còn room for improvement.')
    else:
        print(f'✗ Score = {best_row["composite"]:.4f}. Cần cải thiện thêm.')


if __name__ == '__main__':
    main()
