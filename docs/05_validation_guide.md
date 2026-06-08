# docs/05_validation_guide.md — Phase 3: Validation & Threshold Tuning

## Mục tiêu

Dùng `kaggle_metric.score_detailed()` để:
1. Đo score thật trên valid set
2. Grid search tìm conf/iou threshold tốt nhất cho YOLO
3. Lưu `best_config.json` cho Phase 4

---

## scripts/03_validate.py — Full Implementation Guide

```python
#!/usr/bin/env python3
"""
Phase 3: Validation với official metric + threshold grid search.
"""
import os, json, gc, time, itertools
from pathlib import Path
import torch
import pandas as pd
import sys

ART     = Path(os.getenv('HTR_ART_DIR', './htr_artifacts'))
LOGS    = ART / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

# Thêm repo root vào path để import kaggle_metric
sys.path.insert(0, str(Path(__file__).parent.parent))
from kaggle_metric import score_detailed

# Grid search space
CONF_GRID = [float(x) for x in os.getenv(
    'CONF_GRID', '0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30').split(',')]
IOU_GRID  = [float(x) for x in os.getenv(
    'IOU_GRID',  '0.35,0.40,0.45,0.50,0.55,0.60').split(',')]

N_VALID   = int(os.getenv('N_VALID',  '200'))   # số ảnh valid dùng để tune
OCR_BATCH = int(os.getenv('OCR_BATCH', '8'))


def build_valid_gt_df(art: Path, n: int) -> pd.DataFrame:
    """
    Tạo solution DataFrame từ valid_split.jsonl.
    Format: [image, regions_json]
    """
    rows = []
    with open(art / 'valid_split.jsonl', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n or not line.strip():
                continue
            r = json.loads(line)
            img_name = (r.get('image') or r.get('file_name')
                       or r.get('id') or '')
            regions = r.get('regions') or r.get('annotations') or []
            rows.append({
                'image': img_name,
                'regions': json.dumps(regions, ensure_ascii=False),
            })
    return pd.DataFrame(rows)


def run_pipeline_on_valid(
    valid_rows: list,
    detector,
    ocr_model,
    ocr_processor,
    conf: float,
    iou: float,
    img_index: dict,
) -> pd.DataFrame:
    """
    Chạy pipeline trên valid set với conf/iou cụ thể.
    
    Import reading_order_sort và ocr_regions từ scripts/04_inference.py.
    """
    # Import từ inference module
    sys.path.insert(0, str(Path(__file__).parent))
    from inference_utils import (
        yolo_detect_page, reading_order_sort,
        ocr_regions, regions_to_submission_json
    )
    # AGENT: tách các hàm yolo_detect_page, reading_order_sort, ocr_regions,
    # regions_to_submission_json vào scripts/inference_utils.py
    # để dùng chung ở cả Phase 3 và Phase 4.
    
    rows = []
    current_bs = OCR_BATCH
    
    for row in valid_rows:
        img_name = row.get('image') or row.get('file_name') or ''
        img_path = img_index.get(img_name)
        
        if img_path is None:
            rows.append({'image': img_name, 'regions': '[]'})
            continue
        
        regions = yolo_detect_page(str(img_path), detector, conf, iou)
        if not regions:
            rows.append({'image': img_name, 'regions': '[]'})
            continue
        
        regions = reading_order_sort(regions)
        regions, current_bs = ocr_regions(
            str(img_path), regions, ocr_model, ocr_processor, current_bs)
        
        rows.append({
            'image': img_name,
            'regions': regions_to_submission_json(regions),
        })
    
    return pd.DataFrame(rows)


def main():
    print('=' * 60)
    print('PHASE 3 — VALIDATION & THRESHOLD GRID SEARCH')
    print('=' * 60)
    
    # Load models (dùng lại từ Phase 4 utils)
    sys.path.insert(0, str(Path(__file__).parent))
    from inference_utils import load_yolo_detector, load_qwen3_model
    
    detector = load_yolo_detector()
    ocr_model, ocr_processor = load_qwen3_model()
    
    # Load valid data
    cfg = json.load(open(ART / 'config.json'))
    img_root = Path(cfg['TRAIN_IMG'])
    img_index = {p.name: p for p in img_root.iterdir() if p.is_file()}
    
    valid_rows = []
    with open(ART / 'valid_split.jsonl', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= N_VALID or not line.strip(): continue
            valid_rows.append(json.loads(line))
    print(f'[valid] Using {len(valid_rows)} validation images')
    
    gt_df = build_valid_gt_df(ART, N_VALID)
    
    # Grid search
    best_score = -1
    best_conf  = DEFAULT_CONF = 0.25
    best_iou   = DEFAULT_IOU  = 0.45
    results = []
    
    total_combos = len(CONF_GRID) * len(IOU_GRID)
    print(f'[grid] Testing {total_combos} combinations...')
    
    for combo_i, (conf, iou) in enumerate(itertools.product(CONF_GRID, IOU_GRID)):
        t0 = time.time()
        
        pred_df = run_pipeline_on_valid(
            valid_rows, detector, ocr_model, ocr_processor,
            conf=conf, iou=iou, img_index=img_index,
        )
        
        try:
            r = score_detailed(gt_df, pred_df, 'image')
        except Exception as e:
            print(f'  [err] conf={conf} iou={iou}: {e}')
            continue
        
        elapsed = time.time() - t0
        results.append({
            'conf': conf, 'iou': iou,
            'composite': r['composite_score'],
            'det_f1': r['detection_f1'],
            'class_acc': r['classification_accuracy'],
            'region_cer': r['region_cer'],
            'page_cer': r['page_cer'],
        })
        
        print(
            f'  [{combo_i+1}/{total_combos}] conf={conf:.2f} iou={iou:.2f} '
            f'→ composite={r["composite_score"]:.4f} '
            f'(DetF1={r["detection_f1"]:.3f} RegCER={r["region_cer"]:.3f} '
            f'PageCER={r["page_cer"]:.3f}) [{elapsed:.0f}s]'
        )
        
        if r['composite_score'] > best_score:
            best_score = r['composite_score']
            best_conf  = conf
            best_iou   = iou
        
        # Clear cache giữa các runs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results
    results_df = pd.DataFrame(results).sort_values('composite', ascending=False)
    results_df.to_csv(ART / 'grid_search_results.csv', index=False)
    
    best_config = {
        'conf': best_conf,
        'iou':  best_iou,
        'best_composite_score': best_score,
        'n_valid_images': N_VALID,
    }
    json.dump(best_config, open(ART / 'best_config.json', 'w'), indent=2)
    
    print('\n' + '=' * 60)
    print(f'BEST CONFIG: conf={best_conf}, iou={best_iou}')
    print(f'BEST SCORE:  {best_score:.4f}')
    print(f'Saved to: {ART}/best_config.json')
    print('=' * 60)
    print('\nTop 5 configurations:')
    print(results_df.head(5).to_string(index=False))


if __name__ == '__main__':
    main()
```

---

## AGENT: Tạo scripts/inference_utils.py

File này chứa tất cả shared functions giữa Phase 3 và Phase 4:

```python
# scripts/inference_utils.py
# Shared utilities cho Phase 3 (validation) và Phase 4 (inference)
# AGENT: move các hàm sau vào file này:
#   - load_yolo_detector()
#   - yolo_detect_page()
#   - reading_order_sort()       ← CRITICAL: dùng bucket_size=15
#   - load_qwen3_model()
#   - ocr_crops_batch()          ← với OOM recovery
#   - ocr_regions()
#   - regions_to_submission_json()
#   - try_load_paddle_ocr()
#   - paddle_ocr_crop()
#   - ensemble_ocr_with_paddle()
```
