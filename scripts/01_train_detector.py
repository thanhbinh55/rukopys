#!/usr/bin/env python3
"""
scripts/01_train_detector.py — Phase 1: Fine-tune YOLOv8x.
Hardware: 1× L40 48GB, 64GB RAM, 8 vCPU.

Cải tiến so với phiên bản gốc:
- Better silver data handling (normalize bbox formats)
- TensorRT export tự động sau khi train
- Improved class imbalance detection
"""
import os, json, shutil, yaml
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ===== CONSTANTS =====
DEVICE     = 'cuda:0'
ROOT       = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART        = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
LOGS       = ART / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

MODEL_NAME  = os.getenv('YOLO_MODEL',    'yolov8x.pt')
EPOCHS      = int(os.getenv('EPOCHS',    '80'))
IMGSZ       = int(os.getenv('IMGSZ',     '1280'))
BATCH       = int(os.getenv('BATCH',     '12'))     # L40 48GB với imgsz=1280
WORKERS     = int(os.getenv('WORKERS',   '4'))      # 8 vCPU → dùng 4
PATIENCE    = int(os.getenv('PATIENCE',  '25'))
SAVE_PERIOD = int(os.getenv('SAVE_PERIOD','10'))
USE_SILVER  = os.getenv('USE_SILVER', '1') == '1'

YOLO_ROOT  = ART / 'det_yolo'
DATASET    = YOLO_ROOT / 'dataset'
CLASSES    = ['handwritten','printed','formula','table','annotation','image','graph']
CLS2ID     = {c: i for i, c in enumerate(CLASSES)}


def normalize_bbox(bbox, img_w: int, img_h: int):
    """Normalize bbox về YOLO format [cx, cy, w, h] normalized 0-1."""
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]

    # Handle [cx, cy, w, h] format (nếu đã normalize)
    if all(0 <= v <= 1 for v in [x1, y1, x2, y2]) and x2 <= 1 and y2 <= 1:
        # Có thể đã là YOLO format — kiểm tra
        if x2 < 0.5 and y2 < 0.5:  # likely w, h not x2, y2
            return x1, y1, x2, y2  # đã là cx, cy, w, h

    # Standard [x1, y1, x2, y2] absolute
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    cx = (x1 + x2) / (2 * img_w)
    cy = (y1 + y2) / (2 * img_h)
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def build_yolo_dataset():
    """Build YOLO dataset structure từ train/valid splits."""
    cfg           = json.load(open(ART / 'config.json'))
    train_img_root= Path(cfg['TRAIN_IMG'])
    silver_img_root= Path(cfg['SILVER_IMG'])

    for split in ['train', 'valid']:
        (DATASET / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Build image index với multiple lookup strategies
    def build_img_index(img_root: Path) -> dict:
        idx = {}
        if not img_root.exists():
            print(f'[warn] Image dir not found: {img_root}')
            return idx
        for p in img_root.iterdir():
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                idx[p.name]   = p   # full name: img_001.jpg
                idx[p.stem]   = p   # stem: img_001
        return idx

    img_index = build_img_index(train_img_root)

    def find_img(name: str, index: dict) -> Path | None:
        if not name:
            return None
        base = Path(name).name
        return (index.get(base)
                or index.get(Path(base).stem)
                or index.get(name))

    def write_split(jsonl_path: Path, split: str, img_lookup: dict):
        n_img = n_box = n_missing = n_empty = 0

        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                name = (r.get('image') or r.get('file_name')
                       or r.get('filename') or r.get('id') or '')
                imgp = find_img(name, img_lookup)
                if not imgp:
                    n_missing += 1
                    continue

                try:
                    with Image.open(imgp) as im:
                        W, H = im.size
                except Exception:
                    continue

                regions = r.get('regions') or r.get('annotations') or []
                yolo_lines = []
                for reg in regions:
                    bbox = (reg.get('bbox') or reg.get('bbox_2d')
                            or reg.get('box') or reg.get('bounding_box'))
                    typ  = reg.get('type', 'handwritten').lower().strip()

                    if not bbox or typ not in CLS2ID:
                        continue

                    norm = normalize_bbox(bbox, W, H)
                    if norm is None:
                        continue

                    cx, cy, bw, bh = norm
                    yolo_lines.append(
                        f'{CLS2ID[typ]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

                if not yolo_lines:
                    n_empty += 1
                    continue

                # Symlink image (tiết kiệm disk)
                dst = DATASET / 'images' / split / imgp.name
                if not dst.exists():
                    try:
                        os.symlink(imgp.resolve(), dst)
                    except (OSError, NotImplementedError):
                        shutil.copy2(imgp, dst)

                # Label file
                lbl_path = DATASET / 'labels' / split / (imgp.stem + '.txt')
                lbl_path.write_text('\n'.join(yolo_lines), encoding='utf-8')

                n_img += 1
                n_box += len(yolo_lines)

        print(f'[yolo] {split:5s}: {n_img:,} images, {n_box:,} boxes '
              f'| missing={n_missing} empty={n_empty}')
        return n_img

    print('[yolo] Building YOLO dataset...')
    n_train = write_split(ART / 'train_split.jsonl', 'train', img_index)
    n_valid = write_split(ART / 'valid_split.jsonl', 'valid', img_index)

    # Silver data
    silver_meta = Path(cfg.get('SILVER_META', ''))
    if USE_SILVER and silver_meta.exists() and silver_img_root.exists():
        silver_index = {**img_index, **build_img_index(silver_img_root)}
        n_silver = write_split(silver_meta, 'train', silver_index)
        print(f'[yolo] Silver: +{n_silver:,} images thêm vào train')
    elif not USE_SILVER:
        print('[yolo] Silver disabled by USE_SILVER=0')
    else:
        print('[yolo] Silver not found; training detector with gold data only')

    # Kiểm tra class imbalance
    print('\n[yolo] Checking class distribution...')
    from collections import Counter
    counter = Counter()
    for lbl_file in (DATASET / 'labels' / 'train').glob('*.txt'):
        for ln in lbl_file.read_text().strip().split('\n'):
            if ln.strip():
                try:
                    counter[int(ln.split()[0])] += 1
                except (ValueError, IndexError):
                    pass
    for cls_id, cls_name in enumerate(CLASSES):
        count = counter.get(cls_id, 0)
        max_class_count = max(counter.values(), default=1)
        bar = '█' * min(40, int(40 * count / max(max_class_count, 1)))
        print(f'  {cls_name:<15s}: {count:6,} {bar}')

    # cls weight recommendation
    min_count = min((v for v in counter.values() if v > 0), default=1)
    max_count = max(counter.values(), default=1)
    suggested_cls = min(1.5, 0.7 * (max_count / max(min_count, 1)) ** 0.2)
    print(f'\n[yolo] Suggested cls weight: {suggested_cls:.2f} (imbalance ratio: {max_count/min_count:.0f}×)')

    # Write data.yaml
    data_yaml = {
        'path':  str(DATASET.resolve()),
        'train': 'images/train',
        'val':   'images/valid',
        'nc':    len(CLASSES),
        'names': {i: c for i, c in enumerate(CLASSES)},
    }
    yaml_path = YOLO_ROOT / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    print(f'\n[yolo] data.yaml → {yaml_path}')
    return yaml_path


def main():
    from ultralytics import YOLO
    import torch

    print('PHASE 1 — TRAIN YOLO DETECTOR')
    print('=' * 50)
    print(f'Model: {MODEL_NAME} | imgsz={IMGSZ} | batch={BATCH} | epochs={EPOCHS}')
    print(f'Device: {DEVICE}')
    print(f'USE_SILVER={int(USE_SILVER)}')
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'GPU: {torch.cuda.get_device_name(0)} ({vram:.0f}GB)')

    # Disk check
    free_gb = shutil.disk_usage('/').free / 1e9
    print(f'Disk free: {free_gb:.1f}GB')
    if free_gb < 15:
        raise RuntimeError(f'Only {free_gb:.1f}GB free — need at least 15GB')

    yaml_path = build_yolo_dataset()

    # Load model — resume nếu có checkpoint
    last_pt = YOLO_ROOT / 'last.pt'
    runs_last = YOLO_ROOT / 'runs' / 'yolov8x_htr' / 'weights' / 'last.pt'

    if last_pt.exists():
        print(f'\n[yolo] Resuming from: {last_pt}')
        model = YOLO(str(last_pt))
        resume = True
    elif runs_last.exists():
        print(f'\n[yolo] Resuming from: {runs_last}')
        model = YOLO(str(runs_last))
        resume = True
    else:
        print(f'\n[yolo] Starting fresh training: {MODEL_NAME}')
        model = YOLO(MODEL_NAME)
        resume = False

    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(YOLO_ROOT / 'runs'),
        name='yolov8x_htr',

        # Memory management
        cache=False,            # 64GB RAM không đủ để cache

        # Augmentation — tối ưu cho documents
        flipud=0.0,             # KHÔNG flip dọc (text sẽ ngược)
        fliplr=0.0,             # KHÔNG flip ngang (text sẽ mirror)
        degrees=2.0,            # Rotate nhẹ (documents thường thẳng)
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
        copy_paste=0.1,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        # Convergence
        patience=PATIENCE,
        cos_lr=True,
        close_mosaic=20,        # Tắt mosaic 20 epoch cuối
        optimizer='AdamW',
        lr0=5e-4,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=5e-4,

        # Loss weights
        cls=0.7,                # Tăng classification weight cho 7 classes
        box=7.5,
        dfl=1.5,

        # Checkpoint
        save_period=SAVE_PERIOD,
        resume=resume,
        amp=True,               # AMP cho L40
        plots=True,
        exist_ok=True,
        verbose=True,
    )

    # Copy best weights lên root
    best_src = YOLO_ROOT / 'runs' / 'yolov8x_htr' / 'weights' / 'best.pt'
    if best_src.exists():
        shutil.copy2(best_src, YOLO_ROOT / 'best.pt')
        shutil.copy2(YOLO_ROOT / 'runs' / 'yolov8x_htr' / 'weights' / 'last.pt',
                     YOLO_ROOT / 'last.pt')
        print(f'\n[done] best.pt → {YOLO_ROOT}/best.pt')

    # Export TensorRT (optional — 3× faster inference)
    best_pt = YOLO_ROOT / 'best.pt'
    if best_pt.exists():
        try:
            print('\n[export] Exporting TensorRT FP16 engine (~5 min)...')
            export_model = YOLO(str(best_pt))
            export_model.export(
                format='engine',
                half=True,
                device=0,
                workspace=8,
                simplify=True,
            )
            engine_src = best_pt.with_suffix('.engine')
            if engine_src.exists():
                shutil.move(str(engine_src), str(YOLO_ROOT / 'best.engine'))
            print(f'[export] TensorRT engine → {YOLO_ROOT}/best.engine')
        except Exception as e:
            print(f'[export] TensorRT failed (non-critical, sẽ dùng PyTorch): {e}')

    # Validation metrics
    print('\n[validate] Running final validation...')
    try:
        val_model = YOLO(str(YOLO_ROOT / 'best.pt'))
        val_res = val_model.val(data=str(yaml_path), device=DEVICE, verbose=False)
        map50    = val_res.box.map50
        map50_95 = val_res.box.map
        print(f'[validate] mAP@0.5:    {map50:.4f}  (target: ≥0.85)')
        print(f'[validate] mAP@0.5-95: {map50_95:.4f}')
        per_class = val_res.box.maps
        if per_class is not None:
            for i, (cls_name, cls_map) in enumerate(zip(CLASSES, per_class)):
                flag = '⚠' if cls_map < 0.70 else '✓'
                print(f'  {flag} {cls_name:<15s}: {cls_map:.3f}')

        # Log results
        with open(LOGS / 'phase1.log', 'w') as f:
            json.dump({
                'status': 'done',
                'map50': float(map50),
                'map50_95': float(map50_95),
                'epochs_trained': EPOCHS,
            }, f, indent=2)

    except Exception as e:
        print(f'[validate] Validation failed: {e}')

    print('\n[DONE] Phase 1 complete!')
    print(f'       best.pt → {YOLO_ROOT}/best.pt')


if __name__ == '__main__':
    main()
