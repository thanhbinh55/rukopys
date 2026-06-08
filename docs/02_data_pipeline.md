---
title: 02_data_pipeline

---

# docs/02_data_pipeline.md — Phase 0: Data Setup & Synthetic Generation

## scripts/00_setup_inspect.py — Full Guide

```python
#!/usr/bin/env python3
"""
Phase 0: Data audit, train/valid split, config generation.
Output: htr_artifacts/config.json, train_split.jsonl, valid_split.jsonl
"""
import os, json, random, shutil
from pathlib import Path
from collections import Counter, defaultdict

# ===== CONSTANTS =====
ART       = Path(os.getenv('HTR_ART_DIR', './htr_artifacts'))
DATA_ROOT = Path(os.getenv('DATA_ROOT',   './data'))
ART.mkdir(parents=True, exist_ok=True)
(ART / 'logs').mkdir(exist_ok=True)

# Dataset paths — điều chỉnh theo Kaggle dataset name
TRAIN_IMG    = DATA_ROOT / 'train'  / 'images'
TEST_IMG     = DATA_ROOT / 'test'   / 'images'
TRAIN_META   = DATA_ROOT / 'train'  / 'metadata.jsonl'
TEST_META    = DATA_ROOT / 'test'   / 'metadata.jsonl'
SILVER_META  = DATA_ROOT / 'silver' / 'metadata.jsonl'
SILVER_IMG   = DATA_ROOT / 'silver' / 'images'

CLASSES = ['handwritten','printed','formula','table','annotation','image','graph']
VALID_RATIO = 0.10   # 10% validation
random.seed(42)


def read_jsonl(p: Path) -> list:
    rows = []
    if not p.exists():
        print(f'[warn] Not found: {p}')
        return rows
    with open(p, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def get_regions(r: dict) -> list:
    return r.get('regions') or r.get('annotations') or r.get('labels') or []


def audit(train: list, silver: list, test: list):
    """In thông tin dataset để debug."""
    all_regions = [a for r in train for a in get_regions(r)]
    scorable = [
        a for a in all_regions
        if a.get('language','uk') == 'uk'
        and a.get('legibility','legible') == 'legible'
        and a.get('type','handwritten') not in ('image','graph')
        and a.get('text','').strip()
    ]
    print(f'\n[audit] Train pages:    {len(train):,}')
    print(f'[audit] Silver pages:   {len(silver):,}')
    print(f'[audit] Test pages:     {len(test):,}')
    print(f'[audit] Total regions:  {len(all_regions):,}')
    print(f'[audit] Scorable:       {len(scorable):,} ({100*len(scorable)/max(1,len(all_regions)):.1f}%)')
    
    by_type = Counter(a.get('type','handwritten') for a in all_regions)
    print(f'[audit] By type: {dict(by_type)}')
    
    by_source = Counter(r.get('source','unknown') for r in train)
    print(f'[audit] Train by source: {dict(by_source)}')
    
    # Class imbalance warning
    counts = list(by_type.values())
    if counts and max(counts) / max(1, min(counts)) > 10:
        print('[warn] Class imbalance > 10× — consider class weights in YOLO')
    
    return {
        'train_pages': len(train),
        'silver_pages': len(silver),
        'test_pages': len(test),
        'total_regions': len(all_regions),
        'scorable_regions': len(scorable),
        'scorable_ratio': len(scorable)/max(1,len(all_regions)),
        'by_type': dict(by_type),
        'by_source': dict(by_source),
    }


def stratified_split(train: list) -> tuple[list, list]:
    """Split train → train/valid theo source để đảm bảo distribution."""
    by_source = defaultdict(list)
    for r in train:
        by_source[r.get('source','unknown')].append(r)
    
    train_rows, valid_rows = [], []
    for source, rows in by_source.items():
        random.shuffle(rows)
        n_valid = max(20, int(VALID_RATIO * len(rows))) if len(rows) > 50 else max(1, len(rows)//5)
        valid_rows.extend(rows[:n_valid])
        train_rows.extend(rows[n_valid:])
    
    print(f'[split] Train: {len(train_rows):,}  Valid: {len(valid_rows):,}')
    print(f'[split] Valid sources: {Counter(r.get("source","?") for r in valid_rows)}')
    return train_rows, valid_rows


def main():
    print('PHASE 0 — DATA SETUP & AUDIT')
    
    # Load metadata (stream, không load ảnh)
    train  = read_jsonl(TRAIN_META)
    silver = read_jsonl(SILVER_META)
    test   = read_jsonl(TEST_META)
    
    assert train,  f'No train data found at {TRAIN_META}'
    assert test,   f'No test data found at {TEST_META}'
    
    # Audit
    stats = audit(train, silver, test)
    json.dump(stats, open(ART/'scorable_stats.json','w'), indent=2, ensure_ascii=False)
    
    # Split
    train_rows, valid_rows = stratified_split(train)
    
    # Save splits
    def write_jsonl(rows, path):
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    write_jsonl(train_rows, ART/'train_split.jsonl')
    write_jsonl(valid_rows,  ART/'valid_split.jsonl')
    
    # Save valid GT for Phase 3
    gt_rows = []
    for r in valid_rows:
        img_name = r.get('image') or r.get('file_name') or r.get('id') or ''
        gt_rows.append({
            'image': img_name,
            'regions': json.dumps(r.get('regions') or r.get('annotations') or [],
                                  ensure_ascii=False),
        })
    import pandas as pd
    pd.DataFrame(gt_rows).to_csv(ART/'valid_gt.csv', index=False)
    
    # Config
    config = {
        'DATA_ROOT':   str(DATA_ROOT),
        'TRAIN_IMG':   str(TRAIN_IMG),
        'TEST_IMG':    str(TEST_IMG),
        'TRAIN_META':  str(TRAIN_META),
        'TEST_META':   str(TEST_META),
        'SILVER_META': str(SILVER_META),
        'SILVER_IMG':  str(SILVER_IMG),
        'ART':         str(ART),
        'classes':     CLASSES,
    }
    json.dump(config, open(ART/'config.json','w'), indent=2, ensure_ascii=False)
    print(f'[done] config.json → {ART}/config.json')


if __name__ == '__main__':
    main()
```

---

## scripts/00b_generate_synthetic.py — TRDG Synthetic Data

```python
#!/usr/bin/env python3
"""
Phase 0B: Generate synthetic crops với TRDG.
Output: htr_artifacts/rec_data/synthetic/  + synth_manifest.csv
"""
import os, json, random, time
from pathlib import Path
import pandas as pd

ART      = Path(os.getenv('HTR_ART_DIR', './htr_artifacts'))
OUT_DIR  = ART / 'rec_data' / 'synthetic'
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_GEN    = int(os.getenv('N_SYNTH', '100000'))
FONT_DIR = Path(os.getenv('FONT_DIR', './fonts'))


def load_texts(art: Path) -> list:
    """Load text strings từ gold data + word list."""
    texts = []
    
    # Từ gold data
    for jsonl_path in [art/'train_split.jsonl']:
        if not jsonl_path.exists(): continue
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                for reg in (r.get('regions') or r.get('annotations') or []):
                    t = reg.get('text','').strip()
                    if 5 <= len(t) <= 100:
                        texts.append(t)
    
    # Ukrainian word list (nếu có)
    for wl_path in [Path('./data/ukr_words.txt'), Path('/kaggle/input/ukr-words/words.txt')]:
        if wl_path.exists():
            words = [w.strip() for w in open(wl_path,'r',encoding='utf-8') if w.strip()]
            # Tạo câu ngẫu nhiên
            extras = [' '.join(random.choices(words, k=random.randint(4,14)))
                      for _ in range(50000)]
            texts.extend(extras)
            print(f'[synth] Added {len(extras):,} generated sentences from word list')
            break
    
    random.shuffle(texts)
    print(f'[synth] Total text sources: {len(texts):,}')
    return texts


def generate(texts: list, n: int, font_dir: Path) -> list:
    """Generate synthetic crops với TRDG."""
    try:
        from trdg.generators import GeneratorFromStrings
    except ImportError:
        print('[synth] TRDG not installed. Run: pip install trdg')
        print('[synth] Skipping synthetic generation.')
        return []
    
    # Font paths
    font_paths = []
    if font_dir.exists():
        font_paths = [str(p) for p in font_dir.glob('**/*.ttf')]
        font_paths += [str(p) for p in font_dir.glob('**/*.otf')]
    print(f'[synth] Fonts found: {len(font_paths)}')
    
    gen_texts = (texts * ((n // len(texts)) + 1))[:n]
    random.shuffle(gen_texts)
    
    generator = GeneratorFromStrings(
        strings=gen_texts,
        count=n,
        fonts=font_paths if font_paths else [],
        language='uk',
        size=64,                # height của output
        skewing_angle=3,
        random_skew=True,
        blur=1,
        random_blur=True,
        background_type=0,     # Gaussian noise background
        distorsion_type=0,
        is_handwritten=False,
        width=-1,              # auto width
        alignment=0,
        text_color='#2a2a2a',
        orientation=0,
        space_width=1.0,
        fit=True,
    )
    
    manifest = []
    progress_path = ART / 'logs' / 'synth_progress.json'
    
    for i, (img, label) in enumerate(generator):
        if i >= n:
            break
        
        img_name = f'synth_{i:08d}.jpg'
        img_path = OUT_DIR / img_name
        
        # Resize nếu height != 64
        if img.height != 64:
            ratio = 64 / img.height
            new_w = min(1920, int(img.width * ratio))
            img = img.resize((new_w, 64))
        
        img.save(str(img_path), 'JPEG', quality=90)
        manifest.append({
            'image_path': str(img_path),
            'text': label,
            'weight': 0.3,
            'source': 'synthetic_trdg',
        })
        
        if (i + 1) % 5000 == 0:
            # Save progress checkpoint
            json.dump({'done': i+1, 'total': n}, open(progress_path,'w'))
            print(f'[synth] {i+1:,}/{n:,} images generated')
    
    return manifest


def main():
    print('PHASE 0B — SYNTHETIC DATA GENERATION (TRDG)')
    
    texts = load_texts(ART)
    if not texts:
        print('[synth] No texts found. Run Phase 0 first.')
        return
    
    manifest = generate(texts, N_GEN, FONT_DIR)
    
    if manifest:
        df = pd.DataFrame(manifest)
        df.to_csv(ART / 'rec_data' / 'synth_manifest.csv', index=False)
        print(f'[done] {len(manifest):,} synthetic crops → {OUT_DIR}')
        print(f'[done] Manifest: {ART}/rec_data/synth_manifest.csv')
    else:
        print('[done] No synthetic data generated (TRDG not installed or no texts)')


if __name__ == '__main__':
    main()
```

# docs/03_detector_guide.md — Phase 1: YOLO Detector

## scripts/01_train_detector.py — Full Guide

```python
#!/usr/bin/env python3
"""
Phase 1: Fine-tune YOLOv8x để detect regions trong documents.
Hardware: 1× L40 48GB, 64GB RAM, 8 vCPU.
"""
import os, json, shutil, time, yaml
from pathlib import Path
from PIL import Image

# ===== CONSTANTS =====
DEVICE  = 'cuda:0'
ART     = Path(os.getenv('HTR_ART_DIR', './htr_artifacts'))
LOGS    = ART / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

# YOLO config — L40 48GB optimal
MODEL_NAME = os.getenv('YOLO_MODEL',   'yolov8x.pt')
EPOCHS     = int(os.getenv('EPOCHS',   '80'))
IMGSZ      = int(os.getenv('IMGSZ',    '1280'))
BATCH      = int(os.getenv('BATCH',    '12'))      # L40 48GB với imgsz=1280
WORKERS    = int(os.getenv('WORKERS',  '4'))       # 8 vCPU → max 6, dùng 4
PATIENCE   = int(os.getenv('PATIENCE', '25'))
SAVE_PERIOD= int(os.getenv('SAVE_PERIOD', '10'))

YOLO_ROOT  = ART / 'det_yolo'
DATASET    = YOLO_ROOT / 'dataset'
CLASSES    = ['handwritten','printed','formula','table','annotation','image','graph']
CLS2ID     = {c: i for i, c in enumerate(CLASSES)}


def build_yolo_dataset():
    """Build YOLO dataset structure từ train/valid splits."""
    from pathlib import Path
    
    cfg   = json.load(open(ART/'config.json'))
    train_img_root = Path(cfg['TRAIN_IMG'])
    silver_img_root = Path(cfg['SILVER_IMG'])
    
    for split in ['train', 'valid']:
        (DATASET/'images'/split).mkdir(parents=True, exist_ok=True)
        (DATASET/'labels'/split).mkdir(parents=True, exist_ok=True)
    
    img_index = {p.name: p for p in train_img_root.iterdir() if p.is_file()}
    stem_index = {p.stem: p for p in img_index.values()}
    
    def find_img(name):
        if not name: return None
        base = Path(name).name
        return img_index.get(base) or stem_index.get(Path(base).stem)
    
    def write_split(jsonl_path, split, img_root_override=None):
        n_img = n_box = missing = empty = 0
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                name = (r.get('image') or r.get('file_name')
                       or r.get('filename') or r.get('id') or '')
                imgp = find_img(name)
                if not imgp:
                    missing += 1
                    continue
                try:
                    im = Image.open(imgp)
                    W, H = im.size
                    im.close()
                except Exception:
                    continue
                
                regions = r.get('regions') or r.get('annotations') or []
                lines = []
                for reg in regions:
                    bbox = reg.get('bbox') or reg.get('bbox_2d') or reg.get('box')
                    typ  = reg.get('type','handwritten').lower()
                    if not bbox or len(bbox) != 4 or typ not in CLS2ID:
                        continue
                    x1, y1, x2, y2 = map(float, bbox)
                    x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(W, x2); y2 = min(H, y2)
                    if x2-x1 < 5 or y2-y1 < 5: continue
                    xc = (x1+x2)/(2*W); yc = (y1+y2)/(2*H)
                    bw = (x2-x1)/W;     bh = (y2-y1)/H
                    lines.append(f'{CLS2ID[typ]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}')
                
                if not lines:
                    empty += 1
                    continue
                
                # Symlink hoặc copy image
                dst = DATASET/'images'/split/imgp.name
                if not dst.exists():
                    try:
                        os.symlink(imgp.resolve(), dst)
                    except Exception:
                        shutil.copy2(imgp, dst)
                
                # Write label file
                (DATASET/'labels'/split/(imgp.stem+'.txt')).write_text(
                    '\n'.join(lines), encoding='utf-8')
                n_img += 1
                n_box += len(lines)
        
        print(f'[yolo] {split}: {n_img:,} images, {n_box:,} boxes, {missing} missing, {empty} empty')
        return n_img
    
    n_train = write_split(ART/'train_split.jsonl', 'train')
    n_valid = write_split(ART/'valid_split.jsonl', 'valid')
    
    # Silver data (nếu có bbox annotations)
    silver_meta = Path(cfg.get('SILVER_META', ''))
    if silver_meta.exists():
        silver_index = {p.name: p for p in silver_img_root.iterdir() if p.is_file()}
        img_index.update(silver_index)
        stem_index.update({p.stem: p for p in silver_index.values()})
        n_silver = write_split(silver_meta, 'train')  # thêm vào train
        print(f'[yolo] Silver: +{n_silver:,} images added to train')
    
    # Write YAML
    data_yaml = {
        'path':  str(DATASET.resolve()),
        'train': 'images/train',
        'val':   'images/valid',
        'names': {i: c for i, c in enumerate(CLASSES)},
        'nc':    len(CLASSES),
    }
    yaml_path = YOLO_ROOT / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, allow_unicode=True)
    
    print(f'[yolo] data.yaml → {yaml_path}')
    return yaml_path


def main():
    from ultralytics import YOLO
    
    print('PHASE 1 — TRAIN YOLO DETECTOR')
    print(f'Model: {MODEL_NAME} | imgsz={IMGSZ} | batch={BATCH} | epochs={EPOCHS}')
    
    # Build dataset
    yaml_path = build_yolo_dataset()
    
    # Load model
    # Check for existing checkpoint để resume
    last_pt = YOLO_ROOT / 'last.pt'
    if last_pt.exists():
        print(f'[yolo] Resuming from: {last_pt}')
        model = YOLO(str(last_pt))
        resume = True
    else:
        model = YOLO(MODEL_NAME)  # download nếu chưa có
        resume = False
    
    # Train
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(YOLO_ROOT/'runs'),
        name='yolov8x_htr',
        
        # Memory
        cache=False,           # KHÔNG cache — 64GB RAM không đủ
        
        # Augmentation — tối ưu cho documents
        flipud=0.0,            # KHÔNG flip dọc (text sẽ ngược)
        fliplr=0.0,            # KHÔNG flip ngang (text sẽ mirror)
        degrees=2.0,           # rotate nhẹ
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
        close_mosaic=20,
        optimizer='AdamW',
        lr0=5e-4,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=5e-4,
        
        # Loss weights
        cls=0.7,               # tăng classification weight
        
        # Checkpoint & logging
        save_period=SAVE_PERIOD,
        resume=resume,
        amp=True,              # automatic mixed precision
        plots=True,
        exist_ok=True,
        verbose=True,
    )
    
    # Copy best weights lên ART root
    best_src = YOLO_ROOT / 'runs' / 'yolov8x_htr' / 'weights' / 'best.pt'
    if best_src.exists():
        shutil.copy2(best_src, YOLO_ROOT / 'best.pt')
        print(f'[done] best.pt → {YOLO_ROOT}/best.pt')
    
    # Export TensorRT (optional, ~5 min, 3× faster inference)
    best_pt = YOLO_ROOT / 'best.pt'
    if best_pt.exists():
        try:
            print('[export] Exporting TensorRT FP16 engine...')
            export_model = YOLO(str(best_pt))
            export_model.export(
                format='engine',
                half=True,
                device=0,
                workspace=8,
                simplify=True,
            )
            print(f'[export] TensorRT engine saved: {YOLO_ROOT}/best.engine')
        except Exception as e:
            print(f'[export] TensorRT export failed (non-critical): {e}')
    
    print('[DONE] Phase 1 complete!')


if __name__ == '__main__':
    main()