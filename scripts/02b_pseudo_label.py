#!/usr/bin/env python3
"""
scripts/02b_pseudo_label.py — Phase 2B: Pseudo-labeling silver data.

Dùng trained Qwen3-VL để OCR silver data, lọc bằng confidence heuristic,
tạo ra pseudo-labeled crops có quality tốt hơn silver annotations gốc.

Output: htr_artifacts/rec_data/pseudo_manifest.csv
Thời gian: ~2-3h trên L40 cho 80k silver samples

Chạy SAU Phase 2, TRƯỚC Phase 2 (retrain).
"""
import os, gc, json, time, shutil
from pathlib import Path
import torch
import pandas as pd
from PIL import Image

ROOT    = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART     = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
LOGS    = ART / 'logs'
DEVICE  = 'cuda:0'

MAX_SILVER  = int(os.getenv('MAX_SILVER',  '80000'))
OCR_BATCH   = int(os.getenv('OCR_BATCH',   '8'))
MIN_LEN     = int(os.getenv('MIN_LEN',     '2'))     # min chars để giữ
MAX_LEN     = int(os.getenv('MAX_LEN',     '300'))
CONF_THRESH = float(os.getenv('CONF_THRESH','0.0'))  # language model confidence (placeholder)

# Import từ inference_utils
import sys
sys.path.insert(0, str(Path(__file__).parent))


def load_silver_crops(art: Path, max_n: int) -> list:
    """Load silver data và tạo crop list."""
    cfg = json.load(open(art / 'config.json'))
    silver_meta = Path(cfg.get('SILVER_META', ''))
    silver_img_root = Path(cfg.get('SILVER_IMG', ''))

    if not silver_meta.exists():
        print(f'[pseudo] No silver metadata found: {silver_meta}')
        return []

    img_idx = {}
    if silver_img_root.exists():
        for p in silver_img_root.iterdir():
            if p.is_file():
                img_idx[p.name] = p
                img_idx[p.stem] = p

    records = []
    print(f'[pseudo] Loading silver metadata...')

    with open(silver_meta, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or len(records) >= max_n:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            img_name = (row.get('image') or row.get('file_name')
                       or row.get('filename') or row.get('id') or '')
            img_path = img_idx.get(img_name) or img_idx.get(Path(img_name).stem)
            if not img_path:
                continue

            for reg in (row.get('regions') or row.get('annotations') or []):
                bbox  = reg.get('bbox') or reg.get('bbox_2d')
                rtype = reg.get('type', 'handwritten')
                lang  = reg.get('language', 'uk')

                if not bbox or rtype in ('image', 'graph') or lang != 'uk':
                    continue

                records.append({
                    'source_image': str(img_path),
                    'bbox': bbox,
                    'type': rtype,
                    'silver_text': reg.get('text', '').strip(),
                })

    print(f'[pseudo] Silver crops to process: {len(records):,}')
    return records


def crop_image(img_path: str, bbox: list, W: int, H: int) -> Image.Image:
    """Crop và resize image."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = 4
    x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad);  y2 = min(H, y2 + pad)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return Image.new('RGB', (128, 64), 'white')

    img = Image.open(img_path).convert('RGB')
    crop = img.crop((x1, y1, x2, y2))

    if crop.height > 128:
        ratio = 128 / crop.height
        new_w = max(32, min(1920, int(crop.width * ratio)))
        crop = crop.resize((new_w, 128), Image.LANCZOS)

    return crop


def is_valid_prediction(text: str) -> bool:
    """Kiểm tra prediction có hợp lệ không."""
    if not text or len(text.strip()) < MIN_LEN:
        return False
    if len(text.strip()) > MAX_LEN:
        return False

    # Quá nhiều ký tự lạ → reject
    text = text.strip()
    cyrillic_count = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    latin_count = sum(1 for c in text if c.isalpha() and not ('\u0400' <= c <= '\u04FF'))
    digit_count = sum(1 for c in text if c.isdigit())

    total_alpha = cyrillic_count + latin_count + digit_count
    if total_alpha == 0:
        return False

    # Ít nhất 30% là ký tự có nghĩa
    if total_alpha / max(len(text), 1) < 0.3:
        return False

    # Quá nhiều lặp lại (hallucination)
    words = text.split()
    if len(words) > 3:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:  # > 60% repeated words
            return False

    return True


def run_pseudo_labeling(model, processor, records: list,
                         output_dir: Path) -> list:
    """Chạy inference trên silver crops và lọc kết quả."""
    from inference_utils import ocr_crops_batch

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    skipped = 0
    t0 = time.time()

    BATCH = OCR_BATCH
    i = 0
    last_img_path = None
    last_img = None
    last_W = last_H = 0

    print(f'[pseudo] Running inference on {len(records):,} crops...')

    while i < len(records):
        batch_records = records[i:i+BATCH]
        batch_crops = []

        for rec in batch_records:
            img_path = rec['source_image']
            try:
                # Cache last image để tránh re-open
                if img_path != last_img_path:
                    last_img = Image.open(img_path).convert('RGB')
                    last_W, last_H = last_img.size
                    last_img_path = img_path

                crop = last_img.crop([
                    max(0, int(rec['bbox'][0]) - 4),
                    max(0, int(rec['bbox'][1]) - 4),
                    min(last_W, int(rec['bbox'][2]) + 4),
                    min(last_H, int(rec['bbox'][3]) + 4),
                ])
                if crop.height > 128:
                    ratio = 128 / crop.height
                    new_w = max(32, min(1920, int(crop.width * ratio)))
                    crop = crop.resize((new_w, 128), Image.LANCZOS)
                batch_crops.append((crop, rec))
            except Exception as e:
                batch_crops.append((Image.new('RGB', (128, 64), 'white'), rec))

        # OCR inference
        try:
            texts, BATCH = ocr_crops_batch(batch_crops, model, processor, BATCH)
        except Exception as e:
            print(f'[pseudo] Batch error: {e}')
            i += len(batch_records)
            continue

        # Process results
        for j, (text, rec) in enumerate(zip(texts, batch_records)):
            if not is_valid_prediction(text):
                skipped += 1
                continue

            # Save crop
            crop_name = f'pseudo_{len(manifest):08d}.jpg'
            crop_path = output_dir / crop_name

            try:
                batch_crops[j][0].save(str(crop_path), 'JPEG', quality=90)
                manifest.append({
                    'image_path':  str(crop_path),
                    'text':        text.strip(),
                    'weight':      0.7,
                    'source':      'pseudo',
                    'type':        rec.get('type', 'handwritten'),
                    'silver_text': rec.get('silver_text', ''),
                })
            except Exception:
                pass

        i += len(batch_records)

        if i % (BATCH * 20) == 0:
            elapsed = time.time() - t0
            speed = i / max(elapsed, 1e-6)
            eta = (len(records) - i) / max(speed, 1e-6) / 60
            print(f'[pseudo] {i:,}/{len(records):,} | kept={len(manifest):,} '
                  f'| skip={skipped:,} | {speed:.0f}/s | ETA {eta:.1f}min')

        # Cache cleanup
        if i % 1000 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    print(f'\n[pseudo] Done: {len(manifest):,} accepted, {skipped:,} rejected')
    return manifest


def main():
    print('PHASE 2B — PSEUDO-LABELING SILVER DATA')
    print('=' * 50)

    best_ckpt = ART / 'qwen3_lora' / 'best_checkpoint'
    if not best_ckpt.exists():
        raise FileNotFoundError(
            f'No trained model found at {best_ckpt}\n'
            'Run Phase 2 first: python scripts/02_train_recognizer.py')

    # Load model
    print('[pseudo] Loading trained model...')
    from inference_utils import load_qwen3_model
    model, processor = load_qwen3_model(ART)

    # Load silver crops
    records = load_silver_crops(ART, MAX_SILVER)
    if not records:
        print('[pseudo] No silver data to process. Skipping.')
        return

    # Output
    pseudo_dir = ART / 'rec_data' / 'pseudo'
    manifest = run_pseudo_labeling(model, processor, records, pseudo_dir)

    if manifest:
        df = pd.DataFrame(manifest)
        out_path = ART / 'rec_data' / 'pseudo_manifest.csv'
        df.to_csv(out_path, index=False)
        print(f'\n[DONE] Pseudo-labeled data: {len(df):,} crops')
        print(f'       Manifest → {out_path}')
        print(f'\nRetrain với pseudo data:')
        print(f'  USE_PSEUDO=1 python scripts/02_train_recognizer.py')
    else:
        print('[warn] No valid pseudo labels generated.')


if __name__ == '__main__':
    main()
