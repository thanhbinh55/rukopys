#!/usr/bin/env python3
"""
scripts/00c_prepare_hkr.py — Chuẩn bị HKR Cyrillic dataset cho training.

HKR (Handwritten Kazakh & Russian) là external dataset với 12k+ Cyrillic word crops.
Script này:
1. Kiểm tra HKR data có sẵn
2. Convert annotation format → manifest CSV
3. Filter quality (độ dài text, kích thước ảnh)
4. Output: htr_artifacts/hkr_manifest.csv

Usage:
    HKR_DIR=./data/hkr python scripts/00c_prepare_hkr.py
"""
import os
import json
from pathlib import Path
import pandas as pd
from PIL import Image

ROOT = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
HKR_DIR = Path(os.getenv('HKR_DIR', ROOT / 'data' / 'kaggle_input' / 'hkr'))
MAX_HKR = int(os.getenv('MAX_HKR', '50000'))
MIN_TEXT_LEN = int(os.getenv('MIN_TEXT_LEN', '2'))
MAX_TEXT_LEN = int(os.getenv('MAX_TEXT_LEN', '80'))


def find_hkr_data() -> tuple:
    """
    Tìm HKR data theo nhiều format có thể.
    HKR có thể download từ:
    - https://github.com/abdoelsayed2016/HKR_Dataset
    - Kaggle dataset

    Returns:
        (img_dir: Path, annotation: dict) hoặc (None, None)
    """
    # Format 1: annotation.json + images/
    ann_path = HKR_DIR / 'annotation.json'
    if ann_path.exists():
        ann = json.load(open(ann_path, encoding='utf-8'))
        img_dir = HKR_DIR / 'train'
        if not img_dir.exists():
            img_dir = HKR_DIR / 'images'
        if not img_dir.exists():
            img_dir = HKR_DIR
        print(f'[hkr] Found annotation.json with {len(ann)} entries')
        return img_dir, ann

    # Format 2: CSV với image_path + label columns
    for csv_name in ['train.csv', 'annotation.csv', 'labels.csv']:
        csv_path = HKR_DIR / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Normalize column names
            col_map = {}
            for col in df.columns:
                if col.lower() in ('image', 'filename', 'file', 'path'):
                    col_map[col] = 'filename'
                elif col.lower() in ('text', 'label', 'transcript', 'transcription'):
                    col_map[col] = 'text'
            if col_map:
                df = df.rename(columns=col_map)
            ann = {row['filename']: row['text'] for _, row in df.iterrows()
                   if 'filename' in df.columns and 'text' in df.columns}
            img_dir = HKR_DIR
            print(f'[hkr] Found CSV with {len(ann)} entries')
            return img_dir, ann

    # Format 3: directory of images với text files
    img_files = list(HKR_DIR.glob('**/*.jpg')) + list(HKR_DIR.glob('**/*.png'))
    if img_files:
        # Tìm text file tương ứng
        ann = {}
        for img_path in img_files:
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                text = txt_path.read_text(encoding='utf-8').strip()
                ann[img_path.name] = text
        if ann:
            print(f'[hkr] Found {len(ann)} image+txt pairs')
            return HKR_DIR, ann

    return None, None


def filter_and_convert(img_dir: Path, annotation: dict, max_n: int) -> list:
    """
    Filter records theo quality và convert sang manifest format.
    """
    records = []
    skipped = 0

    for fname, text in annotation.items():
        if len(records) >= max_n:
            break

        text = str(text).strip()

        # Filter text quality
        if not text or len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
            skipped += 1
            continue

        # Find image file
        img_path = img_dir / fname
        if not img_path.exists():
            # Try với extension khác
            for ext in ['.jpg', '.png', '.jpeg']:
                alt = img_dir / (Path(fname).stem + ext)
                if alt.exists():
                    img_path = alt
                    break
            else:
                skipped += 1
                continue

        # Quick image validation (không load ảnh vào RAM)
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                if w < 10 or h < 5:  # quá nhỏ
                    skipped += 1
                    continue
        except Exception:
            skipped += 1
            continue

        records.append({
            'image_path': str(img_path.resolve()),
            'text': text,
            'weight': 0.5,        # thấp hơn Ukrainian gold (1.0)
            'source': 'hkr',
            'type': 'handwritten',
        })

        if len(records) % 5000 == 0:
            print(f'[hkr] Processed {len(records):,}...')

    print(f'[hkr] Accepted: {len(records):,}  Skipped: {skipped:,}')
    return records


def main():
    print('=' * 50)
    print('PHASE 0C — HKR DATASET PREPARATION')
    print('=' * 50)

    if not HKR_DIR.exists():
        print(f'[hkr] HKR directory not found: {HKR_DIR}')
        print('[hkr] Download from: https://github.com/abdoelsayed2016/HKR_Dataset')
        print('[hkr] Or: kaggle datasets download -d abdoelsayed2016/hkr-dataset')
        print('[hkr] Set env var: HKR_DIR=/path/to/hkr')
        print('[hkr] Skipping — pipeline will run without HKR data')
        return

    img_dir, annotation = find_hkr_data()
    if annotation is None:
        print(f'[hkr] Could not parse HKR data in {HKR_DIR}')
        print('[hkr] Skipping — pipeline will run without HKR data')
        return

    print(f'[hkr] Image dir: {img_dir}')
    print(f'[hkr] Total annotations: {len(annotation):,}')
    print(f'[hkr] Max to use: {MAX_HKR:,}')

    records = filter_and_convert(img_dir, annotation, MAX_HKR)

    if not records:
        print('[hkr] No valid records found. Skipping.')
        return

    df = pd.DataFrame(records)
    out_path = ART / 'hkr_manifest.csv'
    df.to_csv(out_path, index=False)

    print(f'[done] HKR manifest: {out_path}')
    print(f'[done] {len(df):,} crops ready for training')
    print(f'[done] Sample: {df.iloc[0]["text"]!r}')


if __name__ == '__main__':
    main()
