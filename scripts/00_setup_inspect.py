#!/usr/bin/env python3
"""
scripts/00_setup_inspect.py — Phase 0A: Data audit, train/valid split, config generation.
Output: htr_artifacts/config.json, train_split.jsonl, valid_split.jsonl
Hardware: L40 48GB / 8vCPU / 64GB RAM
"""
import csv, os, json, random, shutil
from pathlib import Path
from collections import Counter, defaultdict

# ===== CONSTANTS =====
ROOT      = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART       = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
DATA_ROOT = Path(os.getenv('DATA_ROOT',   ROOT / 'data' / 'kaggle_input' / 'rukopys-dataset' / 'rukopys_raw'))
ART.mkdir(parents=True, exist_ok=True)
(ART / 'logs').mkdir(exist_ok=True)

TRAIN_IMG   = DATA_ROOT / 'train'  / 'images'
TEST_IMG    = DATA_ROOT / 'test'   / 'images'
TRAIN_META  = DATA_ROOT / 'train'  / 'metadata.jsonl'
TEST_META   = DATA_ROOT / 'test'   / 'metadata.jsonl'
SILVER_META = DATA_ROOT / 'silver' / 'metadata.jsonl'
SILVER_IMG  = DATA_ROOT / 'silver' / 'images'

CLASSES     = ['handwritten','printed','formula','table','annotation','image','graph']
VALID_RATIO = 0.10
random.seed(42)


def read_jsonl(p: Path) -> list:
    rows = []
    if not p.exists():
        print(f'[warn] Not found: {p}')
        return rows
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f'[warn] JSON parse error: {e}')
    return rows


def get_regions(r: dict) -> list:
    return r.get('regions') or r.get('annotations') or r.get('labels') or []


def audit(train: list, silver: list, test: list) -> dict:
    all_regions = [a for r in train for a in get_regions(r)]

    # Scorable = uk language, legible, not image/graph, has text
    scorable = [
        a for a in all_regions
        if a.get('language', 'uk') == 'uk'
        and a.get('legibility', 'legible') == 'legible'
        and a.get('type', 'handwritten') not in ('image', 'graph')
        and a.get('text', '').strip()
    ]

    print(f'\n[audit] ══════════════════════════════════')
    print(f'[audit] Train pages:    {len(train):,}')
    print(f'[audit] Silver pages:   {len(silver):,}')
    print(f'[audit] Test pages:     {len(test):,}')
    print(f'[audit] Total regions:  {len(all_regions):,}')
    print(f'[audit] Scorable:       {len(scorable):,} ({100*len(scorable)/max(1,len(all_regions)):.1f}%)')

    by_type = Counter(a.get('type', 'handwritten') for a in all_regions)
    print(f'[audit] By type:')
    for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f'        {t:<15s}: {cnt:,}')

    by_source = Counter(r.get('source', 'unknown') for r in train)
    print(f'[audit] Train by source: {dict(by_source)}')

    # Class imbalance check
    counts = [by_type.get(c, 0) for c in CLASSES]
    min_c = min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 1
    max_c = max(counts)
    if max_c / max(1, min_c) > 10:
        rare = [CLASSES[i] for i, c in enumerate(counts) if c < max_c * 0.1]
        print(f'[warn] Class imbalance > 10×. Rare classes: {rare}')
        print(f'[warn] Consider: tăng cls=1.0 trong YOLO, thêm doc_gen synthetic pages')

    # Text length stats
    text_lens = [len(a.get('text', '')) for a in scorable]
    if text_lens:
        avg_len = sum(text_lens) / len(text_lens)
        max_len = max(text_lens)
        print(f'[audit] Text length: avg={avg_len:.1f}, max={max_len}')
        if max_len > 512:
            long_count = sum(1 for l in text_lens if l > 256)
            print(f'[warn] {long_count} regions có text > 256 chars → tăng MAX_SEQ_LEN nếu cần')

    stats = {
        'train_pages':      len(train),
        'silver_pages':     len(silver),
        'test_pages':       len(test),
        'total_regions':    len(all_regions),
        'scorable_regions': len(scorable),
        'scorable_ratio':   round(len(scorable) / max(1, len(all_regions)), 4),
        'by_type':          dict(by_type),
        'by_source':        dict(by_source),
        'avg_text_len':     round(sum(text_lens)/max(1, len(text_lens)), 1),
    }
    return stats


def stratified_split(train: list) -> tuple:
    """Split train → train/valid theo source để đảm bảo distribution đồng đều."""
    by_source = defaultdict(list)
    for r in train:
        by_source[r.get('source', 'unknown')].append(r)

    train_rows, valid_rows = [], []
    for source, rows in by_source.items():
        random.shuffle(rows)
        if len(rows) > 50:
            n_valid = max(20, int(VALID_RATIO * len(rows)))
        else:
            n_valid = max(1, len(rows) // 5)
        valid_rows.extend(rows[:n_valid])
        train_rows.extend(rows[n_valid:])

    print(f'\n[split] Train: {len(train_rows):,}  Valid: {len(valid_rows):,}')
    print(f'[split] Valid sources: {dict(Counter(r.get("source","?") for r in valid_rows))}')
    return train_rows, valid_rows


def write_jsonl(rows: list, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[save] {path.name}: {len(rows):,} rows')


def main():
    print('PHASE 0A — DATA SETUP & AUDIT')
    print('=' * 50)

    # Disk check
    free_gb = shutil.disk_usage('/').free / 1e9
    print(f'[disk] Free space: {free_gb:.1f}GB')
    if free_gb < 20:
        print(f'[warn] Low disk space! Minimum 20GB recommended.')

    # Load metadata (streaming, không load ảnh)
    print(f'\n[load] Reading metadata from: {DATA_ROOT}')
    train  = read_jsonl(TRAIN_META)
    silver = read_jsonl(SILVER_META)
    test   = read_jsonl(TEST_META)

    if not train:
        raise RuntimeError(f'No train data found at {TRAIN_META}\nCheck DATA_ROOT env var.')
    if not test:
        raise RuntimeError(f'No test data found at {TEST_META}')

    # Audit
    stats = audit(train, silver, test)
    json.dump(stats, open(ART / 'scorable_stats.json', 'w'), indent=2, ensure_ascii=False)

    # Split
    train_rows, valid_rows = stratified_split(train)

    # Save splits
    write_jsonl(train_rows, ART / 'train_split.jsonl')
    write_jsonl(valid_rows,  ART / 'valid_split.jsonl')

    # Save valid GT CSV cho Phase 3
    gt_rows = []
    for r in valid_rows:
        img_name = r.get('image') or r.get('file_name') or r.get('id') or ''
        gt_rows.append({
            'image':   img_name,
            'regions': json.dumps(r.get('regions') or r.get('annotations') or [],
                                  ensure_ascii=False),
        })
    with open(ART / 'valid_gt.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'regions'])
        writer.writeheader()
        writer.writerows(gt_rows)
    print(f'[save] valid_gt.csv: {len(gt_rows):,} rows')

    # Verify images accessible
    print('\n[verify] Checking image accessibility...')
    sample_rows = train_rows[:10]
    found = missing = 0
    for r in sample_rows:
        name = r.get('image') or r.get('file_name') or r.get('id') or ''
        img_path = TRAIN_IMG / name if name else None
        if img_path is not None and not img_path.exists():
            img_path = TRAIN_IMG / Path(name).name
        if img_path is not None and img_path.exists():
            found += 1
        else:
            missing += 1
    print(f'[verify] Sample images: {found}/10 found', '✓' if missing == 0 else f'({missing} missing)')

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
        'n_train':     len(train_rows),
        'n_valid':     len(valid_rows),
        'n_test':      len(test),
    }
    json.dump(config, open(ART / 'config.json', 'w'), indent=2, ensure_ascii=False)

    print('\n[DONE] Phase 0A complete!')
    print(f'       config.json      → {ART}/config.json')
    print(f'       train_split.jsonl → {ART}/train_split.jsonl ({len(train_rows):,} pages)')
    print(f'       valid_split.jsonl → {ART}/valid_split.jsonl ({len(valid_rows):,} pages)')
    print(f'       valid_gt.csv      → {ART}/valid_gt.csv')

    # Log
    with open(ART / 'logs' / 'phase0.log', 'w') as f:
        f.write(json.dumps({'status': 'done', **stats}, indent=2))


if __name__ == '__main__':
    main()
