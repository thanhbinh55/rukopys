#!/usr/bin/env python3
"""
Print the effective run configuration before launching expensive phases.

This script is read-only. It checks dataset presence, optional silver data,
training toggles, and artifact readiness.
"""
import json
import os
from pathlib import Path


ROOT = Path(os.getenv('PROJECT_ROOT', Path.cwd()))


def default_data_root() -> Path:
    kaggle_root = ROOT / 'data' / 'kaggle_input' / 'rukopys-dataset'
    return kaggle_root / 'rukopys_raw'


DATA_ROOT = Path(os.getenv('DATA_ROOT', default_data_root()))
ART = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))


def env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default) == '1'


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())


def exists(path: Path) -> str:
    return 'yes' if path.exists() else 'no'


def status(ok: bool) -> str:
    return 'ready' if ok else 'missing'


def print_dataset():
    train_meta = DATA_ROOT / 'train' / 'metadata.jsonl'
    test_meta = DATA_ROOT / 'test' / 'metadata.jsonl'
    silver_meta = DATA_ROOT / 'silver' / 'metadata.jsonl'
    sample = DATA_ROOT / 'sample_submission.csv'

    train_rows = count_jsonl(train_meta)
    test_rows = count_jsonl(test_meta)
    silver_rows = count_jsonl(silver_meta)

    print('DATA')
    print(f'  DATA_ROOT: {DATA_ROOT}')
    print(f'  train:  {train_rows if train_rows is not None else "missing"} rows')
    print(f'  test:   {test_rows if test_rows is not None else "missing"} rows')
    print(f'  silver: {silver_rows if silver_rows is not None else "missing/optional"} rows')
    print(f'  sample_submission.csv: {exists(sample)}')
    if train_rows is None or test_rows is None:
        print('  next step: bash setup_kaggle_dataset.sh && source .env.kaggle')
    print()
    return train_rows, test_rows, silver_rows


def print_toggles(silver_rows):
    use_silver = env_bool('USE_SILVER', '1')
    effective_silver = use_silver and silver_rows is not None and silver_rows > 0

    print('TRAINING TOGGLES')
    print(f'  MODEL_ID: {os.getenv("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")}')
    print(f'  USE_SILVER: {int(use_silver)} -> {"active" if effective_silver else "inactive"}')
    print(f'  MAX_SILVER: {os.getenv("MAX_SILVER", "80000")}')
    print(f'  USE_SYNTH:  {os.getenv("USE_SYNTH", "1")}')
    print(f'  USE_HKR:    {os.getenv("USE_HKR", "1")}')
    print(f'  USE_PSEUDO: {os.getenv("USE_PSEUDO", "0")}')
    print(f'  CURRICULUM: {os.getenv("CURRICULUM", "0")}')
    print(f'  USE_FLASH_ATTN: {os.getenv("USE_FLASH_ATTN", "auto")}')
    print(f'  BATCH_SIZE: {os.getenv("BATCH_SIZE", "2")}')
    print(f'  GRAD_ACCUM: {os.getenv("GRAD_ACCUM", "16")}')
    print(f'  GRAD_CHECKPT: {os.getenv("GRAD_CHECKPT", "1")}')
    print()
    print('EFFECT')
    if effective_silver:
        print('  Phase 1: silver pages will be added to YOLO train split.')
        print('  Phase 2: silver OCR crops will be added with weight=0.5.')
        print('  Phase 2B: pseudo-labeling can run if requested.')
    elif use_silver:
        print('  USE_SILVER is on, but no silver metadata exists; code will skip silver.')
    else:
        print('  Silver is disabled by USE_SILVER=0.')
    print()


def print_artifacts(train_rows, test_rows):
    config_path = ART / 'config.json'
    train_split = ART / 'train_split.jsonl'
    valid_split = ART / 'valid_split.jsonl'
    detector = ART / 'det_yolo' / 'best.pt'
    qwen_best = ART / 'qwen3_lora' / 'best_checkpoint'
    best_config = ART / 'best_config.json'

    print('ARTIFACTS')
    print(f'  HTR_ART_DIR: {ART}')
    print(f'  config.json: {exists(config_path)}')
    print(f'  train_split.jsonl: {exists(train_split)}')
    print(f'  valid_split.jsonl: {exists(valid_split)}')
    print(f'  detector best.pt: {exists(detector)}')
    print(f'  qwen3 best_checkpoint: {exists(qwen_best)}')
    print(f'  best_config.json: {exists(best_config)}')
    print()

    print('PHASE READINESS')
    print(f'  Phase 0A setup_inspect: {status(train_rows is not None and test_rows is not None)}')
    print(f'  Phase 1 train_detector: {status(config_path.exists() and train_split.exists() and valid_split.exists())}')
    print(f'  Phase 2 train_recognizer: {status(config_path.exists() and train_split.exists() and valid_split.exists())}')
    print(f'  Phase 3 validate: {status(detector.exists() and qwen_best.exists())}')
    print(f'  Phase 4 inference: {status(detector.exists() and qwen_best.exists())}')
    if not best_config.exists():
        print('  Note: Phase 4 can run without best_config.json, but will use default YOLO thresholds.')
    print()

    if config_path.exists():
        cfg = json.load(open(config_path, encoding='utf-8'))
        print('SAVED PHASE 0A CONFIG')
        for key in ['DATA_ROOT', 'TRAIN_IMG', 'TEST_IMG', 'SILVER_META', 'SILVER_IMG', 'n_train', 'n_valid', 'n_test']:
            if key in cfg:
                print(f'  {key}: {cfg[key]}')


def main():
    print('RUKOPYS RUN CONFIG')
    print('=' * 60)
    train_rows, test_rows, silver_rows = print_dataset()
    print_toggles(silver_rows)
    print_artifacts(train_rows, test_rows)


if __name__ == '__main__':
    main()
