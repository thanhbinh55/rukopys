#!/usr/bin/env python3
"""
scripts/check_env.py — Kiểm tra môi trường trước khi chạy bất kỳ script nào.
AGENT: Chạy file này TRƯỚC TIÊN. Sửa mọi lỗi '✗ FAIL' trước khi tiếp tục.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --fix  # tự động fix những gì có thể
"""
import sys
import os
import shutil
import subprocess
import argparse
from pathlib import Path


ROOT = Path(os.getenv('PROJECT_ROOT', Path.cwd()))


def default_data_root() -> Path:
    return ROOT / 'data' / 'kaggle_input' / 'rukopys-dataset' / 'rukopys_raw'


def default_art_dir() -> Path:
    return ROOT / 'local_working' / 'htr_artifacts'


def run_python_import(module: str) -> bool:
    """Check nếu module import được."""
    result = subprocess.run(
        [sys.executable, '-c', f'import {module}'],
        capture_output=True,
    )
    return result.returncode == 0


def check_cuda() -> dict:
    """Kiểm tra CUDA và GPU."""
    checks = {}
    try:
        import torch
        checks['PyTorch available'] = True
        checks['CUDA available'] = torch.cuda.is_available()

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            checks[f'GPU name (got: {gpu_name})'] = True
            checks[f'VRAM >= 40GB (got: {vram_gb:.1f}GB)'] = vram_gb >= 40
            checks['BF16 support'] = torch.cuda.is_bf16_supported()

            # TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            checks['TF32 support'] = True
        else:
            checks['VRAM >= 40GB'] = False
            checks['BF16 support'] = False
    except ImportError:
        checks['PyTorch available'] = False
        checks['CUDA available'] = False
        checks['VRAM >= 40GB'] = False
        checks['BF16 support'] = False

    return checks


def check_disk() -> dict:
    """Kiểm tra disk space."""
    free_gb = shutil.disk_usage('/').free / 1e9
    return {
        f'Disk >= 50GB free (got: {free_gb:.1f}GB)': free_gb >= 50,
    }


def check_ram() -> dict:
    """Kiểm tra RAM."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    total_kb = int(line.split()[1])
                    total_gb = total_kb / (1024 ** 2)
                    return {f'RAM >= 32GB (got: {total_gb:.1f}GB)': total_gb >= 32}
    except Exception:
        pass
    return {'RAM check': True}  # Skip nếu không đọc được


def check_packages() -> dict:
    """Kiểm tra các packages cần thiết."""
    required = {
        'torch': 'torch',
        'transformers': 'transformers',
        'peft (LoRA)': 'peft',
        'ultralytics (YOLO)': 'ultralytics',
        'PIL (Pillow)': 'PIL',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yaml': 'yaml',
        'rapidfuzz': 'rapidfuzz',
        'qwen_vl_utils': 'qwen_vl_utils',
    }
    optional = {
        'flash_attn (FA2)': 'flash_attn',
        'trdg (synthetic)': 'trdg',
        'paddleocr': 'paddleocr',
        'accelerate': 'accelerate',
        'datasets': 'datasets',
    }

    results = {}
    for name, module in required.items():
        results[f'[REQUIRED] {name}'] = run_python_import(module)
    for name, module in optional.items():
        results[f'[OPTIONAL] {name}'] = run_python_import(module)
    return results


def check_paths() -> dict:
    """Kiểm tra các paths quan trọng."""
    data = Path(os.getenv('DATA_ROOT', default_data_root()))

    checks = {
        f'Data root exists ({data})': data.exists(),
        f'Train images exist ({data}/train/images)': (data / 'train' / 'images').exists(),
        f'Train metadata exists ({data}/train/metadata.jsonl)': (data / 'train' / 'metadata.jsonl').exists(),
        f'Test images exist ({data}/test/images)': (data / 'test' / 'images').exists(),
        f'Test metadata exists ({data}/test/metadata.jsonl)': (data / 'test' / 'metadata.jsonl').exists(),
    }

    # kaggle_metric.py phải có
    metric_path = Path('kaggle_metric.py')
    if not metric_path.exists():
        metric_path = Path('..') / 'kaggle_metric.py'
    checks['kaggle_metric.py exists'] = metric_path.exists()

    return checks


def check_artifacts() -> dict:
    """Kiểm tra các artifacts từ từng phase."""
    art = Path(os.getenv('HTR_ART_DIR', default_art_dir()))

    return {
        '[Phase 0] config.json': (art / 'config.json').exists(),
        '[Phase 0] train_split.jsonl': (art / 'train_split.jsonl').exists(),
        '[Phase 0] valid_split.jsonl': (art / 'valid_split.jsonl').exists(),
        '[Phase 0B] synth_manifest.csv': (art / 'rec_data' / 'synth_manifest.csv').exists(),
        '[Phase 1] det_yolo/best.pt': (art / 'det_yolo' / 'best.pt').exists(),
        '[Phase 2] qwen3_lora/best_checkpoint': (art / 'qwen3_lora' / 'best_checkpoint').exists(),
        '[Phase 3] best_config.json': (art / 'best_config.json').exists(),
        '[Phase 4] submission.csv': (art / 'submission.csv').exists(),
    }


def auto_fix():
    """Fix những gì có thể tự động."""
    art = Path(os.getenv('HTR_ART_DIR', default_art_dir()))

    # Tạo thư mục artifacts
    for subdir in ['logs', 'det_yolo', 'qwen3_lora', 'rec_data/crops', 'rec_data/synthetic']:
        p = art / subdir
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f'[fix] Created: {p}')

    print('[fix] Directory structure created.')
    print('[fix] Run missing packages: pip install transformers peft ultralytics qwen-vl-utils rapidfuzz')
    print('[fix] Flash Attention: pip install flash-attn --no-build-isolation')
    print('[fix] PaddleOCR: pip install paddlepaddle-gpu paddleocr')


def main():
    parser = argparse.ArgumentParser(description='Check RUKOPYS environment')
    parser.add_argument('--fix', action='store_true', help='Auto-fix what can be fixed')
    parser.add_argument('--artifacts', action='store_true', help='Show artifact status')
    args = parser.parse_args()

    if args.fix:
        auto_fix()
        print()

    print('=' * 60)
    print('RUKOPYS ENVIRONMENT CHECK')
    print('=' * 60)

    all_groups = [
        ('Hardware', {**check_cuda(), **check_disk(), **check_ram()}),
        ('Packages', check_packages()),
        ('Data Paths', check_paths()),
    ]

    if args.artifacts:
        all_groups.append(('Artifacts', check_artifacts()))

    total_ok = total_fail = 0
    required_failures = []

    for group_name, checks in all_groups:
        print(f'\n--- {group_name} ---')
        for name, passed in checks.items():
            status = '✓' if passed else '✗ FAIL'
            print(f'  {status:<10} {name}')
            if passed:
                total_ok += 1
            else:
                total_fail += 1
                if '[REQUIRED]' in name or group_name == 'Hardware':
                    required_failures.append(name)

    print()
    print('=' * 60)
    if total_fail == 0:
        print('✓  ALL CHECKS PASSED — Environment is ready!')
    else:
        print(f'✗  {total_fail} check(s) failed, {total_ok} passed')
        if required_failures:
            print('\nCRITICAL failures (must fix before running):')
            for f in required_failures:
                print(f'  - {f}')
    print('=' * 60)

    # Return exit code — useful for CI
    sys.exit(0 if total_fail == 0 else 1)


if __name__ == '__main__':
    main()
