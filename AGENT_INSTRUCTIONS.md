# AGENT_INSTRUCTIONS.md — Kim Chỉ Nam Cho Codex
## ĐỌC FILE NÀY TRƯỚC KHI LÀM BẤT CỨ ĐIỀU GÌ

---

## 1. Phần cứng thực tế (KHÔNG ĐƯỢC BỎ QUA)

| Resource | Giá trị | Ràng buộc code |
|---|---|---|
| GPU | 1× NVIDIA L40 | `device = "cuda:0"` duy nhất, KHÔNG `device_map="auto"` multi-GPU |
| VRAM | 48GB | Fit Qwen3-VL-8B BF16 + LoRA rank-64, KHÔNG cần quantize |
| System RAM | 64GB | **KHÔNG** `cache='ram'` cho YOLO; lazy-load tất cả dataset |
| vCPU | 8 cores | `dataloader_num_workers = 4` (không quá 6) |
| Disk | 200GB | `save_total_limit = 2`; xóa intermediate crops sau khi train xong |
| OS | Ubuntu 24 | Flash Attention 2 + TF32 đều support |

### Quy tắc bộ nhớ cứng (vi phạm → OOM crash)

```python
# FORBIDDEN — sẽ dùng hết 64GB RAM:
cache = 'ram'                    # KHÔNG trong YOLO
df = pd.read_csv(huge_file)      # KHÔNG load toàn bộ, dùng chunked read
images = [Image.open(p) for p in all_paths]  # KHÔNG load hết ảnh

# REQUIRED — lazy loading:
class LazyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return Image.open(self.paths[idx])   # mở từng ảnh khi cần
```

---

## 2. Thứ tự triển khai (TUẦN TỰ, KHÔNG BỎ BƯỚC)

```
BƯỚC 0A: scripts/00_setup_inspect.py
  → Output: htr_artifacts/config.json, train_split.jsonl, valid_split.jsonl
  → Checkpoint: htr_artifacts/logs/phase0.log
  → Kiểm tra: assert Path('htr_artifacts/config.json').exists()

BƯỚC 0B: scripts/00b_generate_synthetic.py  [có thể chạy song song với 0A]
  → Output: htr_artifacts/rec_data/synthetic/  (~100k crops)
  → Checkpoint: htr_artifacts/logs/synth_progress.json
  → Kiểm tra: assert count(htr_artifacts/rec_data/synthetic/*.jpg) > 10000

BƯỚC 1: scripts/01_train_detector.py
  → Input: htr_artifacts/config.json, train_split.jsonl
  → Output: htr_artifacts/det_yolo/best.pt, htr_artifacts/det_yolo/args.yaml
  → Checkpoint: htr_artifacts/det_yolo/runs/ (auto-resume với --resume)
  → Kiểm tra: assert Path('htr_artifacts/det_yolo/best.pt').exists()

BƯỚC 2: scripts/02_train_recognizer.py
  → Input: htr_artifacts/config.json, rec_data/
  → Output: htr_artifacts/qwen3_lora/  (adapter weights)
  → Checkpoint: htr_artifacts/qwen3_lora/checkpoint-*/  (auto-resume)
  → Kiểm tra: assert any(Path('htr_artifacts/qwen3_lora').glob('checkpoint-*'))

BƯỚC 3: scripts/03_validate.py
  → Input: best.pt + qwen3_lora/best_checkpoint + valid_split.jsonl
  → Output: htr_artifacts/best_config.json  (optimal conf/iou thresholds)
  → Kiểm tra: assert Path('htr_artifacts/best_config.json').exists()

BƯỚC 4: scripts/04_inference.py
  → Input: best_config.json + test metadata
  → Output: htr_artifacts/submission.csv
  → Kiểm tra: wc -l htr_artifacts/submission.csv  (== test_count + 1)
```

---

## 3. Cấu trúc artifact bắt buộc

```
htr_artifacts/
├── config.json                  # paths, classes, dataset stats
├── train_split.jsonl            # training records
├── valid_split.jsonl            # validation records
├── logs/
│   ├── phase0.log
│   ├── phase1.log
│   ├── phase2.log
│   └── inference.log
├── det_yolo/
│   ├── best.pt                  # YOLO best weights
│   ├── last.pt                  # YOLO last checkpoint
│   ├── args.yaml                # training args (for resume)
│   └── dataset/                 # YOLO dataset structure
│       ├── images/{train,valid}/
│       └── labels/{train,valid}/
├── rec_data/
│   ├── train_manifest.csv       # image_path, text, weight, source
│   ├── valid_manifest.csv
│   ├── crops/                   # gold crop images
│   └── synthetic/               # TRDG synthetic crops
├── qwen3_lora/
│   ├── checkpoint-500/          # intermediate checkpoints
│   ├── checkpoint-1000/
│   ├── best_checkpoint/         # symlink → best eval checkpoint
│   └── training_log.json
├── best_config.json             # optimal thresholds from Phase 3
└── submission.csv               # final output
```

---

## 4. Error handling bắt buộc

### 4.1 CUDA OOM trong inference

```python
# Pattern này PHẢI có trong mọi inference loop:
def safe_infer_batch(batch, model, processor, current_bs):
    try:
        return run_batch(batch, model, processor), current_bs
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        gc.collect()
        new_bs = max(1, current_bs // 2)
        print(f'[OOM] batch={current_bs} → retry bs={new_bs}')
        if new_bs == current_bs:   # đã là 1 mà vẫn OOM
            return [''] * len(batch), 1  # skip gracefully
        return safe_infer_batch(batch, model, processor, new_bs)
```

### 4.2 Checkpoint resume

```python
# Mọi training script PHẢI support resume:
last_ckpt = find_latest_checkpoint(OUTPUT_DIR)
trainer.train(resume_from_checkpoint=last_ckpt)
# Hoặc với YOLO:
model.train(..., resume=True if last_ckpt else False)
```

### 4.3 Disk space guard

```python
import shutil
def check_disk(min_gb=10, path='/'):
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_gb:
        raise RuntimeError(f'[DISK] Only {free_gb:.1f}GB free! Need {min_gb}GB minimum.')
    return free_gb

# Gọi trước mỗi checkpoint save:
check_disk(min_gb=10)
```

---

## 5. Constants toàn cục (COPY VÀO MỌI SCRIPT)

```python
# ===== HARDWARE CONSTANTS — L40 48GB / 8vCPU / 64GB RAM =====
DEVICE          = 'cuda:0'
VRAM_GB         = 48
RAM_GB          = 64
N_CPU           = 8
DL_WORKERS      = 4          # max 6
USE_BF16        = True       # L40 Ada native BF16
USE_TF32        = True       # Ada matmul TF32
USE_FLASH_ATTN  = True       # L40 support FA2
GRAD_CHECKPT    = False      # 48GB đủ → tắt để nhanh hơn 30%
SAVE_LIMIT      = 2          # bảo vệ 200GB disk

# ===== PATHS =====
ART             = Path(os.getenv('HTR_ART_DIR', './htr_artifacts'))
DATA_ROOT       = Path(os.getenv('DATA_ROOT',   './data'))
```

---

## 6. Kiểm tra môi trường (chạy đầu tiên)

```python
# scripts/check_env.py — Codex PHẢI tạo và chạy file này trước
import torch, subprocess, shutil
from pathlib import Path

checks = {
    'CUDA available': torch.cuda.is_available(),
    'GPU name L40': 'L40' in torch.cuda.get_device_name(0),
    'VRAM >= 40GB': torch.cuda.get_device_properties(0).total_memory / 1e9 >= 40,
    'BF16 support': torch.cuda.is_bf16_supported(),
    'Disk >= 50GB': shutil.disk_usage('/').free / 1e9 >= 50,
    'flash-attn installed': bool(subprocess.run(
        ['python', '-c', 'import flash_attn'], capture_output=True).returncode == 0),
}

for name, passed in checks.items():
    status = '✓' if passed else '✗ FAIL'
    print(f'  {status}  {name}')

all_ok = all(checks.values())
print('\n' + ('=== ENV OK ===' if all_ok else '=== ENV HAS ISSUES ==='))
```

---

## 7. Thông tin về dataset và metric

- **Score = 0.15×DetF1 + 0.05×ClassAcc + 0.30×(1−CER) + 0.50×(1−PageCER)**
- PageCER chiếm **50%** tổng điểm → Reading order cực kỳ quan trọng
- Metric tự normalize text (Latin↔Cyrillic lookalike, dashes, quotes) → đừng waste time post-process
- Region types: `handwritten`, `printed`, `formula`, `table`, `annotation`, `image`, `graph`
- Chỉ submit `bbox`, `type`, `text` — KHÔNG cần `language`, `legibility`
- Nếu ảnh không có region: submit `[]`
- File `kaggle_metric.py` (trong repo gốc) có hàm `score_detailed()` → dùng local để debug
