# RUKOPYS HTR — Production Pipeline
### 1× L40 (48GB VRAM) · 8 vCPU · 64GB RAM · 200GB Disk

> **Mục tiêu:** Nâng score từ 0.17 → 0.90+ trên leaderboard RUKOPYS (Ukrainian Handwritten Text Recognition).

---

## Kiến trúc tổng quan

```
┌──────────────────────────────────────────────────────────────────┐
│                      RUKOPYS PIPELINE                            │
│                                                                  │
│  Phase 0A  Phase 0B  Phase 0C  Phase 1  Phase 2  Phase 3  Phase4│
│  Data Prep  Synth    HKR Prep  Detector  Recog   Validate  Infer │
│  & Audit    TRDG     Dataset   YOLOv8x  Qwen3VL  & Tune   Submit │
│                                          8B LoRA  Official  Paddle│
│                                                   Metric   OCR   │
└──────────────────────────────────────────────────────────────────┘
```

## Cấu trúc thư mục đầy đủ

```
rukopys/
├── README.md
├── AGENT_INSTRUCTIONS.md          ← kim chỉ nam cho Codex (ĐỌC TRƯỚC TIÊN)
│
├── docs/
│   ├── 00_algorithm_overview.md   ← mô tả từng thuật toán, kiến trúc 2-stage
│   ├── 01_metric_deep_dive.md     ← phân tích metric chính thức (ĐỌC NGAY SAU AGENT)
│   ├── 02_data_pipeline.md        ← data flow, Phase 0A + 0B chi tiết
│   ├── 03_detector_guide.md       ← Phase 1: YOLOv8x, hyperparams, TensorRT export
│   ├── 04_recognizer_guide.md     ← Phase 2: Qwen3-VL LoRA SFT full implementation
│   ├── 05_validation_guide.md     ← Phase 3: grid search + official metric eval
│   ├── 06_inference_guide.md      ← Phase 4: end-to-end submission, OOM handling
│   ├── 07_external_resources.md  ← TRDG, HKR, Gemma4, PaddleOCR, TrOCR integration
│   └── 08_thunder_gpu_setup.md   ← Setup Thunder GPU + Kaggle dataset
│
├── scripts/
│   ├── install_env.sh             ← Cài đặt toàn bộ environment (chạy một lần)
│   ├── check_env.py               ← Kiểm tra environment (chạy TRƯỚC TIÊN)
│   ├── 00_setup_inspect.py        ← Phase 0A: data audit + train/valid split
│   ├── 00b_generate_synthetic.py  ← Phase 0B: TRDG synthetic crops (100k images)
│   ├── 00c_prepare_hkr.py         ← Phase 0C: HKR Cyrillic dataset preparation
│   ├── 01_train_detector.py       ← Phase 1: YOLOv8x fine-tune
│   ├── 02_train_recognizer.py     ← Phase 2: Qwen3-VL-8B LoRA SFT
│   ├── 03_validate.py             ← Phase 3: grid search + official metric
│   ├── 04_inference.py            ← Phase 4: end-to-end submission
│   └── inference_utils.py         ← Shared utilities (Phase 3 & 4 import từ đây)
├── setup_kaggle_dataset.sh        ← Tải/kiểm tra Kaggle dataset trên GPU server
├── .gitignore                     ← Không commit kaggle.json, dataset, artifacts
│
├── configs/
│   ├── hardware.yaml              ← L40 hardware constants
│   ├── model.yaml                 ← model hyperparameters
│   └── paths.yaml                 ← data/output paths
│
└── kaggle_metric.py               ← official scoring function (copy từ competition)
```

---

## Dataset

Source code **không chứa dataset**. Dữ liệu được tải trên GPU instance từ Kaggle:

```text
https://www.kaggle.com/datasets/bnthanh/rukopys-dataset
```

Sau khi clone repo trên instance, đặt `kaggle.json` ở root repo hoặc `~/.kaggle/kaggle.json`, rồi chạy:

```bash
bash setup_kaggle_dataset.sh
source .env.kaggle
python scripts/show_run_config.py
```

Script sẽ tải dataset vào:

```text
data/kaggle_input/rukopys-dataset/rukopys_raw/
```

`kaggle.json`, `rukopys_raw/`, `data/kaggle_input/`, `local_working/`, artifact train và `submission.csv` đều nằm trong `.gitignore`.

---

## Thứ tự triển khai bắt buộc

```
BƯỚC 0: Cài đặt môi trường
  bash scripts/install_env.sh

BƯỚC 1: Tải data từ Kaggle
  bash setup_kaggle_dataset.sh
  source .env.kaggle
  python scripts/show_run_config.py
  python scripts/check_env.py

BƯỚC 2: Data setup
  python scripts/00_setup_inspect.py          # Phase 0A
  python scripts/00b_generate_synthetic.py    # Phase 0B (song song với 0A)
  python scripts/00c_prepare_hkr.py           # Phase 0C (optional)

BƯỚC 3: Train detector
  python scripts/01_train_detector.py         # Phase 1 (~4h trên L40)

BƯỚC 4: Train recognizer
  python scripts/02_train_recognizer.py       # Phase 2 (~8h trên L40)

BƯỚC 5: Validation & tuning
  python scripts/03_validate.py              # Phase 3 (~2h)

BƯỚC 6: Generate submission
  python scripts/04_inference.py             # Phase 4 (~1-2h)
```

---

## Score trajectory dự kiến

| Sau bước | Score | Thời gian |
|---|---|---|
| Baseline cũ (Kansallisarkisto) | 0.17 | — |
| Phase 1 + Phase 2 (TrOCR-large) | 0.55–0.65 | 4h |
| Phase 1 + Phase 2 (Qwen3-VL-8B LoRA) | 0.78–0.85 | 8h |
| + Synthetic data (TRDG) + HKR | 0.83–0.89 | +2h |
| + PaddleOCR ensemble | 0.87–0.93 | +1h |

---

## Score formula (quan trọng nhất)

```
Score = 0.15 × DetF1 + 0.05 × ClassAcc + 0.30 × (1−CER) + 0.50 × (1−PageCER)
```

**PageCER chiếm 50%** → Reading order sort chính xác là cực kỳ quan trọng.
Xem `docs/01_metric_deep_dive.md` để hiểu từng component.

---

## Debug workflow

```python
from kaggle_metric import score_detailed
import pandas as pd

sol = pd.read_csv('htr_artifacts/valid_gt.csv')
sub = pd.read_csv('htr_artifacts/valid_pred.csv')
r = score_detailed(sol, sub, 'image')

# Diagnosis:
# page_cer >> region_cer → Reading order sai → xem reading_order_sort()
# detection_recall < 0.7  → Giảm YOLO conf threshold
# class_acc < 0.8          → Tăng cls weight trong YOLO
# region_cer > 0.3         → Train Qwen3 thêm epochs / data
```

---

## Checkpoint recovery

Tất cả scripts đều **tự động resume** từ checkpoint:

```bash
# Nếu Phase 2 bị interrupt, chỉ cần chạy lại
python scripts/02_train_recognizer.py
# → Tự detect checkpoint-* và resume

# YOLO tương tự
python scripts/01_train_detector.py
# → Tự detect last.pt và resume với resume=True
```
