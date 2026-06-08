# docs/03_detector_guide.md — Phase 1: YOLO Region Detector

## Mục tiêu

Fine-tune YOLOv8x để detect và classify các vùng (regions) trong trang tài liệu viết tay Ukrainian.
Output: `htr_artifacts/det_yolo/best.pt` — được dùng bởi Phase 3 và Phase 4.

---

## Tại sao YOLOv8x

| Model | mAP@0.5 (doc) | Speed (L40) | VRAM |
|-------|--------------|-------------|------|
| YOLOv8n | ~0.72 | 200 img/s | 4GB |
| YOLOv8l | ~0.85 | 80 img/s | 18GB |
| **YOLOv8x** | **~0.89** | **45 img/s** | **32GB** |
| YOLOv9e | ~0.88 | 35 img/s | 36GB |

YOLOv8x là điểm cân bằng tốt nhất cho L40 48GB với imgsz=1280.

---

## Cấu trúc dataset YOLO

```
htr_artifacts/det_yolo/dataset/
├── images/
│   ├── train/   ← symlinks tới real images (tiết kiệm disk)
│   └── valid/
└── labels/
    ├── train/   ← .txt files, 1 file/image
    └── valid/
```

### Format label file

```
# Mỗi dòng = 1 bounding box
# class_id cx cy w h  (normalized 0–1)
0 0.512345 0.234567 0.456789 0.123456
2 0.123456 0.567890 0.234567 0.345678
```

### Class mapping

```python
CLASSES = ['handwritten', 'printed', 'formula', 'table', 'annotation', 'image', 'graph']
# ID:           0             1          2         3          4           5        6
```

---

## Hyperparameters chi tiết

### Training config tối ưu cho L40 48GB

```yaml
# configs/yolo_train.yaml
model: yolov8x.pt
data: htr_artifacts/det_yolo/data.yaml

# Hardware
device: cuda:0
batch: 12         # L40 48GB với imgsz=1280 → safe margin
workers: 4        # 8 vCPU → dùng 4 để tránh CPU bottleneck
imgsz: 1280       # Document pages thường rộng, cần resolution cao
cache: false      # 64GB RAM không đủ cache toàn bộ dataset

# Training
epochs: 80
patience: 25      # Early stopping
cos_lr: true      # Cosine annealing
close_mosaic: 20  # Tắt mosaic 20 epoch cuối để ổn định
optimizer: AdamW
lr0: 0.0005       # 5e-4
lrf: 0.01
warmup_epochs: 5
weight_decay: 0.0005
save_period: 10

# Loss weights
cls: 0.7          # Tăng classification loss — 7 classes cần phân biệt tốt
box: 7.5          # Default
dfl: 1.5          # Distribution Focal Loss

# Augmentation — CẨN THẬN với document text
flipud: 0.0       # TUYỆT ĐỐI KHÔNG flip dọc — text sẽ ngược
fliplr: 0.0       # TUYỆT ĐỐI KHÔNG flip ngang — text sẽ mirror
degrees: 2.0      # Rotate nhẹ — document thường thẳng
translate: 0.1
scale: 0.3
mosaic: 0.5
copy_paste: 0.1
hsv_h: 0.015
hsv_s: 0.4
hsv_v: 0.3
```

### Lý do tắt flip augmentations

Đây là lỗi phổ biến nhất khi apply YOLO trên documents:
- `flipud=0.0`: Text bị lộn ngược → model học sai pattern
- `fliplr=0.0`: Text bị mirror → chữ Ukrainian sẽ bị đảo chiều

Tài liệu Ukrainian KHÔNG có text đọc từ phải sang trái hay từ dưới lên trên.

---

## Data.yaml format

```yaml
# htr_artifacts/det_yolo/data.yaml
path: /absolute/path/to/htr_artifacts/det_yolo/dataset
train: images/train
val: images/valid
nc: 7
names:
  0: handwritten
  1: printed
  2: formula
  3: table
  4: annotation
  5: image
  6: graph
```

---

## Training command

```bash
# Lần đầu
python scripts/01_train_detector.py

# Resume từ checkpoint (tự động detect)
python scripts/01_train_detector.py
# Script tự kiểm tra htr_artifacts/det_yolo/last.pt và resume

# Override config
EPOCHS=100 BATCH=8 IMGSZ=1280 python scripts/01_train_detector.py
```

---

## TensorRT Export (optional, 3× faster inference)

```python
from ultralytics import YOLO

model = YOLO('htr_artifacts/det_yolo/best.pt')
model.export(
    format='engine',    # TensorRT
    half=True,          # FP16 — L40 support
    device=0,
    workspace=8,        # GB
    simplify=True,
)
# Output: htr_artifacts/det_yolo/best.engine
```

**Note:** Export mất ~5 phút nhưng inference nhanh hơn 3×. Nên export trước Phase 4.

---

## Checkpoint Resume

YOLO tự động hỗ trợ resume:

```python
from ultralytics import YOLO
from pathlib import Path

last_pt = Path('htr_artifacts/det_yolo/runs/yolov8x_htr/weights/last.pt')
if last_pt.exists():
    model = YOLO(str(last_pt))
    model.train(resume=True)
else:
    model = YOLO('yolov8x.pt')
    model.train(data=yaml_path, ...)
```

---

## Validation metrics

Sau khi train, kiểm tra:

```bash
# Quick validation
python -c "
from ultralytics import YOLO
model = YOLO('htr_artifacts/det_yolo/best.pt')
results = model.val(data='htr_artifacts/det_yolo/data.yaml', device='cuda:0')
print('mAP50:', results.box.map50)
print('mAP50-95:', results.box.map)
print('Per-class mAP:', results.box.maps)
"
```

### Targets

| Metric | Target | Ghi chú |
|--------|--------|---------|
| mAP@0.5 | ≥ 0.85 | Detection quality |
| DetF1 trên valid (kaggle metric) | ≥ 0.85 | Dùng `score_detailed()` |
| ClassAcc | ≥ 0.85 | 7-class classification |

---

## Debug: Class imbalance

Nếu một số classes (formula, table, graph) có ít examples:

```python
# Kiểm tra distribution trong dataset
from pathlib import Path
from collections import Counter

label_dir = Path('htr_artifacts/det_yolo/dataset/labels/train')
class_counts = Counter()
for lbl_file in label_dir.glob('*.txt'):
    for line in lbl_file.read_text().strip().split('\n'):
        if line:
            cls_id = int(line.split()[0])
            class_counts[cls_id] += 1

CLASSES = ['handwritten','printed','formula','table','annotation','image','graph']
for cls_id, count in sorted(class_counts.items()):
    print(f'{CLASSES[cls_id]:15s}: {count:,}')
```

Nếu imbalance > 10×:
1. Thêm doc_gen synthetic pages cho classes thiếu
2. Tăng `cls=0.7` lên `cls=1.0` trong training config
3. Sử dụng weighted sampling trong YOLO (cần custom dataloader)

---

## File outputs sau Phase 1

```
htr_artifacts/det_yolo/
├── best.pt           ← QUAN TRỌNG NHẤT — dùng cho inference
├── last.pt           ← checkpoint cuối, dùng để resume
├── best.engine       ← TensorRT (optional, faster inference)
├── data.yaml         ← dataset config
├── args.yaml         ← training args (để reproducibility)
└── runs/
    └── yolov8x_htr/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv    ← training metrics per epoch
        ├── val_batch*.jpg ← validation visualization
        └── confusion_matrix.png
```
