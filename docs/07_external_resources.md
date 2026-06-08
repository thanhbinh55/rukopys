# docs/07_external_resources.md — External Resources & Integration Guide

## Tổng quan

File này hướng dẫn chi tiết cách tích hợp các nguồn tài nguyên bên ngoài vào pipeline RUKOPYS.
Mỗi nguồn được mô tả rõ: mục đích, cách cài đặt, cách tích hợp, và trọng số (weight) trong training.

---

## 1. TextRecognitionDataGenerator (TRDG)

**Link:** https://github.com/Belval/TextRecognitionDataGenerator

### Mục đích

Sinh dữ liệu tổng hợp (synthetic) cho OCR recognizer. Render text Cyrillic lên nền ảnh giả với font đa dạng, noise, blur, skew. Giúp model học ký tự Ukrainian mà không cần annotation thủ công.

### Cài đặt

```bash
pip install trdg
# Cài thêm fonts Cyrillic — bắt buộc để generate đúng Ukrainian
mkdir -p ./fonts/cyrillic
# Download từ Google Fonts (filter: Ukrainian support)
# Fonts tốt nhất: Roboto, Noto Sans, Open Sans, Marmelad (Google Fonts)
# Handwriting fonts: Caveat, Kalam, Dekko (đều có Cyrillic)
```

### Config tối ưu cho Ukrainian

```python
from trdg.generators import GeneratorFromStrings

generator = GeneratorFromStrings(
    strings=texts,               # list[str] từ gold data + word list
    count=100000,
    fonts=cyrillic_font_paths,   # PHẢI có Cyrillic — không dùng Latin fonts
    language='uk',
    size=64,                     # height in pixels
    skewing_angle=3,
    random_skew=True,
    blur=1,
    random_blur=True,
    background_type=0,           # 0=Gaussian noise, 1=plain white, 2=quasicrystal
    distorsion_type=0,           # 0=none, 1=sin, 2=cos, 3=random
    is_handwritten=False,        # TRDG's built-in handwriting mode — optional
    width=-1,                    # auto width
    alignment=0,
    text_color='#2a2a2a',
    orientation=0,
    space_width=1.0,
    fit=True,
)
```

### Weight trong training

```python
weight = 0.3  # thấp hơn real data (1.0) vì synthetic không hoàn hảo
```

### Nguồn text content

```
1. Gold annotations từ train_split.jsonl (text của từng region)
2. Ukrainian word list: UberText corpus (https://lang.org.ua/en/corpora/#anchor11)
3. Kobza corpus (Ukrainian literature public domain)
4. Dictation GT texts (2020–2025, public domain)
```

### Tích hợp vào pipeline

Scripts: `00b_generate_synthetic.py` → output: `htr_artifacts/rec_data/synthetic/`
Manifest: `htr_artifacts/rec_data/synth_manifest.csv` (columns: image_path, text, weight, source)

---

## 2. doc_gen — Document Layout Generator

**Link:** https://github.com/mhlzcu/doc_gen

### Mục đích

Generate full-page document layout images với các vùng text, table, formula, image được arrange ngẫu nhiên. Dùng để tăng training data cho **YOLO detector** — đặc biệt cho các classes hiếm như `formula`, `table`, `annotation`.

### Cài đặt

```bash
git clone https://github.com/mhlzcu/doc_gen ./tools/doc_gen
cd ./tools/doc_gen
pip install -r requirements.txt
```

### Cách dùng

```python
# Generate 5000 synthetic document pages
python tools/doc_gen/generate.py \
    --output_dir htr_artifacts/det_yolo/synth_pages \
    --n_pages 5000 \
    --classes handwritten printed formula table annotation \
    --width 2480 --height 3508 \
    --lang uk

# Convert sang YOLO format
python tools/doc_gen/to_yolo.py \
    --input_dir htr_artifacts/det_yolo/synth_pages \
    --output_dir htr_artifacts/det_yolo/dataset/synth \
    --classes handwritten printed formula table annotation image graph
```

### Weight trong YOLO training

Dùng `copy_paste=0.1` augmentation trong YOLO config để mix synthetic vào training batch.
Hoặc thêm trực tiếp vào `dataset/images/train/` và `dataset/labels/train/`.

### Lưu ý

- Synthetic detector data ít quan trọng hơn synthetic OCR data
- Nếu real training data đủ (>5k pages), có thể bỏ qua doc_gen
- Ưu tiên dùng doc_gen cho classes `formula` và `table` vì dataset thật thường thiếu

---

## 3. HKR Dataset — Handwritten Kazakh & Russian

**Link:** https://github.com/abdoelsayed2016/HKR_Dataset

### Mục đích

12,000+ word-level crop images viết tay tiếng Kazakh và Russian — Cyrillic rất gần với Ukrainian.
Dùng làm **additional real data** cho OCR recognizer. Không cần YOLO (đã được crop sẵn).

### Download

```bash
# Kaggle dataset
kaggle datasets download -d abdoelsayed2016/hkr-dataset
unzip hkr-dataset.zip -d data/hkr/

# Hoặc từ GitHub (train/test splits có sẵn)
git clone https://github.com/abdoelsayed2016/HKR_Dataset ./data/hkr
```

### Format dữ liệu

```
data/hkr/
├── train/
│   ├── 000001.jpg  (word crop, ~150×64 px)
│   └── ...
├── test/
│   └── ...
└── annotation.json   # {"000001.jpg": "слово", ...}
```

### Convert sang manifest format

```python
import json, pandas as pd
from pathlib import Path

ann = json.load(open('data/hkr/annotation.json'))
records = []
for fname, text in ann.items():
    img_path = Path('data/hkr/train') / fname
    if img_path.exists():
        records.append({
            'image_path': str(img_path),
            'text': text,
            'weight': 0.5,       # thấp hơn Ukrainian gold (1.0)
            'source': 'hkr',
        })

df = pd.DataFrame(records)
df.to_csv('htr_artifacts/hkr_manifest.csv', index=False)
print(f'HKR: {len(df):,} crops')
```

### Weight trong training

```python
weight = 0.5  # giảm vì là Kazakh/Russian, không phải Ukrainian hoàn toàn
max_hkr = 50000  # không dùng quá nhiều để không lấn át Ukrainian data
```

---

## 4. Qwen3-VL-8B-Instruct (Main Recognizer)

**Link:** https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

### Tại sao chọn Qwen3-VL-8B

- State-of-the-art VLM 2025, vượt Qwen2.5-VL-7B trên tất cả OCR benchmarks
- Cyrillic pretraining mạnh hơn TrOCR rất nhiều
- Fit L40 48GB ở BF16 với LoRA rank-64 và batch=4
- Flash Attention 2 support native trên Ada Lovelace

### Cài đặt

```bash
pip install "transformers>=4.57.0" "peft>=0.13.0" qwen-vl-utils
pip install flash-attn --no-build-isolation  # FA2 — PHẢI install riêng

# Download model (sẽ cache tự động khi load)
python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3-VL-8B-Instruct')"
```

### Load pattern cho inference

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
import torch

processor = AutoProcessor.from_pretrained(
    'Qwen/Qwen3-VL-8B-Instruct',
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen3-VL-8B-Instruct',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    device_map={'': 'cuda:0'},      # KHÔNG 'auto' — single GPU
)

# PHẢI tắt thinking mode
if hasattr(model.generation_config, 'enable_thinking'):
    model.generation_config.enable_thinking = False
```

### LoRA config tối ưu

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,           # = 2r
    use_rslora=True,          # Rank-Stabilized LoRA — stable với r=64
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],
    lora_dropout=0.05,
    bias='none',
)
```

### VRAM usage trên L40 48GB

| Config | VRAM |
|--------|------|
| BF16 base model | ~16GB |
| + LoRA r=64 | ~17GB |
| + Batch=4 + activations | ~35–40GB |
| Headroom | ~8–13GB ✓ |

---

## 5. Google Gemma 4 (Alternative Recognizer)

**Link:** https://huggingface.co/collections/google/gemma-4

### Các variants có thể dùng trên L40 48GB

| Model | Params | VRAM (BF16) | Ghi chú |
|-------|--------|-------------|---------|
| gemma-4-E2B-it | 5B | ~10GB | Nhanh nhất, baseline |
| gemma-4-E4B-it | 8B MoE | ~15GB | Cân bằng speed/accuracy |
| gemma-4-26B-A4B-it | 27B | ~40GB | Tốt nhất fit L40, cần careful |

### Khi nào dùng Gemma thay Qwen3-VL

- Khi Qwen3-VL chưa available trên HuggingFace Hub
- Khi muốn ensemble 2 model khác architecture
- Gemma 4 mạnh hơn trên printed text; Qwen3 mạnh hơn handwriting

### Load Gemma 4

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

processor = AutoProcessor.from_pretrained('google/gemma-4-E4B-it')
model = AutoModelForImageTextToText.from_pretrained(
    'google/gemma-4-E4B-it',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
    attn_implementation='flash_attention_2',
)
```

### LoRA config cho Gemma 4

```python
LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,              # Gemma nhỏ hơn → r thấp hơn là đủ
    lora_alpha=64,
    use_rslora=True,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05,
)
```

---

## 6. PaddleOCR (Fallback Ensemble)

**Link:** https://github.com/PaddlePaddle/PaddleOCR

### Mục đích

Dùng làm **fallback** khi Qwen3-VL uncertain (output rỗng hoặc quá ngắn).
PaddleOCR inference nhanh hơn Qwen3 ~10× nên overhead negligible.

### Cài đặt

```bash
pip install paddlepaddle-gpu paddleocr
# Hoặc CPU version:
pip install paddlepaddle paddleocr

# Download Cyrillic models tự động khi khởi tạo lần đầu
python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='ru')"
```

### Sử dụng

```python
from paddleocr import PaddleOCR
import numpy as np

# lang='ru' = Russian Cyrillic, gần nhất với Ukrainian
ocr = PaddleOCR(
    lang='ru',
    use_gpu=True,
    use_angle_cls=True,   # detect rotated text
    show_log=False,
)

def paddle_ocr_crop(crop_img, ocr_engine):
    arr = np.array(crop_img)
    result = ocr_engine.ocr(arr, cls=True)
    if not result or not result[0]:
        return '', 0.0
    texts, confs = [], []
    for line in result[0]:
        texts.append(line[1][0])
        confs.append(line[1][1])
    return ' '.join(texts), min(confs) if confs else 0.0
```

### Ensemble strategy

```
Qwen3-VL output:
  len >= 3 chars → dùng Qwen3 text (tin tưởng)
  len < 3 chars  → chạy PaddleOCR
                   → nếu PaddleOCR conf > 0.5 → dùng PaddleOCR
                   → ngược lại → dùng Qwen3 (dù ngắn)
```

---

## 7. TrOCR — microsoft/trocr-base-handwritten

**Link:** https://huggingface.co/microsoft/trocr-base-handwritten

### Vai trò trong pipeline

**Không phải main model** — dùng để:
1. Làm baseline comparison
2. Ensemble với Qwen3-VL cho printed text regions
3. Fallback khi không có đủ VRAM cho Qwen3

### Load TrOCR

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained(
    'microsoft/trocr-base-handwritten',
    torch_dtype=torch.float16,
).to('cuda:0')

def trocr_infer(crop_img, processor, model):
    pixel_values = processor(crop_img, return_tensors='pt').pixel_values.to('cuda:0')
    with torch.no_grad():
        ids = model.generate(pixel_values, max_new_tokens=128)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
```

### Fine-tuning TrOCR trên Ukrainian (nếu cần)

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir='htr_artifacts/trocr_finetuned',
    per_device_train_batch_size=32,  # TrOCR nhỏ → batch lớn được
    learning_rate=5e-5,
    num_train_epochs=5,
    predict_with_generate=True,
    generation_max_length=128,
    bf16=True,
    save_total_limit=2,
)
```

---

## 8. Kansallisarkisto Cyrillic HTR Model

**Link:** https://huggingface.co/Kansallisarkisto

### Vai trò

Model được train trên 30k+ dòng chữ viết tay Cyrillic archival từ National Archives of Finland.
Rất tốt cho **archival/historical documents** (phù hợp với ЦДАВО split trong RUKOPYS).

### Models có sẵn

```
Kansallisarkisto/cyrillic-htr-model  — chính, recognition
```

### Load và sử dụng

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained('Kansallisarkisto/cyrillic-htr-model')
model = AutoModelForVision2Seq.from_pretrained(
    'Kansallisarkisto/cyrillic-htr-model',
    torch_dtype=torch.float16,
).to('cuda:0')
```

### Khi nào dùng Kansallisarkisto

- Ensemble riêng cho test images có nguồn gốc archival (ЦДАВО 1919–1935)
- Nếu detect được document source type, route archival pages qua model này
- Combine text với Qwen3-VL output qua voting CER

---

## 9. Chiến lược tích hợp tổng thể

### Data pipeline (thứ tự ưu tiên)

```
Priority 1: Gold annotations (weight=1.0)
Priority 2: Silver annotations (weight=0.5)
Priority 3: HKR Cyrillic dataset (weight=0.5, max 50k)
Priority 4: TRDG synthetic (weight=0.3, max 100k)
Priority 5: doc_gen pages cho YOLO (boost rare classes)
```

### Model ensemble strategy

```
Page inference:
  1. YOLO detect all regions
  2. For each region:
     a. Qwen3-VL-8B-LoRA → text + implicit confidence
     b. If output < 3 chars → PaddleOCR fallback
     c. (Optional) If region.source == 'archive' → Kansallisarkisto ensemble
  3. Reading order sort (bucket_size=15)
  4. Build submission JSON
```

### Cài đặt toàn bộ một lần

```bash
#!/bin/bash
# install_all.sh — chạy một lần để setup environment

# Deep learning core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.57.0" "peft>=0.13.0" "accelerate>=0.30.0"

# Flash Attention 2 (bắt buộc cho L40 Ada Lovelace)
pip install flash-attn --no-build-isolation

# Vision utilities
pip install qwen-vl-utils  # Qwen-VL image processing
pip install timm           # Vision backbone utilities

# OCR
pip install trdg           # Synthetic data generation
pip install paddlepaddle-gpu paddleocr  # Fallback OCR

# YOLO
pip install ultralytics    # YOLOv8/YOLOv9/YOLOv10

# Data
pip install pandas numpy pillow scikit-learn rapidfuzz
pip install datasets huggingface_hub

# Monitoring
pip install tensorboard    # optional, training visualization

echo "Environment setup complete!"
```

---

## 10. Pseudo-labeling với Dictation Ground Truth

### Kỹ thuật nâng cao — không cần annotation bbox

Ukrainian National Dictation texts (2020–2025) là **public domain** và **canonical text đã biết**. Mỗi năm có 1 bài dictation với text cố định cho hàng nghìn người viết.

```python
# Chiến lược pseudo-labeling:
# 1. Load dictation images (không có bbox annotation)
# 2. Dùng YOLO detect lines
# 3. OCR từng line với Qwen3-VL
# 4. Align predicted text với known dictation GT text bằng dynamic programming
# 5. Gán GT text chính xác vào từng line crop → training data mới

from rapidfuzz import fuzz

def align_dictation_lines(ocr_lines, gt_text_words):
    """
    Align YOLO-detected lines với known dictation GT.
    Trả về list (line_crop, gt_text) pairs.
    """
    # Dynamic programming alignment
    # Chi tiết implementation: xem scripts/pseudo_label_dictation.py
    pass
```

Kỹ thuật này cho phép tạo **thêm hàng nghìn high-quality training samples** từ dictation data mà không cần manual annotation.

---

## 11. Checklist tích hợp

Trước khi chạy Phase 2 (train recognizer), kiểm tra:

```python
# scripts/check_data_sources.py
from pathlib import Path

checks = {
    'Gold train data': Path('htr_artifacts/train_split.jsonl').exists(),
    'Gold crops extracted': any(Path('htr_artifacts/rec_data/crops').glob('*.jpg')),
    'Synthetic TRDG': Path('htr_artifacts/rec_data/synth_manifest.csv').exists(),
    'HKR dataset': Path('htr_artifacts/hkr_manifest.csv').exists(),
    'Qwen3-VL model cached': any(Path.home().glob('.cache/huggingface/hub/models--Qwen*')),
    'Flash Attention installed': __import__('subprocess').run(
        ['python', '-c', 'import flash_attn'], capture_output=True).returncode == 0,
    'PaddleOCR installed': __import__('subprocess').run(
        ['python', '-c', 'from paddleocr import PaddleOCR'], capture_output=True).returncode == 0,
}

for name, ok in checks.items():
    print(f"  {'✓' if ok else '✗ MISSING':<12} {name}")
```
