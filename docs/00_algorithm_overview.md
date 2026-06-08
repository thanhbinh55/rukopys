# docs/00_algorithm_overview.md — Mô Tả Thuật Toán

## Tổng quan kiến trúc 2-stage

Pipeline theo thiết kế **Detect → Recognize** cổ điển nhưng được nâng cấp với các thành phần state-of-the-art:

```
Ảnh trang tài liệu
       │
       ▼
┌─────────────────┐
│  Stage 1        │  YOLOv8x fine-tuned
│  DETECTOR       │  → bbox [x1,y1,x2,y2]
│                 │  → region type (7 classes)
└────────┬────────┘
         │  crop từng region
         ▼
┌─────────────────┐
│  Stage 2        │  Qwen3-VL-8B + LoRA
│  RECOGNIZER     │  → text per region
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reading Order  │  Bucket sort y//15 → sort x
│  Sort           │  (mirror metric logic)
└────────┬────────┘
         │
         ▼
   submission.csv
```

---

## Module 1: YOLOv8x Detector

### Thuật toán

**YOLOv8** (You Only Look Once v8) là single-stage object detector dùng kiến trúc CSP (Cross Stage Partial) + anchor-free detection head. So với YOLOv5/v7:
- Anchor-free → không cần tune anchor sizes
- Decoupled head cho classification và regression
- DFL (Distribution Focal Loss) cho bbox regression chính xác hơn

**Tại sao chọn YOLOv8x (xlarge)?**
- Nano (n) / Small (s): nhanh nhưng bỏ sót region nhỏ
- Large (l): tốt, nhưng xlarge tốt hơn 3–5% mAP
- L40 48GB fit YOLOv8x với imgsz=1280, batch=12

### Hyperparameters tối ưu cho L40

```yaml
model: yolov8x.pt
epochs: 80
imgsz: 1280        # document pages rất rộng
batch: 12          # 48GB VRAM fit tốt
cache: false       # KHÔNG cache — 64GB RAM không đủ
workers: 4
patience: 25
cos_lr: true
close_mosaic: 20
flipud: 0.0        # KHÔNG flip dọc — text sẽ lộn
fliplr: 0.0        # KHÔNG flip ngang — text sẽ mirror
degrees: 2.0       # rotate nhẹ — document thường thẳng
mosaic: 0.5
copy_paste: 0.1
cls: 0.7           # tăng classification weight
optimizer: AdamW
lr0: 5e-4
warmup_epochs: 5
```

### Lý do tắt flip augmentation

Đây là lỗi phổ biến nhất với document detection. Khi flip ngang, chữ viết tay mirror → model học pattern sai. Tài liệu Ukrainian không có chữ đọc từ phải sang trái.

---

## Module 2: Qwen3-VL-8B Recognizer

### Thuật toán

**Qwen3-VL** là Vision-Language Model (VLM) sử dụng kiến trúc:
- **Visual encoder:** Patch-based ViT (Vision Transformer) xử lý ảnh thành visual tokens
- **Language decoder:** Causal LM (GPT-like) sinh text token by token
- **Cross-attention:** Kết nối visual tokens với text generation

Qwen3-VL-8B là phiên bản mới nhất (2025), vượt Qwen2.5-VL-7B trên tất cả OCR benchmarks (~+6% DocVQA, +8% TextVQA). Đặc biệt mạnh hơn trên Cyrillic script.

### Tại sao VLM tốt hơn TrOCR cho task này?

| | TrOCR-large | Qwen3-VL-8B |
|---|---|---|
| Params | 558M | 8B |
| Architecture | Encoder-Decoder | Decoder-only + ViT |
| Context | image → text | text context + image |
| Cyrillic pretrain | Rất ít | Nhiều (đa ngôn ngữ) |
| Reading complex layout | Kém | Tốt hơn nhiều |
| L40 fit (BF16) | Batch=64 | Batch=6 |
| CER on Ukrainian | ~20% | ~8% (estimated) |

### LoRA Fine-tuning

**LoRA (Low-Rank Adaptation):** Thay vì fine-tune toàn bộ 8B params, ta thêm 2 ma trận nhỏ A (r×d) và B (d×r) vào mỗi attention layer. Chỉ train A và B (~140M params = 1.75% tổng).

**RSLoRA (Rank-Stabilized LoRA):** Cải tiến của LoRA chuẩn — chia learning rate cho √r thay vì r, giúp training stable hơn với rank cao (64).

**Config tối ưu cho L40 48GB:**
```python
LoraConfig(
    r = 64,              # rank — cao hơn = expressiveness hơn, L40 fit được
    lora_alpha = 128,    # = 2r, học nhanh hơn
    use_rslora = True,   # stable hơn với r=64
    target_modules = [   # áp dụng LoRA lên tất cả attention + MLP
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ],
    lora_dropout = 0.05,
)
```

### Flash Attention 2

FA2 tái cứu trúc attention computation: thay vì materialize toàn bộ attention matrix O(n²), FA2 tính block-by-block trong SRAM của GPU. Kết quả:
- 2–4× faster attention
- Memory O(n) thay vì O(n²)
- L40 (Ada Lovelace) support FA2 native

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    attn_implementation = 'flash_attention_2',  # BẮT BUỘC trên L40
    torch_dtype = torch.bfloat16,
)
```

### Thinking Mode — PHẢI TẮT

Qwen3 có "thinking" mode (chain-of-thought nội bộ). Với OCR task, thinking tokens:
1. Tốn thêm 200–500 tokens VRAM per sample
2. Không cải thiện accuracy (task không cần reasoning)
3. Làm chậm inference 3–5×

```python
# Tắt trong generation:
model.generate(..., enable_thinking=False)
# Hoặc trong config:
generation_config.enable_thinking = False
```

---

## Module 3: Data Augmentation

### 3.1 TRDG Synthetic Crops (cho Recognizer)

**TextRecognitionDataGenerator** tạo text line images từ fonts + text strings. Dùng để tăng diversity training data cho OCR recognizer.

**Pipeline:**
1. Lấy tất cả text strings từ gold + silver data
2. Thêm Ukrainian word list
3. Generate với font Cyrillic/handwriting
4. Augment: noise, blur, skew, ink variation
5. Weight = 0.3 (thấp hơn real data)

### 3.2 Document Layout Generator (cho Detector)

`doc_gen` tạo full-page layout images với diverse region arrangements. Giúp YOLO học detect table/formula/annotation regions ngay cả khi dataset thật thiếu examples.

### 3.3 HKR Dataset (Cyrillic handwriting thật)

12k+ word images handwriting Kazakh/Russian — Cyrillic rất gần Ukrainian. Dùng làm additional real data cho recognizer với weight = 0.5.

---

## Module 4: Reading Order

### Tại sao quan trọng

PageCER tính trên **toàn trang text ghép lại** theo thứ tự đọc. Nếu thứ tự sai, CER tăng vọt dù text per region đúng hoàn toàn.

### Thuật toán metric (từ kaggle_metric.py)

```python
# Metric dùng bucket sort:
scorable.sort(key=lambda r: (
    ((r["bbox"][1] + r["bbox"][3]) / 2) // 15,  # bucket 15px
    (r["bbox"][0] + r["bbox"][2]) / 2,           # then x
))
```

### Implementation trong pipeline

```python
def reading_order_sort(regions, bucket_size=15):
    """
    Mirror chính xác thuật toán trong kaggle_metric._build_page_text().
    Đây là hàm duy nhất đúng — đừng dùng hàm sort khác.
    """
    scorable = [r for r in regions
                if r.get('type') not in ('image', 'graph')]
    non_scorable = [r for r in regions
                    if r.get('type') in ('image', 'graph')]
    scorable.sort(key=lambda r: (
        int(((r['bbox'][1] + r['bbox'][3]) / 2) // bucket_size),
        (r['bbox'][0] + r['bbox'][2]) / 2,
    ))
    return scorable + non_scorable
```

**QUAN TRỌNG:** Thứ tự đọc chỉ áp dụng cho scorable regions. `image/graph` nên giữ trong submission để không mất điểm detection/classification; PageCER sẽ bỏ qua chúng.

---

## Module 5: PaddleOCR Fallback (Ensemble)

### Chiến lược ensemble

Không cần ensemble toàn bộ — chỉ dùng PaddleOCR cho crops mà Qwen3 uncertain:

```
Qwen3-VL-8B inference:
   ├── confidence >= 0.7 → dùng Qwen3 text trực tiếp
   └── confidence < 0.7  → chạy PaddleOCR (Cyrillic model)
                           → chọn text có ít lỗi hơn
```

PaddleOCR inference nhanh hơn Qwen3 ~10× → overhead negligible.

### PaddleOCR model cho Cyrillic

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    lang='ru',           # Russian Cyrillic — closest to Ukrainian
    use_gpu=True,
    use_angle_cls=True,
    show_log=False,
)
```

---

## Module 6: Text Normalization

**KHÔNG cần post-process** các lỗi sau — metric tự normalize:
- Latin c/o/p/x/e → Cyrillic với/о/р/х/е
- Em-dash/en-dash → hyphen
- Quote variants → " hoặc '
- ~~strikethrough~~ markers
- Unicode super/subscript → ^/_

**CẦN đảm bảo** pipeline output đúng format:
- UTF-8 encoding
- Không có null bytes
- Text cho region type `formula` và `table` theo đúng convention

---

## Module 7: Local Validation (score_detailed)

Dùng `kaggle_metric.score_detailed()` để debug từng component:

```python
result = score_detailed(sol_df, sub_df, 'image')
# result keys:
# composite_score, detection_f1, detection_precision, detection_recall,
# classification_accuracy, region_cer, page_cer,
# n_matched_regions, n_false_positives, n_false_negatives
```

**Workflow debug:**
1. Nếu `detection_f1` thấp → tune YOLO conf/iou threshold
2. Nếu `region_cer` cao → OCR model chưa tốt, cần thêm epochs
3. Nếu `page_cer` cao nhưng `region_cer` thấp → reading order sai
4. Nếu `classification_accuracy` thấp → YOLO nhầm region type
