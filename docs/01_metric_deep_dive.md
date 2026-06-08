# docs/01_metric_deep_dive.md — Phân Tích Metric Chính Thức

## Công thức

```
Score = 0.15 × DetF1 + 0.05 × ClassAcc + 0.30 × (1 − CER) + 0.50 × (1 − PageCER)
```

## Phân tích từng component

### Component 1: Detection F1 (15%)

```python
# Greedy IoU matching với threshold = 0.5
# Type-agnostic: chỉ đo bbox quality, không phân biệt region type
det_f1 = 2 * precision * recall / (precision + recall)
```

**Implication cho code:**
- YOLO confidence threshold nên tune để maximize recall (nhiều match) hơn precision
- IoU 0.5 là threshold thoải mái — không cần bbox pixel-perfect
- Submit thêm bbox (FP) ít bị phạt hơn bỏ sót bbox (FN) vì PageCER sẽ penalize nặng hơn

**Target:** DetF1 ≥ 0.85

---

### Component 2: Classification Accuracy (5%)

```python
# Chỉ tính trên IoU-matched pairs
# Nếu pred bbox match GT bbox (IoU ≥ 0.5), thì check type có đúng không
class_acc = correct_type_count / total_matched_count
```

**Implication cho code:**
- Training YOLO với `cls=0.7` (tăng classification loss weight)
- 7 classes: `handwritten`, `printed`, `formula`, `table`, `annotation`, `image`, `graph`
- Nhầm `handwritten` vs `printed` ảnh hưởng ít (cả 2 đều scorable)
- Nhầm `image`/`graph` vs text types ảnh hưởng nhiều (image/graph excluded khỏi CER)

**Target:** ClassAcc ≥ 0.85

---

### Component 3: Per-Region CER (30%)

```python
# Chỉ tính trên matched regions MÀ _is_scorable() == True
def _is_scorable(region):
    if region.get("type") in ("image", "graph"): return False
    if region.get("language", "uk") == "other":  return False  # GT-only field
    if region.get("legibility", "legible") == "illegible": return False  # GT-only
    return True

# CER = Levenshtein(pred, gt) / len(gt)
# CER = 1.0 nếu không có region nào match (penalize empty submission)
```

**Implication cho code:**
- Text cho `image`/`graph` regions có thể để `""` — không bị score
- `language` và `legibility` là GT-only — KHÔNG cần predict
- Nếu submission thiếu matches, CER default = 1.0 (worst) → PHẢI có detections

**Target:** Region CER ≤ 0.15 (1 − CER ≥ 0.85)

---

### Component 4: Page CER (50%) — QUAN TRỌNG NHẤT

```python
# Build page text từ tất cả scorable regions, sort theo bucket:
scorable.sort(key=lambda r: (
    ((r["bbox"][1] + r["bbox"][3]) / 2) // 15,
    (r["bbox"][0] + r["bbox"][2]) / 2,
))
page_text = "\n".join(r["text"] for r in scorable)

# Symmetric: pred regions matched với non-scorable GT bị DROP
pred_drop = {pi for gi, pi in matched if not _is_scorable(gt[gi])}
pred_page = _build_page_text(pred, drop_indices=pred_drop)

# PageCER = Levenshtein(pred_page, gt_page) / len(gt_page)
```

**CRITICAL INSIGHTS:**

**1. Bucket size = 15px:** Metric nhóm các regions có center_y cách nhau ≤ 15px vào cùng "dòng". Sau đó sort theo center_x trong mỗi bucket. Pipeline của bạn PHẢI mirror chính xác logic này.

**2. Pred regions cho non-scorable GT bị DROP:** Nếu YOLO predict bbox cho vùng `image`/`graph` và text không rỗng, text đó bị loại khỏi pred_page. An toàn — không bị phạt.

**3. PageCER không phụ thuộc vào bbox granularity:** Dù bạn split 1 line thành 5 bbox hay gộp 5 line thành 1 bbox, chỉ cần text cuối cùng đúng thì PageCER vẫn tốt.

**4. PageCER > RegionCER:** Trong thực tế, PageCER thường cao hơn RegionCER vì reading order errors và missing detections đều ảnh hưởng.

**Target:** Page CER ≤ 0.10 (1 − PageCER ≥ 0.90)

---

## Text Normalization — Những gì ĐƯỢC tha thứ

Metric apply `_normalize_text()` cho cả GT lẫn prediction trước khi tính CER:

| Lỗi OCR | Có được normalize không? | Ví dụ |
|---|---|---|
| Latin c → Cyrillic с | ✅ ĐÃ xử lý | "cocна" → "сосна" |
| Latin o,p,x,e,A,B,C... | ✅ ĐÃ xử lý | "Pосія" → "Росія" |
| Em-dash/en-dash | ✅ ĐÃ xử lý | "—" → "-" |
| Curly quotes «» | ✅ ĐÃ xử lý | "«Привіт»" → '"Привіт"' |
| Strikethrough ~~text~~ | ✅ ĐÃ xử lý | lấy text bên trong |
| x_{3} vs x_3 | ✅ ĐÃ xử lý (formula) | đều OK |
| x² vs x^2 | ✅ ĐÃ xử lý (formula) | đều OK |
| Whitespace collapse | ✅ ĐÃ xử lý | "a  b" → "a b" |
| Sai chính tả thật | ❌ KHÔNG xử lý | "пагода"≠"погода" |
| Sai ký tự hoàn toàn | ❌ KHÔNG xử lý | "к" ≠ "г" |

**Kết luận:** Đừng waste compute để post-process Latin/Cyrillic lookalike hay dashes. Tập trung vào accuracy thật sự.

---

## Submission Format

```csv
image,regions
page_001.jpg,"[{""bbox"":[50,100,850,130],""type"":""handwritten"",""text"":""Доброго ранку""}]"
page_002.jpg,"[]"
```

**Bắt buộc:**
- Column `image` (tên file, không phải full path)
- Column `regions` (JSON array, có thể là `[]`)
- Mỗi region: `bbox` [x1,y1,x2,y2] + `type` + `text`
- Tất cả test images phải có trong submission (dù empty)

**Không cần:**
- `language` field
- `legibility` field

---

## Debug workflow với score_detailed()

```python
# Sau Phase 3, dùng script này để hiểu vấn đề:
import sys
sys.path.insert(0, '.')
from kaggle_metric import score_detailed
import pandas as pd

sol = pd.read_csv('htr_artifacts/valid_gt.csv')
sub = pd.read_csv('htr_artifacts/valid_pred.csv')
r = score_detailed(sol, sub, 'image')

print(f"COMPOSITE:  {r['composite_score']:.4f}")
print(f"Det F1:     {r['detection_f1']:.4f}  (P={r['detection_precision']:.3f} R={r['detection_recall']:.3f})")
print(f"Cls Acc:    {r['classification_accuracy']:.4f}")
print(f"Region CER: {r['region_cer']:.4f}  → +{1-r['region_cer']:.4f}")
print(f"Page CER:   {r['page_cer']:.4f}   → +{1-r['page_cer']:.4f}")
print(f"FP: {r['n_false_positives']}  FN: {r['n_false_negatives']}  Matched: {r['n_matched_regions']}")

# DIAGNOSIS:
if r['page_cer'] > r['region_cer'] + 0.1:
    print(">>> READING ORDER VẤN ĐỀ — xem lại reading_order_sort()")
if r['detection_recall'] < 0.7:
    print(">>> YOLO RECALL THẤP — giảm conf threshold hoặc train thêm")
if r['classification_accuracy'] < 0.8:
    print(">>> CLASS ERROR — tăng cls weight trong YOLO training")
if r['region_cer'] > 0.3:
    print(">>> OCR KÉM — train Qwen3-VL thêm epochs hoặc cần more data")
```
