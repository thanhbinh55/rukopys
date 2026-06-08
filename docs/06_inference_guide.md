# docs/06_inference_guide.md — Phase 4: End-to-End Inference & Submission

## Mục tiêu

Chạy full pipeline trên test set: YOLO detect → Qwen3-VL recognize → sort by reading order → build submission CSV. Xử lý OOM gracefully, log mọi lỗi.

---

## scripts/04_inference.py — Full Implementation Guide

### 4.1 Setup và config

```python
#!/usr/bin/env python3
"""
Phase 4: End-to-end inference → submission.csv
Pipeline: YOLO detect regions → Qwen3-VL OCR → reading order → CSV
Hardware: 1× L40 48GB, 64GB RAM, 8 vCPU.
"""
import os, gc, json, logging, re, time, shutil
from pathlib import Path
from typing import Optional
import torch
import pandas as pd
from PIL import Image

# ===== CONSTANTS =====
DEVICE        = 'cuda:0'
ART           = Path(os.getenv('HTR_ART_DIR',  './htr_artifacts'))
DATA_ROOT     = Path(os.getenv('DATA_ROOT',    './data'))
OUTPUT_CSV    = ART / 'submission.csv'
LOGS_DIR      = ART / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Inference batch sizes — tune dựa trên VRAM usage
YOLO_BATCH    = int(os.getenv('YOLO_BATCH',   '8'))    # 8 pages/batch
OCR_BATCH_INIT= int(os.getenv('OCR_BATCH',    '8'))    # crops/batch (start conservative)
MAX_NEW_TOKENS= int(os.getenv('MAX_TOKENS',   '256'))

# YOLO thresholds (load từ best_config.json nếu có)
DEFAULT_CONF  = float(os.getenv('YOLO_CONF',  '0.25'))
DEFAULT_IOU   = float(os.getenv('YOLO_IOU',   '0.45'))

# Classes phải khớp với YOLO training
CLASSES = ['handwritten','printed','formula','table','annotation','image','graph']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'inference.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)
```

### 4.2 Load configs từ Phase 3

```python
def load_inference_config() -> dict:
    """
    Load optimal conf/iou thresholds từ Phase 3 grid search.
    Fallback về defaults nếu chưa chạy Phase 3.
    """
    config_path = ART / 'best_config.json'
    if config_path.exists():
        cfg = json.load(open(config_path))
        log.info(f'[config] Loaded from Phase 3: conf={cfg["conf"]}, iou={cfg["iou"]}')
        return cfg
    else:
        log.warning(f'[config] best_config.json not found — using defaults')
        return {'conf': DEFAULT_CONF, 'iou': DEFAULT_IOU}

def load_best_thresholds():
    cfg = load_inference_config()
    return cfg['conf'], cfg['iou']
```

### 4.3 YOLO Detector với TensorRT fallback

```python
def load_yolo_detector():
    """
    Load YOLO model. Ưu tiên TensorRT engine nếu có (3× faster).
    """
    from ultralytics import YOLO
    
    best_pt = ART / 'det_yolo' / 'best.pt'
    engine  = ART / 'det_yolo' / 'best.engine'
    
    if not best_pt.exists():
        raise FileNotFoundError(
            f'YOLO weights not found: {best_pt}\n'
            f'Chạy Phase 1 trước: python scripts/01_train_detector.py'
        )
    
    # Try TensorRT engine first (3× faster inference)
    if engine.exists():
        try:
            log.info(f'[yolo] Loading TensorRT engine: {engine}')
            det = YOLO(str(engine))
            log.info('[yolo] TensorRT engine loaded OK')
            return det
        except Exception as e:
            log.warning(f'[yolo] TensorRT failed: {e} — falling back to PyTorch')
    
    log.info(f'[yolo] Loading PyTorch weights: {best_pt}')
    det = YOLO(str(best_pt))
    
    # Export TensorRT nếu chưa có (chạy 1 lần, save để dùng lại)
    try:
        log.info('[yolo] Exporting TensorRT engine (one-time, ~5 min)...')
        det.export(format='engine', half=True, device=0, workspace=8, simplify=True)
        log.info(f'[yolo] TensorRT engine saved: {engine}')
    except Exception as e:
        log.warning(f'[yolo] TensorRT export failed: {e}')
    
    return det


def yolo_detect_page(img_path: str, detector, conf: float, iou: float) -> list:
    """
    Detect regions trong 1 trang.
    
    Returns:
        list of dict: [{'bbox': [x1,y1,x2,y2], 'type': str, 'conf': float}]
    """
    try:
        results = detector.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            imgsz=1280,
            device=DEVICE,
            verbose=False,
        )
        if not results or len(results) == 0:
            return []
        
        regions = []
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            region_type = CLASSES[cls_id] if cls_id < len(CLASSES) else 'handwritten'
            
            regions.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'type': region_type,
                'conf': conf_score,
                'text': '',  # sẽ được fill bởi OCR
            })
        
        return regions
    
    except Exception as e:
        log.error(f'[yolo] Detection failed for {img_path}: {e}')
        return []
```

### 4.4 Reading Order Sort — mirror metric chính xác

```python
def reading_order_sort(regions: list, bucket_size: int = 15) -> list:
    """
    Sắp xếp regions theo thứ tự đọc.
    
    CRITICAL: Mirror CHÍNH XÁC thuật toán trong kaggle_metric._build_page_text():
        scorable.sort(key=lambda r: (
            ((r["bbox"][1] + r["bbox"][3]) / 2) // 15,
            (r["bbox"][0] + r["bbox"][2]) / 2,
        ))
    
    Đây là hàm duy nhất đúng — KHÔNG thay thế bằng heuristic khác.
    
    Args:
        regions: list of region dicts với key 'bbox' = [x1,y1,x2,y2]
        bucket_size: phải = 15 để match metric (đừng thay đổi)
    
    Returns:
        Sorted regions theo reading order
    """
    if not regions:
        return regions
    
    # Filter: scorable regions (mirror _is_scorable)
    scorable = [r for r in regions
                if r.get('type') not in ('image', 'graph')]
    non_scorable = [r for r in regions
                    if r.get('type') in ('image', 'graph')]
    
    # Sort theo bucket_y rồi center_x — chính xác metric logic
    scorable.sort(key=lambda r: (
        int(((r['bbox'][1] + r['bbox'][3]) / 2) // bucket_size),
        (r['bbox'][0] + r['bbox'][2]) / 2,
    ))
    
    # Non-scorable giữ nguyên thứ tự (không ảnh hưởng PageCER)
    return scorable + non_scorable
```

### 4.5 Qwen3-VL OCR với OOM protection

```python
def load_qwen3_model():
    """Load Qwen3-VL model với LoRA weights từ Phase 2."""
    from transformers import AutoProcessor
    from peft import PeftModel
    
    lora_dir = ART / 'qwen3_lora' / 'best_checkpoint'
    if not lora_dir.exists():
        # Thử tìm checkpoint gần nhất
        checkpoints = sorted((ART / 'qwen3_lora').glob('checkpoint-*'))
        if checkpoints:
            lora_dir = checkpoints[-1]
            log.warning(f'[ocr] best_checkpoint not found, using: {lora_dir}')
        else:
            raise FileNotFoundError(
                f'Qwen3-VL weights not found in {ART}/qwen3_lora/\n'
                f'Chạy Phase 2 trước: python scripts/02_train_recognizer.py'
            )
    
    model_id = os.getenv('MODEL_ID', 'Qwen/Qwen3-VL-8B-Instruct')
    log.info(f'[ocr] Loading base model: {model_id}')
    
    processor = AutoProcessor.from_pretrained(str(lora_dir))
    
    from transformers import Qwen3VLForConditionalGeneration
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map={'': DEVICE},
    )
    
    # Load LoRA adapter
    log.info(f'[ocr] Loading LoRA from: {lora_dir}')
    model = PeftModel.from_pretrained(base_model, str(lora_dir))
    model = model.merge_and_unload()   # merge LoRA vào base weights
    model.eval()
    
    # Tắt thinking mode
    if hasattr(model, 'generation_config'):
        if hasattr(model.generation_config, 'enable_thinking'):
            model.generation_config.enable_thinking = False
    
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    log.info(f'[ocr] Model loaded. VRAM: {vram_used:.1f}GB / 48GB')
    
    return model, processor


# System prompt (phải khớp với Phase 2)
SYSTEM_PROMPT = (
    "You are a specialized Ukrainian handwritten text recognition system. "
    "Your task: transcribe exactly what is handwritten in the provided image. "
    "Rules:\n"
    "1. Output ONLY the transcribed text — no explanations, no formatting\n"
    "2. Preserve original Ukrainian spelling and punctuation exactly\n"
    "3. For illegible characters, output your best guess\n"
    "4. Use Cyrillic characters (not Latin lookalikes) for Ukrainian text\n"
    "5. Keep numbers, punctuation, and special characters as written"
)


def ocr_crops_batch(
    crops: list,    # list of (PIL.Image, region_dict)
    model,
    processor,
    current_bs: int,
) -> tuple[list, int]:
    """
    OCR một batch crops với OOM recovery.
    
    Returns:
        (texts: list[str], used_batch_size: int)
    """
    from qwen_vl_utils import process_vision_info
    
    images = [c[0] for c in crops]
    
    try:
        messages_list = []
        for img in images:
            messages_list.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": "Transcribe:"},
                ]},
            ])
        
        texts_prompt = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        all_image_inputs = []
        for m in messages_list:
            image_inputs, _ = process_vision_info(m)
            if image_inputs:
                all_image_inputs.extend(image_inputs)
        
        inputs = processor(
            text=texts_prompt,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors='pt',
        ).to(DEVICE)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,           # greedy cho consistency
                use_cache=True,
                enable_thinking=False,     # Qwen3: tắt thinking tokens
            )
        
        # Decode chỉ phần output (không lặp lại input)
        input_len = inputs['input_ids'].shape[1]
        decoded = processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )
        
        texts = [d.strip() for d in decoded]
        return texts, current_bs
    
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        
        # OOM recovery
        torch.cuda.empty_cache()
        gc.collect()
        new_bs = max(1, current_bs // 2)
        log.warning(f'[OOM] batch={current_bs} → retry with bs={new_bs}')
        
        if len(crops) <= 1:
            # Single crop vẫn OOM → return empty
            log.error(f'[OOM] Single crop OOM — skipping')
            return [''], 1
        
        # Chia batch làm đôi và retry
        mid = len(crops) // 2
        texts1, bs1 = ocr_crops_batch(crops[:mid], model, processor, new_bs)
        texts2, bs2 = ocr_crops_batch(crops[mid:], model, processor, new_bs)
        return texts1 + texts2, min(bs1, bs2)


def ocr_regions(
    img_path: str,
    regions: list,
    model,
    processor,
    batch_size: int,
) -> tuple[list, int]:
    """
    OCR tất cả regions của 1 page.
    
    Returns:
        (regions_with_text, final_batch_size)
    """
    if not regions:
        return regions, batch_size
    
    try:
        page_img = Image.open(img_path).convert('RGB')
        W, H = page_img.size
    except Exception as e:
        log.error(f'[ocr] Cannot open {img_path}: {e}')
        return regions, batch_size
    
    # Chỉ OCR scorable regions
    scorable_idx = [i for i, r in enumerate(regions)
                    if r.get('type') not in ('image', 'graph')]
    
    if not scorable_idx:
        return regions, batch_size
    
    # Crop images
    crops = []
    for idx in scorable_idx:
        r = regions[idx]
        x1, y1, x2, y2 = r['bbox']
        # Clamp + small padding
        pad = 4
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
        if x2 - x1 < 5 or y2 - y1 < 5:
            crops.append((Image.new('RGB', (64, 32), 'white'), r))
            continue
        crop = page_img.crop((x1, y1, x2, y2))
        # Resize nếu crop quá lớn
        if crop.height > 128:
            ratio = 128 / crop.height
            new_w = min(1920, int(crop.width * ratio))
            crop = crop.resize((new_w, 128), Image.LANCZOS)
        crops.append((crop, r))
    
    # Batch inference
    all_texts = []
    i = 0
    while i < len(crops):
        batch = crops[i:i+batch_size]
        texts, batch_size = ocr_crops_batch(batch, model, processor, batch_size)
        all_texts.extend(texts)
        i += len(batch)
    
    # Gán text vào regions
    for j, idx in enumerate(scorable_idx):
        if j < len(all_texts):
            regions[idx]['text'] = all_texts[j]
    
    return regions, batch_size
```

### 4.6 PaddleOCR Fallback (optional ensemble)

```python
def try_load_paddle_ocr():
    """Load PaddleOCR nếu available, return None nếu không."""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            lang='ru',               # Russian Cyrillic ≈ Ukrainian
            use_gpu=True,
            use_angle_cls=True,
            show_log=False,
        )
        log.info('[paddle] PaddleOCR loaded (Russian Cyrillic model)')
        return ocr
    except ImportError:
        log.info('[paddle] PaddleOCR not installed — skipping ensemble')
        return None
    except Exception as e:
        log.warning(f'[paddle] PaddleOCR load failed: {e}')
        return None


def paddle_ocr_crop(crop_img: Image.Image, paddle_ocr) -> tuple[str, float]:
    """
    OCR 1 crop với PaddleOCR.
    Returns: (text, min_confidence)
    """
    import numpy as np
    try:
        arr = np.array(crop_img)
        result = paddle_ocr.ocr(arr, cls=True)
        if not result or not result[0]:
            return '', 0.0
        texts = []
        min_conf = 1.0
        for line in result[0]:
            text = line[1][0]
            conf = line[1][1]
            texts.append(text)
            min_conf = min(min_conf, conf)
        return ' '.join(texts), min_conf
    except Exception as e:
        return '', 0.0


def ensemble_ocr_with_paddle(
    qwen_text: str,
    crop_img: Image.Image,
    paddle_ocr,
    qwen_conf_threshold: float = 0.7,
) -> str:
    """
    Ensemble strategy: dùng PaddleOCR chỉ khi cần.
    Qwen3-VL là main model; PaddleOCR là fallback cho uncertain cases.
    
    Heuristic: nếu Qwen output ngắn bất thường hoặc empty → thử PaddleOCR.
    """
    if paddle_ocr is None:
        return qwen_text
    
    # Qwen text có vẻ OK → dùng trực tiếp
    if len(qwen_text.strip()) >= 3:
        return qwen_text
    
    # Qwen output empty/very short → thử PaddleOCR
    paddle_text, paddle_conf = paddle_ocr_crop(crop_img, paddle_ocr)
    if paddle_text.strip() and paddle_conf > 0.5:
        log.debug(f'[ensemble] Qwen empty → using PaddleOCR: "{paddle_text[:30]}"')
        return paddle_text
    
    return qwen_text
```

### 4.7 Build submission

```python
def regions_to_submission_json(regions: list) -> str:
    """
    Convert regions list → JSON string cho submission CSV.
    Chỉ giữ fields cần thiết: bbox, type, text.
    """
    output = []
    for r in regions:
        output.append({
            'bbox': [int(x) for x in r['bbox']],
            'type': r.get('type', 'handwritten'),
            'text': r.get('text', '').strip(),
        })
    return json.dumps(output, ensure_ascii=False)
```

### 4.8 Main inference loop

```python
def main():
    log.info('=' * 70)
    log.info('PHASE 4 — INFERENCE & SUBMISSION GENERATION')
    log.info(f'Device: {DEVICE}  |  OCR batch init: {OCR_BATCH_INIT}')
    log.info('=' * 70)
    
    # Disk check
    free_gb = shutil.disk_usage('/').free / 1e9
    log.info(f'[disk] Free: {free_gb:.1f}GB')
    
    # Load configs
    conf_threshold, iou_threshold = load_best_thresholds()
    log.info(f'[config] YOLO conf={conf_threshold}, iou={iou_threshold}')
    
    # Load test metadata
    cfg = json.load(open(ART / 'config.json'))
    data_root  = Path(cfg['DATA_ROOT'])
    test_img   = Path(cfg['TEST_IMG'])
    test_meta  = Path(cfg['TEST_META'])
    
    test_rows = []
    with open(test_meta, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_rows.append(json.loads(line))
    log.info(f'[data] Test images: {len(test_rows):,}')
    
    # Build image index
    img_index = {p.name: p for p in test_img.iterdir() if p.is_file()}
    stem_index = {}
    for p in img_index.values():
        stem_index.setdefault(p.stem, p)
    log.info(f'[data] Test image index: {len(img_index):,}')
    
    # Load models
    log.info('[models] Loading YOLO detector...')
    detector = load_yolo_detector()
    
    log.info('[models] Loading Qwen3-VL recognizer...')
    ocr_model, ocr_processor = load_qwen3_model()
    
    # Optional: PaddleOCR ensemble
    paddle_ocr = try_load_paddle_ocr()
    
    # Inference loop
    submission_rows = []
    current_bs = OCR_BATCH_INIT
    t_start = time.time()
    
    for i, row in enumerate(test_rows):
        img_name = (row.get('image') or row.get('file_name')
                   or row.get('filename') or row.get('id') or '')
        img_path = img_index.get(img_name) or stem_index.get(Path(img_name).stem)
        
        if img_path is None:
            log.warning(f'[{i+1}/{len(test_rows)}] Image not found: {img_name}')
            submission_rows.append({'image': img_name, 'regions': '[]'})
            continue
        
        # Step 1: Detect
        regions = yolo_detect_page(str(img_path), detector, conf_threshold, iou_threshold)
        
        if not regions:
            log.warning(f'[{i+1}] No regions detected: {img_name}')
            submission_rows.append({'image': img_name, 'regions': '[]'})
            continue
        
        # Step 2: Reading order sort TRƯỚC khi OCR
        # (sort bbox để có context về thứ tự, OCR vẫn per-region)
        regions = reading_order_sort(regions)
        
        # Step 3: OCR
        regions, current_bs = ocr_regions(
            str(img_path), regions, ocr_model, ocr_processor, current_bs)
        
        # Step 4: Ensemble với PaddleOCR (optional)
        if paddle_ocr is not None:
            try:
                page_img = Image.open(str(img_path)).convert('RGB')
                W, H = page_img.size
                for r in regions:
                    if r.get('type') in ('image', 'graph'):
                        continue
                    x1, y1, x2, y2 = r['bbox']
                    crop = page_img.crop((max(0,x1-4), max(0,y1-4),
                                          min(W,x2+4), min(H,y2+4)))
                    r['text'] = ensemble_ocr_with_paddle(
                        r.get('text', ''), crop, paddle_ocr)
            except Exception as e:
                log.warning(f'[paddle] Ensemble failed for {img_name}: {e}')
        
        # Step 5: Build submission row
        regions_json = regions_to_submission_json(regions)
        submission_rows.append({'image': img_name, 'regions': regions_json})
        
        # Progress log
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            speed = (i + 1) / elapsed * 60  # pages/min
            eta_min = (len(test_rows) - i - 1) / max(speed/60, 1e-6) / 60
            vram_gb = torch.cuda.memory_allocated(0) / 1e9
            log.info(
                f'[{i+1}/{len(test_rows)}] '
                f'{img_name} | {len(regions)} regions | '
                f'OCR_batch={current_bs} | '
                f'{speed:.1f} pages/min | '
                f'ETA={eta_min:.1f}min | '
                f'VRAM={vram_gb:.1f}GB'
            )
    
    # Save submission
    sub_df = pd.DataFrame(submission_rows)
    
    # Đảm bảo tất cả test images có trong submission
    all_test_names = set(r.get('image') or r.get('file_name') or r.get('id', '')
                         for r in test_rows)
    submitted_names = set(sub_df['image'])
    missing_names = all_test_names - submitted_names
    if missing_names:
        log.warning(f'[submit] Adding {len(missing_names)} missing images with empty regions')
        missing_rows = [{'image': n, 'regions': '[]'} for n in missing_names]
        sub_df = pd.concat([sub_df, pd.DataFrame(missing_rows)], ignore_index=True)
    
    sub_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f'[DONE] submission.csv saved: {OUTPUT_CSV}')
    log.info(f'[DONE] Total images: {len(sub_df)}')
    log.info(f'[DONE] Total time: {(time.time()-t_start)/60:.1f} min')
    
    # Validate submission format
    validate_submission(sub_df, test_rows)


def validate_submission(sub_df: pd.DataFrame, test_rows: list):
    """Quick validation trước khi nộp."""
    log.info('[validate] Checking submission format...')
    
    errors = 0
    for _, row in sub_df.iterrows():
        try:
            regions = json.loads(row['regions'])
            assert isinstance(regions, list)
            for r in regions:
                assert 'bbox' in r, f"Missing bbox in {row['image']}"
                assert 'type' in r, f"Missing type in {row['image']}"
                assert 'text' in r, f"Missing text in {row['image']}"
                assert len(r['bbox']) == 4, f"Invalid bbox in {row['image']}"
        except Exception as e:
            log.error(f'[validate] Invalid row {row["image"]}: {e}')
            errors += 1
    
    if errors == 0:
        log.info('[validate] ✓ All rows valid')
    else:
        log.error(f'[validate] ✗ {errors} invalid rows — fix before submitting!')
    
    # Check coverage
    expected_count = len(test_rows)
    actual_count = len(sub_df)
    log.info(f'[validate] Coverage: {actual_count}/{expected_count} images')
    if actual_count < expected_count:
        log.error(f'[validate] ✗ Missing {expected_count - actual_count} images!')


if __name__ == '__main__':
    main()
```

---

## Monitoring và debug

```bash
# Monitor VRAM trong khi inference:
watch -n 3 nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv,noheader

# Check inference progress:
tail -f htr_artifacts/logs/inference.log

# Validate submission trước khi nộp:
python -c "
import pandas as pd, json
df = pd.read_csv('htr_artifacts/submission.csv')
print('Rows:', len(df))
print('Sample regions:', df['regions'].iloc[0][:100])
print('Empty regions count:', df['regions'].eq('[]').sum())
"
```

## Common issues

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `FileNotFoundError: best.pt` | Chưa chạy Phase 1 | `python scripts/01_train_detector.py` |
| `FileNotFoundError: qwen3_lora` | Chưa chạy Phase 2 | `python scripts/02_train_recognizer.py` |
| OOM sau nhiều pages | VRAM fragmentation | Restart model, `torch.cuda.empty_cache()` |
| submission.csv thiếu images | image index miss | Kiểm tra `test_img` path trong config.json |
| Score thấp dù OCR OK | Reading order sai | Đảm bảo `reading_order_sort(bucket_size=15)` |
