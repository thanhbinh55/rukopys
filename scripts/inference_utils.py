#!/usr/bin/env python3
"""
scripts/inference_utils.py — Shared utilities cho Phase 3 & Phase 4.

BUG FIXES so với phiên bản gốc:
1. reading_order_sort: sort scorable regions theo đúng metric, append non-scorable
   ở cuối để giữ detection/classification credit trong submission.
2. ensemble_ocr_with_paddle: confidence heuristic tốt hơn
   → Kiểm tra Cyrillic ratio, không chỉ độ dài text
3. ocr_crops_batch: xử lý đúng hơn khi process_vision_info unavailable
"""
import os, gc, json, logging, time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# ===== CONSTANTS =====
DEVICE         = os.getenv('DEVICE', 'cuda:0')
ROOT           = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART            = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
MAX_NEW_TOKENS = int(os.getenv('MAX_TOKENS', '256'))
OCR_BATCH_INIT = int(os.getenv('OCR_BATCH', '8'))
DEFAULT_CONF   = float(os.getenv('YOLO_CONF', '0.25'))
DEFAULT_IOU    = float(os.getenv('YOLO_IOU', '0.45'))
CLASSES        = ['handwritten','printed','formula','table','annotation','image','graph']
USE_FLASH_ATTN = os.getenv('USE_FLASH_ATTN', 'auto').lower()

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

log = logging.getLogger(__name__)


def get_attention_implementation() -> str:
    if USE_FLASH_ATTN in ('0', 'false', 'no', 'off', 'sdpa'):
        return 'sdpa'
    try:
        import flash_attn  # noqa: F401
        return 'flash_attention_2'
    except Exception:
        if USE_FLASH_ATTN in ('1', 'true', 'yes', 'on', 'flash_attention_2'):
            raise RuntimeError(
                'USE_FLASH_ATTN requested but flash_attn is not importable. '
                'Install flash-attn or set USE_FLASH_ATTN=0.'
            )
        log.warning('[ocr] flash_attn not found; using SDPA attention fallback')
        return 'sdpa'


# ─────────────────────────────────────────────
# IMAGE PATH RESOLUTION
# ─────────────────────────────────────────────

def build_image_index(img_root: Path) -> dict:
    """
    Index image files robustly for metadata values like:
    - "abc.jpg"
    - "images/abc.jpg"
    - "abc"
    """
    index = {}
    for p in Path(img_root).rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(img_root).as_posix()
        keys = {
            p.name,
            p.stem,
            rel,
            f'images/{p.name}',
        }
        for k in keys:
            index.setdefault(k, p)
    return index


def resolve_image_path(image_name: str, img_index: dict) -> Optional[Path]:
    if not image_name:
        return None
    image_name = str(image_name)
    return (
        img_index.get(image_name)
        or img_index.get(Path(image_name).name)
        or img_index.get(Path(image_name).stem)
    )


# ─────────────────────────────────────────────
# YOLO DETECTOR
# ─────────────────────────────────────────────

def load_yolo_detector(art: Path = ART):
    from ultralytics import YOLO

    best_pt = art / 'det_yolo' / 'best.pt'
    engine  = art / 'det_yolo' / 'best.engine'

    if not best_pt.exists():
        raise FileNotFoundError(
            f'YOLO weights not found: {best_pt}\n'
            f'Run Phase 1: python scripts/01_train_detector.py')

    if engine.exists():
        try:
            log.info(f'[yolo] Loading TensorRT engine...')
            return YOLO(str(engine))
        except Exception as e:
            log.warning(f'[yolo] TensorRT failed: {e} → fallback to PyTorch')

    log.info(f'[yolo] Loading PyTorch: {best_pt}')
    det = YOLO(str(best_pt))

    # Try export TensorRT (one-time)
    if not engine.exists():
        try:
            log.info('[yolo] Exporting TensorRT (~5 min)...')
            det.export(format='engine', half=True, device=0, workspace=8, simplify=True)
            engine_src = best_pt.with_suffix('.engine')
            if engine_src.exists():
                import shutil
                shutil.move(str(engine_src), str(engine))
            log.info(f'[yolo] TensorRT saved: {engine}')
        except Exception as e:
            log.warning(f'[yolo] TensorRT export failed: {e}')

    return det


def yolo_detect_page(img_path: str, detector, conf: float, iou: float,
                     use_tta: bool = False) -> list:
    """
    Detect regions trong 1 trang.
    use_tta: Test-Time Augmentation — chậm hơn 2× nhưng tốt hơn ~2-3% DetF1
    """
    try:
        results = detector.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            imgsz=1280,
            device=DEVICE,
            augment=use_tta,   # TTA multi-scale
            verbose=False,
        )

        if not results or not results[0].boxes:
            return []

        regions = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id     = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            region_type = CLASSES[cls_id] if cls_id < len(CLASSES) else 'handwritten'

            regions.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'type': region_type,
                'conf': conf_score,
                'text': '',
            })

        return regions

    except Exception as e:
        log.error(f'[yolo] Detection failed for {img_path}: {e}')
        return []


# ─────────────────────────────────────────────
# READING ORDER SORT — CRITICAL BUG FIX
# ─────────────────────────────────────────────

def reading_order_sort(regions: list, bucket_size: int = 15) -> list:
    """
    Sắp xếp regions theo thứ tự đọc.

    Metric chỉ dùng scorable regions (không phải image/graph) để build page text,
    nhưng submission vẫn nên giữ image/graph để không mất điểm detection/class.

    Mirror CHÍNH XÁC kaggle_metric._build_page_text():
        scorable.sort(key=lambda r: (
            ((r["bbox"][1] + r["bbox"][3]) / 2) // 15,
            (r["bbox"][0] + r["bbox"][2]) / 2,
        ))

    NOTE: Non-scorable (image/graph) được append cuối; PageCER sẽ bỏ qua chúng.
    """
    if not regions:
        return regions

    scorable = [r for r in regions if r.get('type') not in ('image', 'graph')]
    non_scorable = [r for r in regions if r.get('type') in ('image', 'graph')]

    # Sort theo đúng metric logic
    scorable.sort(key=lambda r: (
        int(((r['bbox'][1] + r['bbox'][3]) / 2) // bucket_size),
        (r['bbox'][0] + r['bbox'][2]) / 2,
    ))

    # Giữ non_scorable để metric detection/classification vẫn có prediction.
    return scorable + non_scorable


# ─────────────────────────────────────────────
# QWEN3-VL RECOGNIZER
# ─────────────────────────────────────────────

def load_qwen3_model(art: Path = ART):
    """Load Qwen3-VL với LoRA weights."""
    lora_dir = art / 'qwen3_lora' / 'best_checkpoint'
    if not lora_dir.exists():
        checkpoints = sorted(
            (art / 'qwen3_lora').glob('checkpoint-*'),
            key=lambda p: int(p.name.split('-')[-1]) if p.name.split('-')[-1].isdigit() else 0
        )
        if checkpoints:
            lora_dir = checkpoints[-1]
            log.warning(f'[ocr] best_checkpoint not found → {lora_dir.name}')
        else:
            raise FileNotFoundError(
                f'No Qwen3-VL weights in {art}/qwen3_lora/\n'
                f'Run Phase 2: python scripts/02_train_recognizer.py')

    model_id = os.getenv('MODEL_ID', 'Qwen/Qwen3-VL-8B-Instruct')
    log.info(f'[ocr] Base: {model_id}  |  LoRA: {lora_dir.name}')

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(
        str(lora_dir),
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )

    attn_impl = get_attention_implementation()
    load_kw = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map={'': DEVICE},
    )
    log.info(f'[ocr] attention={attn_impl}')

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kw)
        log.info('[ocr] Loaded Qwen3VLForConditionalGeneration')
    except Exception as e:
        raise RuntimeError(
            f'Cannot load Qwen3-VL model: {model_id}. '
            'Install transformers==4.57.1 and qwen-vl-utils.'
        ) from e

    # Merge LoRA
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model = model.merge_and_unload()
    model.eval()

    # Tắt thinking
    if hasattr(model, 'generation_config'):
        if hasattr(model.generation_config, 'enable_thinking'):
            model.generation_config.enable_thinking = False

    vram = torch.cuda.memory_allocated(0) / 1e9
    log.info(f'[ocr] Loaded. VRAM: {vram:.1f}GB / 48GB')
    return model, processor


def ocr_crops_batch(crops: list, model, processor, current_bs: int) -> tuple:
    """
    OCR batch crops với OOM recovery.
    crops: list of (PIL.Image, region_dict)
    Returns: (texts: list[str], final_batch_size: int)
    """
    images = [c[0] for c in crops]

    try:
        # Try qwen_vl_utils
        try:
            from qwen_vl_utils import process_vision_info
            _has_qvl = True
        except ImportError:
            _has_qvl = False

        messages_list = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": "Transcribe:"},
                ]},
            ]
            for img in images
        ]

        texts_prompt = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]

        if _has_qvl:
            all_imgs = []
            for m in messages_list:
                img_inputs, _ = process_vision_info(m)
                if img_inputs:
                    all_imgs.extend(img_inputs)
        else:
            all_imgs = images

        inputs = processor(
            text=texts_prompt,
            images=all_imgs if all_imgs else None,
            padding=True,
            return_tensors='pt',
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                enable_thinking=False,
            )

        input_len = inputs['input_ids'].shape[1]
        decoded = processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )
        return [d.strip() for d in decoded], current_bs

    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise

        torch.cuda.empty_cache()
        gc.collect()
        new_bs = max(1, current_bs // 2)
        log.warning(f'[OOM] bs={current_bs} → retry bs={new_bs}')

        if len(crops) <= 1:
            log.error('[OOM] Single crop OOM — skipping')
            return [''], 1

        mid = len(crops) // 2
        t1, bs1 = ocr_crops_batch(crops[:mid], model, processor, new_bs)
        t2, bs2 = ocr_crops_batch(crops[mid:], model, processor, new_bs)
        return t1 + t2, min(bs1, bs2)


def ocr_regions(img_path: str, regions: list, model, processor,
                batch_size: int) -> tuple:
    """OCR tất cả scorable regions trong 1 page."""
    if not regions:
        return regions, batch_size

    try:
        page_img = Image.open(img_path).convert('RGB')
        W, H = page_img.size
    except Exception as e:
        log.error(f'[ocr] Cannot open {img_path}: {e}')
        return regions, batch_size

    scorable_idx = [i for i, r in enumerate(regions)
                    if r.get('type') not in ('image', 'graph')]
    if not scorable_idx:
        return regions, batch_size

    crops = []
    for idx in scorable_idx:
        r = regions[idx]
        x1, y1, x2, y2 = r['bbox']
        pad = 4
        x1 = max(0, x1-pad);  y1 = max(0, y1-pad)
        x2 = min(W, x2+pad);  y2 = min(H, y2+pad)

        if x2-x1 < 5 or y2-y1 < 5:
            crops.append((Image.new('RGB', (64, 32), 'white'), r))
            continue

        crop = page_img.crop((x1, y1, x2, y2))

        # Adaptive resize based on region width/height ratio
        if crop.height > 128:
            ratio = 128 / crop.height
            new_w = max(32, min(1920, int(crop.width * ratio)))
            crop = crop.resize((new_w, 128), Image.LANCZOS)

        crops.append((crop, r))

    all_texts = []
    i = 0
    while i < len(crops):
        batch = crops[i:i+batch_size]
        texts, batch_size = ocr_crops_batch(batch, model, processor, batch_size)
        all_texts.extend(texts)
        i += len(batch)

    for j, idx in enumerate(scorable_idx):
        if j < len(all_texts):
            regions[idx]['text'] = all_texts[j]

    return regions, batch_size


# ─────────────────────────────────────────────
# PADDLE OCR FALLBACK
# ─────────────────────────────────────────────

def try_load_paddle_ocr():
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(lang='ru', use_gpu=True, use_angle_cls=True, show_log=False)
        log.info('[paddle] PaddleOCR loaded (Russian Cyrillic)')
        return ocr
    except ImportError:
        log.info('[paddle] PaddleOCR not installed — skipping ensemble')
        return None
    except Exception as e:
        log.warning(f'[paddle] Load failed: {e}')
        return None


def paddle_ocr_crop(crop_img: Image.Image, paddle_ocr) -> tuple:
    import numpy as np
    try:
        arr = np.array(crop_img)
        result = paddle_ocr.ocr(arr, cls=True)
        if not result or not result[0]:
            return '', 0.0
        texts, confs = [], []
        for line in result[0]:
            texts.append(line[1][0])
            confs.append(line[1][1])
        return ' '.join(texts), min(confs) if confs else 0.0
    except Exception:
        return '', 0.0


def _is_confident_prediction(text: str) -> bool:
    """
    BUG FIX: Cải tiến confidence heuristic.
    Thay vì chỉ kiểm tra len >= 3, kiểm tra quality thực sự.
    """
    if not text or not text.strip():
        return False

    text = text.strip()

    # Quá ngắn (nhưng 1-2 char có thể hợp lệ: "5", "B+", "ні")
    if len(text) == 0:
        return False

    # Kiểm tra có ký tự có nghĩa
    meaningful = sum(1 for c in text
                     if c.isalnum() or '\u0400' <= c <= '\u04FF')
    if meaningful == 0:
        return False

    # Phát hiện hallucination: lặp lại token quá nhiều
    words = text.split()
    if len(words) >= 4:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:
            return False  # > 60% words giống nhau → hallucination

    # Quá nhiều ký tự đặc biệt (không phải text thật)
    special = sum(1 for c in text if not c.isalnum() and c not in ' .,!?;:\'-—–()[]')
    if len(text) > 5 and special / len(text) > 0.5:
        return False

    return True


def ensemble_ocr_with_paddle(qwen_text: str, crop_img: Image.Image,
                               paddle_ocr, min_chars: int = 2) -> str:
    """
    BUG FIX: Dùng quality-based confidence thay vì chỉ len check.
    """
    if paddle_ocr is None:
        return qwen_text

    if _is_confident_prediction(qwen_text):
        return qwen_text

    # Qwen không confident → thử PaddleOCR
    paddle_text, paddle_conf = paddle_ocr_crop(crop_img, paddle_ocr)
    if paddle_text.strip() and paddle_conf > 0.5 and _is_confident_prediction(paddle_text):
        log.debug(f'[ensemble] Qwen uncertain → PaddleOCR: "{paddle_text[:30]}"')
        return paddle_text

    return qwen_text


# ─────────────────────────────────────────────
# SUBMISSION FORMAT
# ─────────────────────────────────────────────

def regions_to_submission_json(regions: list) -> str:
    """Convert regions → JSON string cho CSV."""
    output = []
    for r in regions:
        output.append({
            'bbox': [int(x) for x in r['bbox']],
            'type': r.get('type', 'handwritten'),
            'text': r.get('text', '').strip(),
        })
    return json.dumps(output, ensure_ascii=False)


# ─────────────────────────────────────────────
# GENERIC OOM WRAPPER
# ─────────────────────────────────────────────

def safe_infer_batch(batch: list, infer_fn, current_bs: int, max_retries: int = 4):
    """Generic OOM-safe batch inference wrapper."""
    retries = 0
    bs = current_bs

    while retries <= max_retries:
        try:
            results = []
            for i in range(0, len(batch), bs):
                results.extend(infer_fn(batch[i:i+bs]))
            return results, bs
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            gc.collect()
            new_bs = max(1, bs // 2)
            log.warning(f'[OOM] bs={bs} → {new_bs} (retry {retries+1}/{max_retries})')
            if new_bs == bs:
                return [''] * len(batch), 1
            bs = new_bs
            retries += 1

    return [''] * len(batch), 1


# ─────────────────────────────────────────────
# DISK GUARD
# ─────────────────────────────────────────────

def check_disk(min_gb: float = 10.0, path: str = '/') -> float:
    import shutil
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_gb:
        raise RuntimeError(
            f'[DISK] Only {free_gb:.1f}GB free — need {min_gb}GB.\n'
            f'Clean: rm -rf {ART}/det_yolo/dataset/ hoặc old checkpoints')
    return free_gb
