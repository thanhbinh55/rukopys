# docs/04_recognizer_guide.md — Phase 2: Qwen3-VL Recognizer

## Mục tiêu

Fine-tune Qwen3-VL-8B-Instruct với LoRA để nhận dạng chữ viết tay Ukrainian trên crop images. Đây là phase quan trọng nhất, chiếm 80% tổng điểm (CER + PageCER).

---

## scripts/02_train_recognizer.py — Full Implementation Guide

### 2.1 Setup và imports

```python
#!/usr/bin/env python3
"""
Phase 2: Fine-tune Qwen3-VL-8B-Instruct với LoRA cho Ukrainian HTR.
Hardware: 1× L40 48GB, 64GB RAM, 8 vCPU.
"""
import os, gc, json, random, re, time, shutil, logging
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from qwen_vl_utils import process_vision_info

# ===== HARDWARE CONSTANTS =====
DEVICE         = 'cuda:0'
USE_BF16       = True
USE_TF32       = True
USE_FLASH_ATTN = True
GRAD_CHECKPT   = False   # 48GB đủ → tắt để tăng tốc 30%
SAVE_LIMIT     = 2
DL_WORKERS     = 4

# ===== PATHS =====
ART       = Path(os.getenv('HTR_ART_DIR',   './htr_artifacts'))
DATA_ROOT = Path(os.getenv('DATA_ROOT',     './data'))
OUTPUT    = ART / 'qwen3_lora'
LOGS_DIR  = ART / 'logs'
ART.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ===== MODEL CONFIG =====
MODEL_ID = os.getenv('MODEL_ID', 'Qwen/Qwen3-VL-8B-Instruct')

# ===== TRAINING HYPERPARAMS (L40 48GB optimal) =====
BATCH_SIZE    = int(os.getenv('BATCH_SIZE',   '4'))    # per device
GRAD_ACCUM    = int(os.getenv('GRAD_ACCUM',   '8'))    # effective batch = 32
NUM_EPOCHS    = float(os.getenv('NUM_EPOCHS', '3'))
LR            = float(os.getenv('LR',         '2e-4'))  # LoRA cần LR cao hơn full FT
WARMUP_RATIO  = float(os.getenv('WARMUP_RATIO','0.05'))
MAX_NEW_TOKENS = 256
MAX_SEQ_LEN   = 512   # input + output tokens

# ===== LORA CONFIG =====
LORA_R        = int(os.getenv('LORA_R',    '64'))
LORA_ALPHA    = int(os.getenv('LORA_ALPHA','128'))   # = 2×r

# ===== DATA CONFIG =====
MAX_GOLD    = int(os.getenv('MAX_GOLD',    '999999'))  # load tất cả gold
MAX_SILVER  = int(os.getenv('MAX_SILVER',  '80000'))
MAX_SYNTH   = int(os.getenv('MAX_SYNTH',   '100000'))
MAX_HKR     = int(os.getenv('MAX_HKR',     '50000'))
USE_SILVER  = os.getenv('USE_SILVER',  '1') == '1'
USE_SYNTH   = os.getenv('USE_SYNTH',   '1') == '1'
USE_HKR     = os.getenv('USE_HKR',     '1') == '1'
AUG_PROB    = float(os.getenv('AUG_PROB', '0.5'))
```

### 2.2 System Prompt

```python
# System prompt tối ưu cho Ukrainian HTR
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

USER_PROMPT = "Transcribe the handwritten text in this image:"
```

### 2.3 Dataset class (LAZY LOADING — bắt buộc với 64GB RAM)

```python
class HTRCropDataset(Dataset):
    """
    Lazy-loading dataset cho OCR crops.
    KHÔNG load ảnh vào RAM — mỗi __getitem__ mới mở file.
    Phù hợp với 64GB RAM limit.
    """
    
    def __init__(self, manifest_df: pd.DataFrame, processor, augment: bool = False):
        """
        manifest_df: DataFrame với columns [image_path, text, weight]
        processor: Qwen3-VL processor
        augment: bật augmentation cho training set
        """
        # Chỉ lưu paths và labels trong RAM, không lưu ảnh
        self.records = manifest_df[['image_path', 'text', 'weight']].to_dict('records')
        self.processor = processor
        self.augment = augment
        self._verify_sample()
    
    def _verify_sample(self, n=5):
        """Kiểm tra n records đầu để đảm bảo paths tồn tại."""
        missing = 0
        for r in self.records[:n]:
            if not Path(r['image_path']).exists():
                missing += 1
                print(f"[warn] Missing: {r['image_path']}")
        if missing:
            print(f"[warn] {missing}/{n} sample paths missing. Check manifest.")
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        rec = self.records[idx]
        
        # Mở ảnh lazy — chỉ đọc khi cần
        try:
            img = Image.open(rec['image_path']).convert('RGB')
        except Exception as e:
            # Trả về blank image thay vì crash
            print(f"[err] Cannot open {rec['image_path']}: {e}")
            img = Image.new('RGB', (128, 64), color=(255, 255, 255))
        
        # Augmentation chỉ trong training
        if self.augment and random.random() < AUG_PROB:
            img = self._augment(img)
        
        text = rec['text']
        
        return {
            'image': img,
            'text': text,
            'weight': rec.get('weight', 1.0),
        }
    
    def _augment(self, img: Image.Image) -> Image.Image:
        """
        Augmentation nhẹ cho handwritten Ukrainian text.
        Không dùng transforms làm mất chữ.
        """
        # Contrast variation (simulate ink variation)
        if random.random() < 0.3:
            factor = random.uniform(0.75, 1.25)
            img = ImageEnhance.Contrast(img).enhance(factor)
        
        # Brightness (simulate paper yellowing)
        if random.random() < 0.2:
            factor = random.uniform(0.85, 1.15)
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        # Slight blur (simulate poor scan)
        if random.random() < 0.15:
            radius = random.uniform(0.3, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # Very slight rotation (handwriting không hoàn toàn thẳng)
        if random.random() < 0.25:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, expand=False,
                             fillcolor=(255, 255, 255))
        
        # Slight sharpening (simulate high-resolution scan)
        if random.random() < 0.15:
            img = img.filter(ImageFilter.SHARPEN)
        
        return img


def collate_fn(batch, processor):
    """
    Collator cho Qwen3-VL.
    Xử lý multi-image batch với padding.
    """
    messages_list = []
    for item in batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": item['image']},  # PIL Image trực tiếp
                {"type": "text",  "text": USER_PROMPT},
            ]},
            # Teacher forcing: thêm target text vào conversation
            {"role": "assistant", "content": item['text']},
        ]
        messages_list.append(messages)
    
    # Apply chat template cho từng message
    texts = []
    all_images = []
    for messages in messages_list:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
        # Extract images từ messages
        image_inputs, _ = process_vision_info(messages)
        if image_inputs:
            all_images.extend(image_inputs)
    
    # Tokenize batch
    inputs = processor(
        text=texts,
        images=all_images if all_images else None,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors='pt',
    )
    
    # Labels: mask system+user tokens, chỉ train trên assistant response
    labels = inputs['input_ids'].clone()
    # Tìm vị trí bắt đầu của assistant token và mask phần trước
    # AGENT: implement masking logic dựa trên chat template của Qwen3
    # Tham khảo: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs['labels'] = labels
    
    return inputs
```

### 2.4 Load và chuẩn bị dữ liệu

```python
def load_manifest(art: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training + validation manifests.
    Mỗi row: image_path (absolute), text (normalized), weight (float), source (str).
    """
    
    def read_jsonl_to_crops(jsonl_path, img_root, split='gold', max_n=999999):
        """Convert jsonl metadata → crop records."""
        records = []
        # KHÔNG đọc hết vào RAM — stream line by line
        with open(jsonl_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if not line.strip() or len(records) >= max_n:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                img_name = (row.get('image') or row.get('file_name')
                           or row.get('filename') or row.get('id') or '')
                img_path = img_root / img_name
                
                regions = (row.get('regions') or row.get('annotations') or [])
                for reg in regions:
                    bbox = reg.get('bbox') or reg.get('bbox_2d')
                    text = reg.get('text') or reg.get('transcription') or ''
                    rtype = reg.get('type', 'handwritten')
                    lang = reg.get('language', 'uk')
                    legibility = reg.get('legibility', 'legible')
                    
                    # Filter: chỉ lấy scorable regions có text
                    if (not text.strip() or lang != 'uk'
                            or legibility == 'illegible'
                            or rtype in ('image', 'graph')
                            or not bbox or len(bbox) != 4):
                        continue
                    
                    # Tạo crop path (sẽ được generate trong bước tiếp theo)
                    crop_name = f"{img_path.stem}_r{len(records):06d}.jpg"
                    crop_path = art / 'rec_data' / 'crops' / crop_name
                    
                    records.append({
                        'image_path': str(crop_path),
                        'source_image': str(img_path),
                        'bbox': bbox,
                        'text': normalize_ocr_target(text, rtype),
                        'weight': 1.0 if split == 'gold' else 0.5,
                        'source': split,
                        'type': rtype,
                    })
        return records
    
    cfg = json.load(open(art / 'config.json'))
    data_root = Path(cfg['DATA_ROOT'])
    train_img  = Path(cfg['TRAIN_IMG'])
    silver_img = Path(cfg.get('SILVER_IMG', str(data_root / 'silver' / 'images')))
    
    # Gold data
    train_records = read_jsonl_to_crops(
        art / 'train_split.jsonl', train_img, 'gold', MAX_GOLD)
    valid_records = read_jsonl_to_crops(
        art / 'valid_split.jsonl', train_img, 'gold')
    
    # Silver data (optional)
    silver_meta = data_root / 'silver' / 'metadata.jsonl'
    if USE_SILVER and silver_meta.exists():
        silver_records = read_jsonl_to_crops(
            silver_meta, silver_img, 'silver', MAX_SILVER)
        for r in silver_records:
            r['weight'] = 0.5
        train_records.extend(silver_records)
        print(f'[data] +{len(silver_records):,} silver crops')
    
    # Synthetic TRDG (optional)
    synth_manifest = art / 'rec_data' / 'synth_manifest.csv'
    if USE_SYNTH and synth_manifest.exists():
        synth_df = pd.read_csv(synth_manifest).head(MAX_SYNTH)
        synth_records = synth_df.to_dict('records')
        for r in synth_records:
            r['weight'] = 0.3
            r.setdefault('source', 'synthetic')
        train_records.extend(synth_records)
        print(f'[data] +{len(synth_records):,} synthetic crops (TRDG)')
    
    # HKR Cyrillic (optional)
    hkr_manifest = art / 'hkr_manifest.csv'
    if USE_HKR and hkr_manifest.exists():
        hkr_df = pd.read_csv(hkr_manifest).head(MAX_HKR)
        hkr_records = hkr_df.to_dict('records')
        for r in hkr_records:
            r['weight'] = 0.5
            r.setdefault('source', 'hkr')
        train_records.extend(hkr_records)
        print(f'[data] +{len(hkr_records):,} HKR Cyrillic crops')
    
    random.shuffle(train_records)
    train_df = pd.DataFrame(train_records)
    valid_df  = pd.DataFrame(valid_records)
    
    print(f'[data] Total train: {len(train_df):,}  |  Valid: {len(valid_df):,}')
    print(f'[data] Sources: {train_df["source"].value_counts().to_dict()}')
    
    return train_df, valid_df
```

### 2.5 Crop extraction (tách riêng, chạy 1 lần)

```python
def extract_crops(manifest_df: pd.DataFrame, overwrite: bool = False):
    """
    Crop regions từ full-page images theo bbox.
    Chạy 1 lần, lưu crops ra disk → training load từ disk.
    """
    crop_dir = ART / 'rec_data' / 'crops'
    crop_dir.mkdir(parents=True, exist_ok=True)
    
    done = skipped = errors = 0
    
    for idx, row in manifest_df.iterrows():
        crop_path = Path(row['image_path'])
        
        if crop_path.exists() and not overwrite:
            skipped += 1
            continue
        
        try:
            src_img = Image.open(row['source_image']).convert('RGB')
            W, H = src_img.size
            
            x1, y1, x2, y2 = row['bbox']
            # Clamp và add margin
            pad = 4
            x1 = max(0, int(x1) - pad)
            y1 = max(0, int(y1) - pad)
            x2 = min(W, int(x2) + pad)
            y2 = min(H, int(y2) + pad)
            
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            
            crop = src_img.crop((x1, y1, x2, y2))
            
            # Resize: giữ aspect ratio, max height 128px
            max_h = 128
            if crop.height > max_h:
                ratio = max_h / crop.height
                new_w = min(1920, int(crop.width * ratio))
                crop = crop.resize((new_w, max_h), Image.LANCZOS)
            
            crop.save(str(crop_path), 'JPEG', quality=92)
            done += 1
            
        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"[err] crop {idx}: {e}")
        
        if (done + skipped) % 5000 == 0:
            print(f'[crop] done={done:,} skipped={skipped:,} errors={errors}')
    
    print(f'[crop] DONE: {done:,} new, {skipped:,} skipped, {errors} errors')
```

### 2.6 Model loading

```python
def load_model_and_processor():
    """Load Qwen3-VL + apply LoRA."""
    
    print(f'[model] Loading {MODEL_ID} ...')
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,    # min image resolution
        max_pixels=1280 * 28 * 28,   # max: L40 48GB fit 1280px
    )
    
    # Set pad token nếu chưa có
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model với BF16 + Flash Attention 2
    load_kwargs = {
        'torch_dtype': torch.bfloat16,
        'device_map': {'': DEVICE},  # single GPU — KHÔNG 'auto' (multi-GPU)
    }
    if USE_FLASH_ATTN:
        load_kwargs['attn_implementation'] = 'flash_attention_2'
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, **load_kwargs)
    print('[model] Loaded Qwen3VLForConditionalGeneration')
    
    # TF32 cho L40 Ada Lovelace
    if USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # KHÔNG bật gradient checkpointing (48GB VRAM đủ → nhanh hơn 30%)
    if GRAD_CHECKPT:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    
    # Tắt thinking mode (Qwen3)
    if hasattr(model, 'generation_config'):
        if hasattr(model.generation_config, 'enable_thinking'):
            model.generation_config.enable_thinking = False
    
    vram_gb = torch.cuda.memory_allocated(0) / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'[model] After load: {vram_gb:.1f}GB / {total_gb:.0f}GB VRAM')
    print(f'[model] Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B')
    
    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            # Attention
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            # MLP (feedforward)
            'gate_proj', 'up_proj', 'down_proj',
            # Visual encoder attention (quan trọng cho image understanding)
            # AGENT: check model architecture và thêm visual attention modules nếu có
        ],
        lora_dropout=0.05,
        bias='none',
        use_rslora=True,       # Rank-Stabilized LoRA
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total    = sum(p.numel() for p in model.parameters())
    print(f'[lora] Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)')
    
    return model, processor
```

### 2.7 TrainingArguments tối ưu

```python
def get_training_args(output_dir: Path) -> TrainingArguments:
    """TrainingArguments được tune cho L40 48GB / 64GB RAM."""
    
    # Tìm checkpoint gần nhất để resume
    existing_checkpoints = sorted(output_dir.glob('checkpoint-*'))
    resume_from = str(existing_checkpoints[-1]) if existing_checkpoints else None
    if resume_from:
        print(f'[train] Will resume from: {resume_from}')
    
    return TrainingArguments(
        output_dir=str(output_dir),
        
        # Batch & gradient
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,      # effective batch = 32
        
        # Learning rate
        learning_rate=LR,
        lr_scheduler_type='cosine',
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_EPOCHS,
        
        # Precision (L40 BF16 native)
        bf16=USE_BF16,
        fp16=False,
        tf32=USE_TF32,
        
        # Optimizer (fused AdamW nhanh hơn 20% trên Ampere+)
        optim='adamw_torch_fused',
        
        # DataLoader
        dataloader_num_workers=DL_WORKERS,    # 4 — phù hợp 8 vCPU
        dataloader_pin_memory=False,           # 64GB RAM → không cần pin
        dataloader_prefetch_factor=2,          # giảm nếu RAM tight
        
        # Checkpointing
        save_strategy='steps',
        save_steps=200,
        save_total_limit=SAVE_LIMIT,           # chỉ giữ 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        
        # Evaluation
        eval_strategy='steps',
        eval_steps=200,
        
        # Logging
        logging_dir=str(LOGS_DIR / 'tb'),
        logging_steps=20,
        report_to='none',                      # không dùng W&B/TensorBoard
        
        # Misc
        remove_unused_columns=False,           # QUAN TRỌNG cho VLM
        predict_with_generate=False,           # train mode không cần generate
        ddp_find_unused_parameters=False,      # LoRA có unused params
        
        # Gradient checkpointing
        gradient_checkpointing=GRAD_CHECKPT,
    ), resume_from
```

### 2.8 CER metric cho evaluation

```python
def compute_cer_metric(eval_pred, processor):
    """
    Tính CER trên validation set sau mỗi eval step.
    Dùng để chọn best checkpoint.
    """
    from rapidfuzz.distance import Levenshtein
    
    predictions, labels = eval_pred
    
    # Decode predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Clip invalid token IDs
    predictions = predictions.clip(0, processor.tokenizer.vocab_size - 1)
    labels_clean = [
        [l if l != -100 else processor.tokenizer.pad_token_id for l in label]
        for label in labels
    ]
    
    pred_texts = processor.tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    label_texts = processor.tokenizer.batch_decode(
        labels_clean, skip_special_tokens=True)
    
    cers = []
    for pred, gt in zip(pred_texts, label_texts):
        pred = pred.strip()
        gt   = gt.strip()
        if len(gt) == 0:
            continue
        cer = Levenshtein.distance(pred, gt) / max(len(gt), 1)
        cers.append(cer)
    
    mean_cer = sum(cers) / max(len(cers), 1)
    return {'cer': mean_cer, 'n_samples': len(cers)}
```

### 2.9 Main training loop

```python
def main():
    print('=' * 70)
    print('PHASE 2 — TRAIN QWEN3-VL OCR RECOGNIZER')
    print(f'Model: {MODEL_ID}')
    print(f'Device: {DEVICE}  |  BF16: {USE_BF16}  |  FlashAttn: {USE_FLASH_ATTN}')
    print(f'Batch/GPU: {BATCH_SIZE}  |  GradAccum: {GRAD_ACCUM}  |  EffBatch: {BATCH_SIZE*GRAD_ACCUM}')
    print(f'LoRA r={LORA_R} α={LORA_ALPHA} RSLoRA=True')
    print('=' * 70)
    
    # Disk check
    free_gb = shutil.disk_usage('/').free / 1e9
    print(f'[disk] Free: {free_gb:.1f}GB')
    if free_gb < 20:
        raise RuntimeError(f'[disk] Only {free_gb:.1f}GB free — need at least 20GB')
    
    # Load data manifests
    train_df, valid_df = load_manifest(ART)
    
    # Extract crops nếu chưa có
    rebuild = os.getenv('REBUILD_CROPS', '0') == '1'
    print(f'[crop] Extracting crops (rebuild={rebuild})...')
    extract_crops(train_df, overwrite=rebuild)
    extract_crops(valid_df,  overwrite=rebuild)
    
    # Save manifests cho Phase 3
    train_df.to_csv(ART / 'rec_data' / 'train_manifest.csv', index=False)
    valid_df.to_csv( ART / 'rec_data' / 'valid_manifest.csv',  index=False)
    print('[crop] Manifests saved')
    
    # Load model
    model, processor = load_model_and_processor()
    
    # Tạo datasets
    import functools
    train_dataset = HTRCropDataset(train_df, processor, augment=True)
    valid_dataset = HTRCropDataset(valid_df,  processor, augment=False)
    print(f'[data] Train: {len(train_dataset):,}  |  Valid: {len(valid_dataset):,}')
    
    # Training args
    training_args, resume_from = get_training_args(OUTPUT)
    
    # Collator với processor baked in
    collator = functools.partial(collate_fn, processor=processor)
    
    # CER metric
    compute_metrics = functools.partial(compute_cer_metric, processor=processor)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train với auto-resume
    print('[train] Starting training...')
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    
    # Save best model
    best_dir = OUTPUT / 'best_checkpoint'
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    print(f'[train] Best model saved → {best_dir}')
    
    # Save training log
    log = {
        'train_runtime':      train_result.metrics.get('train_runtime'),
        'train_loss':         train_result.metrics.get('train_loss'),
        'train_samples_sec':  train_result.metrics.get('train_samples_per_second'),
        'model_id':           MODEL_ID,
        'lora_r':             LORA_R,
        'effective_batch':    BATCH_SIZE * GRAD_ACCUM,
    }
    json.dump(log, open(ART / 'logs' / 'phase2_training.json', 'w'), indent=2)
    
    # Cleanup intermediate checkpoints để tiết kiệm disk
    print('[cleanup] Removing intermediate checkpoints...')
    for ckpt in OUTPUT.glob('checkpoint-*'):
        if ckpt.name != 'best_checkpoint':
            shutil.rmtree(ckpt)
            print(f'[cleanup] Removed {ckpt}')
    
    print('[DONE] Phase 2 complete!')
    print(f'[DONE] Best model: {best_dir}')
    vram_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f'[DONE] VRAM usage: {vram_gb:.1f}GB / 48GB')


def normalize_ocr_target(text: str, region_type: str = 'handwritten') -> str:
    """
    Chuẩn hóa target text trước khi train.
    Mirror một phần logic của kaggle_metric._normalize_text().
    """
    import unicodedata
    text = unicodedata.normalize('NFKC', str(text))
    # Strikethrough
    text = re.sub(r'~~.*?~~\{(.*?)\}', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    # Whitespace
    text = re.sub(r'\s+', ' ', text.replace('\r', ' ').replace('\n', ' ')).strip()
    return text


if __name__ == '__main__':
    main()
```

---

## Checkpoint Resume

Script tự động resume từ checkpoint cuối cùng. Không cần làm gì thêm — chỉ chạy lại command cũ:

```bash
python scripts/02_train_recognizer.py
# Sẽ tự detect htr_artifacts/qwen3_lora/checkpoint-* và resume
```

---

## Monitoring VRAM

```bash
# Trong terminal khác khi đang train:
watch -n 5 nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu --format=csv,noheader
# Expected: ~35-42GB used out of 48GB
```
