#!/usr/bin/env python3
"""
scripts/02_train_recognizer.py — Phase 2: Fine-tune Qwen3-VL-8B với LoRA.
Hardware: 1× L40 48GB, 64GB RAM, 8 vCPU.

Cải tiến quan trọng:
1. FIX: Label masking — chỉ train trên assistant response tokens
2. Curriculum learning: silver/synthetic → gold (2-stage)
3. WeightedRandomSampler thay vì loss weighting
4. Dynamic OCR resolution
"""
import os, gc, json, random, re, time, shutil, logging, argparse
from pathlib import Path
from typing import Optional
from functools import partial

import torch
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ===== HARDWARE CONSTANTS =====
DEVICE          = 'cuda:0'
USE_BF16        = True
USE_TF32        = True
USE_FLASH_ATTN  = True
GRAD_CHECKPT    = False   # 48GB đủ → tắt để tăng tốc 30%
SAVE_LIMIT      = 2
DL_WORKERS      = 4

# ===== PATHS =====
ROOT     = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
ART      = Path(os.getenv('HTR_ART_DIR', ROOT / 'local_working' / 'htr_artifacts'))
DATA_ROOT= Path(os.getenv('DATA_ROOT',   ROOT / 'data' / 'kaggle_input' / 'rukopys-dataset' / 'rukopys_raw'))
OUTPUT   = ART / 'qwen3_lora'
LOGS_DIR = ART / 'logs'
ART.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ===== MODEL =====
MODEL_ID = os.getenv('MODEL_ID', 'Qwen/Qwen3-VL-8B-Instruct')

# ===== HYPERPARAMS (L40 48GB) =====
BATCH_SIZE    = int(os.getenv('BATCH_SIZE',    '4'))
GRAD_ACCUM    = int(os.getenv('GRAD_ACCUM',    '8'))
NUM_EPOCHS    = float(os.getenv('NUM_EPOCHS',  '3'))
LR            = float(os.getenv('LR',          '2e-4'))
WARMUP_RATIO  = float(os.getenv('WARMUP_RATIO','0.05'))
MAX_NEW_TOKENS= 256
MAX_SEQ_LEN   = 512

# ===== LORA =====
LORA_R     = int(os.getenv('LORA_R',     '64'))
LORA_ALPHA = int(os.getenv('LORA_ALPHA', '128'))  # = 2r

# ===== DATA CONFIG =====
MAX_GOLD   = int(os.getenv('MAX_GOLD',   '999999'))
MAX_SILVER = int(os.getenv('MAX_SILVER', '80000'))
MAX_SYNTH  = int(os.getenv('MAX_SYNTH',  '100000'))
MAX_HKR    = int(os.getenv('MAX_HKR',    '50000'))
USE_SILVER = os.getenv('USE_SILVER', '1') == '1'
USE_SYNTH  = os.getenv('USE_SYNTH',  '1') == '1'
USE_HKR    = os.getenv('USE_HKR',    '1') == '1'
USE_PSEUDO = os.getenv('USE_PSEUDO', '0') == '1'
AUG_PROB   = float(os.getenv('AUG_PROB', '0.5'))

# ===== CURRICULUM =====
CURRICULUM = os.getenv('CURRICULUM', '0') == '1'  # Override bởi --curriculum flag

# ===== PROMPTS =====
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


def normalize_ocr_target(text: str, region_type: str = 'handwritten') -> str:
    import unicodedata
    text = unicodedata.normalize('NFKC', str(text))
    text = re.sub(r'~~.*?~~\{(.*?)\}', r'\1', text)  # strikethrough correction
    text = re.sub(r'~~(.*?)~~', r'\1', text)           # plain strikethrough
    text = re.sub(r'\s+', ' ', text.replace('\r', ' ').replace('\n', ' ')).strip()
    return text


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class HTRCropDataset(Dataset):
    """Lazy-loading dataset cho OCR crops. KHÔNG load ảnh vào RAM."""

    def __init__(self, records: list, processor, augment: bool = False):
        self.records   = records
        self.processor = processor
        self.augment   = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec['image_path']).convert('RGB')
            # Adaptive resize dựa trên text length
            text_len = len(rec.get('text', ''))
            target_h = 64 if text_len < 15 else (96 if text_len < 50 else 128)
            if img.height != target_h:
                ratio = target_h / max(1, img.height)
                new_w = max(32, min(1920, int(img.width * ratio)))
                img = img.resize((new_w, target_h), Image.LANCZOS)
        except Exception:
            img = Image.new('RGB', (256, 64), color=(255, 255, 255))

        if self.augment and random.random() < AUG_PROB:
            img = self._augment(img)

        return {'image': img, 'text': rec.get('text', ''), 'weight': rec.get('weight', 1.0)}

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.3:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        if random.random() < 0.2:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.15:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.9)))
        if random.random() < 0.25:
            angle = random.uniform(-3, 3)
            img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        if random.random() < 0.1:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(1.2, 2.0))
        return img


# ─────────────────────────────────────────────
# COLLATE — FIX LABEL MASKING (BUG FIX CHÍNH)
# ─────────────────────────────────────────────

def collate_fn(batch: list, processor) -> dict:
    """
    Collator với đúng label masking.

    CRITICAL FIX: Chỉ train trên assistant response tokens.
    Mask (-100) tất cả system + user tokens.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    messages_list = []
    for item in batch:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": [
                {"type": "image", "image": item['image']},
                {"type": "text",  "text":  USER_PROMPT},
            ]},
            {"role": "assistant", "content": item['text']},
        ]
        messages_list.append(messages)

    texts = [
        processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=False)
        for m in messages_list
    ]

    # Collect images
    all_images = []
    if process_vision_info is not None:
        for m in messages_list:
            img_inputs, _ = process_vision_info(m)
            if img_inputs:
                all_images.extend(img_inputs)
    else:
        for item in batch:
            all_images.append(item['image'])

    inputs = processor(
        text=texts,
        images=all_images if all_images else None,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors='pt',
    )

    # ── LABEL MASKING (robust — tìm assistant token thật sự) ─────
    # Tạo labels = copy của input_ids
    labels = inputs['input_ids'].clone()

    # Mask padding tokens trước
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Encode chuỗi "<|im_start|>assistant" theo Qwen3 chat template để lấy
    # token IDs làm pattern tìm kiếm. Đây là cách duy nhất đáng tin cậy vì
    # số lượng special tokens trước/sau nội dung phụ thuộc vào tokenizer version.
    # Qwen3 dùng: <|im_start|>assistant\n{text}<|im_end|>
    try:
        # Lấy token IDs cho marker "<|im_start|>assistant"
        # add_special_tokens=False để không thêm BOS ngoài ý muốn
        assistant_header_ids = processor.tokenizer.encode(
            '<|im_start|>assistant', add_special_tokens=False)
    except Exception:
        assistant_header_ids = []

    for batch_idx in range(len(batch)):
        seq = inputs['input_ids'][batch_idx].tolist()
        target_text = batch[batch_idx]['text']

        if not target_text.strip():
            labels[batch_idx] = -100
            continue

        # Tìm vị trí cuối cùng của pattern "<|im_start|>assistant" trong seq
        # (cuối cùng vì có thể có nhiều turns, ta muốn turn assistant cuối)
        start_response = _find_last_sublist(seq, assistant_header_ids)

        if start_response == -1 or not assistant_header_ids:
            # Fallback: encode assistant text + 1 newline token, đếm từ cuối
            # Newline "\n" sau "assistant" trong Qwen3 template là 1 token
            target_ids = processor.tokenizer.encode(
                target_text, add_special_tokens=False)
            target_len = len(target_ids)
            seq_len = sum(1 for t in seq if t != processor.tokenizer.pad_token_id)
            # +2: newline token sau "assistant" + EOS token cuối
            start_response = max(0, seq_len - target_len - 2)
        else:
            # start_response trỏ tới đầu "<|im_start|>assistant"
            # Bỏ qua header + newline token (thường là 1 token "\n")
            start_response = start_response + len(assistant_header_ids) + 1

        labels[batch_idx, :start_response] = -100

    inputs['labels'] = labels
    return inputs


def _find_last_sublist(seq: list, sublist: list) -> int:
    """
    Tìm vị trí bắt đầu của lần xuất hiện CUỐI CÙNG của sublist trong seq.
    Trả về -1 nếu không tìm thấy hoặc sublist rỗng.
    """
    if not sublist:
        return -1
    n, m = len(seq), len(sublist)
    result = -1
    for i in range(n - m + 1):
        if seq[i:i + m] == sublist:
            result = i
    return result


class WeightedTrainer(Trainer):
    """Trainer dùng WeightedRandomSampler khi manifest có weight."""

    def __init__(self, *args, train_sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_sample_weights = train_sample_weights

    def _get_train_sampler(self):
        if self.train_sample_weights is None:
            return super()._get_train_sampler()
        return WeightedRandomSampler(
            self.train_sample_weights,
            num_samples=len(self.train_sample_weights),
            replacement=True,
        )


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_manifest(art: Path, curriculum_stage: int = 0) -> tuple:
    """
    Load training + validation manifests.
    curriculum_stage=0: tất cả data
    curriculum_stage=1: chỉ gold + silver (loại synthetic/hkr)
    curriculum_stage=2: chỉ gold (finest quality)
    """
    cfg = json.load(open(art / 'config.json'))
    train_img_root = Path(cfg['TRAIN_IMG'])
    silver_img_root = Path(cfg.get('SILVER_IMG', ''))

    def build_img_index(root: Path) -> dict:
        if not root.exists():
            return {}
        idx = {}
        for p in root.iterdir():
            if p.is_file():
                idx[p.name] = p
                idx[p.stem] = p
        return idx

    img_idx = build_img_index(train_img_root)

    def read_crops_from_jsonl(jsonl_path, split_name, max_n=999999, weight=1.0):
        records = []
        if not jsonl_path.exists():
            return records
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or len(records) >= max_n:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                img_name = (row.get('image') or row.get('file_name')
                           or row.get('filename') or row.get('id') or '')
                img_path = img_idx.get(img_name) or img_idx.get(Path(img_name).stem)
                if not img_path:
                    continue

                for reg in (row.get('regions') or row.get('annotations') or []):
                    bbox    = reg.get('bbox') or reg.get('bbox_2d')
                    text    = reg.get('text', '').strip()
                    rtype   = reg.get('type', 'handwritten')
                    lang    = reg.get('language', 'uk')
                    legible = reg.get('legibility', 'legible')

                    if (not text or lang != 'uk' or legible == 'illegible'
                            or rtype in ('image', 'graph') or not bbox):
                        continue

                    # Tạo crop name deterministc
                    crop_name = f'{Path(img_name).stem}_r{len(records):06d}.jpg'
                    crop_path = art / 'rec_data' / 'crops' / crop_name

                    records.append({
                        'image_path':   str(crop_path),
                        'source_image': str(img_path),
                        'bbox':         bbox,
                        'text':         normalize_ocr_target(text, rtype),
                        'weight':       weight,
                        'source':       split_name,
                        'type':         rtype,
                    })
        return records

    train_records = read_crops_from_jsonl(art / 'train_split.jsonl', 'gold', MAX_GOLD, 1.0)
    valid_records = read_crops_from_jsonl(art / 'valid_split.jsonl', 'gold', 999999, 1.0)

    print(f'[data] Gold train crops: {len(train_records):,}')
    print(f'[data] Gold valid crops: {len(valid_records):,}')

    if curriculum_stage < 2:
        # Silver
        if USE_SILVER:
            silver_meta = Path(cfg.get('SILVER_META', ''))
            if silver_meta.exists():
                silver_idx = {**img_idx, **build_img_index(silver_img_root)}
                old_idx = img_idx.copy()
                img_idx.update(silver_idx)
                silver_recs = read_crops_from_jsonl(silver_meta, 'silver', MAX_SILVER, 0.5)
                img_idx.clear(); img_idx.update(old_idx)
                train_records.extend(silver_recs)
                print(f'[data] +{len(silver_recs):,} silver crops (weight=0.5)')

    if curriculum_stage == 0:
        # Synthetic TRDG
        if USE_SYNTH:
            synth_csv = art / 'rec_data' / 'synth_manifest.csv'
            if synth_csv.exists():
                synth_df = pd.read_csv(synth_csv).head(MAX_SYNTH)
                synth_recs = synth_df.to_dict('records')
                for r in synth_recs:
                    r.setdefault('weight', 0.3)
                    r.setdefault('source', 'synthetic')
                train_records.extend(synth_recs)
                print(f'[data] +{len(synth_recs):,} synthetic crops (weight=0.3)')

        # HKR
        if USE_HKR:
            hkr_csv = art / 'hkr_manifest.csv'
            if hkr_csv.exists():
                hkr_df = pd.read_csv(hkr_csv).head(MAX_HKR)
                hkr_recs = hkr_df.to_dict('records')
                for r in hkr_recs:
                    r.setdefault('weight', 0.5)
                    r.setdefault('source', 'hkr')
                train_records.extend(hkr_recs)
                print(f'[data] +{len(hkr_recs):,} HKR crops (weight=0.5)')

        # Pseudo-labeled silver
        if USE_PSEUDO:
            pseudo_csv = art / 'rec_data' / 'pseudo_manifest.csv'
            if pseudo_csv.exists():
                pseudo_df = pd.read_csv(pseudo_csv)
                pseudo_recs = pseudo_df.to_dict('records')
                for r in pseudo_recs:
                    r.setdefault('weight', 0.7)
                    r.setdefault('source', 'pseudo')
                train_records.extend(pseudo_recs)
                print(f'[data] +{len(pseudo_recs):,} pseudo-labeled crops (weight=0.7)')

    random.shuffle(train_records)

    print(f'\n[data] Total train: {len(train_records):,}  |  Valid: {len(valid_records):,}')
    if train_records:
        from collections import Counter
        source_counts = Counter(r.get('source', 'unknown') for r in train_records)
        print(f'[data] Sources: {dict(source_counts)}')

    return pd.DataFrame(train_records), pd.DataFrame(valid_records)


def extract_crops(manifest_df: pd.DataFrame, overwrite: bool = False):
    """Crop regions từ full-page images theo bbox. Chạy 1 lần."""
    crop_dir = ART / 'rec_data' / 'crops'
    crop_dir.mkdir(parents=True, exist_ok=True)

    needed = [r for r in manifest_df.to_dict('records')
              if (not Path(r['image_path']).exists() or overwrite)
              and r.get('source_image') and r.get('bbox')]
    if not needed:
        print(f'[crop] All crops exist, skipping extraction.')
        return

    done = errors = 0
    print(f'[crop] Extracting {len(needed):,} crops...')

    for rec in needed:
        crop_path = Path(rec['image_path'])
        try:
            src = Image.open(rec['source_image']).convert('RGB')
            W, H = src.size

            x1, y1, x2, y2 = [int(v) for v in rec['bbox']]
            pad = 4
            x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
            x2 = min(W, x2 + pad);  y2 = min(H, y2 + pad)

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = src.crop((x1, y1, x2, y2))

            # Resize: max height 128px
            if crop.height > 128:
                ratio = 128 / crop.height
                new_w = max(32, min(1920, int(crop.width * ratio)))
                crop = crop.resize((new_w, 128), Image.LANCZOS)

            crop.save(str(crop_path), 'JPEG', quality=92)
            done += 1

        except Exception as e:
            errors += 1
            if errors < 5:
                print(f'[crop] Error: {e}')

        if done % 5000 == 0:
            print(f'[crop] {done:,} done, {errors} errors')

    print(f'[crop] Done: {done:,} extracted, {errors} errors')


def load_model_and_processor():
    """Load Qwen3-VL + apply LoRA."""
    print(f'[model] Loading {MODEL_ID}...')

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    load_kwargs = {'torch_dtype': torch.bfloat16, 'device_map': {'': DEVICE}}
    if USE_FLASH_ATTN:
        load_kwargs['attn_implementation'] = 'flash_attention_2'

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **load_kwargs)
        print('[model] Loaded with Qwen3VLForConditionalGeneration')
    except Exception as e:
        raise RuntimeError(
            f'Cannot load {MODEL_ID}. '
            'Try: pip install "transformers>=4.57.0" qwen-vl-utils'
        ) from e

    if USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Tắt thinking mode
    if hasattr(model, 'generation_config'):
        if hasattr(model.generation_config, 'enable_thinking'):
            model.generation_config.enable_thinking = False

    vram = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f'[model] {params:.2f}B params | VRAM: {vram:.1f}GB / {total:.0f}GB')

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=['q_proj','k_proj','v_proj','o_proj',
                        'gate_proj','up_proj','down_proj'],
        lora_dropout=0.05,
        bias='none',
        use_rslora=True,  # RSLoRA stable với r=64
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f'[lora] Trainable: {trainable/1e6:.1f}M / {total_p/1e6:.0f}M '
          f'({100*trainable/total_p:.2f}%)')

    return model, processor


def train_one_stage(model, processor, train_df: pd.DataFrame,
                    valid_df: pd.DataFrame, stage_name: str,
                    n_epochs: float, output_dir: Path):
    """Train một stage."""
    print(f'\n[train] Stage: {stage_name} | {len(train_df):,} train / {len(valid_df):,} valid')

    train_dataset = HTRCropDataset(train_df.to_dict('records'), processor, augment=True)
    valid_dataset = HTRCropDataset(valid_df.to_dict('records'), processor, augment=False)

    if 'weight' in train_df.columns:
        weight_values = train_df['weight'].astype(float).tolist()
    else:
        weight_values = [1.0] * len(train_df)
    sample_weights = torch.tensor(weight_values, dtype=torch.float)
    print(f'[train] Weighted sampler enabled | '
          f'min={sample_weights.min():.2f} max={sample_weights.max():.2f}')

    collator = partial(collate_fn, processor=processor)

    # Find resume checkpoint
    checkpoints = sorted(output_dir.glob('checkpoint-*'),
                         key=lambda p: int(p.name.split('-')[-1]))
    resume_from = str(checkpoints[-1]) if checkpoints else None
    if resume_from:
        print(f'[train] Resuming from: {Path(resume_from).name}')

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        # Batch
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        # LR
        learning_rate=LR,
        lr_scheduler_type='cosine',
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=n_epochs,

        # Precision
        bf16=USE_BF16, fp16=False, tf32=USE_TF32,

        # Optimizer
        optim='adamw_torch_fused',

        # DataLoader
        dataloader_num_workers=DL_WORKERS,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=2,

        # Checkpointing
        save_strategy='steps',
        save_steps=200,
        save_total_limit=SAVE_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,

        # Eval
        eval_strategy='steps',
        eval_steps=200,

        # Logging
        logging_dir=str(LOGS_DIR / 'tb'),
        logging_steps=20,
        report_to='none',

        # Misc
        remove_unused_columns=False,
        gradient_checkpointing=GRAD_CHECKPT,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        train_sample_weights=sample_weights,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # Save best
    best_dir = output_dir / 'best_checkpoint'
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    print(f'[train] Best checkpoint → {best_dir}')

    return trainer.state.best_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum', action='store_true',
                        help='Dùng curriculum learning: silver/synth → gold')
    parser.add_argument('--rebuild-crops', action='store_true',
                        help='Force re-extract crops')
    args = parser.parse_args()

    use_curriculum = args.curriculum or CURRICULUM

    print('PHASE 2 — TRAIN QWEN3-VL OCR RECOGNIZER')
    print('=' * 60)
    print(f'Model: {MODEL_ID}')
    print(f'LoRA r={LORA_R} α={LORA_ALPHA} RSLoRA=True')
    print(f'Batch/GPU={BATCH_SIZE} | GradAccum={GRAD_ACCUM} | EffBatch={BATCH_SIZE*GRAD_ACCUM}')
    print(f'Curriculum: {use_curriculum}')
    print('=' * 60)

    # Disk check
    free_gb = shutil.disk_usage('/').free / 1e9
    if free_gb < 20:
        raise RuntimeError(f'Only {free_gb:.1f}GB free — need 20GB+')

    # Load model
    model, processor = load_model_and_processor()

    if use_curriculum:
        # ── Stage 1: Warm-up trên noisy data (silver + synthetic) ──
        print('\n═══ CURRICULUM STAGE 1: Silver + Synthetic ═══')
        train_df_1, valid_df = load_manifest(ART, curriculum_stage=0)
        # Filter: chỉ dùng silver/synthetic/hkr
        mask = train_df_1['source'].isin(['silver', 'synthetic', 'hkr', 'synthetic_trdg'])
        train_df_stage1 = train_df_1[mask].reset_index(drop=True)
        if len(train_df_stage1) > 0:
            extract_crops(train_df_stage1, overwrite=args.rebuild_crops)
            extract_crops(valid_df, overwrite=False)
            stage1_dir = OUTPUT / 'stage1'
            stage1_dir.mkdir(exist_ok=True)
            train_one_stage(model, processor, train_df_stage1, valid_df,
                           'Stage1-Noisy', n_epochs=1.0, output_dir=stage1_dir)

        # ── Stage 2: Fine-tune trên gold data ──
        print('\n═══ CURRICULUM STAGE 2: Gold Data ═══')
        train_df_2, valid_df = load_manifest(ART, curriculum_stage=2)
        if len(train_df_2) > 0:
            extract_crops(train_df_2, overwrite=args.rebuild_crops)
            train_one_stage(model, processor, train_df_2, valid_df,
                           'Stage2-Gold', n_epochs=NUM_EPOCHS, output_dir=OUTPUT)

    else:
        # ── Standard training (tất cả data từ đầu) ──
        train_df, valid_df = load_manifest(ART, curriculum_stage=0)
        if len(train_df) == 0:
            raise RuntimeError('No training data found. Run Phase 0 first.')

        extract_crops(train_df, overwrite=args.rebuild_crops)
        extract_crops(valid_df, overwrite=False)

        # Save manifests
        (ART / 'rec_data').mkdir(exist_ok=True)
        train_df.to_csv(ART / 'rec_data' / 'train_manifest.csv', index=False)
        valid_df.to_csv(ART / 'rec_data' / 'valid_manifest.csv', index=False)

        train_one_stage(model, processor, train_df, valid_df,
                       'Standard', n_epochs=NUM_EPOCHS, output_dir=OUTPUT)

    # Cleanup intermediate checkpoints
    for ckpt in OUTPUT.glob('checkpoint-*'):
        if ckpt.is_dir() and ckpt.name != 'best_checkpoint':
            shutil.rmtree(ckpt)
            print(f'[cleanup] Removed {ckpt.name}')

    # Save log
    json.dump({
        'status': 'done',
        'model_id': MODEL_ID,
        'lora_r': LORA_R,
        'curriculum': use_curriculum,
        'effective_batch': BATCH_SIZE * GRAD_ACCUM,
    }, open(LOGS_DIR / 'phase2_training.json', 'w'), indent=2)

    vram = torch.cuda.memory_allocated(0) / 1e9
    print(f'\n[DONE] Phase 2 complete! VRAM: {vram:.1f}GB / 48GB')
    print(f'       Best checkpoint → {OUTPUT}/best_checkpoint')


if __name__ == '__main__':
    main()
