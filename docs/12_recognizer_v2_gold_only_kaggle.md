# Recognizer V2 Gold-Only Kaggle Runbook

Do not start paid or long training until the frozen baseline diagnostic in
`docs/10_gold_only_v2_experiment_tracker.md` is complete.

## Stage 1: No-Training V2 Ablation

Purpose: reuse the existing baseline LoRA and measure only:

- type-aware prompts;
- shared train/inference image preprocessing;
- type-aware generation budgets;
- natural-text repetition penalty.

Run the same full diagnostic notebook from the latest source revision, but use
a different output directory:

```bash
export MAX_TOKENS_HANDWRITTEN=128
export MAX_TOKENS_PRINTED=160
export MAX_TOKENS_FORMULA=256
export MAX_TOKENS_TABLE=384
export MAX_TOKENS_ANNOTATION=96
export NATURAL_REPETITION_PENALTY=1.05
export NATURAL_NO_REPEAT_NGRAM=0

python scripts/08_full_validation_diagnostic.py \
  --output-dir /kaggle/working/full_validation_v2_inference \
  --modes detector,gt_ocr,e2e,report \
  --conf 0.20 \
  --iou 0.45 \
  --imgsz 1024 \
  --ocr-batch 1 \
  --save-every 1 \
  --time-budget-hours 10.5
```

Compare against the frozen baseline:

```bash
python scripts/11_compare_validation_runs.py \
  --baseline /kaggle/input/BASELINE_DIAGNOSTIC/full_validation_report.json \
  --candidate /kaggle/working/full_validation_v2_inference/full_validation_report.json \
  --output /kaggle/working/v2_inference_comparison.md
```

Decision:

- Keep a V2 inference change only when full PageCER improves or a target type
  improves without materially reducing composite score.
- If formula/table improve but natural text regresses, split the configuration
  rather than accepting one global behavior.

## Stage 2: Gold-Only Continuation Training

Required inputs:

```text
/kaggle/input/datasets/bnthanh/rukopys-dataset
/kaggle/input/datasets/ngovietan/htr-02-train-recognizer
```

For continuation shards after the first run, also attach the previous V2
training output dataset.

### Cell 1: Clone And Install

```python
import os, sys, subprocess
from pathlib import Path

ROOT = Path("/kaggle/working/rukopys")
REPO = "https://github.com/thanhbinh55/rukopys.git"

def run(command):
    print("$", command)
    subprocess.run(command, shell=True, executable="/bin/bash", check=True)

if ROOT.exists():
    run(f"git -C {ROOT} fetch origin main")
    run(f"git -C {ROOT} reset --hard origin/main")
else:
    run(f"git clone {REPO} {ROOT}")

run(f"""
{sys.executable} -m pip install -q -U pip wheel setuptools packaging
{sys.executable} -m pip install -q \
  'transformers==4.57.1' \
  'peft==0.17.1' \
  'accelerate>=1.0.0,<2.0.0' \
  qwen-vl-utils timm bitsandbytes \
  pandas numpy pillow rapidfuzz pyyaml tensorboard
""")
os.chdir(ROOT)
```

### Cell 2: Stage Baseline And Optional Resume Checkpoint

```python
!python scripts/10_prepare_kaggle_gold_training.py
```

The baseline adapter is copied to:

```text
/kaggle/working/rukopys/local_working/htr_artifacts/qwen3_lora/baseline_checkpoint
```

Any attached V2 `checkpoint-*` containing `trainer_state.json` is copied into
the working output directory and resumed automatically.

### Cell 3: Configure One Safe Training Shard

Use `TARGET_MAX_STEPS=300` for shard 1, `600` for shard 2, and approximately
`691` for the final shard. A 300-step target is intentionally below Kaggle's
12-hour limit.

```python
import os
from pathlib import Path

ROOT = Path("/kaggle/working/rukopys")
ART = ROOT / "local_working" / "htr_artifacts"
TARGET_MAX_STEPS = 300  # next runs: 600, then 691

os.environ.update({
    "PROJECT_ROOT": str(ROOT),
    "HTR_ART_DIR": str(ART),
    "DATA_ROOT": "/kaggle/input/datasets/bnthanh/rukopys-dataset/rukopys_raw",
    "HF_HOME": "/tmp/hf_cache",
    "TRANSFORMERS_CACHE": "/tmp/hf_cache",
    "PYTHONPATH": f"{ROOT}:{ROOT / 'scripts'}",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
    "CUDA_VISIBLE_DEVICES": "0,1",

    "USE_SILVER": "0",
    "USE_SYNTH": "0",
    "USE_HKR": "0",
    "USE_PSEUDO": "0",
    "CURRICULUM": "0",

    "INIT_LORA_DIR": str(ART / "qwen3_lora" / "baseline_checkpoint"),
    "QLORA_4BIT": "1",
    "USE_FLASH_ATTN": "0",
    "USE_BF16": "0",
    "USE_TF32": "0",
    "GPU_MAX_MEMORY": "13GiB",
    "CPU_MAX_MEMORY": "24GiB",
    "OPTIM": "paged_adamw_8bit",

    "BATCH_SIZE": "1",
    "GRAD_ACCUM": "32",
    "GRAD_CHECKPT": "1",
    "NUM_EPOCHS": "1",
    "LEARNING_RATE": "5e-5",
    "WARMUP_RATIO": "0.03",
    "MAX_STEPS": str(TARGET_MAX_STEPS),

    "SAVE_STEPS": "50",
    "EVAL_STEPS": "200",
    "SAVE_LIMIT": "3",
    "PERIODIC_EVAL_MAX": "512",
    "TRAIN_TIME_BUDGET_HOURS": "8.5",
    "CLEANUP_CHECKPOINTS": "0",

    "TYPE_BALANCE": "1",
    "TYPE_WEIGHT_CAP": "4.0",
    "CROP_JITTER_PX": "3",
    "CROP_JITTER_PROB": "0.35",
    "AUG_PROB": "0.5",

    "MIN_PIXELS": str(128 * 28 * 28),
    "MAX_PIXELS": str(384 * 28 * 28),
    "MAX_SEQ_LEN": "512",
})

Path("/tmp/hf_cache").mkdir(parents=True, exist_ok=True)
```

### Cell 4: Train

```python
!python scripts/02_train_recognizer.py
```

The script:

- loads the old adapter using `INIT_LORA_DIR`;
- does not initialize a new LoRA;
- uses only gold train crops;
- keeps the fixed validation split separate;
- saves every 50 optimizer steps;
- stops cleanly at the configured time budget;
- keeps the latest three trainer checkpoints;
- never deletes checkpoints unless explicitly requested.

### Cell 5: Verify Output

```python
from pathlib import Path
import json

ART = Path("/kaggle/working/rukopys/local_working/htr_artifacts")
checkpoints = sorted((ART / "qwen3_lora").glob("checkpoint-*"))
for checkpoint in checkpoints:
    required = [
        checkpoint / "adapter_model.safetensors",
        checkpoint / "trainer_state.json",
        checkpoint / "optimizer.pt",
    ]
    print(checkpoint.name, {p.name: p.exists() for p in required})

manifest = {
    "checkpoints": [p.name for p in checkpoints],
    "baseline_adapter": str(ART / "qwen3_lora" / "baseline_checkpoint"),
}
(Path("/kaggle/working") / "gold_v2_checkpoint_manifest.json").write_text(
    json.dumps(manifest, indent=2)
)
```

Do not create a second compressed copy of all checkpoints unless enough disk
space remains. Kaggle already includes files under `/kaggle/working` in the
version output.

## Stage 3: Official Checkpoint Selection

Do not promote `best_eval_loss_checkpoint` automatically.

For every retained `checkpoint-*`:

1. run full 143-page GT-box OCR;
2. rank by RegionCER and PageCER;
3. run end-to-end validation for the best one or two;
4. select the highest official composite score;
5. copy only that adapter to `qwen3_lora/best_checkpoint`.

Example:

```bash
python scripts/08_full_validation_diagnostic.py \
  --adapter-dir "$CHECKPOINT" \
  --output-dir "/kaggle/working/eval_$(basename "$CHECKPOINT")" \
  --modes gt_ocr,report \
  --ocr-batch 1 \
  --time-budget-hours 4
```

Only after official selection should Phase 4 test inference use the V2 adapter.
