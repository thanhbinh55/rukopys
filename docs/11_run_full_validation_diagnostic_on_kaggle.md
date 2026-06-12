# Run Full Validation Diagnostic On Kaggle

This run measures the existing baseline. It does not train and does not alter
the detector or LoRA adapter.

## Required Kaggle Inputs

Attach:

```text
/kaggle/input/datasets/bnthanh/rukopys-dataset
/kaggle/input/datasets/bnthanh/htr-01-train-detector-output
/kaggle/input/datasets/ngovietan/htr-02-train-recognizer
```

Use `GPU T4 x2`, Internet on, and a fresh notebook version.

## Cell 1: Clone And Install

The diagnostic scripts must already be committed to the repository before this
cell is run.

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
  qwen-vl-utils timm bitsandbytes ultralytics \
  pandas numpy pillow rapidfuzz pyyaml
""")

os.chdir(ROOT)
print("source revision:")
run("git rev-parse HEAD")
```

## Cell 2: Stage Existing Models

```python
import os
from pathlib import Path

ROOT = Path("/kaggle/working/rukopys")
os.environ.update({
    "PROJECT_ROOT": str(ROOT),
    "HTR_ART_DIR": str(ROOT / "local_working" / "htr_artifacts"),
    "HF_HOME": "/tmp/hf_cache",
    "TRANSFORMERS_CACHE": "/tmp/hf_cache",
    "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    "PYTHONPATH": f"{ROOT}:{ROOT / 'scripts'}",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "USE_FLASH_ATTN": "0",
    "QLORA_4BIT": "1",
    "GPU_MAX_MEMORY": "13GiB",
    "CPU_MAX_MEMORY": "24GiB",
    "MIN_PIXELS": str(128 * 28 * 28),
    "MAX_PIXELS": str(384 * 28 * 28),
    "MAX_TOKENS": "96",
    "OCR_BATCH": "1",
})

!python scripts/09_prepare_kaggle_diagnostic.py
!nvidia-smi
```

## Cell 3: Run The Full Baseline Diagnostic

This cell is resumable. Re-running it skips pages already present in each
progress JSONL.

```python
!python scripts/08_full_validation_diagnostic.py \
  --modes detector,gt_ocr,e2e,report \
  --conf 0.20 \
  --iou 0.45 \
  --imgsz 1024 \
  --ocr-batch 1 \
  --save-every 1 \
  --time-budget-hours 10.5
```

Expected outputs:

```text
/kaggle/working/full_validation_diagnostic/
  detector_progress.jsonl
  gt_ocr_progress.jsonl
  e2e_progress.jsonl
  valid_detector_only.csv
  valid_gt_box_ocr.csv
  valid_end_to_end.csv
  full_validation_report.json
  full_validation_report.md
  diagnostic.log
```

## Cell 4: Verify And Bundle

```python
from pathlib import Path
import json

OUT = Path("/kaggle/working/full_validation_diagnostic")
report = json.loads((OUT / "full_validation_report.json").read_text())
print(json.dumps(report["completed_pages"], indent=2))
print((OUT / "full_validation_report.md").read_text()[:12000])

!tar -czf /kaggle/working/full_validation_diagnostic.tgz \
  -C /kaggle/working full_validation_diagnostic
!ls -lh /kaggle/working/full_validation_diagnostic.tgz
```

Expected completion:

```json
{
  "detector": 143,
  "gt_ocr": 143,
  "e2e": 143
}
```

## How To Interpret The Three Results

### Detector-only

This determines whether Phase 1 is already sufficient:

- overall F1/precision/recall;
- recall and false negatives per GT type;
- class confusion, especially formula versus handwritten;
- duplicate overlap count.

### GT-box OCR

Every crop uses the exact validation ground-truth box and type. Detection,
classification, and crop location are therefore controlled. Its Region CER is
the cleanest measurement of the current recognizer.

### End-to-end

This uses detector boxes and predicted types. The difference from GT-box OCR
measures the combined cost of:

- missing/extra boxes;
- class mistakes;
- imperfect crop boundaries;
- duplicate boxes;
- reading order.

## Decision Rules

| Observation | Conclusion | Next action |
|---|---|---|
| GT-box RegionCER remains above 0.18 | Recognizer itself is weak | Patch/train Recognizer V2 first |
| GT-box is good but end-to-end is much worse | Detector crop/order is dominant | Tune padding, dedup, order, then Detector V2 if required |
| Formula/table CER is much worse | Current prompt/normalization is damaging structured text | Type-aware prompt and structured target patch |
| Rare-type recall/class accuracy is low | Detector Phase 1 is weak on imbalance | Detector V2 after Recognizer V2 |
| PageCER is much higher than RegionCER | Missing/duplicate/order errors dominate | Inspect worst-page report and visual overlays |

After the run, copy the complete output directory back into:

```text
analysis_outputs/full_validation_diagnostic/
```

Then update `docs/10_gold_only_v2_experiment_tracker.md` with the measured
numbers before applying the Recognizer V2 patch.
