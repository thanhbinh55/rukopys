#!/bin/bash
# scripts/13_run_v2_ablation.sh
# Run No-Training Recognizer V2 Ablation on Kaggle.

# Configure Recognizer V2 settings
export MAX_TOKENS_HANDWRITTEN=128
export MAX_TOKENS_PRINTED=160
export MAX_TOKENS_FORMULA=256
export MAX_TOKENS_TABLE=384
export MAX_TOKENS_ANNOTATION=96
export NATURAL_REPETITION_PENALTY=1.05
export NATURAL_NO_REPEAT_NGRAM=0

echo "=== Staging baseline models ==="
python scripts/09_prepare_kaggle_diagnostic.py

echo "=== Running Validation Diagnostic with Recognizer V2 Settings ==="
python scripts/08_full_validation_diagnostic.py \
  --output-dir /kaggle/working/full_validation_v2_inference \
  --modes detector,gt_ocr,e2e,report \
  --conf 0.20 \
  --iou 0.45 \
  --imgsz 1024 \
  --ocr-batch 1 \
  --save-every 1 \
  --time-budget-hours 10.5

echo "=== Comparing Candidate (V2 settings) against Baseline ==="
# On Kaggle, compare the V2 results against the baseline.
# The baseline can be read from either:
# 1. Staged baseline inputs at /kaggle/input/BASELINE_DIAGNOSTIC/full_validation_report.json
# 2. Sequential execution output folder at /kaggle/working/full_validation_diagnostic/full_validation_report.json
if [ -f "/kaggle/input/BASELINE_DIAGNOSTIC/full_validation_report.json" ]; then
  python scripts/11_compare_validation_runs.py \
    --baseline /kaggle/input/BASELINE_DIAGNOSTIC/full_validation_report.json \
    --candidate /kaggle/working/full_validation_v2_inference/full_validation_report.json \
    --output /kaggle/working/v2_inference_comparison.md
elif [ -f "/kaggle/working/full_validation_diagnostic/full_validation_report.json" ]; then
  python scripts/11_compare_validation_runs.py \
    --baseline /kaggle/working/full_validation_diagnostic/full_validation_report.json \
    --candidate /kaggle/working/full_validation_v2_inference/full_validation_report.json \
    --output /kaggle/working/v2_inference_comparison.md
else
  # Try to search under /kaggle/input for the baseline report
  BASELINE_REPORT=$(find /kaggle/input -name "full_validation_report.json" | head -n 1)
  if [ -n "$BASELINE_REPORT" ]; then
    python scripts/11_compare_validation_runs.py \
      --baseline "$BASELINE_REPORT" \
      --candidate /kaggle/working/full_validation_v2_inference/full_validation_report.json \
      --output /kaggle/working/v2_inference_comparison.md
  else
    echo "Warning: Could not locate baseline full_validation_report.json to compare!"
  fi
fi

if [ -f "/kaggle/working/v2_inference_comparison.md" ]; then
  echo "=== Comparison Result ==="
  cat /kaggle/working/v2_inference_comparison.md
fi

echo "=== Creating tarball bundle ==="
tar -czf /kaggle/working/full_validation_v2_inference.tgz \
  -C /kaggle/working full_validation_v2_inference
echo "Done! Output bundle saved to /kaggle/working/full_validation_v2_inference.tgz"
