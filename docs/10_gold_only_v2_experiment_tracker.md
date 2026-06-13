# RUKOPYS Gold-Only V2 Experiment Tracker

Last updated: 2026-06-12

This is the source of truth for the next optimization cycle. Update this file
after every experiment so that model changes are never mixed with unmeasured
assumptions.

## Goal And Guardrails

- Best public leaderboard result so far: `0.66754`
  (`submission_v3_text_strong.csv`).
- Target: approach `0.90`, without using silver data in this cycle.
- Keep the existing 143-page validation split fixed.
- Never train on any validation page or validation crop.
- Measure the current baseline before patching Recognizer V2.
- Change one coherent subsystem at a time.
- Select checkpoints using the official end-to-end metric, not `eval_loss`
  alone.
- Every long Kaggle job must be resumable and stop before the 12-hour limit.

## Current Measured Baseline

The baseline metrics measured on the full 143 validation pages (Kaggle commit `9c5897d`):

### End-to-End Baseline (YOLO Detector -> Qwen3-VL Recognizer V1)

| Metric | Value | Reliability |
|---|---:|---|
| Composite | 0.709221 | High: full 143 validation pages |
| Detector F1 | 0.912793 | High: full 143 validation pages |
| Class accuracy | 0.966914 | High: full 143 validation pages |
| Region CER | 0.323964 | High: full 143 validation pages |
| Page CER | 0.357710 | High: full 143 validation pages |

### Ground-Truth Box OCR Baseline (Isolates Recognizer Quality)

| Metric | Value | Reliability |
|---|---:|---|
| Composite | 0.757555 | High: full 143 validation pages |
| Region CER | 0.331693 | High: full 143 validation pages |
| Page CER | 0.285874 | High: full 143 validation pages |

Per-type Region CER in GT-box OCR:
- handwritten: 0.3394 (2,323 regions)
- formula: 0.2691 (376 regions)
- annotation: 0.5443 (66 regions)
- printed: 0.2037 (53 regions)
- table: 0.2454 (13 regions)
- graph/image: n/a

Full validation contains:

| Type | Regions |
|---|---:|
| handwritten | 2,323 |
| formula | 376 |
| annotation | 66 |
| printed | 53 |
| table | 13 |
| image | 10 |
| graph | 11 |

Sources: 35 dictation, 20 archive, 20 university, and 68 school pages.


## Root-Cause Register

| ID | Phase | Finding | Root cause | Planned fix | Verification |
|---|---|---|---|---|---|
| RC-01 | Phase 2 + Phase 4 | Every type receives a handwritten prompt | One global `SYSTEM_PROMPT` and `USER_PROMPT` | Type-aware prompts shared by train and inference | Prompt contract unit test; full GT-box CER by type |
| RC-02 | Phase 2 | Region `type` disappears before collate | `HTRCropDataset.__getitem__` omits `type` | Return `type`; collator uses it | Dataset/collator contract test |
| RC-03 | Phase 2 | Training resize depends on answer length | `target_h` is selected from ground-truth text length | Shared image-only preprocessing | Train/inference preprocessing parity test |
| RC-04 | Phase 2 | Formula/table structure is flattened | Target normalization replaces all newlines with spaces | Type-aware normalization preserving structured lines | Formula/table normalization tests |
| RC-05 | Phase 4 | One token budget for all types | Global `MAX_NEW_TOKENS` | Per-type generation budgets and type-bucketed batches | Generation-config test; GT-box CER by type |
| RC-06 | Phase 4 | Transformers can reject inference call | Unsupported `enable_thinking=False` is passed to `generate()` | Remove it from model kwargs; set config only when supported | Smoke inference |
| RC-07 | Phase 2 | Rare gold types are under-sampled | Every gold record has weight 1.0 | Capped square-root inverse-frequency sampler | Sampling-distribution report |
| RC-08 | Phase 2 -> Phase 4 | Train crops are cleaner than detector crops | Training only uses fixed GT boxes with four-pixel padding | Safe crop jitter/padding augmentation | GT-box vs jittered-box OCR ablation |
| RC-09 | Phase 3 | Validation result is over-optimistic | First eight pages are all dictation | Full 143-page diagnostic | Full diagnostic report |
| RC-10 | Phase 2 | Evaluation consumes about 31 minutes/run | All 2,802 crops evaluated every 100 steps | Less frequent lightweight eval; full metric at checkpoints | Runtime log and checkpoint metric table |
| RC-11 | Phase 4 | Long OCR loops hurt PageCER | Greedy generation can repeat phrases/dashes | Type-safe generation controls plus `text_strong` cleanup | Full validation PageCER |
| RC-12 | Phase 1/4 | Formula/handwritten duplicates | Detector emits overlapping cross-class boxes | Validate class-aware dedup; do not blindly delete boxes | Detector-only confusion/overlap report |

## Master Checklist

Status values: `[ ] pending`, `[~] running`, `[x] passed`, `[!] blocked`,
`[-] rejected`.

### A. Baseline Inventory

- [x] A1. Confirm raw train/validation data are available locally.
  - Result: 1,330 train images and fixed 143-page validation split available.
- [x] A2. Confirm model weights available locally.
  - Result: blocked locally. Detector and LoRA weights exist only in Kaggle
    input datasets; local checkout has no `best.pt` or adapter weights.
- [x] A3. Confirm local runtime can execute GPU inference.
  - Result: no. Local Python 3.14 has no torch/transformers/ultralytics and the
    M4 Mac has 16 GB unified memory. Real diagnostic inference will run on
    Kaggle T4x2; local is used for static analysis and tests.
- [x] A4. Preserve current baseline source before Recognizer V2 changes.
  - Result: current Phase 2/4 behavior documented in RC-01 through RC-12.

### B. Full Baseline Diagnostic

- [x] B1. Stage detector and recognizer artifacts from Kaggle inputs.
  - Required inputs:
    - `/kaggle/input/datasets/bnthanh/rukopys-dataset`
    - `/kaggle/input/datasets/bnthanh/htr-01-train-detector-output`
    - `/kaggle/input/datasets/ngovietan/htr-02-train-recognizer`
- [x] B2. Detector-only inference on all 143 validation pages.
  - Outputs: detector cache, F1/precision/recall, per-type recall, confusion matrix, overlap report.
  - Result: Detector F1 = 0.912793, 93 overlap pairs.
- [x] B3. GT-box OCR on all scorable validation regions.
  - Output: `valid_gt_box_ocr.csv` and resumable progress JSONL.
  - Result: Region CER = 0.331693, Page CER = 0.285874.
- [x] B4. End-to-end OCR using cached detector predictions.
  - Output: `valid_end_to_end.csv`.
  - Result: Region CER = 0.323964, Page CER = 0.357710, Composite = 0.709221.
- [x] B5. Produce full diagnostic report.
  - Saved under `analysis_outputs/full_validation_diagnostic/`.
- [x] B6. Compare GT-box and end-to-end results.
  - Decision: GT-box Region CER is extremely high (0.331693 vs target 0.12). We must prioritize Recognizer V2 prompt, normalization, and training changes first.


### C. Recognizer V2 Patch

- [~] C1. Add one shared type-aware prompt map.
  - Implementation prepared locally; pending baseline/V2 inference ablation.
- [~] C2. Preserve `type` through dataset, collator, and inference.
  - Implementation prepared locally; pending GPU smoke test.
- [~] C3. Replace label-length resize with shared image-only preprocessing.
  - Implementation prepared locally; pending GPU comparison.
- [~] C4. Preserve formula/table newlines and PSV structure.
  - Contract test passes; pending real formula/table CER.
- [~] C5. Add per-type token budgets.
  - Contract test passes; pending runtime and hallucination measurement.
- [x] C6. Remove unsupported generation kwargs.
  - Source no longer passes `enable_thinking` to `generate()`.
- [~] C7. Add capped type-balanced sampling.
  - Weight calculation test passes; pending training sampler report.
- [~] C8. Add safe GT-box crop jitter for training only.
  - Implementation prepared with environment-controlled probability/size.
- [x] C9. Add unit/regression tests for all contracts.
  - Seven local contract tests pass.
- [ ] C10. Run no-training ablation on baseline checkpoint.
  - Compare old vs V2 inference on the same cached validation set.
  - This measures prompt/preprocessing/generation changes before training.

### D. Gold-Only Continuation Training

- [~] D1. Stage the existing baseline `best_checkpoint`.
  - Staging script prepared; pending Kaggle execution.
- [~] D2. Verify `INIT_LORA_DIR` loads a trainable existing adapter.
  - Source path implemented; pending GPU smoke log.
- [ ] D3. Verify no validation image/crop enters training.
- [~] D4. Train one gold-only continuation epoch.
  - `USE_SILVER=0`
  - `USE_SYNTH=0`
  - `USE_HKR=0`
  - `USE_PSEUDO=0`
  - `LEARNING_RATE=5e-5`
  - `NUM_EPOCHS=1`
  - `SAVE_STEPS=50`
  - `EVAL_STEPS=200`
  - `SAVE_LIMIT=3`
  - T4 plan: resumable targets around step 300, 600, then 691; do not try
    to force the full epoch into one 12-hour session.
- [ ] D5. Save checkpoint manifest and periodic backup bundle.
- [ ] D6. Evaluate checkpoint 50/100/... using the same full diagnostic cache.
- [ ] D7. Select the best checkpoint by official composite and PageCER.

### E. Final Decision

- [ ] E1. Retrain detector only if full diagnostic proves it is necessary.
- [ ] E2. Run full test inference with resume every image.
- [ ] E3. Apply the validated `text_strong` policy.
- [ ] E4. Validate 385 rows, JSON, coverage, and no empty outputs.
- [ ] E5. Submit and record public/private leaderboard score.

## Experiment Log

| Date | Experiment | Code/config | Result | Decision |
|---|---|---|---|---|
| 2026-06-12 | Inventory | Local checkout | No local model weights or ML runtime | Run real diagnostics on Kaggle |
| 2026-06-12 | Existing eight-page validation audit | Baseline Phase 3 | Sample is dictation-only; rare types absent | Full 143-page validation is mandatory |
| 2026-06-12 | Existing postprocess leaderboard comparison | V3 candidates | `text_strong=0.66754`; stronger bbox deletion is worse | OCR is higher priority than detector retraining |
| 2026-06-12 | Baseline diagnostic launch | Kaggle commit `9c5897d` | Running on `bnthanh/htr-full-validation-diagnostic` | Do not interpret V2 changes until baseline report completes |
| 2026-06-12 | Recognizer V2 contract tests | Local pure-Python tests | 7/7 passed | Proceed to GPU ablation after baseline report |
| 2026-06-13 | Baseline diagnostic report | Kaggle commit `9c5897d` | E2E Composite: 0.709221, GT Region CER: 0.331693 | Prioritize Recognizer V2; next is Stage 1 V2 ablation |

## Result Template For Each New Run

```text
Experiment ID: EXP-01-BASELINE
Purpose: Full validation diagnostic of the baseline system (YOLO detector and Qwen3-VL V1 recognizer) on 143 validation pages.
Checkpoint/source revision: 9c5897d24b63cdcffb9c2fc8f7aeb5955729d53f
Changed variables: none
Fixed variables: default baseline thresholds (conf=0.20, iou=0.45), imgsz=1024, max_tokens=96, dataloader defaults
Input artifacts:
  - bnthanh/rukopys-dataset
  - bnthanh/htr-01-train-detector-output (detector)
  - ngovietan/htr-02-train-recognizer (recognizer)
Runtime/hardware: Kaggle GPU T4 x2
Completion: Completed successfully (143/143 pages for detector, gt_ocr, e2e)

Composite: 0.709221 (E2E) / 0.757555 (GT-box OCR)
Detector F1 / precision / recall: 0.912793 / 0.884287 / 0.943198
Class accuracy: 0.966914
Region CER: 0.323964 (E2E) / 0.331693 (GT-box OCR)
Page CER: 0.357710 (E2E) / 0.285874 (GT-box OCR)

Per-type Region CER (GT-box OCR / E2E):
- handwritten: 0.3394 / 0.3333
- formula: 0.2691 / 0.2255
- annotation: 0.5443 / 0.5127
- printed: 0.2037 / 0.4929
- table: 0.2454 / 0.2067

Per-source PageCER (GT-box OCR / E2E):
- archive: 0.3486 / 0.4016
- dictation: 0.2761 / 0.2664
- school: 0.2602 / 0.3497
- university: 0.3275 / 0.5007

Failure examples:
- images/195812b7-a9bc-4cb1-b688-4d13261acc7a.jpg (university, E2E PageCER=1.2405, predicted regions=36/20, high class confusion and extra boxes)
- images/3dd87640-0760-48cd-900c-a10d930b497b.jpg (dictation, GT-box PageCER=0.9388, E2E PageCER=1.2226, severe OCR errors)
Runtime: ~2.5 hours total execution
Disk/VRAM peak: Normal, within constraints

Conclusion:
Baseline successfully measured.
The primary bottleneck is the recognizer (GT Region CER is 0.331693, far from target <=0.12).
Detector class confusion (e.g. formula vs handwritten) and duplicate box overlaps (93 overlap pairs) also degrade PageCER.
Pass/fail: Pass (baseline set)
Next decision: Proceed to Stage 1 No-Training V2 Ablation on Kaggle to evaluate prompt/preprocessing/generation config improvements before training.
```
