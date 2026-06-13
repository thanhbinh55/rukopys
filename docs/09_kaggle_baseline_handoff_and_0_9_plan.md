# RUKOPYS Kaggle Handoff And 0.9 Plan

Last updated: 2026-06-12

This document is a compact handoff for continuing the current RUKOPYS HTR work from Phase 0 through the first full Kaggle leaderboard submission. It is written so another AI/chatbox can understand what has been run, what broke, what was fixed, what the current bottlenecks are, and what to optimize next.

## Current State

The baseline pipeline now works end to end on Kaggle and has produced a complete submission:

- Phase 0/0A data audit and split: completed.
- Phase 1 YOLO detector training: completed and exported detector artifacts.
- Phase 2 Qwen3-VL recognizer training: completed enough to export a LoRA `best_checkpoint`.
- Phase 3 threshold validation: completed as a quick sanity grid on 8 validation images.
- Phase 4 test inference/submission: completed for `385 / 385` images.
- User-reported Kaggle leaderboard score after the raw full submission: about `0.61-0.63`.
- User-reported score after Postprocess V2: `0.66`.

Historical note:

- An earlier partial submission scored `0.51` because only `237 / 385` test images had real predictions and `148` rows were empty `[]`.
- The current full run removed that coverage problem: `submission.csv` has `385 / 385` non-empty rows, `0` invalid JSON rows, and `0` missing sample-submission images.

Important result: the model is no longer blocked by Kaggle mechanics or incomplete output. The next bottleneck is model/pipeline quality: OCR errors, repeated text hallucination, duplicate overlapping boxes, formula/handwritten confusion, and too-small validation.

## Output Artifacts Analyzed

Latest full-run artifacts are under:

```text
rukopys_full_063/input/datasets/ngovietan/htr-output-kaggle-full-ver1-061/
```

The nested artifact directory contains:

```text
rukopys/local_working/htr_artifacts/best_config.json
rukopys/local_working/htr_artifacts/config.json
rukopys/local_working/htr_artifacts/grid_search_results.csv
rukopys/local_working/htr_artifacts/phase3_4_summary.json
rukopys/local_working/htr_artifacts/phase4_progress.jsonl
rukopys/local_working/htr_artifacts/scorable_stats.json
rukopys/local_working/htr_artifacts/submission.csv
rukopys/local_working/htr_artifacts/train_split.jsonl
rukopys/local_working/htr_artifacts/valid_gt.csv
rukopys/local_working/htr_artifacts/valid_split.jsonl
```

Top-level output also contains:

```text
phase3_4_summary.json
submission.csv
submission_partial.csv
```

Local raw data used for cross-checks:

- `rukopys_raw/test/metadata.jsonl`
- `rukopys_raw/train/metadata.jsonl`
- `rukopys_raw/sample_submission.csv`

## What We Ran From Phase 0 To Now

### Phase 0A: Data Setup And Audit

Purpose:

- Read Kaggle dataset from `/kaggle/input/datasets/bnthanh/rukopys-dataset/rukopys_raw`.
- Create `train_split.jsonl`, `valid_split.jsonl`, `valid_gt.csv`, and `config.json`.
- Inspect class imbalance and text length.

Observed data stats:

- Train pages: `1,330`
- Silver pages: `8,202`
- Test pages: `385`
- Total regions: `25,651`
- Scorable regions: `24,893` or `97.04%`
- Type distribution:
  - handwritten: `21,577`
  - formula: `2,892`
  - annotation: `554`
  - printed: `308`
  - table: `142`
  - image: `119`
  - graph: `59`
- Warning: class imbalance is severe; rare classes are `printed`, `table`, `annotation`, `image`, `graph`.
- Warning: some regions have text longer than 256 chars.

### Phase 1: Detector Training

Purpose:

- Train YOLO detector for region bboxes and region types.

What mattered operationally:

- On Kaggle T4, detector training was feasible.
- The output dataset `htr-01-train-detector-output` contains the detector artifacts and source tree.
- YOLO emitted read-only filesystem/cache warnings when using Kaggle input-style paths. The safer approach is to prepare/copy writable training artifacts under `/kaggle/working`.
- Class imbalance remained a known detector risk.

Known Phase 1 direction for improvement:

- Increase class loss weight for rare classes.
- Oversample rare pages.
- Use silver data carefully for detector.
- Train with larger image size/model on paid GPU.

### Phase 2: Recognizer Training

Purpose:

- Fine-tune `Qwen/Qwen3-VL-8B-Instruct` for OCR using LoRA/QLoRA.

Kaggle T4 constraints:

- Single T4 has about 15 GB VRAM; full 8B fine-tuning is not safe.
- T4 x2 does not automatically solve this unless the code properly shards model/training across GPUs.
- The workable Kaggle T4 route was QLoRA, batch size 1, gradient accumulation, and conservative pixels/tokens.

Important environment choices used or recommended:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/tmp/hf_cache
export TRANSFORMERS_CACHE=/tmp/hf_cache
export YOLO_CONFIG_DIR=/kaggle/working/rukopys/local_working/ultralytics
export PYTHONPATH=/kaggle/working/rukopys:/kaggle/working/rukopys/scripts
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

export USE_FLASH_ATTN=0
export QLORA_4BIT=1
export DEVICE_MAP=auto
export GPU_MAX_MEMORY=13GiB
export CPU_MAX_MEMORY=24GiB
export OPTIM=paged_adamw_8bit

export BATCH_SIZE=1
export GRAD_ACCUM=32
export GRAD_CHECKPT=1
export SAVE_STEPS=100
export EVAL_STEPS=100
export SAVE_LIMIT=2
```

What worked:

- Qwen3-VL 8B QLoRA could load and train on Kaggle T4 x2.
- The output dataset `htr-02-train-recognizer` contains `qwen3_lora/best_checkpoint`.
- Checkpoints saved periodically, so timeout did not destroy all progress.

What did not work well:

- Training was slow on T4.
- Full model loading/training on T4 easily OOMs if not using 4-bit/QLoRA.
- Initial notebook errors included using bash heredoc in a Python cell and missing helper script paths.

### Phase 3: Quick Validation

Purpose:

- Find a YOLO confidence/IoU threshold using validation data and the official metric.

Current run:

- Validation size: `8` images only.
- Grid:
  - `conf=0.05, iou=0.45`
  - `conf=0.10, iou=0.45`
  - `conf=0.20, iou=0.45`
- Best:
  - `conf=0.20`
  - `iou=0.45`
  - composite score: `0.842659`
  - DetF1: `0.966361`
  - RegionCER: `0.218372`
  - PageCER: `0.172301`

Interpretation:

- The subset score is promising.
- It is not robust because `N=8` is too small.
- `PageCER` is the dominant component in the competition score, so reading order and repeated/hallucinated text matter a lot.

### Phase 4: Test Inference

Purpose:

- YOLO detect test pages.
- Sort regions in reading order.
- OCR scorable regions with Qwen3-VL LoRA.
- Save submission continuously.

Current full run:

- Time budget: `11.0h`
- Processed images: `385 / 385`
- Complete: `true`
- `submission.csv`: valid shape with all `385` rows.
- Empty rows: `0`
- Invalid JSON rows: `0`
- Missing sample-submission images: `0`
- User-reported leaderboard score: about `0.61`.
- Full-run Phase4 elapsed time for the resume job: `14,540.8s` or about `4.04h`.

Current prediction volume:

- Total predicted regions: `8,934`
- Mean regions per image: `23.21`
- Median regions per image: `23`
- P90 regions per image: `35`
- Max regions in one image: `85`

Predicted type distribution on test:

- handwritten: `6,686`
- formula: `1,076`
- printed: `801`
- annotation: `307`
- image: `31`
- graph: `18`
- table: `15`

Coverage by source is now complete:

- dictation: processed
- archive: processed
- university: processed
- school: processed

This means the current score is a real full-baseline score, not a partial-output artifact.

## Bugs And Fixes Learned

### `enable_thinking` Generation Error

Observed error:

```text
ValueError: The following `model_kwargs` are not used by the model: ['enable_thinking']
```

Cause:

- Current installed `transformers`/Qwen3-VL generation path did not accept `enable_thinking=False` inside `model.generate(...)`.

Fix:

- Remove `enable_thinking=False` from `scripts/inference_utils.py` before importing/running inference.

Hotfix cell:

```python
from pathlib import Path
import sys, importlib

ROOT = Path("/kaggle/working/rukopys")
inf_path = ROOT / "scripts" / "inference_utils.py"
src = inf_path.read_text()
src = src.replace("                enable_thinking=False,\n", "")
inf_path.write_text(src)

if "inference_utils" in sys.modules:
    import inference_utils
    importlib.reload(inference_utils)
```

### Kaggle Output And Timeout

What we learned:

- `kaggle kernels output ...` only downloads committed output, not the live `/kaggle/working` of a currently running job.
- During Save & Commit, you generally cannot run diagnostic cells in parallel.
- Files saved under `/kaggle/working` are what matter for Kaggle output commit.
- Phase2 checkpoints saved every `SAVE_STEPS=100`; each checkpoint is an adapter/state snapshot, not a nested copy of all previous checkpoints.
- Phase4 saved `submission.csv` and `phase4_progress.jsonl` every 5 images, which protected the partial output.

### Disk And Cache

Current Phase3/4 output was small:

- `submission.csv`: about `845 KB`
- `phase3_4_artifacts.tgz`: about `557 KB`
- `/kaggle/working` used under 1 GB in the analyzed run.

Best practice:

- Put HF/base-model cache in `/tmp` or ephemeral storage, not `/kaggle/working`.
- Keep `/kaggle/working` for final outputs, checkpoints, logs, and small bundles.
- Avoid saving full base model weights to output; save LoRA adapters/checkpoints only.

## Current Strengths

1. End-to-end Kaggle pipeline is now proven.

The full `385 / 385` submission was valid and scored on the leaderboard. The project is past the mechanics stage.

2. Output formatting and resume logic are stable.

The full run had:

- `385 / 385` rows.
- `385 / 385` progress entries.
- `0` empty rows.
- `0` invalid JSON rows.
- `0` missing sample-submission images.

3. Quick validation says the detector can be strong.

On the small 8-image Phase3 grid:

- Best threshold: `conf=0.20`, `iou=0.45`.
- DetF1: `0.966`.
- RegionCER: `0.218`.
- PageCER: `0.172`.
- Composite: `0.8427`.

This result is not representative enough, but it proves the pipeline can score high on at least some validation pages.

4. T4 resume inference is viable.

The resumed 148-image run completed in about `4.04h`, safely under the 12h Kaggle limit. For inference-only continuation, T4 can work if progress is reused and Phase3 is skipped.

5. The current artifacts are sufficient for deeper offline analysis.

The latest full-run dataset contains the full submission, progress JSONL, split files, threshold grid, best config, and data stats.

## Current Weaknesses

### 1. Full Coverage Only Raised LB From 0.51 To About 0.61-0.63

This is the most important new signal.

Before full coverage, the optimistic guess was that completing the missing `148` rows might push the baseline much higher. It did improve the score, but only to about `0.61`, so the current bottleneck is now quality, not mechanics.

Likely contributors:

- OCR is still noisy.
- Page-level concatenation is hurt by repeated/hallucinated text.
- Duplicate detections repeat the same text in the page string.
- Formula/handwritten overlaps are common.
- Phase3 threshold was selected on only 8 images.

### 2. Duplicate/Overlapping Detections Are Much Worse In The Full Run

Current full-run predictions:

- Images with IoU >= 0.80 overlapping region pairs: `102 / 385`.
- Overlap pairs: `201`.
- Most common duplicate pairs:
  - formula vs handwritten: `80`
  - handwritten vs formula: `71`
  - handwritten vs annotation: `8`
  - handwritten vs printed: `8`
  - printed vs handwritten: `8`
  - annotation vs handwritten: `8`

This is dangerous because PageCER builds a page-level string. If the same visual line is predicted twice as `formula` and `handwritten`, the page text may contain duplicate content and lose a lot of the `0.50` PageCER component.

### 3. OCR Repetition And Hallucination Are Clear

Current full-run stats:

- Text length mean: `35.05`
- Text length median: `31`
- Text length P90: `63`
- Max text length: `289`
- Text predictions >= 180 chars: `75`

Observed repeated/hallucinated patterns include:

- repeated `відповідно до встановлених норм та правил...`
- repeated `— — — — ...`
- repeated sentence fragments in long handwritten regions
- repeated `перервах, перервах...`

This likely hurts both RegionCER and PageCER. For the next run, generation constraints and post-OCR cleanup should be treated as first-class improvements, not polish.

### 4. Phase3 Validation Is Too Small To Trust

`N_VALID=8` was only a smoke test. The `0.8427` validation score did not translate to the full leaderboard (`~0.61`), meaning the quick grid is over-optimistic or not representative.

Need:

- Phase3 on at least `50` pages.
- Ideally all `143` validation pages.
- Per-source metrics: dictation, archive, university, school.
- Per-type diagnostics, especially formula/printed/annotation.

### 5. Predicted Type Distribution Suggests Class Confusion

Full test predicted type counts:

- handwritten: `6,686`
- formula: `1,076`
- printed: `801`
- annotation: `307`
- image: `31`
- graph: `18`
- table: `15`

By source:

- dictation: mostly handwritten, almost no formula.
- archive: many printed regions.
- university: many formula regions.
- school: formula and handwritten dominate.

The high number of formula/handwritten overlap pairs suggests the detector/classifier cannot reliably decide whether math-like handwriting should be `formula` or `handwritten`. Because classification is only 5% but duplicated OCR text hurts PageCER, postprocessing should prioritize page text quality over keeping every ambiguous duplicate.

### 6. Current Recognizer Was Trained Under T4 Constraints

The Qwen3-VL recognizer is a QLoRA/T4-compatible baseline. That was the right move to make the pipeline run, but it is unlikely to be enough for `0.9`.

Need a stronger recognizer run on L40/H100:

- more representative training data
- better LoRA settings
- selected silver curriculum
- stricter generation controls
- validation CER tracked by source/type

### 7. Current Phase4 Is Still Sequential

The resume job was fine, but full inference from scratch on T4 would still be risky. For future experiments, build sharding and caching as default behavior:

- cache YOLO detections
- cache OCR outputs
- support `SHARD_ID/NUM_SHARDS`
- merge shard submissions deterministically
- save every image

## Score Formula Reminder

Official score:

```text
Score = 0.15 * Detection_F1
      + 0.05 * ClassAcc
      + 0.30 * (1 - RegionCER)
      + 0.50 * (1 - PageCER)
```

To reach `0.90`, a realistic target is:

- DetF1: `>= 0.97`
- ClassAcc: `>= 0.90`
- RegionCER: `<= 0.12`
- PageCER: `<= 0.08-0.10`

Example:

```text
0.15 * 0.97 + 0.05 * 0.90 + 0.30 * 0.88 + 0.50 * 0.92
= 0.9145
```

So the main battle is PageCER and OCR, not just detector mAP.

## Immediate Next Step After Full Baseline

The next best step is **not** to blindly retrain. The next step is to produce a reliable diagnostic loop so every expensive GPU run has a measurable target.

Priority order:

1. Run a larger Phase3 validation using the current model.
2. Add duplicate/overlap suppression and OCR repetition cleanup, then re-score validation.
3. Only then train Detector V2 and Recognizer V2.

Minimum diagnostic run:

```python
RUN_PHASE3 = True
RUN_PHASE4 = False
PHASE3_N_VALID = 50
CONF_GRID = "0.10,0.15,0.20,0.25,0.30"
IOU_GRID = "0.40,0.45,0.50,0.55"
OCR_BATCH = 1
MAX_TOKENS = 96
USE_TTA = False
```

Better diagnostic run on L40/H100:

```python
RUN_PHASE3 = True
RUN_PHASE4 = False
PHASE3_N_VALID = 143
CONF_GRID = "0.10,0.15,0.20,0.25,0.30,0.35"
IOU_GRID = "0.40,0.45,0.50,0.55,0.60"
OCR_BATCH = 2
MAX_TOKENS = 96
MAX_PIXELS = 512 * 28 * 28
```

Before any new leaderboard submission, run a submission sanity report:

```text
rows == 385
progress rows == 385
empty rows == 0
invalid JSON rows == 0
overlap pairs IoU>=0.80 < current 201
texts >=180 chars < current 75
no repeated dash-only OCR outputs
```

## Next Kaggle Optimization Strategy

Use Kaggle T4 only for smoke tests and resumed inference. Use L40/H100 for real training and larger validation.

For Kaggle T4 inference or smoke tests:

```python
RUN_PHASE3 = False
RUN_PHASE4 = True
TIME_BUDGET_HOURS = 11.0
SAVE_EVERY_IMAGES = 1
OCR_BATCH = 1
MAX_TOKENS = 96
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 384 * 28 * 28
YOLO_IMGSZ = 1024
USE_TTA = False
DEFAULT_YOLO_CONF = 0.20
DEFAULT_YOLO_IOU = 0.45
```

For future full inference, add sharding instead of relying on one long sequential job:

```text
NUM_SHARDS=2 or 4
SHARD_ID=0..NUM_SHARDS-1
```

Each shard writes:

```text
submission_shard_{SHARD_ID}.csv
phase4_progress_shard_{SHARD_ID}.jsonl
```

Then merge by `sample_submission.csv` order.

## Advanced Pipeline For 0.9

### Stage A: Diagnostic And Postprocessing First

Goal:

- Improve the current `0.66` postprocessed baseline without expensive retraining.
- Identify which metric component blocks progress.

Actions:

- Run Phase3 on `50-143` validation pages.
- Save `valid_pred_current.csv` for manual inspection.
- Add duplicate overlap suppression for IoU >= `0.80`.
- Add repetition cleanup for OCR outputs.
- Re-score using `kaggle_metric.score_detailed`.

Current implementation:

- Script: `scripts/05_diagnostic_postprocess_v2.py`.
- Input used locally: `rukopys_full_063/input/datasets/ngovietan/htr-output-kaggle-full-ver1-061/submission.csv`.
- Output candidate: `analysis_outputs/postprocess_v2/submission_postprocess_v2.csv`.
- Diagnostic report: `analysis_outputs/postprocess_v2/diagnostic_postprocess_v2_report.md`.
- Validity check: `385` rows, `0` invalid JSON rows, `0` empty rows.

Why this comes first:

- Current full submission has `201` high-overlap duplicate pairs.
- Current full submission has `75` OCR texts with length >= 180 chars, many showing repetition/hallucination.
- These can damage PageCER badly and can be tested without new training.

Local quick result from Postprocess V2:

- Regions: `8934 -> 8743` (`191` removed).
- Images with IoU >= `0.80` overlaps: `102 -> 9`.
- IoU >= `0.80` overlap pairs: `201 -> 10`.
- Texts >= `180` chars: `75 -> 46`.
- Actions: `191` duplicate regions removed, `46` OCR strings cleaned.
- User-reported leaderboard result: approximately `0.61 -> 0.66`.

Postprocess V3 is now implemented in:

```text
scripts/06_postprocess_v3_candidates.py
```

It fixes an important V2 issue: the generic repetition cleaner could shorten
valid formula/table content. V3 always starts from the original full
submission and writes four independent candidates:

```text
analysis_outputs/postprocess_v3/submission_v3_safe.csv
analysis_outputs/postprocess_v3/submission_v3_balanced.csv
analysis_outputs/postprocess_v3/submission_v3_text_strong.csv
analysis_outputs/postprocess_v3/submission_v3_dedup_strong.csv
```

V3 candidate summary:

| Profile | Regions | IoU >= .80 pairs | Texts >= 180 | Regions removed by train-GT audit |
|---|---:|---:|---:|---:|
| safe | 8744 | 11 | 38 | 0 / 25651 |
| balanced | 8689 | 11 | 38 | 2 / 25651 |
| text_strong | 8689 | 11 | 34 | 2 / 25651 |
| dedup_strong | 8587 | 11 | 38 | 6 / 25651 |

Recommended submit order:

1. `safe`
2. `balanced`
3. `text_strong`
4. `dedup_strong`

Do not automatically apply dictionary spelling correction, hard text-length
caps, or broad type reclassification without representative validation
predictions. The train-GT distribution contains legitimate long text,
multilingual archive text, formulas, and nested regions that those rules can
damage.

### Stage B: Detector V2

Goal:

- Improve recall/classification while reducing duplicate/overlapping detections.

Recommended paid GPU:

- L40 48GB minimum.
- H100 if available for final competition-aligned run.

Training changes:

- Use larger detector: `yolov8x` or current best Ultralytics x-size model available in the repo environment.
- Increase image size: `1280` if memory allows.
- Increase class loss weight: start `cls=1.0`, test `cls=1.2-1.5`.
- Oversample rare classes:
  - formula
  - printed
  - table
  - annotation
  - graph/image
- Add silver pages for detector in a controlled way:
  - first `MAX_SILVER=2000`
  - then `MAX_SILVER=4000-8000` only if validation improves.
- Use `close_mosaic` in later epochs.
- Save best by validation mAP and run the official metric afterward.

Postprocessing changes:

- Add class-aware duplicate suppression after YOLO.
- If two boxes overlap IoU >= `0.80` and OCR/page order would duplicate text, keep the higher-confidence box or prefer the class with stronger confidence.
- For `formula` vs `handwritten`, keep both only if boxes differ enough or their OCR text differs materially.
- Consider Weighted Boxes Fusion only for TTA/ensembles; otherwise start with smarter NMS.

Target:

- DetF1 `>=0.97`.
- Rare class recall visibly better.
- High-overlap duplicate pairs under `80` on the full test-style prediction report.

### Stage C: Recognizer V2

Goal:

- Reduce RegionCER from current quick-valid `~0.218` toward `0.10-0.14`.
- Reduce PageCER toward `0.08-0.10`.

Recommended GPU:

- Best: H100 80GB, because competition final environment is 1x H100.
- Good: L40 48GB for LoRA/BF16 experiments.
- Avoid relying on T4 for final recognizer training.

Training approach:

Important guard before any Recognizer V2 run:

- `scripts/02_train_recognizer.py` must not silently initialize a fresh LoRA if the goal is to continue from the current `best_checkpoint`.
- The script now supports `INIT_LORA_DIR=/path/to/qwen3_lora/best_checkpoint`.
- If `INIT_LORA_DIR` is set, it loads the old adapter with `PeftModel.from_pretrained(..., is_trainable=True)`.
- If `INIT_LORA_DIR` is not set, it keeps the old behavior and creates a fresh LoRA adapter.
- For continued training from the `0.61` baseline adapter, use a lower LR such as `LR=1e-4` or `LR=5e-5`.

Example:

```bash
export INIT_LORA_DIR=/kaggle/input/datasets/ngovietan/htr-output-kaggle-full-ver1-061/rukopys/local_working/htr_artifacts/qwen3_lora/best_checkpoint
export LR=1e-4
python scripts/02_train_recognizer.py
```

1. Gold-only warm start:
   - 2-3 epochs.
   - BF16 if available.
   - LoRA rank 64 or 128.
   - `batch_size=1-2`, `grad_accum=16-32`.
   - `gradient_checkpointing=True`.

2. Silver curriculum:
   - Add silver gradually.
   - Start with high-quality sampled silver subset.
   - Cap by time, for example `MAX_SILVER=2000`, then scale.

3. Synthetic/HKR:
   - Use only if validation CER improves.
   - Do not mix everything blindly.
   - Keep source/type metrics to prevent synthetic overfit.

4. OCR prompt/generation controls:
   - `do_sample=False`.
   - lower `max_new_tokens`, e.g. `96` first, `128` only for long boxes.
   - add `no_repeat_ngram_size` if supported by the generation path.
   - add mild `repetition_penalty`, e.g. `1.03-1.08`, and validate.
   - keep formula prompts separate if code supports type-aware prompts.

5. Pixel sizing:
   - Use `MAX_PIXELS=512*28*28` on L40/H100 if memory allows.
   - Keep `384*28*28` on T4.

Checkpointing:

```bash
export SAVE_STEPS=100
export EVAL_STEPS=100
export SAVE_LIMIT=3
```

Also save:

- `trainer_state.json`
- `adapter_model.safetensors`
- tokenizer/processor files
- config and env snapshot

### Stage D: Inference V2

Goal:

- Full 385-image inference under `11h30m`.
- Keep output valid even on timeout.

Changes:

- Resume-aware by default.
- Save every image.
- Add `TIME_BUDGET_HOURS=11.0`.
- Add sharding:
  - `SHARD_ID`
  - `NUM_SHARDS`
  - deterministic merge at the end.
- Cache detections separately so OCR failures do not require rerunning YOLO.
- Cache OCR outputs by image/region hash if possible.
- Use source-aware ordering if validation shows source-specific PageCER failures.

Speed targets:

- T4 full inference from scratch is risky.
- L40/H100 or sharded Kaggle runs should target `60-80 sec/image` or better.

How to reach it:

- Run on L40/H100.
- Increase OCR batch only after memory is stable.
- Avoid Phase3 in the same inference run unless necessary.
- Do not download/cache base model into `/kaggle/working`.
- Keep generation tokens controlled.

## Recommended Next Run Sequence

### Run 1: Full Diagnostic Validation

Use L40/H100 if possible; T4 only if time is acceptable.

Purpose:

- Re-score current detector/recognizer on `50-143` validation pages.
- Produce per-source/per-type metrics.
- Confirm whether the dominant loss is detector recall, classification, RegionCER, or PageCER.

Expected runtime:

- T4: can be slow; use `N=50`.
- L40/H100: use full `N=143` if possible.

Deliverables:

- `grid_search_results_full.csv`
- `valid_pred_current.csv`
- per-page error table sorted by PageCER
- duplicate-overlap report
- long/repeated OCR report

### Run 2: Postprocessing V1 Without Retraining

Purpose:

- Improve current model score using cheap changes before expensive training.

Implement and validate:

- class-aware duplicate suppression for IoU >= `0.80`
- formula/handwritten overlap policy
- repeated phrase cleanup
- repeated dash cleanup
- max text length clamp or repetition detector
- source/type-aware generation `max_new_tokens`

Already available locally:

```bash
python3 scripts/05_diagnostic_postprocess_v2.py
```

This writes:

```text
analysis_outputs/postprocess_v2/submission_postprocess_v2.csv
analysis_outputs/postprocess_v2/diagnostic_postprocess_v2_report.md
analysis_outputs/postprocess_v2/diagnostic_postprocess_v2_report.json
```

Success target:

- Reduce overlap pairs from `201` to under `80`.
- Reduce texts >=180 chars from `75` to under `25`.
- Improve validation PageCER without hurting detection too much.

### Run 3: Detector V2

Use L40/H100.

Purpose:

- Improve bbox/class quality and reduce ambiguous duplicate regions.

Training recipe:

- larger YOLO model if available
- `imgsz=1280`
- rare-class oversampling
- `cls=1.0-1.5`
- selected silver for detector
- validate using official metric, not only mAP

Expected runtime:

- L40: likely `4-8h`
- H100: safer for iteration

### Run 4: Recognizer V2

Use H100 if possible; L40 minimum for serious run.

Purpose:

- Move RegionCER from current quick-valid `~0.218` toward `0.10-0.14`.
- Move PageCER toward `0.08-0.10`.

Training recipe:

- gold-only warm start, 2-3 epochs
- LoRA rank 64/128
- BF16 on L40/H100
- selected silver curriculum after gold stabilizes
- checkpoint every 100 steps
- keep only top 2-3 checkpoints plus best adapter
- evaluate by source/type every checkpoint if possible

### Run 5: Final Inference V2

Use H100/L40 or Kaggle shards.

Purpose:

- Full 385-image inference with improved detector/recognizer/postprocessing.
- Verify success gates before submit.

## Concrete Success Gates

Do not rely on leaderboard alone. Check these before each submit:

```text
submission rows == 385
progress rows == 385
missing images == 0
invalid JSON rows == 0
empty rows == 0
processed images == 385
phase4 complete == true
overlap duplicate count < 80, current is 201
texts >=180 chars < 25, current is 75
no extreme repeated dash/text examples
```

Validation targets for a serious 0.9 attempt:

```text
Phase3 N >= 50, ideally 143
DetF1 >= 0.97
ClassAcc >= 0.90
RegionCER <= 0.12-0.14
PageCER <= 0.08-0.10
```

## Most Important Takeaway

The baseline is now fully operational and complete, but a full submission only reached about `0.61`. That means the next phase is quality optimization, not Kaggle plumbing.

The best path to `0.9` is:

1. Build a reliable full-validation diagnostic loop.
2. Fix duplicate overlaps and OCR repetition without retraining.
3. Train Detector V2 with rare-class and duplicate-awareness.
4. Train Recognizer V2 on L40/H100 with gold plus selected silver curriculum.
5. Run final inference with shard/resume/checkpoint behavior by default.

Do not spend paid GPU hours until validation can explain exactly which component is holding the score down.
