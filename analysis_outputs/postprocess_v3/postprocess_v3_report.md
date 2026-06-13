# Postprocess V3 Candidate Report

Input: `rukopys_full_063/input/datasets/ngovietan/htr-output-kaggle-full-ver1-061/submission.csv`

All candidates start from the original full submission, not from V2.

## Candidate Summary

| Profile | Regions | IoU>=.80 pairs | IoA>=.95 pairs | Texts >=180 | GT regions removed |
|---|---:|---:|---:|---:|---:|
| `safe` | 8744 | 11 | 435 | 38 | 0 |
| `balanced` | 8689 | 11 | 374 | 38 | 2 |
| `text_strong` | 8689 | 11 | 374 | 34 | 2 |
| `dedup_strong` | 8587 | 11 | 277 | 38 | 6 |

## Profiles

- `safe`: corrected V2 behavior; does not clean formulas/tables.
- `balanced`: adds high-confidence containment and cross-class deduplication.
- `text_strong`: truncates stronger OCR generation loops; higher PageCER upside and risk.
- `dedup_strong`: removes more overlapping/contained regions; higher detection risk.

## Recommended Submission Order

1. `safe` - lowest measured GT risk and fixes the V2 formula/table cleanup bug.
2. `balanced` - removes additional high-confidence contained duplicates.
3. `text_strong` - tests whether remaining PageCER is dominated by generation loops.
4. `dedup_strong` - highest geometry cleanup upside but also the highest detection risk.

Do not apply blind spelling correction, hard text-length caps, or broad type
reclassification without validation predictions. Those operations can alter
legitimate Ukrainian, archive-language, formula, and table content.

## Actions

### `safe`

- Output: `analysis_outputs/postprocess_v3/submission_v3_safe.csv`
- Actions: `{'high_iou_duplicate': 190, 'token_loop_collapsed': 46, 'character_loop_collapsed': 3, 'unicode_sanitized': 387, 'dash_filler_collapsed': 2}`
- Train GT audit: `{'available': True, 'affected_pages': 144, 'regions_before': 25651, 'regions_after': 25651, 'unicode_sanitized': 336, 'token_loop_collapsed': 1}`

### `balanced`

- Output: `analysis_outputs/postprocess_v3/submission_v3_balanced.csv`
- Actions: `{'high_iou_duplicate': 190, 'token_loop_collapsed': 46, 'character_loop_collapsed': 3, 'contained_text_duplicate': 50, 'unicode_sanitized': 387, 'dash_filler_collapsed': 2, 'cross_class_duplicate': 5}`
- Train GT audit: `{'available': True, 'affected_pages': 146, 'regions_before': 25651, 'regions_after': 25649, 'unicode_sanitized': 336, 'token_loop_collapsed': 1, 'contained_text_duplicate': 2}`

### `text_strong`

- Output: `analysis_outputs/postprocess_v3/submission_v3_text_strong.csv`
- Actions: `{'high_iou_duplicate': 190, 'token_loop_collapsed': 50, 'character_loop_collapsed': 3, 'contained_text_duplicate': 50, 'unicode_sanitized': 387, 'dash_filler_collapsed': 2, 'cross_class_duplicate': 5}`
- Train GT audit: `{'available': True, 'affected_pages': 146, 'regions_before': 25651, 'regions_after': 25649, 'unicode_sanitized': 336, 'token_loop_collapsed': 1, 'contained_text_duplicate': 2}`

### `dedup_strong`

- Output: `analysis_outputs/postprocess_v3/submission_v3_dedup_strong.csv`
- Actions: `{'high_iou_duplicate': 189, 'token_loop_collapsed': 46, 'character_loop_collapsed': 3, 'contained_text_duplicate': 50, 'strong_containment_duplicate': 99, 'unicode_sanitized': 387, 'dash_filler_collapsed': 2, 'strong_cross_class_duplicate': 4, 'cross_class_duplicate': 5}`
- Train GT audit: `{'available': True, 'affected_pages': 149, 'regions_before': 25651, 'regions_after': 25645, 'unicode_sanitized': 336, 'token_loop_collapsed': 1, 'strong_containment_duplicate': 4, 'contained_text_duplicate': 2}`
