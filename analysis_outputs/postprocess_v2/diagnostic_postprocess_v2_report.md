# Diagnostic + Postprocess V2 Report

Input: `rukopys_full_063/input/datasets/ngovietan/htr-output-kaggle-full-ver1-061/submission.csv`
Output candidate: `analysis_outputs/postprocess_v2/submission_postprocess_v2.csv`

## Summary

- Regions: 8934 -> 8743 (-191)
- Images with IoU>=0.80 overlaps: 102 -> 9 (-93)
- IoU>=0.80 overlap pairs: 201 -> 10 (-191)
- Texts >=180 chars: 75 -> 46 (-29)
- Actions: `{'duplicates_removed': 191, 'text_cleaned': 46}`

## Top Overlap Type Pairs Before

- ['formula', 'handwritten']: 80
- ['handwritten', 'formula']: 71
- ['handwritten', 'annotation']: 8
- ['handwritten', 'printed']: 8
- ['printed', 'handwritten']: 8
- ['annotation', 'handwritten']: 8
- ['formula', 'printed']: 4
- ['annotation', 'image']: 3
- ['printed', 'formula']: 3
- ['printed', 'annotation']: 2
- ['formula', 'image']: 1
- ['annotation', 'printed']: 1

## Long OCR Examples Before

- `b95befcb-d245-4e81-afa6-6f3aedda8639.jpg` printed len=289: відповідно до встановлених норм та правил, відповідно до встановлених норм та правил, відповідно до встановлених норм та правил, відповідно до встановлених норм та правил, відповід
- `607c9477-058c-48ff-beca-330836b5ce60.jpg` handwritten len=281: так само як до різних мандрівок. Він відчуває, що він відчуває, що не зможе відчути нічого на іншій планеті, свої картини, свої картини, свої картини, свої картини, свої картини, с
- `607c9477-058c-48ff-beca-330836b5ce60.jpg` handwritten len=266: відповідно до встановлених нормативів та вимог до виробництва, а також відповідно до встановлених нормативів та вимог до виробництва, а також відповідно до встановлених нормативів 
- `c7f47b50-f77b-419f-9e95-029d0b5e106e.jpg` printed len=255: — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — 
- `c7f47b50-f77b-419f-9e95-029d0b5e106e.jpg` printed len=255: — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — 
- `66b9b5aa-1b36-4adc-9312-6c36328421c0.jpg` handwritten len=255: відповідно, якую вони встигли тільки відповідно до зазначених умов, але не відповідно до зазначених умов, що передбачають, зворотній відносини, з максимального відносини, зворотній
