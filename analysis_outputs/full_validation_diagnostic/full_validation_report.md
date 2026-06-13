# Full Validation Diagnostic

Completed pages: `{'detector': 143, 'gt_ocr': 143, 'e2e': 143} / 143`

## detector

| Metric | Value |
|---|---:|
| Completed pages | 143 |
| Detector F1 | 0.912793 |
| Detector precision | 0.884287 |
| Detector recall | 0.943198 |
| Class accuracy | 0.966914 |
| IoU >= 0.80 overlap pairs | 93 |

### By type

| Type | GT | Pred | Recall | Class acc | FN | FP |
|---|---:|---:|---:|---:|---:|---:|
| annotation | 66 | 63 | 0.7576 | 0.8000 | 16 | 18 |
| formula | 376 | 492 | 0.9441 | 0.9746 | 21 | 83 |
| graph | 11 | 8 | 0.4545 | 1.0000 | 6 | 3 |
| handwritten | 2323 | 2400 | 0.9582 | 0.9704 | 97 | 220 |
| image | 10 | 9 | 0.6000 | 0.8333 | 4 | 4 |
| printed | 53 | 59 | 0.7170 | 0.9211 | 15 | 24 |
| table | 13 | 11 | 0.7692 | 1.0000 | 3 | 0 |

## gt_ocr

| Metric | Value |
|---|---:|
| Composite | 0.757555 |
| Detector F1 | 1.000000 |
| Detector precision | 1.000000 |
| Detector recall | 1.000000 |
| Class accuracy | 1.000000 |
| Region CER | 0.331693 |
| Page CER | 0.285874 |

### By type

| Type | GT | Pred | Recall | Class acc | Region CER | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| annotation | 66 | 66 | 1.0000 | 1.0000 | 0.5443 | 0 | 0 |
| formula | 376 | 376 | 1.0000 | 1.0000 | 0.2691 | 0 | 0 |
| graph | 11 | 11 | 1.0000 | 1.0000 | n/a | 0 | 0 |
| handwritten | 2323 | 2323 | 1.0000 | 1.0000 | 0.3394 | 0 | 0 |
| image | 10 | 10 | 1.0000 | 1.0000 | n/a | 0 | 0 |
| printed | 53 | 53 | 1.0000 | 1.0000 | 0.2037 | 0 | 0 |
| table | 13 | 13 | 1.0000 | 1.0000 | 0.2454 | 0 | 0 |

### By source

| Source | Composite | DetF1 | Class acc | Region CER | Page CER |
|---|---:|---:|---:|---:|---:|
| archive | 0.7087 | 1.0000 | 1.0000 | 0.3901 | 0.3486 |
| dictation | 0.7631 | 1.0000 | 1.0000 | 0.3293 | 0.2761 |
| school | 0.7788 | 1.0000 | 1.0000 | 0.3038 | 0.2602 |
| university | 0.7320 | 1.0000 | 1.0000 | 0.3474 | 0.3275 |

### Worst pages

- `images/3dd87640-0760-48cd-900c-a10d930b497b.jpg` source=dictation PageCER=0.9388, regions=37/37
- `images/19e05d79-f994-4046-927b-ea43cbab55dd.jpg` source=dictation PageCER=0.8665, regions=26/26
- `images/40cbd188-3b8c-4adb-afd0-cfce1bf2b88f.jpg` source=dictation PageCER=0.8562, regions=33/33
- `images/6f63c9f2-c961-5562-af68-5a96bd8614d8.jpg` source=school PageCER=0.7657, regions=24/24
- `images/18af2afd-4107-4598-b314-93c6af313422.jpg` source=dictation PageCER=0.7617, regions=37/37
- `images/ec9dd609-3242-59f9-9403-0dcc1d1eea4c.jpg` source=school PageCER=0.7581, regions=21/21
- `images/22bc6edb-5bc7-4b39-a700-1f5e906ae7fd.jpg` source=university PageCER=0.6462, regions=17/17
- `images/3279216f-a680-446e-9e13-14cd3618f718.jpg` source=university PageCER=0.6365, regions=35/35
- `images/dbd41bf6-ac5e-5382-a0f9-07c323f10d13.jpg` source=school PageCER=0.6161, regions=24/24
- `images/5bbc8721-d32e-41ac-b1f6-5d452b3c3698.jpg` source=dictation PageCER=0.5744, regions=38/38
- `images/5656dd23-484d-5120-ae78-e4b59997b416.jpg` source=school PageCER=0.5674, regions=25/25
- `images/d0273ef9-34ef-502b-b2b8-4dcc9992ca5e.jpg` source=school PageCER=0.5411, regions=18/18
- `images/410d65cc-de2e-5c29-acf8-015f0f2a57a4.jpg` source=school PageCER=0.5238, regions=3/3
- `images/1063ccb6-122f-4f58-8ca4-34898a45d2dd.jpg` source=archive PageCER=0.5072, regions=24/24
- `images/fe9ae2cf-47a3-5509-9d0e-68e518df903d.jpg` source=school PageCER=0.4944, regions=10/10
- `images/76f39d4a-393a-5ad8-9435-8547c9072f7f.jpg` source=school PageCER=0.4859, regions=22/22
- `images/eb2163e8-34c6-4ae0-bcef-1bf937225449.jpg` source=archive PageCER=0.4814, regions=27/27
- `images/4147d461-2a04-4ed2-864c-10c9dc7c7092.jpg` source=archive PageCER=0.4761, regions=22/22
- `images/3440f410-c95a-4624-90f0-d9c1a96cfaec.jpg` source=university PageCER=0.4612, regions=29/29
- `images/3d0acb86-1ebf-5ab8-882c-b9ce02d0f95b.jpg` source=school PageCER=0.4599, regions=16/16

## e2e

| Metric | Value |
|---|---:|
| Composite | 0.709221 |
| Detector F1 | 0.912793 |
| Detector precision | 0.884287 |
| Detector recall | 0.943198 |
| Class accuracy | 0.966914 |
| Region CER | 0.323964 |
| Page CER | 0.357710 |

### By type

| Type | GT | Pred | Recall | Class acc | Region CER | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| annotation | 66 | 63 | 0.7576 | 0.8000 | 0.5127 | 16 | 18 |
| formula | 376 | 492 | 0.9441 | 0.9746 | 0.2255 | 21 | 83 |
| graph | 11 | 8 | 0.4545 | 1.0000 | n/a | 6 | 3 |
| handwritten | 2323 | 2400 | 0.9582 | 0.9704 | 0.3333 | 97 | 220 |
| image | 10 | 9 | 0.6000 | 0.8333 | n/a | 4 | 4 |
| printed | 53 | 59 | 0.7170 | 0.9211 | 0.4929 | 15 | 24 |
| table | 13 | 11 | 0.7692 | 1.0000 | 0.2067 | 3 | 0 |

### By source

| Source | Composite | DetF1 | Class acc | Region CER | Page CER |
|---|---:|---:|---:|---:|---:|
| archive | 0.6769 | 0.9498 | 0.9877 | 0.3807 | 0.4016 |
| dictation | 0.7644 | 0.9593 | 0.9886 | 0.3191 | 0.2664 |
| school | 0.7144 | 0.8878 | 0.9618 | 0.3066 | 0.3497 |
| university | 0.6294 | 0.8661 | 0.9237 | 0.3211 | 0.5007 |

### Worst pages

- `images/195812b7-a9bc-4cb1-b688-4d13261acc7a.jpg` source=university PageCER=1.2405, regions=36/20
- `images/3dd87640-0760-48cd-900c-a10d930b497b.jpg` source=dictation PageCER=1.2226, regions=32/37
- `images/1f58c72c-4616-4719-a4d3-bf1c6140eb08.jpg` source=archive PageCER=1.0982, regions=9/5
- `images/a148fed5-320e-59bc-a30d-12324def3390.jpg` source=school PageCER=0.9756, regions=27/27
- `images/5bbc8721-d32e-41ac-b1f6-5d452b3c3698.jpg` source=dictation PageCER=0.8665, regions=34/38
- `images/5ee2046f-1a0d-58b2-8997-ef1e17ba3f14.jpg` source=school PageCER=0.8268, regions=20/8
- `images/fe9ae2cf-47a3-5509-9d0e-68e518df903d.jpg` source=school PageCER=0.7978, regions=17/10
- `images/a111d6d7-056b-5bd7-b359-c457ae1e4f7b.jpg` source=school PageCER=0.7970, regions=23/15
- `images/e1808133-425d-50ee-8461-fb18ba92a47d.jpg` source=school PageCER=0.7675, regions=44/27
- `images/8a8a6dd9-1040-5b11-9c3c-848571dfc1cd.jpg` source=school PageCER=0.7391, regions=3/3
- `images/6f63c9f2-c961-5562-af68-5a96bd8614d8.jpg` source=school PageCER=0.7104, regions=29/24
- `images/26e0c337-b806-4a49-b9c2-456fb67708aa.jpg` source=university PageCER=0.6862, regions=17/13
- `images/3f4abc35-7daa-53b2-8bc8-9f57bdcd390e.jpg` source=school PageCER=0.6296, regions=28/1
- `images/7bd45413-5209-4283-9bfe-119eadcf9f48.jpg` source=university PageCER=0.6186, regions=30/22
- `images/276f9e4a-cd02-5eac-b578-b8eeae89d4c9.jpg` source=school PageCER=0.6162, regions=37/30
- `images/22bc6edb-5bc7-4b39-a700-1f5e906ae7fd.jpg` source=university PageCER=0.6062, regions=22/17
- `images/c30c3eba-5005-4a9b-9ebc-3c18d1d601d3.jpg` source=university PageCER=0.6029, regions=31/30
- `images/3440f410-c95a-4624-90f0-d9c1a96cfaec.jpg` source=university PageCER=0.5822, regions=35/29
- `images/dbd41bf6-ac5e-5382-a0f9-07c323f10d13.jpg` source=school PageCER=0.5806, regions=23/24
- `images/c7f116b6-f582-5d98-ba13-f23590079de8.jpg` source=school PageCER=0.5671, regions=13/12
