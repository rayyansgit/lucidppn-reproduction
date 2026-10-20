# Multi-seed summary: training recipe x part detector

| Test top-1 (mean ± std) | 5-epoch part detector | 28-epoch part detector |
|---|---|---|
| Code defaults (Jump 4) | 83.1 ± 1.0 (4 runs) | 83.0 ± 0.7 (3 runs) |
| Paper recipe | 81.0 ± 1.7 (3 runs) | 81.4 ± 2.0 (3 runs) |

| Configuration | Runs | LucidPPN test top-1 (final epoch) | ShapeTexNet only | Colour gain (clean) | Change under hue shift 0.5 | Notes |
|---|---|---|---|---|---|---|
| Jump 4: code defaults, 5-ep parts | 4 | 83.1 ± 1.0 | 82.1 ± 0.9 | 1.5 ± 0.4 | -3.3 ± 0.7 | original seed 1 (accuracy only), rerun seed 1, seed2, seed3 |
| Jump 7: code defaults, 28-ep parts | 3 | 83.0 ± 0.7 | 81.5 ± 0.6 | 1.5 ± 0.8 | -4.4 ± 0.5 | seed1, seed2, seed3 |
| Jump 5: paper recipe, 5-ep parts | 3 | 81.0 ± 1.7 | 79.6 ± 0.9 | 1.4 ± 0.9 | -4.0 ± 3.7 | seed1, seed2, seed3 |
| Jump 6: paper recipe, 28-ep parts | 3 | 81.4 ± 2.0 | 80.4 ± 1.9 | 1.1 ± 0.4 | -4.5 ± 2.6 | seed1, seed2, seed3 |

All numbers are % on the 652 test images of the 25-class subset; mean ± sample std over runs. Colour gain = LucidPPN − ShapeTexNet on clean images; hue-shift change = LucidPPN accuracy at max hue shift 0.5 (mean of 5 perturbation seeds) − clean accuracy.

![Test accuracy per epoch](summary.png)
