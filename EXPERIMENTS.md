# Experiment Log — Labeled Improvement Jumps

All runs: 25-class CUB subset (750 train / 652 test images), MetiNet with ResNet18 backbone.
Random-chance baseline (25 classes): 4.0%.

## Jump 1 — Baseline: forward pass only, no training
No weights updated. Confirms architecture/data pipeline correctness only.
Log: `MetiNet/forward_pass_log.txt`

## Jump 2 — First real training, no part-detection maps
10 epochs, part_weight=0 (dummy mode), batch_size=16, freeze_epochs=2.
**Result: 33.9-34.2% test top-1**
Log: `MetiNet/training_log_25class_10epoch.txt`

## Jump 3 — Full 3-stage pipeline: added part-detection maps
PDiscoNet (5 epochs) generated real part maps; MetiNet retrained 10 epochs, part_weight=1.0, batch_size=16, freeze_epochs=2.
**Result: 41.6% test top-1** (+7-8pts vs Jump 2). Train-test gap ~10pts (vs ~20pts in Jump 2) — part supervision reduced overfitting.
Logs: `part_detection/train_pdisco_log.txt`, `part_detection/generate_maps_log.txt`, `MetiNet/training_log_25class_10epoch_WITH_parts.txt`

## Jump 4 — Longer training + delayed unfreeze + larger batch (bundled change)
Same part maps as Jump 3. Three changes applied together as one hypothesis, not isolated:
- Epochs: 10 -> 82
- Freeze_epochs: 2 -> 10
- Batch size: 16 -> 32
**Result: 81.9% test top-1 (final epoch)**, +40 points over Jump 3. (The highest single epoch, 83.6% at epoch 68, was
picked using test accuracy, so it is not reported as a result.)

Notable pattern: accuracy climbed steadily through ~epoch 35-40, then plateaued/fluctuated in the 80-84% range for the remaining ~45 epochs. 
Train accuracy saturated near 98-99% by ~epoch 45. Train-test gap widened back to ~15-18pts (from Jump 3's ~10pts) — longer training recovered 
most of the overfitting reduction Jump 3 achieved via the part-loss regularization. Final-epoch checkpoint was not the best-epoch checkpoint (best: epoch 68 at 83.6%) — this run 
did not implement best-checkpoint saving. We deliberately report the final epoch anyway: without a separate validation
split, choosing an epoch by its test accuracy would be selection on the test set.

**Correction to our own reading of Jump 4 (Stage 3 review).** In all three runs, test accuracy sits at chance
(Jump 2: 3.8%, Jump 3: 5.2%, Jump 4: 4.8% at the last frozen epoch) and only starts rising once the backbone is
unfrozen (`analysis/figures/training_curves.png`). The upstream optimiser setup keeps only blocks named `layer4.2`
trainable during warm-up, and ResNet18 has only `layer4.0` and `layer4.1`. So during warm-up only the 1×1 prototype
layer (and ColorNet from epoch 3) learns, and that is apparently not enough to move the prediction off chance. We have
not checked why. One plausible cause is the 25×8 one-hot BCE targets pushing every sigmoid score towards 0. The
updated `train_quick.py` now logs ShapeTexNet accuracy every epoch, which would show whether the shape branch learns
anything during warm-up. Effectively, Jump 4 is "72 unfrozen epochs at batch 32" versus Jump 3's "8 unfrozen epochs
at batch 16". The delayed unfreeze cost 10 epochs and contributed nothing. After 10 unfrozen epochs (epoch 20),
Jump 4 is at 58%, consistent with Jump 3's 41.6% after 8 unfrozen epochs at the smaller batch size.

Log: `MetiNet/training_log_JUMP4_25class_80epoch_batch32.txt`

## Summary table
| Jump | Part maps | Epochs | Freeze epochs | Batch | Test top-1 (final) | Best epoch |
|---|---|---|---|---|---|---|
| 2 | No | 10 | 2 | 16 | 33.9-34.2% | — |
| 3 | Yes | 10 | 2 | 16 | 41.6% | 10 |
| 4 | Yes | 82 | 10 | 32 | 81.9% | 68 (83.6%) |

## Stage 4 — Our experiment (type A: ablation of the colour branch)

**Question (one sentence):** how much of our LucidPPN's test accuracy comes from the ColorNet branch, and how much
robustness to colour changes does that cost?

**Why it matters:** the paper's main claim is that keeping colour separate from shape/texture makes explanations less
ambiguous, at a small cost in accuracy (Table 2: ShapeTexNet 80.4% vs LucidPPN 81.8% on CUB) and robustness (Table 3:
under hue perturbation LucidPPN drops 81.9 → 71.7% while ShapeTexNet stays at 80.3%). We test whether the same
trade-off holds in our reduced setting (25 classes, ResNet18, 7×7 maps).

**Design:**
- Model: the Jump 4 checkpoint, unchanged.
- Variable 1, branch: the full LucidPPN prediction vs the ShapeTexNet branch alone (mean over parts of the grayscale
  prototype scores, the paper's P_S).
- Variable 2, colour perturbation: maximum random hue rotation of 0 (clean), 0.1, 0.25 and 0.5. Luminance is
  preserved, using the authors' `random_hue_jitter`. Each perturbed level is repeated over 3 random seeds
  (mean ± std).
- Also measured: the share of test images where adding colour turns a wrong ShapeTexNet prediction right ("helped")
  or a right one wrong ("hurt").
- No retraining is needed for the "no colour branch" condition: ColorNet multiplies `proto_features.detach()`
  (`metinet.py`), so no gradient from the colour loss reaches ShapeTexNet. ShapeTexNet inside the trained model is the
  model we would get by training without ColorNet (up to randomness).
- Script: `MetiNet/eval_ablation.py`. Outputs: `MetiNet/results/ablation.{json,md}` and `ablation_hue.png`.

**Expected outcome if our model behaves like the paper's:** the colour branch adds a small amount of clean accuracy,
ShapeTexNet accuracy stays flat as the hue shift grows, and LucidPPN's accuracy drops with the hue shift.

**Results** (Jump 4 rerun checkpoint, see below; 652 test images; hue rows mean ± std over 3 seeds):

| Max hue shift | LucidPPN top-1 | ShapeTexNet top-1 | Colour helped | Colour hurt |
|---|---|---|---|---|
| 0.0 | 84.0 | 83.0 | 1.8 | 0.8 |
| 0.1 | 84.2 ± 0.1 | 83.0 | 1.8 | 0.7 ± 0.1 |
| 0.25 | 83.3 ± 0.5 | 83.0 | 1.7 ± 0.2 | 1.4 ± 0.6 |
| 0.5 | 80.4 ± 0.9 | 83.0 | 1.0 ± 0.4 | 3.6 ± 0.6 |

![Colour-branch ablation](MetiNet/results/ablation_hue.png)

**Answer:** colour adds about 1 point on clean images (+1.1; paper +1.4). This rests on 12 images fixed vs 5 broken,
McNemar p ≈ 0.14, so it is not significant on its own. Under strong hue changes LucidPPN loses 3.6 points
(paper −10.2) and drops below ShapeTexNet, which is exactly flat at every hue level as the design guarantees. Full
interpretation is in `REPORT.md` Section 8.

**Jump 4 rerun (reproducibility check).** The Stage 3 checkpoint was lost with the Colab session, so Jump 4 was
retrained with the unmodified scripts and settings. Final test top-1: **84.0%** (original 81.9%). The difference comes
from GPU non-determinism alone. Log: `MetiNet/training_log_JUMP4_rerun.txt`.

**Optional follow-up (B):** re-run the Jump 4 schedule with `--part_weight 0` to test whether part supervision still
matters with long training. Jumps 2 and 3 only compared these at 10 epochs.

## Methodology note
Jump 4 bundles three changes rather than isolating them (time-constrained). Given the plateau pattern observed, a natural, cheap follow-up would be re-running at ~40 epochs with 
best-checkpoint saving — likely reaching similar accuracy in roughly half the training time, and avoiding the late-training overfitting regrowth. This would be a reasonable candidate 
for the Week 4 planned hyperparameter-study experiment.
