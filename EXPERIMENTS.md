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
**Result: 81.9% test top-1 (final epoch), 83.6% (best epoch, #68)** — +40pts vs Jump 3, now in the paper's reported accuracy range.

Notable pattern: accuracy climbed steadily through ~epoch 35-40, then plateaued/fluctuated in the 80-84% range for the remaining ~45 epochs. 
Train accuracy saturated near 98-99% by ~epoch 45. Train-test gap widened back to ~15-18pts (from Jump 3's ~10pts) — longer training recovered 
most of the overfitting reduction Jump 3 achieved via the part-loss regularization. Final-epoch checkpoint was not the best-epoch checkpoint (best: epoch 68 at 83.6%) — this run 
did not implement best-checkpoint saving, a known limitation.

Log: `MetiNet/training_log_JUMP4_25class_80epoch_batch32.txt`

## Summary table
| Jump | Part maps | Epochs | Freeze epochs | Batch | Test top-1 (final) | Best epoch |
|---|---|---|---|---|---|---|
| 2 | No | 10 | 2 | 16 | 33.9-34.2% | — |
| 3 | Yes | 10 | 2 | 16 | 41.6% | 10 |
| 4 | Yes | 82 | 10 | 32 | 81.9% | 68 (83.6%) |

## Methodology note
Jump 4 bundles three changes rather than isolating them (time-constrained). Given the plateau pattern observed, a natural, cheap follow-up would be re-running at ~40 epochs with 
best-checkpoint saving — likely reaching similar accuracy in roughly half the training time, and avoiding the late-training overfitting regrowth. This would be a reasonable candidate 
for the Week 4 planned hyperparameter-study experiment.
