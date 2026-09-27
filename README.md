# LucidPPN Reproduction

Team: Rayyan Saeed (31629), Laksh Kumar (30598), Ahmad Mustafa Khan (30496)

## Paper
LucidPPN: Unambiguous Prototypical Parts Network for User-centric Interpretable Computer Vision.
Pach, Lewandowska, Tabor, Zieliński, Rymarczyk. ICLR 2025.
Paper: https://openreview.net/forum?id=BM9qfolt6p | Code: https://github.com/mateuszpach/LucidPPN

## What the model does
LucidPPN splits each input image into a grayscale version (shape/texture) and a color version, learns separate visual "prototypes" for each, and combines both to classify the image — while keeping color-based evidence separate from shape/texture-based evidence, so predictions are easier to explain (e.g. "this matched on belly color" vs "this matched on wing shape").

## The full pipeline, and what we ran
The method has three stages. We implemented and ran all three, at reduced scale (25 of 200 CUB classes) to fit our compute budget:
1. **Part detection (PDiscoNet)** — locates 8 semantic bird parts per image. Trained 5 epochs on our subset, then generated part-location maps for all train/test images.
2. **MetiNet training** — the main LucidPPN model (grayscale + color branches, prototype learning), trained using the real part maps from stage 1.
3. **Evaluation** — classification accuracy on held-out test images.

## Results — best result: 81.9% test top-1 accuracy (83.6% best epoch)
We ran four labeled experiments, each building on the last. Full details, per-epoch analysis, and honest caveats for every run: **[`EXPERIMENTS.md`](EXPERIMENTS.md)**.

| Jump | What changed | Test top-1 |
|---|---|---|
| 1 | Forward pass only, untrained | n/a (sanity check) |
| 2 | First real training, no part-detection maps | 33.9-34.2% |
| 3 | Added real part-detection maps (full 3-stage pipeline) | 41.6% |
| 4 | Longer training (82 epochs) + delayed backbone unfreeze + larger batch | **81.9%** (final), 83.6% (best epoch) |

Random-chance baseline for 25 classes: 4.0%.

**Key findings** (see `EXPERIMENTS.md` for full detail): part-detection supervision (Jump 3) reduced overfitting, not just raw accuracy — the train-test gap roughly halved. Longer training (Jump 4) drove the largest accuracy gain but partially eroded that overfitting reduction, and accuracy plateaued after ~epoch 40 — the model likely converges in about half the epochs we used, a concrete lead for future efficiency tuning.

## Honest comparison to the paper
Our best result (81.9-83.6%) is within range of typical prototype-network CUB accuracy, achieved on 25/200 classes with reduced backbones (ResNet18/ResNet101 vs. the paper's ConvNeXt-Tiny) and reduced image resolution in the part-detection stage (224 vs. 448). We have not yet tested on the full 200-class dataset, so this is not a like-for-like comparison to the paper's headline number — it demonstrates the pipeline and method work correctly at meaningful scale, not a matched benchmark.

## Provenance
| Component | Source |
|---|---|
| PDiscoNet model/training code (`part_detection/nets.py`, `train.py`, `datasets.py`) | Reused as-is from https://github.com/mateuszpach/LucidPPN |
| MetiNet model code (`metinet/metinet.py`), training loop (`metinet/train.py`), data loading (`util/data.py`) | Reused as-is from https://github.com/mateuszpach/LucidPPN |
| `prepare_cub_subset.py` | Written by us |
| `forward_pass_test.py` | Written by us (adapted from logic in the authors' `MetiNet/main.py`) |
| `part_detection/train_pdisco_quick.py`, `generate_maps_quick.py` | Adapted from the authors' `part_detection/main.py` and `save_maps.py` — reuse their real training/inference functions; our own driver code for class-filtering and reduced epoch counts |
| `MetiNet/train_quick.py` | Adapted from the authors' `MetiNet/main.py` — reuses their real `train_metinet` function and loss; our own driver (optimizer/scheduler setup, epoch loop, hyperparameter sweeps for Jumps 2-4). Evaluation loop rewritten to skip part-segmentation IoU metrics (crash without full-dataset maps) while reusing their `topk_accuracy` function. |

## What's NOT done
- Full 200-class training (we used 25)
- Paper's actual backbone (ConvNeXt-Tiny) and full image resolution (448)
- Best-checkpoint saving (Jump 4's final epoch was not its best — known limitation, see `EXPERIMENTS.md`)
- Our planned ablation experiment (removing the color branch) — natural next step

## How to reproduce
1. `pip install -r requirements_pip.txt` (or use Colab, torch/torchvision preinstalled)
2. `python prepare_cub_subset.py` (expects `CUB_200_2011/` in the working directory)
3. `cd part_detection && python train_pdisco_quick.py`
4. `cd part_detection && python generate_maps_quick.py`
5. `cd MetiNet && python forward_pass_test.py` (sanity check)
6. `cd MetiNet && python train_quick.py` (edit epochs/freeze_epochs/batch_size at top of file to reproduce a specific Jump — see `EXPERIMENTS.md`)
