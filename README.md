# LucidPPN Reproduction

Team: Rayyan Saeed (31629), Laksh Kumar (30598), Ahmad Mustafa Khan (30496)

## Paper
LucidPPN: Unambiguous Prototypical Parts Network for User-centric Interpretable Computer Vision.
Pach, Lewandowska, Tabor, Zieliński, Rymarczyk. ICLR 2025.
Paper: https://openreview.net/forum?id=BM9qfolt6p | Code: https://github.com/mateuszpach/LucidPPN

## What the model does
LucidPPN splits each input image into a grayscale version (shape/texture) and a color version, learns separate visual "prototypes" for each, and combines both to classify the image — while keeping color-based evidence separate from shape/texture-based evidence, so predictions are easier to explain (e.g. "this matched on belly color" vs "this matched on wing shape").

## The full pipeline, and what we ran
The method has three stages. We implemented and ran all three, at reduced scale (25 of 200 CUB classes, few epochs instead of the paper's 28/60+) to fit our compute budget:

1. **Part detection (PDiscoNet)** — a ResNet101-based model that learns to locate 8 semantic bird parts per image. We trained this for 5 epochs on our 25-class subset (`part_detection/train_pdisco_quick.py`), then generated part-location maps for every train/test image (`part_detection/generate_maps_quick.py`).
2. **MetiNet training** — the main LucidPPN model (grayscale + color branches, prototype learning). We trained for 10 epochs using the real part maps from stage 1 (`MetiNet/train_quick.py`).
3. **Evaluation** — classification accuracy on held-out test images, computed with the authors' own `topk_accuracy` function.

## Results

| Setup | Classes | Train imgs | Epochs | Train acc | Test top-1 acc |
|---|---|---|---|---|---|
| No part maps (part_weight=0) | 25 | 750 | 10 | 51.5-54.5% | 33.9-34.2% |
| **Full pipeline, with part maps** | 25 | 750 | 10 | 51.2% | **41.6%** |

Random-chance baseline for 25 classes: 4.0%.

**Key finding:** adding real part-detection supervision improved test accuracy by ~7-8 points (34%→41.6%) and roughly *halved the train-test gap* (~20pts→~10pts), while train accuracy stayed flat. This means the part maps aren't just adding signal — they're acting as a regularizer, pushing the model to learn prototypes tied to genuine bird anatomy rather than memorizing incidental patterns in the small training set. This matches the method's core motivation and gives us confidence the reproduction reflects the actual mechanism the paper describes, not just a black-box result.

Full logs: `MetiNet/training_log_25class_10epoch_WITH_parts.txt` (with maps), `MetiNet/training_log_25class_10epoch.txt` (without, for comparison), `part_detection/train_pdisco_log.txt`, `part_detection/generate_maps_log.txt`.

## Honest comparison to the paper
The paper reports CUB accuracy well above our 41.6% (prototype networks on full CUB typically reach 80%+). Attributable to:
- 25 classes / 750 train images vs. the paper's full 200-class, ~6,000-image split
- 10 epochs (MetiNet) / 5 epochs (PDiscoNet) vs. the authors' 60+ / 28
- ResNet18 (MetiNet) and reduced image size 224 (PDiscoNet, vs. their 448) — smaller/faster choices for speed, not the paper's ConvNeXt-Tiny/full-resolution setup

This is an expected outcome at this compute budget. The direction and structure of our result (part supervision measurably reduces overfitting) reproduces the qualitative mechanism of the method even though the absolute accuracy is far below paper-scale numbers.

## Provenance
| Component | Source |
|---|---|
| PDiscoNet model/training code (`part_detection/nets.py`, `train.py`, `datasets.py`) | Reused as-is from https://github.com/mateuszpach/LucidPPN |
| MetiNet model code (`metinet/metinet.py`), training loop (`metinet/train.py`), data loading (`util/data.py`) | Reused as-is from https://github.com/mateuszpach/LucidPPN |
| `prepare_cub_subset.py` | Written by us |
| `forward_pass_test.py` | Written by us (adapted from logic embedded in the authors' `MetiNet/main.py`) |
| `part_detection/train_pdisco_quick.py` | Adapted from the authors' `part_detection/main.py` — reuses their real `train`/`validation` functions; our own driver filters to 25 classes and cuts epochs 28→5 |
| `part_detection/generate_maps_quick.py` | Adapted from the authors' `part_detection/save_maps.py` — reuses their `CUBDataset` and `save_maps` as-is, pointed at our subset and checkpoint |
| `MetiNet/train_quick.py` | Adapted from the authors' `MetiNet/main.py` — reuses their real `train_metinet` function and loss; our own driver (optimizer/scheduler setup, epoch loop) since the original assumes wandb and hardcoded cluster paths. Evaluation rewritten to skip part-segmentation IoU metrics (which crash without full-dataset part maps) while reusing their `topk_accuracy` for classification accuracy. |

## What's NOT done
- Full 200-class training (we used 25)
- Full epoch counts (paper: 28 PDiscoNet / 60+ MetiNet, ours: 5 / 10)
- Paper's actual backbones (ConvNeXt-Tiny) and full image resolution (448)
- Our planned ablation experiment (removing the color branch) — next step

## How to reproduce
1. `pip install -r requirements_pip.txt` (or use Colab, torch/torchvision preinstalled)
2. `python prepare_cub_subset.py` (expects `CUB_200_2011/` in the working directory, downloads not included in this script — see script comments)
3. `cd part_detection && python train_pdisco_quick.py` (~3.5 min on T4)
4. `cd part_detection && python generate_maps_quick.py` (~1 min)
5. `cd MetiNet && python forward_pass_test.py` (sanity check)
6. `cd MetiNet && python train_quick.py` (~4 min, full pipeline with part maps)
