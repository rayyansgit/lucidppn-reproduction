# LucidPPN Reproduction — Milestone 2 (Data Pipeline + Forward Pass)

Team: Rayyan Saeed (31629), Laksh Kumar (30598), Ahmad Mustafa Khan (30496)

## Paper
LucidPPN: Unambiguous Prototypical Parts Network for User-centric Interpretable Computer Vision.
Pach, Lewandowska, Tabor, Zieliński, Rymarczyk. ICLR 2025.
Paper: https://openreview.net/forum?id=BM9qfolt6p | Code: https://github.com/mateuszpach/LucidPPN

## What the model does
LucidPPN splits each input image into a grayscale version (shape/texture) and a color version, learns separate visual "prototypes" for each, and combines both to classify the image — while keeping color-based evidence separate from shape/texture-based evidence, so predictions are easier to explain (e.g. "this matched on belly color" vs "this matched on wing shape").

## What we did for this milestone
1. Downloaded CUB-200-2011 (11,788 images, 200 bird species) from Caltech.
2. Built a data pipeline (`prepare_cub_subset.py`) that selects a subset of 8 species, crops each image to its bounding box, and splits into train/test — producing 424 images in a folder structure usable by the authors' data loader.
3. Loaded the official MetiNet model code (`get_network`, `MetiNet`) from the authors' repository and ran one real forward pass on a batch from our data (`forward_pass_test.py`).

## Result
Forward pass succeeded on a batch of 8 images (8 classes, ResNet18 backbone):
- Input shape: `[8, 3, 224, 224]`
- Output logits shape: `[8, 8]`
- No NaNs in output

Full log: `MetiNet/forward_pass_log.txt`

## What's NOT done yet (planned for Week 4)
- **Part-segmentation stage**: the official pipeline first trains a separate part-detection model (`part_detection/run_training.sh`, ~28 epochs) to generate body-part masks, which MetiNet normally uses as an additional input. We ran MetiNet in the code's built-in "dummy" mode (no part maps) to prove the core model runs — this is a real, documented simplification, not the full pipeline.
- Full training (60+ epochs per the authors' defaults) and accuracy comparison against the paper.
- Our planned ablation experiment (removing the color branch).

## Provenance
| Component | Source |
|---|---|
| MetiNet model code (`metinet/metinet.py`), data loading (`util/data.py`) | Reused as-is from https://github.com/mateuszpach/LucidPPN |
| `prepare_cub_subset.py` | Written by us |
| `forward_pass_test.py` | Written by us (adapted the forward-pass logic embedded in the authors' `main.py` into a standalone script, bypassing hardcoded cluster paths and the full training loop) |

## How to reproduce
1. `pip install -r requirements_pip.txt` (or use Colab, which has torch/torchvision preinstalled)
2. `python prepare_cub_subset.py` (downloads/expects `CUB_200_2011/` in the working directory — see script)
3. `cd MetiNet && python forward_pass_test.py`
