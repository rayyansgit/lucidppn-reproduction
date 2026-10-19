# LucidPPN Reproduction

Team: Rayyan Saeed (31629), Laksh Kumar (30598), Ahmad Mustafa Khan (30496)

## Run the demo
Needs Python 3.10-3.13. No GPU, dataset or Colab is required: the trained model and example images are in the repo.
- **Windows:** double-click `run_demo.bat`
- **macOS / Linux:** `bash run_demo.sh`

The first run installs its dependencies into `.venv` (about 5-10 minutes). The app then opens at
http://localhost:8501. Details, the REST API and screenshots: [`deploy/README.md`](deploy/README.md).

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

## Results — 81.9% test top-1 accuracy (final epoch of Jump 4)
We ran four labeled experiments, each building on the last. Full details, per-epoch analysis, and honest caveats for every run: **[`EXPERIMENTS.md`](EXPERIMENTS.md)**.

| Jump | What changed | Test top-1 |
|---|---|---|
| 1 | Forward pass only, untrained | n/a (sanity check) |
| 2 | First real training, no part-detection maps | 33.9-34.2% |
| 3 | Added real part-detection maps (full 3-stage pipeline) | 41.6% |
| 4 | Longer training (82 epochs) + delayed backbone unfreeze + larger batch | **81.9%** |

Random-chance baseline for 25 classes: 4.0%.

We report the **final-epoch** accuracy only. The highest single epoch (83.6%, epoch 68) was picked by looking at
test accuracy, which is selection on the test set, so it is not a fair estimate.

![Training curves](analysis/figures/training_curves.png)

**Key findings** (see `EXPERIMENTS.md` for full detail):
- Part-detection supervision (Jump 3) reduced overfitting, not just raw accuracy: the train-test gap roughly halved.
- In every run, test accuracy stays at chance (~4-5%) until the backbone is unfrozen (dashed lines in the figure).
  With ResNet18 the warm-up phase trains only the 1×1 prototype layer: the upstream code unfreezes a block called
  `layer4.2`, and ResNet18 has no such block. So Jump 4's gain comes from training **72 unfrozen epochs instead of 8**,
  not from delaying the unfreeze. Those 10 frozen epochs were wasted.
- Accuracy plateaus from about epoch 40, while train accuracy approaches 99% and the train-test gap widens again.

## Honest comparison to the paper
| | Paper (Table 1) | Ours |
|---|---|---|
| Classes | 200 | 25 (first 25 CUB species) |
| Test top-1 | **81.5 ± 0.4%** (3 runs) | **81.9%** (1 run) |
| ShapeTexNet backbone | ConvNeXt-Tiny, 28×28 feature maps | ResNet18, 7×7 feature maps |
| Object parts K | 12 (best model) | 8 |
| PDiscoNet | 448 px, 28 epochs | 224 px, 5 epochs |
| Schedule | 40 epochs, batch 64, lr 0.002 → 0.0002, 15 frozen | 82 epochs, batch 32, lr 0.0005 → 0.00005, 10 frozen |

The numbers look the same, but they don't measure the same thing. Telling 25 species apart is much easier than
telling 200 apart, and our model is smaller with coarser feature maps. So our 81.9% shows that the pipeline and method
work end to end at a meaningful scale. It is not evidence that we matched the paper. On 200 classes we would expect a
clearly lower number with this setup.

## Provenance
Full log: [`PROVENANCE.md`](PROVENANCE.md). Summary:

| Component | Source |
|---|---|
| PDiscoNet model/training code (`part_detection/nets.py`, `train.py`, `datasets.py`) | Reused as-is from https://github.com/mateuszpach/LucidPPN |
| MetiNet model code (`metinet/metinet.py`), training loop (`metinet/train.py`), data loading (`util/data.py`) | Reused as-is from https://github.com/mateuszpach/LucidPPN |
| `prepare_cub_subset.py` | Written by us |
| `forward_pass_test.py` | Written by us (adapted from logic in the authors' `MetiNet/main.py`) |
| `part_detection/train_pdisco_quick.py`, `generate_maps_quick.py` | Adapted from the authors' `part_detection/main.py` and `save_maps.py` — reuse their real training/inference functions; our own driver code for class-filtering and reduced epoch counts |
| `MetiNet/train_quick.py` | Adapted from the authors' `MetiNet/main.py` — reuses their real `train_metinet` function and loss; our own driver (optimizer/scheduler setup, epoch loop, hyperparameter sweeps for Jumps 2-4). Evaluation loop rewritten to skip part-segmentation IoU metrics (crash without full-dataset maps) while reusing their `topk_accuracy` function. |

## Stage 4 — our experiment: colour-branch ablation
**Question:** how much of our LucidPPN's accuracy comes from the ColorNet branch, and how much robustness to colour
changes does that cost? `MetiNet/eval_ablation.py` scores the Jump 4 checkpoint as the full model and as the
grayscale ShapeTexNet branch alone, on clean and hue-perturbed test images (paper Tables 2-3). Results go to
`MetiNet/results/`. Run it with `notebooks/stage4_stage5_colab.ipynb`. Details are in `EXPERIMENTS.md`.

## Stage 5 — deployment
A Streamlit web app and a FastAPI REST API. See [`deploy/README.md`](deploy/README.md).

## What's NOT done
- Full 200-class training (we used 25)
- Paper's actual backbone (ConvNeXt-Tiny), K=12 parts, and full PDiscoNet resolution (448)
- Multiple seeds (the paper reports mean ± std over 3 runs; we have 1 run)
- Part-alignment IoU metric (upstream `test.py` needs full-dataset maps)

## How to reproduce
1. `pip install -r requirements_pip.txt` (or use Colab, torch/torchvision preinstalled)
2. `python prepare_cub_subset.py` (expects `CUB_200_2011/` in the working directory)
3. `cd part_detection && python train_pdisco_quick.py`
4. `cd part_detection && python generate_maps_quick.py`
5. `cd MetiNet && python forward_pass_test.py` (sanity check)
6. `cd MetiNet && python train_quick.py` (defaults = Jump 4; e.g. `--epochs 8 --freeze_epochs 2 --batch_size 16` for Jump 3 — see `EXPERIMENTS.md`; add `--num_workers 0` on Windows)
7. `cd MetiNet && python eval_ablation.py` (Stage 4)
8. `streamlit run deploy/app.py` or `uvicorn deploy.api:app` (Stage 5)

Or run steps 2-8 on Colab with [`notebooks/stage4_stage5_colab.ipynb`](notebooks/stage4_stage5_colab.ipynb).
