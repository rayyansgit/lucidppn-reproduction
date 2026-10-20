# LucidPPN Reproduction

Team: Rayyan Saeed (31629), Laksh Kumar (30598), Ahmad Mustafa Khan (30496)

Paper: Pach, Lewandowska, Tabor, Zieliński, Rymarczyk. *LucidPPN: Unambiguous Prototypical Parts Network for
User-centric Interpretable Computer Vision.* ICLR 2025.
[Paper](https://openreview.net/forum?id=BM9qfolt6p) | [Official code](https://github.com/mateuszpach/LucidPPN)

> **Branches:** `main` holds Milestone 2 (reproduction). This branch, **`m3&m4`**, holds Milestones 3-4: our
> experiment (colour-branch ablation and follow-up study, `EXPERIMENTS.md`, `experiments/`) and the deployment
> (`deploy/`). Final report: [`REPORT.pdf`](REPORT.pdf).

## Run the demo
Needs Python 3.10-3.13. No GPU, dataset or Colab: the trained model and example images are in the repository.
- **Windows:** double-click `run_demo.bat` (in PowerShell: `.\run_demo.bat`)
- **macOS / Linux:** `bash run_demo.sh`

The first run installs dependencies into `.venv` (5-10 minutes); the app then opens at http://localhost:8501.
App, REST API and worked example: [`deploy/README.md`](deploy/README.md).

## What LucidPPN does
A prototypical-parts network classifies an image by matching its parts to parts of training images ("this looks like
that"). LucidPPN makes these explanations unambiguous in two ways: a **shape branch** sees only a grayscale copy of the
image and a separate **colour branch** sees only pixel colours, so every part's evidence splits into shape and colour;
and part maps from PDiscoNet tie prototype *k* to the same object part in every class.

## What we did
We ran the full three-stage pipeline (PDiscoNet part discovery, then the two-branch MetiNet, then evaluation) on the
first 25 CUB-200-2011 species (750 train / 652 test images), with the authors' model and loss code unchanged and our own
data preparation, training and evaluation drivers. Every file is tagged in [`PROVENANCE.md`](PROVENANCE.md); every run
is logged in [`EXPERIMENTS.md`](EXPERIMENTS.md) and [`experiments/`](experiments/README.md).

| | Paper | Ours |
|---|---|---|
| Classes | 200 | 25 |
| Shape backbone | ConvNeXt-Tiny, 28 x 28 feature maps | ResNet18, also 28 x 28 (the authors' ResNet code uses stride 1 in its last two stages) |
| Parts K | 12 (best model) | 8 |
| PDiscoNet (authors' scripts) | 448 px, 28 epochs | 224 px, 5 epochs (28 in the follow-up study) |

## Results
All numbers are test top-1 at the final epoch on 652 images; mean ± std over runs. Random chance is 4%.

**Reproduction.** Our Jump 4 configuration reaches **83.1 ± 1.0%** over 4 runs (81.9, 84.0, 83.7, 82.8). The paper
reports 81.5 ± 0.4% on 200 classes. The numbers look alike but are not comparable: 25 classes is a much easier task,
and our backbone is weaker.

![Training curves](analysis/figures/training_curves.png)

**Our experiment: ablation of the colour branch.** The shape branch never receives gradients from the colour branch
(stop-gradient in the fusion), so the shape-only model is obtained without retraining. Over 12 trained models:

| Test images | Colour gain, original colours | Full model, hue shift up to 0.5 | Shape only, hue shift up to 0.5 |
|---|---|---|---|
| All 25 species | +1.4 (positive in 12 of 12 models; paper +1.4) | −4.1 | 0.0 |
| 12 most colourful species | +1.7 | −7.5 | 0.0 |
| 12 least colourful species | +1.3 | −0.9 | 0.0 |
| Paper, 200 species | +1.4 | −10.2 | 0.0 |

The colour branch adds a small, consistent gain and becomes a liability when colours are misleading, almost entirely
on colourful species. Our first 25 species are mostly grey, brown and black, which is consistent with our smaller
average cost than the paper's.

**Follow-up study (3 seeds per setting).**
- The paper's training recipe (from the authors' `run_training.sh`) gives 81.0-81.4%, about 2 points below our Jump 4
  recipe: on 750 images its 40 epochs make 4x fewer weight updates.
- A part detector trained for the paper's 28 epochs finds clearly better parts (NMI 24.1 → 31.0) but leaves accuracy
  unchanged (83.1 vs 83.0). It does place the prototypes more consistently on the same body parts (keypoint NMI 17.6 →
  20.8 and 18.1 → 22.6). This keypoint score is our adaptation of PDiscoNet's metric and is not reported in the
  LucidPPN paper.
- Several prototypes often concentrate on the head, the most distinctive region; the demo flags parts that share a
  spot.

## Not done
- Full 200-class training, the ConvNeXt-Tiny backbone, K = 12 parts, PDiscoNet at 448 px
- The paper's user studies, the other three datasets, the PIP-Net comparison
- A validation split: we report final epochs only, never a test-selected best epoch

## Reproduce
1. `pip install -r requirements_pip.txt` (Colab has torch preinstalled)
2. `python prepare_cub_subset.py` (expects `CUB_200_2011/` in the repository root)
3. `cd part_detection && python train_pdisco_quick.py && python generate_maps_quick.py`
4. `cd MetiNet && python forward_pass_test.py` (sanity check)
5. `cd MetiNet && python train_quick.py` (defaults = Jump 4; add `--num_workers 0` on Windows)
6. `cd MetiNet && python eval_ablation.py` and `python eval_full.py` (experiment and full evaluation)
7. `streamlit run deploy/app.py` or `uvicorn deploy.api:app` (deployment)

Steps 2-7 also run on Colab with [`notebooks/stage4_stage5_colab.ipynb`](notebooks/stage4_stage5_colab.ipynb).
