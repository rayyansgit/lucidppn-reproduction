# Provenance

Paper: Pach, Lewandowska, Tabor, Zieliński, Rymarczyk. *LucidPPN: Unambiguous Prototypical Parts Network for
User-centric Interpretable Computer Vision.* ICLR 2025. https://openreview.net/forum?id=BM9qfolt6p

Official code: https://github.com/mateuszpach/LucidPPN. This repository is a fork of it; the upstream history ends
at commit `1e230c7` ("Merge pull request #1"), and every commit after that is ours.

Tags: **reused as-is** = unchanged upstream code; **adapted** = upstream logic with real changes (different driver,
data, training loop or evaluation); **written by us** = new code.

## Code

| Component | File(s) | Status | Notes |
|---|---|---|---|
| PDiscoNet model, losses, training/validation functions, CUB dataset | `part_detection/nets.py`, `train.py`, `datasets.py`, `save_maps.py`, `lib.py` | reused as-is | Upstream vendors PDiscoNet (van der Klis et al., ICCV 2023). |
| PDiscoNet training driver | `part_detection/train_pdisco_quick.py` | adapted from `part_detection/main.py` | Our driver: filters CUB to 25 classes, 224 px instead of 448, 5 epochs instead of 28, wandb disabled. Reuses upstream `train`/`validation`, optimiser groups and loss weights. |
| Part-map generation driver | `part_detection/generate_maps_quick.py` | adapted from `part_detection/save_maps.py` | Our driver over our cropped 25-class subset; reuses upstream `save_maps`. |
| MetiNet (LucidPPN) model: ShapeTexNet, ColorNet, fusion | `MetiNet/metinet/metinet.py`, `MetiNet/features/*` | reused as-is | |
| MetiNet training step and losses | `MetiNet/metinet/train.py` | reused as-is | |
| Data loading and augmentation, hue perturbation, top-k metric | `MetiNet/util/data.py`, `MetiNet/util/func.py` | reused as-is | |
| CUB subset preparation (bbox crop, 25 classes) | `prepare_cub_subset.py` | written by us | |
| Forward-pass sanity check | `MetiNet/forward_pass_test.py` | written by us | Model construction follows upstream `MetiNet/main.py`. |
| MetiNet training driver (Jumps 2-4, Stage 4 runs) | `MetiNet/train_quick.py` | adapted from `MetiNet/main.py` | Our optimiser/scheduler setup and epoch loop; reuses upstream `train_metinet`. Evaluation loop rewritten (upstream `test.py` needs full-dataset part maps for IoU). Added: command-line overrides, per-epoch ShapeTexNet accuracy, JSON history. |
| Colour-branch ablation and hue-robustness evaluation (Stage 4) | `MetiNet/eval_ablation.py` | written by us | Uses upstream `random_hue_jitter` (via `create_datasets`) and `topk_accuracy`. |
| Training-curve plots | `analysis/plot_training_curves.py` | written by us | |
| Deployment: inference wrapper, Streamlit app, FastAPI API | `deploy/*` | written by us | Preprocessing copied from the test-time branch of upstream `util/data.py`. |
| Colab notebook | `notebooks/stage4_stage5_colab.ipynb` | written by us | |
| One-click demo launchers | `run_demo.bat`, `run_demo.sh` | written by us | |
| Demo example images | `deploy/examples/` | CUB-200-2011 (Wah et al., 2011) | One bounding-box-cropped test image per species, made by `prepare_cub_subset.py`. |
| Trained model | `MetiNet/checkpoint_quick_train.pt` | trained by us | Jump 4 configuration, rerun on Colab (84.0% test top-1); used by Stage 4 and the demo. |
| `PIPNet/`, `MetiNet/main.py`, `MetiNet/metinet/test.py`, visualisation utilities | | upstream, not used | Kept from the fork; not part of our pipeline. |

## Results

| Result | Source |
|---|---|
| LucidPPN CUB 81.5 ± 0.4 % (200 classes, ConvNeXt-Tiny), ShapeTexNet 80.4 %, hue-perturbed numbers | reported by the authors (paper Tables 1-3) |
| Everything in `EXPERIMENTS.md`, `MetiNet/*log*.txt`, `part_detection/*log*.txt`, `MetiNet/results/` | obtained by our team, on Google Colab |

## AI assistance
The Stage 3 review, `PROVENANCE.md`, the Colab notebook and the
report draft were written with help from an AI assistant. The team ran all experiments.
