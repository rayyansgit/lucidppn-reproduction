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
| PDiscoNet driver with configurable epochs / output folder | `part_detection/train_pdisco_long.py` | adapted from `part_detection/main.py` | Same setup as `train_pdisco_quick.py` (left unchanged) plus command-line options and a Windows-safe `__main__` guard; used for the 28-epoch detector. |
| Part-map generation for any detector | `part_detection/generate_maps.py` | adapted from `part_detection/save_maps.py` | Same as `generate_maps_quick.py` (left unchanged) with command-line options. |
| Part-quality evaluation (NMI, ARI, keypoint regression) | `part_detection/eval_part_quality.py` | adapted from `part_detection/evaluate_cub.py` | Reuses upstream `eval_kpr` and `eval_nmi_ari` unchanged; filters upstream's 200-class CUB loader to our 25 classes. |
| Full evaluation: top-1/top-5, confusion matrix, IoU, keypoint alignment | `MetiNet/eval_full.py` | adapted from `MetiNet/metinet/test.py` | IoU/IoP/IoT computed with the authors' definition; keypoint NMI/ARI on prototype maps is ours, using the metric from PDiscoNet's `evaluate_cub.py`. |
| Evaluation summary and figures | `analysis/summarize_eval.py` | written by us | |
| Multi-seed summary, colourfulness-vs-hue-sensitivity analysis | `analysis/summarize_runs.py`, `analysis/colour_by_species.py` | written by us | Colourfulness metric from Hasler & Süsstrunk (2003). |
| Overnight experiment queue | `experiments/run_queue.sh` | written by us | |
| Deployment: inference wrapper, Streamlit app, FastAPI API | `deploy/*`, `.streamlit/config.toml` | written by us | Preprocessing follows the test-time branch of upstream `util/data.py`. Per-part colour evidence, accepted-colour grid (trained colour branch evaluated on an RGB grid), part comparison and the in-app ablation view are our additions. |
| Prototype patches ("this looks like that") | `deploy/build_prototypes.py`, `deploy/prototypes/` | written by us | Patches cropped from our CUB training images at each prototype's strongest activation, following the paper's prototype visualisation. |
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
| Everything in `experiments/` (Jumps 5-7, 28-epoch part detector, part quality, multi-seed Stage 4, colourfulness analysis) | obtained by our team, on a local RTX 2080 laptop GPU |
| Colour ablation split by species colourfulness (12 most vs 12 least colourful species), shown in the app and the report | computed with the AI assistant from our per-species results in `experiments/*/ablation/ablation.json`; no new runs |

## AI assistance
The Stage 3 review, `PROVENANCE.md`, the Colab notebook and the
report draft were written with help from an AI assistant. The team ran all experiments.
We also used the AI assistant to help train the models overnight and automate that process (`experiments/run_queue.sh`),
to find the paper's training settings (in the authors' `MetiNet/run_training.sh`) and to implement them in our runs.
For the final submission an AI assistant (Claude) reviewed the repository and re-checked the reported numbers against
the raw result files, rewrote the Streamlit app's layout and explanation views (`deploy/app.py`,
`deploy/lucid_inference.py`), and drafted the updated documentation, report and slides. The team reviewed and edited
these and is responsible for the content.
