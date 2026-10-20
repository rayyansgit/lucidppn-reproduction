# Experiments after Stage 3: Jumps 5-7, part quality, multi-seed Stage 4

Results and their interpretation: end of [`EXPERIMENTS.md`](../EXPERIMENTS.md) and [`SUMMARY.md`](SUMMARY.md).

Everything here was run locally on an RTX 2080 Max-Q laptop (Windows, PyTorch 2.11 + CUDA 12.8, pandas 2.2.3).
Stage 3's results and logs (`MetiNet/training_log_*.txt`, `MetiNet/results/`) and the demo checkpoint
(`MetiNet/checkpoint_quick_train.pt`) are not modified by anything in this folder.

## Layout
| Folder | What |
|---|---|
| `jump4_rerun/ablation/` | Stage 4 evaluation of the Jump 4 rerun checkpoint (the demo model), re-run with 5 hue seeds and per-species results so it is comparable with Jumps 5-6 |
| `jump5_paper_recipe/seed{1,2,3}/` | Jump 5: the paper's training recipe (authors' `MetiNet/run_training.sh`), Stage 3 part maps |
| `part_detector/pdisco_28ep/` | PDiscoNet trained for the paper's 28 epochs (Stage 3 used 5) |
| `part_detector/part_quality.{md,json}` | Part-quality scores of the 5-epoch vs 28-epoch detector |
| `jump6_paper_recipe_pdisco28/seed{1,2,3}/` | Jump 6: Jump 5 recipe with part maps from the 28-epoch detector |
| `jump4_code_defaults/seed{2,3}/` | Jump 4 configuration (code defaults), extra seeds; seed 1 is the rerun in `MetiNet/checkpoint_quick_train.pt` |
| `jump7_code_defaults_pdisco28/seed{1,2,3}/` | Jump 7: Jump 4 configuration with part maps from the 28-epoch detector |
| `run_queue.sh`, `queue_log.txt` | The overnight queue that ran the 28-epoch maps, part quality, Jumps 6, 4 (seeds 2-3) and 7, and all ablations |
| `REPORT_INSERT.md` | Ready-to-paste text and tables for the report and slides |
| `SUMMARY.md`, `summary.png` | All configurations side by side (mean ± std over seeds) |
| `colour_by_species.{md,png,json}` | Hue sensitivity vs colourfulness, per species |

Each training-run folder holds `config.json` (exact settings), `train_log.txt`, `history.json` (per-epoch numbers),
`tensorboard/`, `ablation/` (Stage 4 results) and `checkpoint.pt`. Checkpoints are not committed (`*.pt` is ignored).

## Configurations
| | Jump 4 / Jump 7 (code defaults) | Jump 5 / Jump 6 (paper recipe) |
|---|---|---|
| Epochs, frozen epochs | 80 (+2), 10 | 40, 15 |
| lr shape branch / ColorNet | 0.0005 / 0.0005 | 0.002 / 0.002 |
| ColorNet delayed | 2 epochs | 0 |
| Part weight α_D | 1.0 | 1.4 |
| Batch size | 32 | 64 |
| Part maps | Jump 4: 5-epoch PDiscoNet; Jump 7: 28-epoch | Jump 5: 5-epoch; Jump 6: 28-epoch |
| Unchanged | ResNet18, K = 8, 25 classes, AdamW, weight decay 0, 224 px, augmentation | same |

## Commands (from the repository root)
```bash
# Part maps from the Stage 3-style 5-epoch detector (saved from the Colab rerun)
cd part_detection && python generate_maps.py --checkpoint cub_quick/cub_quick_25class.pt --out ../CUB_subset_maps && cd ..

# Jump 5, seeds 1-3
cd MetiNet && python train_quick.py --epochs 40 --freeze_epochs 15 --no_color_epochs 0 --lr_net 0.002 --lr_color 0.002 \
    --part_weight 1.4 --batch_size 64 --seed 1 --num_workers 5 --persistent_workers --eval_batch_size 64 \
    --out_dir ../experiments/jump5_paper_recipe/seed1 && cd ..

# 28-epoch part detector, its maps, and part quality of both detectors
cd part_detection && python train_pdisco_long.py --epochs 28 --out_dir ../experiments/part_detector/pdisco_28ep
python generate_maps.py --checkpoint ../experiments/part_detector/pdisco_28ep/pdisco.pt --out ../CUB_subset_maps_pdisco28
python eval_part_quality.py --checkpoints 5ep=cub_quick/cub_quick_25class.pt 28ep=../experiments/part_detector/pdisco_28ep/pdisco.pt \
    --out ../experiments/part_detector/part_quality && cd ..

# Jump 6: as Jump 5 plus --maps_dir ../CUB_subset_maps_pdisco28 --out_dir ../experiments/jump6_paper_recipe_pdisco28/seedN

# Stage 4 colour ablation per model (5 hue seeds; --num_workers 0 on Windows)
cd MetiNet && python eval_ablation.py --checkpoint ../experiments/jump5_paper_recipe/seed1/checkpoint.pt --seeds 5 \
    --num_workers 0 --out_dir ../experiments/jump5_paper_recipe/seed1/ablation && cd ..

# Summaries
python analysis/summarize_runs.py
python analysis/colour_by_species.py --ablations experiments/*/ablation/ablation.json experiments/*/seed*/ablation/ablation.json \
    --out experiments/colour_by_species
```
