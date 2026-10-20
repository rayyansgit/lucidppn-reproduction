#!/usr/bin/env bash
# Overnight queue for the 2x2 study (training recipe x part detector, 3 seeds each), run from the repository root:
#   bash experiments/run_queue.sh
# Order = priority: if anything fails late, the most important results already exist.
# Every step skips work whose output already exists, so the queue can be restarted safely.
set -u
PY=${PY:-/d/lgpu/Scripts/python.exe}
export PYTHONIOENCODING=utf-8
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG="$ROOT/experiments/queue_log.txt"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
clean() { sed -u 's/\r/\n/g' | grep --line-buffered -v "it/s\]\|s/it\]\|^\s*$\|FutureWarning\|UserWarning\|_VF.meshgrid\|upsample_bilinear"; }

PAPER="--epochs 40 --freeze_epochs 15 --no_color_epochs 0 --lr_net 0.002 --lr_color 0.002 --part_weight 1.4 --batch_size 64"
FAST="--num_workers 5 --persistent_workers --eval_batch_size 64"

train() {  # $1 = run folder (relative to experiments/), $2 = seed, $3 = maps dir, $4.. = recipe flags
    local name=$1 d="$ROOT/experiments/$1" seed=$2 maps=$3; shift 3
    if [ -f "$d/history.json" ]; then say "skip $name (done)"; return; fi
    mkdir -p "$d"; say "train $name (seed $seed, maps $maps)"
    (cd "$ROOT/MetiNet" && "$PY" -u train_quick.py "$@" --seed "$seed" --maps_dir "$maps" $FAST --out_dir "$d" \
        --run_name "$(basename "$(dirname "$d")")_$(basename "$d")" 2>&1 | clean > "$d/train_log.txt")
    say "  -> $(grep 'Final test' "$d/train_log.txt" || echo FAILED)"
}

ablate() {  # $1 = run folder (relative to experiments/) containing checkpoint.pt
    local d="$ROOT/experiments/$1"
    if [ -f "$d/ablation/ablation.json" ]; then say "skip ablation $1 (done)"; return; fi
    [ -f "$d/checkpoint.pt" ] || { say "no checkpoint for $1, skipping ablation"; return; }
    mkdir -p "$d/ablation"; say "ablation $1"
    (cd "$ROOT/MetiNet" && "$PY" -u eval_ablation.py --checkpoint "$d/checkpoint.pt" --seeds 5 --num_workers 0 \
        --out_dir "$d/ablation" 2>&1 | grep -v "Warn\|warn" > "$d/ablation/eval_log.txt")
}

# Ask Windows not to idle-sleep while the queue runs (no setting is changed; released when this helper exits)
"$PY" -c "import ctypes, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); time.sleep(12 * 3600)" &
AWAKE=$!
trap 'kill $AWAKE 2>/dev/null' EXIT

# 0. wait for the 28-epoch part detector (started separately) to finish
say "waiting for 28-epoch PDiscoNet"
until grep -q "PDiscoNet training done" "$ROOT/experiments/part_detector/pdisco_28ep/train_log.txt" 2>/dev/null; do sleep 30; done

# 1. part maps from the 28-epoch detector, and part quality of both detectors
if [ ! -d "$ROOT/CUB_subset_maps_pdisco28/test" ]; then
    say "maps from 28-epoch detector"
    (cd "$ROOT/part_detection" && "$PY" generate_maps.py --checkpoint ../experiments/part_detector/pdisco_28ep/pdisco.pt \
        --out ../CUB_subset_maps_pdisco28 2>&1 | clean | tail -3 >> "$LOG")
fi
if [ ! -f "$ROOT/experiments/part_detector/part_quality.json" ]; then
    say "part quality"
    (cd "$ROOT/part_detection" && "$PY" -u eval_part_quality.py --checkpoints 5ep=cub_quick/cub_quick_25class.pt \
        28ep=../experiments/part_detector/pdisco_28ep/pdisco.pt --out ../experiments/part_detector/part_quality 2>&1 \
        | clean > ../experiments/part_detector/part_quality_log.txt)
    say "  -> $(tail -4 "$ROOT/experiments/part_detector/part_quality_log.txt" | tr '\n' ' ')"
fi

# 2. Jump 6: paper recipe + 28-epoch parts
for s in 1 2 3; do train "jump6_paper_recipe_pdisco28/seed$s" $s ../CUB_subset_maps_pdisco28 $PAPER; done
# 3. Jump 4 code defaults, seeds 2 and 3 (seed 1 = the rerun in MetiNet/checkpoint_quick_train.pt)
for s in 2 3; do train "jump4_code_defaults/seed$s" $s ../CUB_subset_maps; done
# 4. Jump 7: code defaults + 28-epoch parts
for s in 1 2 3; do train "jump7_code_defaults_pdisco28/seed$s" $s ../CUB_subset_maps_pdisco28; done

# 5. Stage 4 colour ablation on every new model
for d in jump6_paper_recipe_pdisco28/seed{1,2,3} jump4_code_defaults/seed{2,3} jump7_code_defaults_pdisco28/seed{1,2,3}; do
    ablate "$d"
done

# 6. summaries
(cd "$ROOT" && "$PY" analysis/summarize_runs.py > /dev/null 2>&1; \
 "$PY" analysis/colour_by_species.py --ablations experiments/*/ablation/ablation.json \
     experiments/*/seed*/ablation/ablation.json --out experiments/colour_by_species > /dev/null 2>&1)
say "QUEUE DONE"
