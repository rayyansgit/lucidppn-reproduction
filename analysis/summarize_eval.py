"""
Pools the full evaluations (MetiNet/eval_full.py) of all 12 models into one report.

Run from the repository root:  python analysis/summarize_eval.py
Reads   experiments/jump4_rerun/eval/eval_full.json and experiments/<config>/seed*/eval/eval_full.json
Writes  experiments/EVALUATION.md, experiments/confusion_matrix.png, experiments/part_alignment.png
"""
import glob
import json
import os
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'experiments')
CONFIGS = [  # (label, recipe, part detector, eval files)
    ('Code defaults, 5-epoch parts', 'Code defaults', '5-epoch',
     ['jump4_rerun/eval'] + [f'jump4_code_defaults/seed{s}/eval' for s in (2, 3)]),
    ('Code defaults, 28-epoch parts', 'Code defaults', '28-epoch',
     [f'jump7_code_defaults_pdisco28/seed{s}/eval' for s in (1, 2, 3)]),
    ('Paper recipe, 5-epoch parts', 'Paper recipe', '5-epoch',
     [f'jump5_paper_recipe/seed{s}/eval' for s in (1, 2, 3)]),
    ('Paper recipe, 28-epoch parts', 'Paper recipe', '28-epoch',
     [f'jump6_paper_recipe_pdisco28/seed{s}/eval' for s in (1, 2, 3)]),
]


def ms(values, scale=1.0, digits=1):
    values = [scale * v for v in values]
    return f'{st.mean(values):.{digits}f} ± {st.stdev(values):.{digits}f}' if len(values) > 1 else f'{values[0]:.{digits}f}'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    runs = {}
    for label, _, _, dirs in CONFIGS:
        runs[label] = [json.load(open(os.path.join(EXP, d, 'eval_full.json'))) for d in dirs
                       if os.path.exists(os.path.join(EXP, d, 'eval_full.json'))]
    all_runs = [r for rs in runs.values() for r in rs]
    classes = all_runs[0]['classification']['classes']
    pretty = [c.replace('_', ' ') for c in classes]

    lines = ['# Full evaluation of all 12 models', '',
             'Test set: 652 images of the 25-class subset. Classification metrics as in the paper (top-1, top-5); part '
             'alignment with the paper\'s IoU (prototype map vs the PDiscoNet mask the model was trained with, '
             'definition from the authors\' `metinet/test.py`) and with keypoint NMI/ARI (which part\'s prototype is '
             'strongest at each of CUB\'s 15 human-annotated keypoints; 7,571 visible keypoints; comparable across '
             'models). Mean ± std over models.', '',
             '## Classification and part alignment', '',
             '| Configuration | Models | Top-1 | Top-5 | ShapeTexNet top-1 | ShapeTexNet top-5 | IoU with own masks | '
             'IoP | IoT | Keypoint NMI ↑ | Keypoint ARI ↑ |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for label, rs in runs.items():
        c = [r['classification'] for r in rs]; i = [r['iou_own_masks'] for r in rs]; k = [r['keypoint_alignment'] for r in rs]
        lines.append(f"| {label} | {len(rs)} | {ms([x['lucid_top1'] for x in c], 100)} | {ms([x['lucid_top5'] for x in c], 100)} | "
                     f"{ms([x['shape_top1'] for x in c], 100)} | {ms([x['shape_top5'] for x in c], 100)} | "
                     f"{ms([x['iou'] for x in i], 1, 3)} | {ms([x['iop'] for x in i], 1, 3)} | {ms([x['iot'] for x in i], 1, 3)} | "
                     f"{ms([x['nmi'] for x in k])} | {ms([x['ari'] for x in k])} |")
    lines += ['', 'IoU is measured against each model\'s own training masks, which differ between the 5-epoch and the '
              '28-epoch detector (different, unsupervised parts), so IoU values are comparable within a column of '
              'part detectors only. Keypoint NMI/ARI uses the same human annotations for every model.', '']

    # per-species accuracy and most confused pairs, pooled over all 12 models
    conf = sum(np.array(r['classification']['confusion']) for r in all_runs)
    acc = conf.diagonal() / conf.sum(1)
    order = np.argsort(acc)
    lines += ['## Per-species accuracy (all 12 models pooled)', '', '| Species | Top-1 | Most often confused with |', '|---|---|---|']
    for i in order:
        off = conf[i].copy(); off[i] = 0
        j = int(off.argmax())
        conf_txt = f'{pretty[j]} ({100 * off[j] / conf[i].sum():.0f}% of its images)' if off[j] else '—'
        lines.append(f'| {pretty[i]} | {100 * acc[i]:.1f}% | {conf_txt} |')
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and conf[i, j]:
                pairs.append((conf[i, j] / conf[i].sum(), i, j))
    pairs.sort(reverse=True)
    lines += ['', '## Most confused pairs', '', '| True species | Predicted as | Share of its test images |', '|---|---|---|']
    lines += [f'| {pretty[i]} | {pretty[j]} | {100 * p:.1f}% |' for p, i, j in pairs[:10]]
    lines += ['', '![Confusion matrix](confusion_matrix.png)', '', '![Part alignment](part_alignment.png)']
    with open(os.path.join(EXP, 'EVALUATION.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:12]))

    # confusion matrix figure (row-normalised, pooled)
    norm = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(100 * norm, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(len(classes)), pretty, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), pretty, fontsize=7)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if norm[i, j] >= 0.05:
                ax.text(j, i, f'{100 * norm[i, j]:.0f}', ha='center', va='center', fontsize=5.5,
                        color='white' if norm[i, j] > 0.5 else 'black')
    ax.set(xlabel='Predicted species', ylabel='True species',
           title='Confusion matrix, 12 models pooled (% of each species\' test images)')
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(EXP, 'confusion_matrix.png'), dpi=150)

    # part-alignment figure: keypoint NMI per configuration, one dot per model
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colours = {'5-epoch': '#8c8c8c', '28-epoch': '#3d7be8'}
    for x, (label, recipe, parts, _) in enumerate(CONFIGS):
        vals = [r['keypoint_alignment']['nmi'] for r in runs[label]]
        ax.bar(x, st.mean(vals), color=colours[parts], alpha=0.35, width=0.6)
        ax.scatter([x] * len(vals), vals, color=colours[parts], zorder=3)
    ax.set_xticks(range(len(CONFIGS)), [f'{c[1]}\n{c[2]} parts' for c in CONFIGS], fontsize=8)
    ax.set_ylabel('Keypoint NMI (higher = better aligned)')
    ax.set_title('Do prototypes sit on the same bird parts? (dots = models)')
    ax.set_ylim(0, max(r['keypoint_alignment']['nmi'] for r in all_runs) * 1.25); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(EXP, 'part_alignment.png'), dpi=150)


if __name__ == '__main__':
    main()
