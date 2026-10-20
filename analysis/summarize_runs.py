"""
Pools the multi-seed runs in experiments/ (Jumps 5-6) with the Jump 4 baseline into one table and one figure.

Run from the repository root:  python analysis/summarize_runs.py
Reads   experiments/<config>/seed*/history.json and experiments/<config>/seed*/ablation/ablation.json
Writes  experiments/SUMMARY.md and experiments/summary.png
"""
import glob
import json
import os
import re
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'experiments')
# (folder, label, recipe, part detector) -- the 2x2 study; Jump 4's seed-1 runs are added separately below
CONFIGS = [
    ('jump4_code_defaults', 'Jump 4: code defaults, 5-ep parts', 'Code defaults (Jump 4)', '5-epoch'),
    ('jump7_code_defaults_pdisco28', 'Jump 7: code defaults, 28-ep parts', 'Code defaults (Jump 4)', '28-epoch'),
    ('jump5_paper_recipe', 'Jump 5: paper recipe, 5-ep parts', 'Paper recipe', '5-epoch'),
    ('jump6_paper_recipe_pdisco28', 'Jump 6: paper recipe, 28-ep parts', 'Paper recipe', '28-epoch'),
]
LINE = re.compile(r'Epoch (\d+)/\d+ \| loss=[\d.]+ \| train_acc=([\d.]+) \| test_top1=([\d.]+)')


def ms(values, scale=100):
    values = [scale * v for v in values]
    if len(values) == 1:
        return f'{values[0]:.1f}'
    return f'{st.mean(values):.1f} ± {st.stdev(values):.1f}'


def log_curve(path):
    with open(os.path.join(ROOT, path), encoding='utf-8', errors='ignore') as f:
        rows = [tuple(map(float, m.groups())) for m in LINE.finditer(f.read().replace('\r', '\n'))]
    return [r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows]


def new_cell():
    return {'final': [], 'shape': [], 'gain': [], 'drop': [], 'seeds': 0, 'notes': []}


def add_ablation(d, path):
    with open(path) as f:
        r = json.load(f)['results']
    d['shape'].append(r['0.0']['shapetexnet_top1'][0])
    d['gain'].append(r['0.0']['lucidppn_top1'][0] - r['0.0']['shapetexnet_top1'][0])
    d['drop'].append(r['0.5']['lucidppn_top1'][0] - r['0.0']['lucidppn_top1'][0])


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cells, curves = {}, []
    for key, label, recipe, parts in CONFIGS:
        d = cells[key] = new_cell()
        if key == 'jump4_code_defaults':
            # seed-1 runs: the team's original Stage 3 run (log only, checkpoint lost) and the rerun (= demo model)
            for path, note in [('MetiNet/training_log_JUMP4_25class_80epoch_batch32.txt', 'original seed 1 (accuracy only)'),
                               ('MetiNet/training_log_JUMP4_rerun.txt', 'rerun seed 1')]:
                ep, test, _ = log_curve(path)
                d['final'].append(test[-1]); d['seeds'] += 1; d['notes'].append(note)
                curves.append((label, ep, test))
            rerun_abl = os.path.join(EXP, 'jump4_rerun', 'ablation', 'ablation.json')
            if os.path.exists(rerun_abl):
                add_ablation(d, rerun_abl)
        for seed_dir in sorted(glob.glob(os.path.join(EXP, key, 'seed*'))):
            hist_path = os.path.join(seed_dir, 'history.json')
            if not os.path.exists(hist_path):
                continue
            with open(hist_path) as f:
                hist = json.load(f)['history']
            d['final'].append(hist[-1]['test_top1']); d['seeds'] += 1; d['notes'].append(os.path.basename(seed_dir))
            curves.append((label, [h['epoch'] for h in hist], [h['test_top1'] for h in hist]))
            abl_path = os.path.join(seed_dir, 'ablation', 'ablation.json')
            if os.path.exists(abl_path):
                add_ablation(d, abl_path)
    rows = [(label, cells[key], ', '.join(cells[key]['notes'])) for key, label, _, _ in CONFIGS if cells[key]['seeds']]

    lines = ['| Configuration | Runs | LucidPPN test top-1 (final epoch) | ShapeTexNet only | Colour gain (clean) | '
             'Change under hue shift 0.5 | Notes |', '|---|---|---|---|---|---|---|']
    for label, d, note in rows:
        cell = lambda k: ms(d[k]) if d[k] else '—'
        lines.append(f"| {label} | {d['seeds']} | {cell('final')} | {cell('shape')} | {cell('gain')} | "
                     f"{cell('drop')} | {note} |")
    grid = ['| Test top-1 (mean ± std) | 5-epoch part detector | 28-epoch part detector |', '|---|---|---|']
    for recipe in ['Code defaults (Jump 4)', 'Paper recipe']:
        cell = {parts: cells[key] for key, _, r, parts in CONFIGS if r == recipe}
        fmt = lambda c: f"{ms(c['final'])} ({c['seeds']} runs)" if c['seeds'] else '—'
        grid.append(f"| {recipe} | {fmt(cell['5-epoch'])} | {fmt(cell['28-epoch'])} |")
    lines = grid + [''] + lines
    lines.append('')
    lines.append('All numbers are % on the 652 test images of the 25-class subset; mean ± sample std over runs. '
                 'Colour gain = LucidPPN − ShapeTexNet on clean images; hue-shift change = LucidPPN accuracy at max '
                 'hue shift 0.5 (mean of 5 perturbation seeds) − clean accuracy.')
    os.makedirs(EXP, exist_ok=True)
    with open(os.path.join(EXP, 'SUMMARY.md'), 'w', encoding='utf-8') as f:
        f.write('# Multi-seed summary: training recipe x part detector\n\n' + '\n'.join(lines) + '\n\n![Test accuracy per epoch](summary.png)\n')
    print('\n'.join(lines))

    colours = dict(zip([label for _, label, _, _ in CONFIGS], ['#8c8c8c', '#2f9e66', '#3d7be8', '#e8a33d']))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    seen = set()
    for label, ep, test in curves:
        ax.plot(ep, [100 * t for t in test], color=colours[label], alpha=0.75, lw=1.3,
                label=None if label in seen else label)
        seen.add(label)
    ax.axhline(4.0, color='k', lw=0.8, alpha=0.4)
    ax.set(xlabel='Epoch', ylabel='Test top-1 accuracy (%)', title='Test accuracy per epoch, every run (one line per seed)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, 'summary.png'), dpi=150)


if __name__ == '__main__':
    main()
