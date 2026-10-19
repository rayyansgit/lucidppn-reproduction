"""
Plots train/test accuracy per epoch for Jumps 2-4 from the saved MetiNet training logs.
Run from the repository root:  python analysis/plot_training_curves.py
Writes analysis/figures/training_curves.png
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ('Jump 2: no part maps, 10 ep', 'MetiNet/training_log_25class_10epoch.txt', 2),
    ('Jump 3: + part maps, 10 ep', 'MetiNet/training_log_25class_10epoch_WITH_parts.txt', 2),
    ('Jump 4: + 82 ep, batch 32', 'MetiNet/training_log_JUMP4_25class_80epoch_batch32.txt', 10),
]
LINE = re.compile(r'Epoch (\d+)/\d+ \| loss=([\d.]+) \| train_acc=([\d.]+) \| test_top1=([\d.]+)')


def parse(path):
    with open(os.path.join(ROOT, path), encoding='utf-8', errors='ignore') as f:
        text = f.read().replace('\r', '\n')
    rows = [tuple(map(float, m.groups())) for m in LINE.finditer(text)]
    return [list(col) for col in zip(*rows)]


fig, axes = plt.subplots(1, 2, figsize=(11, 4))
colors = ['#8c8c8c', '#e8a33d', '#3d7be8']
for (label, path, freeze), c in zip(RUNS, colors):
    ep, loss, train, test = parse(path)
    axes[0].plot(ep, [100 * t for t in test], color=c, label=f'{label} (test)')
    axes[0].plot(ep, [100 * t for t in train], color=c, ls=':', alpha=0.8)
    axes[0].axvline(freeze + 0.5, color=c, lw=0.8, ls='--', alpha=0.5)
    axes[1].plot(ep, loss, color=c, label=label)
    print(f'{label}: final test {100 * test[-1]:.1f}%, final train {100 * train[-1]:.1f}%, '
          f'test at end of frozen phase (epoch {freeze}) {100 * test[freeze - 1]:.1f}%')
axes[0].axhline(4.0, color='k', lw=0.8, alpha=0.4)
axes[0].text(1, 6, 'chance (4%)', fontsize=8, alpha=0.6)
axes[0].set(xlabel='Epoch', ylabel='Top-1 accuracy (%)',
            title='Test (solid) and train (dotted) accuracy\ndashed vertical line = backbone unfrozen')
axes[1].set(xlabel='Epoch', ylabel='Training loss', title='Training loss')
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, 'analysis', 'figures'), exist_ok=True)
out = os.path.join(ROOT, 'analysis', 'figures', 'training_curves.png')
fig.savefig(out, dpi=150)
print('saved', out)
