"""
Does hue sensitivity depend on how colourful a species is?

For every species in the 25-class test set:
  * colourfulness: Hasler & Süsstrunk (2003) colourfulness metric of each bounding-box-cropped test image
    (sqrt(std_rg^2 + std_yb^2) + 0.3 * sqrt(mean_rg^2 + mean_yb^2), on 0-255 RGB), averaged per species;
  * hue drop: LucidPPN accuracy on clean images minus accuracy at max hue shift 0.5, per species, from the
    per-species results eval_ablation.py writes; averaged over all models given.
Reports the Spearman rank correlation and a scatter plot.

Run from the repository root:
    python analysis/colour_by_species.py --ablations experiments/*/seed*/ablation/ablation.json --out experiments/colour_by_species
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def colourfulness(path):
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    return np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    cli = argparse.ArgumentParser()
    cli.add_argument('--ablations', nargs='+', required=True, help='ablation.json files with per_class results')
    cli.add_argument('--test_dir', default=os.path.join(ROOT, 'CUB_subset', 'test'))
    cli.add_argument('--hue', default='0.5')
    cli.add_argument('--out', required=True, help='output path without extension')
    opt = cli.parse_args()

    species = sorted(os.listdir(opt.test_dir))
    colour = {s: float(np.mean([colourfulness(p) for p in glob.glob(os.path.join(opt.test_dir, s, '*.jpg'))]))
              for s in species}

    drops, gains = {s: [] for s in species}, {s: [] for s in species}
    used = []
    for path in opt.ablations:
        with open(path) as f:
            res = json.load(f)['results']
        if 'per_class' not in res.get('0.0', {}) or opt.hue not in res:
            continue
        used.append(path)
        for s in species:
            clean, shifted = res['0.0']['per_class'][s], res[opt.hue]['per_class'][s]
            drops[s].append(clean['lucidppn_top1'] - shifted['lucidppn_top1'])
            gains[s].append(clean['lucidppn_top1'] - clean['shapetexnet_top1'])

    x = np.array([colour[s] for s in species])
    y = np.array([100 * np.mean(drops[s]) for s in species])
    g = np.array([100 * np.mean(gains[s]) for s in species])
    rho_drop, p_drop = spearmanr(x, y)
    rho_gain, p_gain = spearmanr(x, g)

    order = np.argsort(-x)
    lines = [f'Models pooled: {len(used)}. Spearman correlation with colourfulness: hue drop rho = {rho_drop:.2f} '
             f'(p = {p_drop:.3f}); clean colour gain rho = {rho_gain:.2f} (p = {p_gain:.3f}).', '',
             '| Species | Colourfulness | Accuracy drop at hue shift ' + opt.hue + ' (pts) | Clean colour gain (pts) |',
             '|---|---|---|---|']
    lines += [f'| {species[i].replace("_", " ")} | {x[i]:.1f} | {y[i]:+.1f} | {g[i]:+.1f} |' for i in order]
    half = len(species) // 2
    top, bottom = order[:half], order[-half:]
    lines += ['', f'Most colourful {half} species: mean drop {y[top].mean():.1f} pts, mean colour gain {g[top].mean():+.1f} pts. '
                  f'Least colourful {half}: mean drop {y[bottom].mean():.1f} pts, mean colour gain {g[bottom].mean():+.1f} pts.']
    os.makedirs(os.path.dirname(os.path.abspath(opt.out)), exist_ok=True)
    with open(opt.out + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(opt.out + '.json', 'w') as f:
        json.dump({'ablations': used, 'spearman_drop': [rho_drop, p_drop], 'spearman_gain': [rho_gain, p_gain],
                   'species': {s: {'colourfulness': colour[s], 'hue_drop': float(100 * np.mean(drops[s])),
                                   'colour_gain': float(100 * np.mean(gains[s]))} for s in species}}, f, indent=1)
    print('\n'.join(lines))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, color='#3d7be8')
    for i, s in enumerate(species):  # label only the stand-out species; the grey cluster stays unlabelled
        if x[i] > 45 or abs(y[i]) > 4 or x[i] < 20:
            ax.annotate(s.replace('_', ' '), (x[i], y[i]), fontsize=7.5, xytext=(4, 2), textcoords='offset points')
    ax.annotate('grey / brown / black species\n(albatrosses, cormorants, auklets,\nblackbirds, catbirds, ...)',
                (31, 1), xytext=(24, 12), fontsize=7.5, color='#555', arrowprops=dict(arrowstyle='->', color='#999'))
    ax.axhline(0, color='k', lw=0.8, alpha=0.4)
    ax.set(xlabel='Colourfulness of the species\' test images (Hasler & Süsstrunk)',
           ylabel=f'LucidPPN accuracy drop at hue shift {opt.hue} (pts)',
           title=f'Hue sensitivity vs colourfulness, per species (Spearman ρ = {rho_drop:.2f}, p = {p_drop:.3f})')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(opt.out + '.png', dpi=150)


if __name__ == '__main__':
    main()
