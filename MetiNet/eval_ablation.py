"""
Stage 4 experiment (ablation of the colour branch), evaluation only.

Question: how much of our LucidPPN's test accuracy comes from the ColorNet branch, and how much
robustness to colour changes does that cost?

For one trained checkpoint we evaluate, on the 25-class CUB test set:
  * LucidPPN      - the full model's prediction (ColorNet x ShapeTexNet, mean over parts)
  * ShapeTexNet   - the grayscale branch alone (mean over parts of the ShapeTexNet resemblances, P_S in the paper)
under increasing hue perturbation (hue rotated at random, luminance preserved, the authors' random_hue_jitter).
This mirrors Tables 2 and 3 of the paper.

Why no retraining is needed for the "no ColorNet" model: in MetiNet.forward the ColorNet output is multiplied by
proto_features.detach(), so no gradient from the colour loss reaches ShapeTexNet. ShapeTexNet inside the jointly
trained model is therefore trained exactly as it would be with the ColorNet removed.

Usage (from MetiNet/):
  python eval_ablation.py --checkpoint checkpoint_quick_train.pt
Outputs: results/ablation.json, results/ablation.md, results/ablation_hue.png
"""
import sys, os, json, argparse
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
from util.data import create_datasets
from util.func import topk_accuracy
from metinet.metinet import MetiNet, get_network

cli = argparse.ArgumentParser()
cli.add_argument('--checkpoint', default='checkpoint_quick_train.pt')
cli.add_argument('--test_dir', default='../CUB_subset/test')
cli.add_argument('--train_dir', default='../CUB_subset/train')
cli.add_argument('--hue_levels', type=float, nargs='+', default=[0.0, 0.1, 0.25, 0.5])
cli.add_argument('--seeds', type=int, default=3, help='repeats of each random hue perturbation')
cli.add_argument('--num_workers', type=int, default=2)
cli.add_argument('--out_dir', default='results')
cli.add_argument('--limit', type=int, default=0, help='evaluate only the first N test images (smoke test)')
opt = cli.parse_args()

args = argparse.Namespace(net='resnet18', disable_pretrained=True, num_classes=25, num_parts=8, bias=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_net():
    feature_net, add_on, pool, classification, color_net, _ = get_network(args.num_classes, args)
    net = MetiNet(args.num_classes, args.num_parts, feature_net, add_on, pool, classification, color_net)
    state = torch.load(opt.checkpoint, map_location='cpu')
    # train_quick.py saves the nn.DataParallel wrapper's state dict ("module." prefix)
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    net.load_state_dict(state)
    return net.to(device).eval()


def make_testset(hue):
    _, _, testset, classes, _, _ = create_datasets(
        train_dir=opt.train_dir, project_dir=opt.train_dir, test_dir=opt.test_dir,
        maps_train_dir='', maps_test_dir='', img_size=224, num_parts=args.num_parts,
        augment_shape_zoom=8, strong_hue_augmentation=hue, crop_augmentation=1.0)
    if opt.limit:
        testset = torch.utils.data.Subset(testset, range(min(opt.limit, len(testset))))
    return testset, classes


@torch.no_grad()
def evaluate(net, testset):
    loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=opt.num_workers)
    n, lucid1, lucid5, shape1, shape5 = 0, 0., 0., 0., 0.
    helped, hurt = 0, 0  # colour branch flips a wrong ShapeTexNet prediction to right / right to wrong
    per_class = torch.zeros(args.num_classes, 3)  # images, LucidPPN correct, ShapeTexNet correct
    for x, x_aug, m, y in loader:
        x, x_aug, y = x.to(device), x_aug.to(device), y.to(device)
        _, proto_pooled, _, _, _, out = net(x, x_aug, m)
        shape_out = proto_pooled.mean(dim=1)
        l1, l5 = topk_accuracy(out, y, topk=[1, 5])
        s1, s5 = topk_accuracy(shape_out, y, topk=[1, 5])
        lucid1 += l1.sum().item(); lucid5 += l5.sum().item()
        shape1 += s1.sum().item(); shape5 += s5.sum().item()
        per_class.index_add_(0, y.cpu(), torch.stack([torch.ones(len(y)), l1.float().cpu(), s1.float().cpu()], 1))
        helped += ((l1 > 0) & (s1 == 0)).sum().item()
        hurt += ((l1 == 0) & (s1 > 0)).sum().item()
        n += len(y)
    return {'lucidppn_top1': lucid1 / n, 'lucidppn_top5': lucid5 / n,
            'shapetexnet_top1': shape1 / n, 'shapetexnet_top5': shape5 / n,
            'colour_helped': helped / n, 'colour_hurt': hurt / n, 'n': n, 'per_class': per_class}


def mean_std(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return t.mean().item(), (t.std().item() if len(vals) > 1 else 0.0)


def main():
    os.makedirs(opt.out_dir, exist_ok=True)
    net = load_net()
    results = {}
    for hue in opt.hue_levels:
        runs = []
        for seed in range(opt.seeds if hue > 0 else 1):  # hue=0 is deterministic
            torch.manual_seed(seed)
            testset, classes = make_testset(hue)
            r = evaluate(net, testset)
            runs.append(r)
            print(f"hue={hue:.2f} seed={seed} | LucidPPN {r['lucidppn_top1']:.4f} | ShapeTexNet {r['shapetexnet_top1']:.4f}"
                  f" | colour helped {r['colour_helped']:.4f}, hurt {r['colour_hurt']:.4f}")
        results[str(hue)] = {k: mean_std([r[k] for r in runs]) for k in runs[0] if k not in ('n', 'per_class')}
        # per-species accuracy, averaged over the perturbation seeds
        pc = torch.stack([r['per_class'] for r in runs]).mean(0)
        results[str(hue)]['per_class'] = {c: {'n': int(pc[i, 0]), 'lucidppn_top1': (pc[i, 1] / pc[i, 0]).item(),
                                              'shapetexnet_top1': (pc[i, 2] / pc[i, 0]).item()}
                                          for i, c in enumerate(classes) if pc[i, 0] > 0}
        results[str(hue)]['n'] = runs[0]['n']
        results[str(hue)]['seeds'] = len(runs)

    with open(os.path.join(opt.out_dir, 'ablation.json'), 'w') as f:
        json.dump({'checkpoint': opt.checkpoint, 'results': results}, f, indent=1)

    pct = lambda ms: f"{100 * ms[0]:.1f} ± {100 * ms[1]:.1f}" if ms[1] else f"{100 * ms[0]:.1f}"
    lines = ['| Max hue shift | LucidPPN top-1 | ShapeTexNet top-1 | Colour helped | Colour hurt |',
             '|---|---|---|---|---|']
    for hue, r in results.items():
        lines.append(f"| {hue} | {pct(r['lucidppn_top1'])} | {pct(r['shapetexnet_top1'])} | "
                     f"{pct(r['colour_helped'])} | {pct(r['colour_hurt'])} |")
    table = '\n'.join(lines)
    with open(os.path.join(opt.out_dir, 'ablation.md'), 'w', encoding='utf-8') as f:
        f.write(table + '\n')
    print('\n' + table)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        hues = [float(h) for h in results]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for key, label, style in [('lucidppn_top1', 'LucidPPN (shape + colour)', 'o-'),
                                  ('shapetexnet_top1', 'ShapeTexNet only (grayscale)', 's--')]:
            m = [100 * results[h][key][0] for h in results]
            s = [100 * results[h][key][1] for h in results]
            ax.errorbar(hues, m, yerr=s, fmt=style, capsize=3, label=label)
        ax.set_xlabel('Maximum random hue shift'); ax.set_ylabel('Test top-1 accuracy (%)')
        ax.set_title('Colour-branch ablation, 25-class CUB'); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(opt.out_dir, 'ablation_hue.png'), dpi=150)
    except ImportError:
        pass


if __name__ == '__main__':
    main()
