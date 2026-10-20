"""
Part-quality scores for one or more PDiscoNet checkpoints on the 25-class CUB subset, using the authors' own metrics
from evaluate_cub.py (reused as-is):
  * NMI / ARI: agreement between the predicted part at each of CUB's 15 annotated keypoints and the keypoint identity
    (higher = parts land on the same body region consistently).
  * Keypoint regression error: a linear map from part centres to keypoint locations, fitted on the train split and
    evaluated on the test split, in % of the bounding-box size (lower = better).
evaluate_cub.py's main() is written for all 200 classes; here the same loader is filtered to the first n classes.

Usage (from part_detection/):
    python eval_part_quality.py --checkpoints old=cub_quick/cub_quick_25class.pt \
        new=../experiments/part_detector/pdisco_28ep/pdisco.pt --out ../experiments/part_detector/part_quality
"""
import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torchvision.models import resnet101
from datasets import CUB200
from nets import IndividualLandmarkNet
from evaluate_cub import eval_kpr, eval_nmi_ari

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def subset(train, n_classes, image_size):
    ds = CUB200(ROOT, train=train, image_size=image_size, evaluate=True)
    pre = '_train' if train else '_test'
    keep = [i for i, l in enumerate(getattr(ds, pre + '_labels')) if l < n_classes]
    for field in ['_data', '_labels', '_parts', '_boxes']:
        setattr(ds, pre + field, [getattr(ds, pre + field)[i] for i in keep])
    return ds


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--checkpoints', nargs='+', required=True, help='name=path pairs')
    cli.add_argument('--out', required=True, help='output path without extension (.json and .md are written)')
    cli.add_argument('--n_classes', type=int, default=25)
    cli.add_argument('--num_parts', type=int, default=8)
    cli.add_argument('--image_size', type=int, default=224)
    cli.add_argument('--num_workers', type=int, default=4)
    opt = cli.parse_args()

    torch.manual_seed(0)
    fit_set, eval_set = subset(True, opt.n_classes, opt.image_size), subset(False, opt.n_classes, opt.image_size)
    print(f"Keypoint-regression fit set: {len(fit_set)} train images | evaluation: {len(eval_set)} test images")
    loader = lambda ds: torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    results = {}
    for item in opt.checkpoints:
        name, path = item.split('=', 1)
        net = IndividualLandmarkNet(resnet101(), opt.num_parts, num_classes=opt.n_classes).cuda()
        net.load_state_dict(torch.load(path, map_location='cuda'))
        net.eval()
        kpr = eval_kpr(net, loader(fit_set), loader(eval_set), opt.num_parts)
        nmi, ari = eval_nmi_ari(net, loader(eval_set))
        results[name] = {'checkpoint': path, 'nmi': nmi, 'ari': ari, 'kpr_error': kpr}
        print(f"{name}: NMI {nmi:.2f} | ARI {ari:.2f} | keypoint regression error {kpr:.2f}%", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(opt.out)), exist_ok=True)
    with open(opt.out + '.json', 'w') as f:
        json.dump(results, f, indent=1)
    lines = ['| Part detector | NMI ↑ | ARI ↑ | Keypoint regression error ↓ |', '|---|---|---|---|']
    lines += [f"| {n} | {r['nmi']:.2f} | {r['ari']:.2f} | {r['kpr_error']:.2f}% |" for n, r in results.items()]
    with open(opt.out + '.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
