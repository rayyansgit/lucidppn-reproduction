"""
Full test-set evaluation of one checkpoint, with the paper's two metric families:
  * classification: top-1 and top-5 accuracy (LucidPPN and ShapeTexNet alone), per-species accuracy and the confusion
    matrix;
  * part alignment against the model's own PDiscoNet masks: IoU between each part's prototype map and PDiscoNet's mask
    for that part, plus intersection over prediction (IoP) and over target (IoT), computed exactly as in the authors'
    MetiNet/metinet/test.py (for each part, the prototype of the class that part votes for; both maps rounded to {0, 1}
    at feature-map size). PDiscoNet's parts are unsupervised and numbered arbitrarily, so this is only meaningful
    against the masks the model was trained with;
  * part alignment against human annotations (comparable across models): at each of CUB's 15 annotated keypoints, the
    part whose prototype (for the predicted class) is most active there; NMI / ARI between that part and the keypoint
    identity, the metric PDiscoNet's authors use for part discovery, applied to LucidPPN's prototype maps.

Usage (from MetiNet/):
    python eval_full.py --checkpoint checkpoint_quick_train.pt --maps ../CUB_subset_maps \
        --out_dir ../experiments/jump4_rerun/eval
Writes <out_dir>/eval_full.json.
"""
import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torchvision.transforms.functional import resize
from util.data import create_datasets
from util.func import topk_accuracy
from metinet.metinet import MetiNet, get_network


def load_net(checkpoint, device):
    args = argparse.Namespace(net='resnet18', disable_pretrained=True, num_classes=25, num_parts=8, bias=False)
    feature_net, add_on, pool, classification, color_net, _ = get_network(25, args)
    net = MetiNet(25, 8, feature_net, add_on, pool, classification, color_net)
    state = torch.load(checkpoint, map_location='cpu')
    net.load_state_dict({k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()})
    return net.to(device).eval()


def keypoints_224():
    """CUB's 15 annotated keypoints per image, moved into the bounding-box crop made by prepare_cub_subset.py and
    scaled to the 224x224 model input: {image file name: [(keypoint id, x, y), ...]} (visible keypoints only)."""
    cub = '../CUB_200_2011'
    names = dict(l.split() for l in open(f'{cub}/images.txt'))
    boxes = {l.split()[0]: [float(v) for v in l.split()[1:]] for l in open(f'{cub}/bounding_boxes.txt')}
    out = {}
    for line in open(f'{cub}/parts/part_locs.txt'):
        img_id, kp, x, y, visible = line.split()
        if visible != '1':
            continue
        bx, by, bw, bh = boxes[img_id]
        u, v = (float(x) - bx) * 224 / bw, (float(y) - by) * 224 / bh
        if 0 <= u < 224 and 0 <= v < 224:
            out.setdefault(os.path.basename(names[img_id]), []).append((int(kp), u, v))
    return out


@torch.no_grad()
def evaluate(net, maps_dir, device, num_workers):
    _, _, testset, classes, _, _ = create_datasets(
        train_dir='../CUB_subset/train', project_dir='../CUB_subset/train', test_dir='../CUB_subset/test',
        maps_train_dir=f'{maps_dir}/train', maps_test_dir=f'{maps_dir}/test', img_size=224, num_parts=8,
        augment_shape_zoom=8, strong_hue_augmentation=0.0, crop_augmentation=1.0)
    loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=num_workers)
    n_cls = len(classes)
    sums = {'lucid_top1': 0., 'lucid_top5': 0., 'shape_top1': 0., 'shape_top5': 0.}
    confusion = torch.zeros(n_cls, n_cls, dtype=torch.long)       # rows: true species, columns: LucidPPN prediction
    iou, iop, iot = torch.zeros(8), torch.zeros(8), torch.zeros(8)
    kps, kp_true, kp_part = keypoints_224(), [], []
    n = 0
    for x, x_aug, m, y in loader:
        x, x_aug, m, y = x.to(device), x_aug.to(device), m.to(device), y.to(device)
        proto_features, proto_pooled, _, _, _, out = net(x, x_aug, m)
        bs, size = len(y), proto_features.shape[-1]
        for key, scores in [('lucid', out), ('shape', proto_pooled.mean(dim=1))]:
            t1, t5 = topk_accuracy(scores, y, topk=[1, 5])
            sums[f'{key}_top1'] += t1.sum().item(); sums[f'{key}_top5'] += t5.sum().item()
        pred = out.argmax(1)
        confusion.index_put_((y.cpu(), pred.cpu()), torch.ones(bs, dtype=torch.long), accumulate=True)
        # keypoint alignment: which part's prototype (of the predicted class) is strongest at each annotated keypoint
        maps = proto_features[torch.arange(bs), :, pred].cpu()            # [bs, parts, size, size]
        for b in range(bs):
            for kp, u, v in kps.get(os.path.basename(testset.imgs[n + b][0]), []):
                kp_true.append(kp)
                kp_part.append(int(maps[b, :, int(v * size / 224), int(u * size / 224)].argmax()))
        for k in range(8):  # authors' definition (metinet/test.py)
            voted = proto_pooled[:, k, :].argmax(dim=1)
            predicted = torch.round(proto_features[torch.arange(bs), k, voted])
            target = torch.round(resize(m[:, k], [size, size], antialias=True))
            inter = torch.minimum(predicted, target).sum((-1, -2))
            iou[k] += torch.nan_to_num(inter / torch.maximum(predicted, target).sum((-1, -2)), nan=1.0, posinf=1.0).sum().cpu()
            iop[k] += torch.nan_to_num(inter / predicted.sum((-1, -2)), nan=1.0, posinf=1.0).sum().cpu()
            iot[k] += torch.nan_to_num(inter / target.sum((-1, -2)), nan=1.0, posinf=1.0).sum().cpu()
        n += bs
    kp_true, kp_part = np.array(kp_true), np.array(kp_part)
    return {
        'classification': {**{k: v / n for k, v in sums.items()},
                           'per_class_top1': {c: (confusion[i, i] / confusion[i].sum()).item()
                                              for i, c in enumerate(classes)},
                           'confusion': confusion.tolist(), 'classes': classes},
        'iou_own_masks': {'iou_per_part': (iou / n).tolist(), 'iou': (iou / n).mean().item(),
                          'iop': (iop / n).mean().item(), 'iot': (iot / n).mean().item()},
        'keypoint_alignment': {'nmi': 100 * normalized_mutual_info_score(kp_true, kp_part),
                               'ari': 100 * adjusted_rand_score(kp_true, kp_part), 'keypoints': len(kp_true),
                               'parts_used': int(len(set(kp_part.tolist())))},
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--checkpoint', required=True)
    cli.add_argument('--maps', required=True, help='the part-map folder this model was trained with')
    cli.add_argument('--out_dir', required=True)
    cli.add_argument('--num_workers', type=int, default=4)
    opt = cli.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_net(opt.checkpoint, device)
    report = {'checkpoint': opt.checkpoint, 'maps': opt.maps, **evaluate(net, opt.maps, device, opt.num_workers)}
    i, k = report['iou_own_masks'], report['keypoint_alignment']
    print(f"IoU with own masks {i['iou']:.3f} (IoP {i['iop']:.3f}, IoT {i['iot']:.3f}) | keypoint NMI {k['nmi']:.1f}, "
          f"ARI {k['ari']:.1f} over {k['keypoints']} keypoints, {k['parts_used']} parts used")
    c = report['classification']
    print(f"top-1 {c['lucid_top1']:.4f} (ShapeTexNet {c['shape_top1']:.4f}) | top-5 {c['lucid_top5']:.4f} "
          f"(ShapeTexNet {c['shape_top5']:.4f})")
    os.makedirs(opt.out_dir, exist_ok=True)
    with open(os.path.join(opt.out_dir, 'eval_full.json'), 'w') as f:
        json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
