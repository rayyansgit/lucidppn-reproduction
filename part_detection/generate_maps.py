"""
Write PDiscoNet part maps for the 25-class CUB subset from any PDiscoNet checkpoint.
Same as generate_maps_quick.py (Stage 3, left unchanged), with command-line options and a __main__ guard.

Usage (from part_detection/):
    python generate_maps.py --checkpoint cub_quick/cub_quick_25class.pt --out ../CUB_subset_maps
    python generate_maps.py --checkpoint ../experiments/part_detector/pdisco_28ep/pdisco.pt --out ../CUB_subset_maps_pdisco28
"""
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torchvision.models import resnet101
from nets import IndividualLandmarkNet
from save_maps import CUBDataset, save_maps  # upstream classes/function, reused as-is

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--checkpoint', required=True)
    cli.add_argument('--out', required=True, help='maps root; train/ and test/ are created inside')
    cli.add_argument('--n_classes', type=int, default=25)
    cli.add_argument('--num_parts', type=int, default=8)
    cli.add_argument('--image_size', type=int, default=224)
    cli.add_argument('--num_workers', type=int, default=4)
    opt = cli.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = IndividualLandmarkNet(resnet101(), opt.num_parts, num_classes=opt.n_classes)
    net.load_state_dict(torch.load(opt.checkpoint, map_location=device))
    net.to(device).eval()

    for split in ['train', 'test']:
        dataset = CUBDataset(os.path.join(ROOT, 'CUB_subset', split), image_size=opt.image_size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
        print(f"Generating maps for {split} set ({len(dataset)} images)...")
        save_maps(loader, net, device, os.path.join(opt.out, split))
    print(f"Done. Maps saved to {opt.out}/{{train,test}}/0..{opt.num_parts - 1}/<class>/<image>")


if __name__ == '__main__':
    main()
