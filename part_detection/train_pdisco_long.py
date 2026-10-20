"""
Train PDiscoNet on the 25-class CUB subset for a configurable number of epochs (paper: 28).
Same setup as train_pdisco_quick.py (Stage 3, 5 epochs), which is left unchanged; this version adds command-line
options, an output folder, and a __main__ guard so data-loading workers also work on Windows.

Usage (from part_detection/):
    python train_pdisco_long.py --epochs 28 --out_dir ../experiments/part_detector/pdisco_28ep
Writes <out_dir>/pdisco.pt, <out_dir>/config.json and TensorBoard logs in <out_dir>/tensorboard.
"""
import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.tensorboard import SummaryWriter
from datasets import CUBDataset
from nets import IndividualLandmarkNet
from train import train, validation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def filter_classes(dataset, n_classes):
    mask = dataset.labels < n_classes
    dataset.ids = dataset.ids[mask]
    dataset.names = dataset.names[mask]
    dataset.labels = dataset.labels[mask]
    dataset.parts = {i: dataset.parts[i] for i in dataset.ids}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--epochs', type=int, default=28)
    cli.add_argument('--n_classes', type=int, default=25)
    cli.add_argument('--num_parts', type=int, default=8)
    cli.add_argument('--image_size', type=int, default=224)
    cli.add_argument('--batch_size', type=int, default=16)
    cli.add_argument('--seed', type=int, default=1)
    cli.add_argument('--num_workers', type=int, default=4)
    cli.add_argument('--out_dir', required=True)
    opt = cli.parse_args()

    out = os.path.abspath(opt.out_dir)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=1)
    cub = os.path.join(ROOT, 'CUB_200_2011')
    os.chdir(out)  # the upstream CUB loader writes testset.tsv into the working directory
    wandb_stub()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    dataset_train = CUBDataset(cub, split=1.0, mode='train', image_size=opt.image_size)
    filter_classes(dataset_train, opt.n_classes)
    dataset_val = CUBDataset(cub, mode='test', train_samples=dataset_train.trainsamples, image_size=opt.image_size)
    filter_classes(dataset_val, opt.n_classes)
    print(f"Train: {len(dataset_train)} images | Val: {len(dataset_val)} images")

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=8, shuffle=True, num_workers=opt.num_workers)

    net = IndividualLandmarkNet(resnet101(weights=ResNet101_Weights.DEFAULT), opt.num_parts,
                                num_classes=opt.n_classes).to(device)

    # Optimiser, learning-rate groups, schedule and loss weights: identical to train_pdisco_quick.py / upstream main.py
    high_lr_layers, med_lr_layers = ["modulation"], ["fc_class_landmarks"]
    lr = 1e-4
    param_dict = [{'params': [], 'lr': lr * 100}, {'params': [], 'lr': lr * 10}, {'params': [], 'lr': lr}]
    for name, p in net.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in high_lr_layers: param_dict[0]['params'].append(p)
        elif layer_name in med_lr_layers: param_dict[1]['params'].append(p)
        else: param_dict[2]['params'].append(p)
    optimizer = torch.optim.Adam(params=param_dict)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_hyperparams = {'l_class': 2, 'l_pres': 1, 'l_equiv': 1, 'l_conc': 1000, 'l_orth': 1}

    writer = SummaryWriter(log_dir=os.path.join(out, 'tensorboard'))
    all_losses = []
    print(f"Training PDiscoNet for {opt.epochs} epochs on {opt.n_classes} classes...")
    for epoch in range(opt.epochs):
        if all_losses:
            net, all_losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn, loss_hyperparams, writer,
                                    all_losses)
        else:
            net, all_losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn, loss_hyperparams, writer)
        scheduler.step()
        print(f'Validation after epoch {epoch} (lr now {scheduler.get_last_lr()[-1]:.2e}):', flush=True)
        validation(device, net, val_loader, epoch, 'pdisco', False, writer)
        torch.cuda.empty_cache()
        torch.save(net.state_dict(), os.path.join(out, 'pdisco.pt'))
    writer.close()
    print(f"PDiscoNet training done. Checkpoint saved to {os.path.join(out, 'pdisco.pt')}")


def wandb_stub():
    try:
        import wandb
        wandb.init(mode="disabled")
    except ImportError:
        pass


if __name__ == '__main__':
    main()
