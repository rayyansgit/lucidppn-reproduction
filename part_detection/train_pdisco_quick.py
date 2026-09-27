import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
import numpy as np
from torchvision.models import resnet101, ResNet101_Weights
from torch.utils.tensorboard import SummaryWriter
from datasets import CUBDataset
from nets import IndividualLandmarkNet
from train import train, validation
import wandb
wandb.init(mode="disabled")  # avoid login prompts / hangs; some internal calls may expect wandb to exist

os.makedirs('cub_quick', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(1)

N_CLASSES = 25
IMAGE_SIZE = 224
EPOCHS = 5
NUM_PARTS = 8
BATCH_SIZE = 16

print("Loading full CUB metadata, filtering to 25 classes...")
dataset_train = CUBDataset('../CUB_200_2011', split=1.0, mode='train', image_size=IMAGE_SIZE)
mask = dataset_train.labels < N_CLASSES
dataset_train.ids = dataset_train.ids[mask]
dataset_train.names = dataset_train.names[mask]
dataset_train.labels = dataset_train.labels[mask]
dataset_train.parts = {i: dataset_train.parts[i] for i in dataset_train.ids}

dataset_val = CUBDataset('../CUB_200_2011', mode='test', train_samples=dataset_train.trainsamples, image_size=IMAGE_SIZE)
mask_val = dataset_val.labels < N_CLASSES
dataset_val.ids = dataset_val.ids[mask_val]
dataset_val.names = dataset_val.names[mask_val]
dataset_val.labels = dataset_val.labels[mask_val]
dataset_val.parts = {i: dataset_val.parts[i] for i in dataset_val.ids}

print(f"Train: {len(dataset_train)} images | Val: {len(dataset_val)} images")

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=8, shuffle=True, num_workers=2)

basenet = resnet101(weights=ResNet101_Weights.DEFAULT)
net = IndividualLandmarkNet(basenet, NUM_PARTS, num_classes=N_CLASSES).to(device)

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

writer = SummaryWriter(log_dir='cub_quick/tb')
all_losses = []
print(f"Training PDiscoNet for {EPOCHS} epochs on {N_CLASSES} classes...")
for epoch in range(EPOCHS):
    if all_losses:
        net, all_losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn, loss_hyperparams, writer, all_losses)
    else:
        net, all_losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn, loss_hyperparams, writer)
    scheduler.step()
    print(f'Validation after epoch {epoch}:')
    validation(device, net, val_loader, epoch, 'cub_quick_25class', False, writer)
    torch.cuda.empty_cache()
    torch.save(net.state_dict(), 'cub_quick/cub_quick_25class.pt')

writer.close()
print("PDiscoNet training done. Checkpoint saved to cub_quick/cub_quick_25class.pt")
