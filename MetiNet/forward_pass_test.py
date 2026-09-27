import sys, os
sys.path.append(os.path.dirname(__file__))

import argparse
import torch
from util.data import create_datasets
from metinet.metinet import get_network, MetiNet

# --- minimal args, only what get_network() and create_datasets() need ---
args = argparse.Namespace(
    net='resnet18',              # smaller/faster than the paper's convnext_tiny_26
    disable_pretrained=False,    # download ImageNet-pretrained weights
    num_classes=8,               # matches our CUB_subset (8 species)
    num_parts=8,                 # paper default
    bias=False,
)

print("Building dataset from CUB_subset ...")
trainset, projectset, testset, classes, train_indices, targets = create_datasets(
    train_dir='../CUB_subset/train',
    project_dir='../CUB_subset/train',
    test_dir='../CUB_subset/test',
    maps_train_dir='',    # empty string -> dummy mode, no part-segmentation maps needed
    maps_test_dir='',
    img_size=224,
    num_parts=8,
    augment_shape_zoom=8,
    strong_hue_augmentation=0.0,
    crop_augmentation=1.0,
)
print(f"Classes found: {classes}")
print(f"Train set size: {len(trainset)}")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, drop_last=True)

print("Building MetiNet model (this downloads pretrained ResNet18 weights, may take a moment) ...")
feature_net, add_on_layers, pool_layer, classification_layer, color_net, num_prototypes = get_network(
    num_classes=8, args=args
)
net = MetiNet(
    num_classes=8,
    num_parts=8,
    feature_net=feature_net,
    add_on_layers=add_on_layers,
    pool_layer=pool_layer,
    classification_layer=classification_layer,
    color_net=color_net,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
print(f"Model built. Device: {device}")

print("Running one forward pass ...")
x, x_aug, m, y = next(iter(trainloader))
x, x_aug = x.to(device), x_aug.to(device)

with torch.no_grad():
    grouped_proto_features, grouped_proto_pooled, grouped_color_features, grouped_color_pooled, agg, out = net(x, x_aug, m)

print("FORWARD PASS SUCCEEDED")
print(f"Input batch shape: {x.shape}")
print(f"Output logits shape: {out.shape}")
print(f"Output values (first sample): {out[0]}")
print(f"Output contains NaN: {torch.isnan(out).any().item()}")
