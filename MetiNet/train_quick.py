import sys, os
sys.path.append(os.path.dirname(__file__))
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from util.data import create_datasets
from util.func import init_weights_xavier
from metinet.metinet import MetiNet, get_network
from metinet.train import train_metinet
from util.func import topk_accuracy

args = argparse.Namespace(
    net='resnet18', disable_pretrained=False, num_classes=25, num_parts=8, bias=False,
    optimizer='Adam', lr_net=0.0005, lr_class=0.05, lr_color=0.0005, weight_decay=0.0,
    part_weight=1.0, proto_class_weight=1.0, color_class_weight=1.0,
    use_classification_layer=False, aggregate='mean',
    epochs=80, freeze_epochs=10, no_color_epochs=2, batch_size=32, seed=1, disable_cuda=False,
)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

print("Loading data...")
trainset, projectset, testset, classes, train_indices, targets = create_datasets(
    train_dir='../CUB_subset/train', project_dir='../CUB_subset/train', test_dir='../CUB_subset/test',
    maps_train_dir='../CUB_subset_maps/train', maps_test_dir='../CUB_subset_maps/test', img_size=224, num_parts=args.num_parts,
    augment_shape_zoom=8, strong_hue_augmentation=0.0, crop_augmentation=1.0,
)
print(f"{len(classes)} classes, {len(trainset)} train images, {len(testset)} test images")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() and not args.disable_cuda else 'cpu')
print(f"Device: {device}")

print("Building model...")
feature_net, add_on_layers, pool_layer, classification_layer, color_net, num_prototypes = get_network(args.num_classes, args)
net = MetiNet(num_classes=args.num_classes, num_parts=args.num_parts, feature_net=feature_net,
             add_on_layers=add_on_layers, pool_layer=pool_layer,
             classification_layer=classification_layer, color_net=color_net).to(device)
net = nn.DataParallel(net, device_ids=[0] if device.type == 'cuda' else None)
net.module._add_on.apply(init_weights_xavier)
torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)

params_to_train, params_backbone = [], []
for name, param in net.module._net.named_parameters():
    (params_to_train if 'layer4.2' in name else params_backbone).append(param)
for param in net.module._add_on.parameters():
    params_to_train.append(param)
params_color = list(net.module._color_net.parameters())
classification_weight = [p for n, p in net.module._classification.named_parameters() if 'weight' in n]

optimizer_net = torch.optim.AdamW([
    {"params": params_backbone, "lr": args.lr_net}, {"params": params_to_train, "lr": args.lr_net}],
    lr=args.lr_net, weight_decay=args.weight_decay)
optimizer_classifier = torch.optim.AdamW([{"params": classification_weight, "lr": args.lr_class}],
    lr=args.lr_class, weight_decay=args.weight_decay)
optimizer_color = torch.optim.AdamW([{"params": params_color, "lr": args.lr_color}],
    lr=args.lr_color, weight_decay=args.weight_decay)

scheduler_net = torch.optim.lr_scheduler.LambdaLR(optimizer_net,
    lr_lambda=lambda e: 0 if e > args.epochs else (0.1 if e > args.freeze_epochs else 1))
scheduler_color = torch.optim.lr_scheduler.LambdaLR(optimizer_color,
    lr_lambda=lambda e: 1 if e > args.no_color_epochs else 0)
scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001)

part_criterion = nn.BCELoss(reduction='mean').to(device)
class_criterion = nn.BCELoss(reduction='mean').to(device)
writer = SummaryWriter(log_dir='runs/quick_train')

backbone_frozen, no_color = True, True
for p in net.module._classification.parameters(): p.requires_grad = True
for p in net.module._color_net.parameters(): p.requires_grad = True
for p in params_to_train: p.requires_grad = True
for p in params_backbone: p.requires_grad = False

total_epochs = args.epochs + args.no_color_epochs
print(f"Training for {total_epochs} epochs WITH part-detection maps (part_weight=1.0)...")
for epoch in range(1, total_epochs + 1):
    if epoch > args.freeze_epochs and backbone_frozen:
        for p in params_backbone: p.requires_grad = True
        backbone_frozen = False
    if epoch > args.no_color_epochs and no_color:
        no_color = False

    train_info = train_metinet(net, trainloader, optimizer_net, optimizer_classifier, optimizer_color,
                               scheduler_net, scheduler_classifier, scheduler_color, no_color,
                               part_criterion, class_criterion, epoch, device,
                               args.use_classification_layer, args.part_weight, args.proto_class_weight,
                               args.color_class_weight, args.num_classes, args.num_parts, args.aggregate)
    net.eval()
    top1_total, top5_total, n_total = 0., 0., 0
    with torch.no_grad():
        for x, x_aug, m, y in testloader:
            x, x_aug, y = x.to(device), x_aug.to(device), y.to(device)
            _, _, _, _, _, out = net(x, x_aug, m, args.use_classification_layer, args.aggregate)
            top1, top5 = topk_accuracy(out, y, topk=[1, 5])
            top1_total += torch.sum(top1).item()
            top5_total += torch.sum(top5).item()
            n_total += len(y)
    eval_info = {'top1_accuracy/ensemble': top1_total / n_total, 'top5_accuracy/ensemble': top5_total / n_total}
    net.train()

    train_acc, test_acc, loss = train_info['train_accuracy/mean'], eval_info['top1_accuracy/ensemble'], train_info['loss/mean']
    print(f"Epoch {epoch}/{total_epochs} | loss={loss:.4f} | train_acc={train_acc:.4f} | test_top1={test_acc:.4f}")
    writer.add_scalar('loss/train', loss, epoch)
    writer.add_scalar('accuracy/train', train_acc, epoch)
    writer.add_scalar('accuracy/test_top1', test_acc, epoch)

writer.close()
torch.save(net.state_dict(), 'checkpoint_quick_train.pt')
print(f"\nDone. Final test top-1 accuracy: {test_acc:.4f}")
