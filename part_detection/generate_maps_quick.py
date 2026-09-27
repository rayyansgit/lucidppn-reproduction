import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
from torchvision.models import resnet101, ResNet101_Weights
from nets import IndividualLandmarkNet
from save_maps import CUBDataset, save_maps  # reuse their real classes/function

N_CLASSES = 25
NUM_PARTS = 8
IMAGE_SIZE = 224
CHECKPOINT = 'cub_quick/cub_quick_25class.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_train = CUBDataset('../CUB_subset/train', image_size=IMAGE_SIZE)
dataset_test = CUBDataset('../CUB_subset/test', image_size=IMAGE_SIZE)
print(f"Train images: {len(dataset_train)} | Test images: {len(dataset_test)}")

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)

basenet = resnet101(weights=ResNet101_Weights.DEFAULT)
net = IndividualLandmarkNet(basenet, NUM_PARTS, num_classes=N_CLASSES)
net.load_state_dict(torch.load(CHECKPOINT, map_location=device))
net.to(device)
net.eval()

print("Generating maps for train set...")
save_maps(train_loader, net, device, '../CUB_subset_maps/train')
print("Generating maps for test set...")
save_maps(test_loader, net, device, '../CUB_subset_maps/test')
print("Done. Maps saved to CUB_subset_maps/{train,test}/0..7/<class>/<image>")
