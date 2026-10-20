"""
Inference wrapper around the trained MetiNet (LucidPPN) checkpoint, shared by the Streamlit app and the FastAPI API.

Preprocessing matches the test-time pipeline in MetiNet/util/data.py exactly:
  Resize(224, 224) -> RGB tensor in [0, 1] for ColorNet, and Grayscale(3) + Normalize(0.445, 0.269) for ShapeTexNet.
Part-segmentation maps are only used by the training loss, never at inference, so a single image is all we need.
"""
import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, 'MetiNet'))
from metinet.metinet import MetiNet, get_network  # noqa: E402

# The first 25 CUB-200-2011 classes, named as prepare_cub_subset.py names the folders. ImageFolder sorts folder
# names, so the model's class index i corresponds to sorted(CUB_25_CLASSES)[i].
CUB_25_CLASSES = [
    'Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet',
    'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird',
    'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting',
    'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow',
    'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant',
]
CLASSES = sorted(CUB_25_CLASSES)
NUM_PARTS = 8
IMG_SIZE = 224

DEFAULT_CHECKPOINT = os.environ.get('LUCID_CHECKPOINT', os.path.join(ROOT, 'MetiNet', 'checkpoint_quick_train.pt'))
PROTOTYPE_DIR = os.path.join(ROOT, 'deploy', 'prototypes')
PATCH = 64                # side of the image patch shown around a prototype's strongest feature-map cell

_resize = transforms.Resize(size=(IMG_SIZE, IMG_SIZE))
_gray_norm = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Normalize(mean=(0.445, 0.445, 0.445), std=(0.269, 0.269, 0.269)),
])


def pretty(name):
    return name.replace('_', ' ')


def load_model(checkpoint=DEFAULT_CHECKPOINT, device='cpu'):
    args = argparse.Namespace(net='resnet18', disable_pretrained=True, num_classes=len(CLASSES),
                              num_parts=NUM_PARTS, bias=False)
    feature_net, add_on, pool, classification, color_net, _ = get_network(len(CLASSES), args)
    net = MetiNet(len(CLASSES), NUM_PARTS, feature_net, add_on, pool, classification, color_net)
    state = torch.load(checkpoint, map_location='cpu')
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    net.load_state_dict(state)
    return net.to(device).eval()


def shift_hue(image: Image.Image, shift: float) -> Image.Image:
    """Rotate hue by `shift` (in [-0.5, 0.5]) while keeping per-pixel luminance, as in the authors' random_hue_jitter
    (deterministic here so the demo is repeatable)."""
    if shift == 0:
        return image
    x = TF.to_tensor(image.convert('RGB'))
    gray = TF.rgb_to_grayscale(x, 3)
    y = TF.adjust_hue(x, shift)
    y = torch.nan_to_num(y * gray / TF.rgb_to_grayscale(y, 3), nan=0.0).clamp(0, 1)
    return TF.to_pil_image(y)


def crop_patch(image224: Image.Image, row: int, col: int, grid: int) -> Image.Image:
    """The PATCH x PATCH region of a 224x224 image centred on cell (row, col) of a grid x grid feature map (28x28 for
    the authors' ResNet18, whose last two stages use stride 1), kept inside the image."""
    cell = IMG_SIZE / grid
    cx, cy = (col + 0.5) * cell, (row + 0.5) * cell
    left = int(min(max(cx - PATCH / 2, 0), IMG_SIZE - PATCH))
    top = int(min(max(cy - PATCH / 2, 0), IMG_SIZE - PATCH))
    return image224.crop((left, top, left + PATCH, top + PATCH))


def load_prototypes():
    """Prototype patches made by build_prototypes.py: {species: {part (1-8): {file, source, activation}}}, or {}."""
    path = os.path.join(PROTOTYPE_DIR, 'index.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)['prototypes']


def preprocess(image: Image.Image):
    image = _resize(image.convert('RGB'))
    x = TF.to_tensor(image)          # ColorNet input, RGB in [0, 1]
    x_aug = _gray_norm(x)            # ShapeTexNet input, grayscale normalised
    return x.unsqueeze(0), x_aug.unsqueeze(0), image


@torch.no_grad()
def predict(net, image: Image.Image, top_k=5, hue_shift=0.0):
    """Returns a JSON-serialisable dict with the top-k classes and, for the top class, per-part evidence."""
    device = next(net.parameters()).device
    image = shift_hue(image, hue_shift)
    x, x_aug, resized = preprocess(image)
    proto_feat, proto_pooled, color_feat, color_pooled, _, out = net(x.to(device), x_aug.to(device), None)
    # out: LucidPPN score per class (mean over parts of the fused colour x shape resemblance)
    # proto_pooled: ShapeTexNet (grayscale) resemblance per part and class; its mean over parts is P_S
    lucid = out[0]
    shape = proto_pooled[0].mean(dim=0)
    # The model scores each species independently (one sigmoid per species, trained with binary cross-entropy), so
    # the scores do not sum to 1. "Share of evidence" normalises them over the 25 species: it sums to 100% and reads
    # like a probability, but it is not a calibrated one.
    share = lucid / lucid.sum()
    order = torch.argsort(lucid, descending=True)[:top_k].tolist()
    top = [{'class': pretty(CLASSES[c]), 'class_index': c, 'lucidppn_score': round(lucid[c].item(), 4),
            'share_of_evidence': round(share[c].item(), 4),
            'shapetexnet_score': round(shape[c].item(), 4)} for c in order]
    c = order[0]
    # Strongest cell of each part's prototype map for the predicted class: where on the bird the evidence is
    grid = proto_feat.shape[-1]
    peak = proto_feat[0, :, c].flatten(1).argmax(dim=1)
    parts = [{'part': k + 1,
              'shape_texture': round(proto_pooled[0, k, c].item(), 4),
              'shape_plus_colour': round(color_pooled[0, k, c].item(), 4),
              'location': [int(peak[k]) // grid, int(peak[k]) % grid], 'grid': grid} for k in range(NUM_PARTS)]
    # Where each part's prototype of the predicted class fires: ShapeTexNet map, upsampled to the image size
    maps = F.interpolate(proto_feat[0, :, c].unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear',
                         align_corners=False)[0].cpu()
    return {
        'prediction': pretty(CLASSES[c]),
        'shapetexnet_prediction': pretty(CLASSES[int(torch.argmax(shape))]),
        'hue_shift': hue_shift,
        'top_k': top,
        'parts': parts,
    }, maps, resized
