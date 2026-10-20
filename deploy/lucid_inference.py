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
import colorsys

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


def shift_hue(x: torch.Tensor, shift: float) -> torch.Tensor:
    """Rotate the hue of an RGB tensor (3 x H x W, values in [0, 1]) by `shift` (in [-0.5, 0.5]) and rescale each
    pixel so its luminance is exactly the original's: the authors' random_hue_jitter (util/func.py) with a fixed
    instead of a random shift. As there, the result is not clipped to [0, 1]; clipping would change the luminance
    of saturated pixels and with it what the shape branch sees."""
    gray = TF.rgb_to_grayscale(x, 3)
    y = TF.adjust_hue(x, shift)
    return torch.nan_to_num(y * gray / TF.rgb_to_grayscale(y, 3), nan=0.0)


def patch_box(row: int, col: int, grid: int):
    """(left, top, right, bottom) of the PATCH x PATCH region of a 224x224 image centred on cell (row, col) of a
    grid x grid feature map (28x28 for the authors' ResNet18, whose last two stages use stride 1), kept inside the
    image."""
    cell = IMG_SIZE / grid
    cx, cy = (col + 0.5) * cell, (row + 0.5) * cell
    left = int(min(max(cx - PATCH / 2, 0), IMG_SIZE - PATCH))
    top = int(min(max(cy - PATCH / 2, 0), IMG_SIZE - PATCH))
    return left, top, left + PATCH, top + PATCH


def crop_patch(image224: Image.Image, row: int, col: int, grid: int) -> Image.Image:
    return image224.crop(patch_box(row, col, grid))


# ColorNet scores one RGB value at a time, so its whole behaviour can be read off by sweeping the colour cube.
_COLOUR_GRID = torch.cartesian_prod(*[torch.linspace(0, 1, 16)] * 3)       # 4,096 colours, RGB in [0, 1]
ACCEPT = 0.5


@torch.no_grad()
def colour_acceptance(net):
    """ColorNet output for every colour of a 16^3 RGB grid: [4096, parts, classes]. Precomputed once per model."""
    device = next(net.parameters()).device
    out = net._color_net(_COLOUR_GRID.to(device)).cpu()
    return out.unflatten(1, (NUM_PARTS, len(CLASSES)))


def accepted_colours(acceptance, part: int, cls: int, max_n: int = 40):
    """Colours that ColorNet accepts (score > 0.5) for part `part` (0-based) of class `cls`, as RGB 0-255 tuples sorted
    by hue, plus the fraction of the colour cube accepted. A fraction near 1 means colour has no say for that part."""
    s = acceptance[:, part, cls]
    ok = s > ACCEPT
    frac = ok.float().mean().item()
    cols = _COLOUR_GRID[ok]
    if len(cols) > max_n:
        cols = cols[torch.linspace(0, len(cols) - 1, max_n).long()]
    hsv = [colorsys.rgb_to_hsv(*c.tolist()) for c in cols]
    order = sorted(range(len(cols)), key=lambda i: (round(hsv[i][0] * 12) if hsv[i][1] > 0.15 else -1, hsv[i][2]))
    return [tuple(int(round(255 * v)) for v in cols[i].tolist()) for i in order], frac


def shared_spots(parts, max_cells: int = 2):
    """Groups of parts whose strongest cells lie within `max_cells` cells (16 px at 28x28) of each other: those
    parts show the same image region, so they are not separate evidence."""
    n = len(parts)
    group = list(range(n))

    def find(i):
        while group[i] != i:
            i = group[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            (r1, c1), (r2, c2) = parts[i]['location'], parts[j]['location']
            if max(abs(r1 - r2), abs(c1 - c2)) <= max_cells:
                group[find(j)] = find(i)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(parts[i]['part'])
    return [sorted(c) for c in clusters.values() if len(c) > 1]


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
def predict(net, image: Image.Image, top_k=5, hue_shift=0.0, compare=None):
    """Returns a JSON-serialisable dict with the top-k classes, per-part evidence for the predicted class and the
    same evidence for a comparison class (`compare`, a class index; default: the runner-up), plus the ShapeTexNet
    maps of the predicted class upsampled to 224x224 and the 224x224 model input."""
    device = next(net.parameters()).device
    if not hasattr(net, '_colour_acceptance'):
        net._colour_acceptance = colour_acceptance(net)
    x, x_aug, resized = preprocess(image)
    if hue_shift:
        # As in the authors' evaluation: resize, then recolour. Only the colour branch's input changes; the luminance,
        # and so the shape branch's grayscale input, is the original image's.
        x = shift_hue(x[0], hue_shift).unsqueeze(0)
        resized = TF.to_pil_image(x[0].clamp(0, 1))     # for display only
    x, x_aug = x.to(device), x_aug.to(device)
    proto_feat, proto_pooled, color_feat, color_pooled, _, out = net(x, x_aug, None)
    # out: LucidPPN score per class (mean over parts of the fused colour x shape resemblance)
    # proto_pooled: ShapeTexNet (grayscale) resemblance per part and class; its mean over parts is P_S
    lucid = out[0]
    shape = proto_pooled[0].mean(dim=0)
    # The model scores each species independently (one sigmoid per species, trained with binary cross-entropy), so
    # the scores do not sum to 1. "Share of evidence" normalises them over the 25 species: it sums to 100% and reads
    # like a probability, but it is not a calibrated one.
    share = lucid / lucid.sum()
    ranked = torch.argsort(lucid, descending=True).tolist()
    top = [{'class': pretty(CLASSES[c]), 'class_index': c, 'lucidppn_score': round(lucid[c].item(), 4),
            'share_of_evidence': round(share[c].item(), 4),
            'shapetexnet_score': round(shape[c].item(), 4)} for c in ranked[:top_k]]

    # The image as ColorNet sees it: downsampled to the feature-map size, as in MetiNet.forward
    grid = proto_feat.shape[-1]
    pixels = (255 * TF.resize(x, [grid, grid], antialias=True)[0]).round().int().cpu()

    def evidence(c):
        """Per part, for class c:
        - shape_texture: ShapeTexNet resemblance (max over the image), at `location`;
        - shape_plus_colour: the fused score the prediction uses (max over the image of shape x colour);
        - colour_match: combined / shape, the share of the shape evidence that the colour check lets through
          (1 = colour fits, 0 = colour vetoes the part), so combined = shape x colour_match;
        - pixel_colour: the colour ColorNet saw (RGB of the 28x28 downsampled image) where the combined score peaks,
          or at the shape peak when colour vetoed the part;
        - colour_space_accepted: share of all colours ColorNet accepts for this part (1.0 = colour has no say)."""
        peak = proto_feat[0, :, c].flatten(1).argmax(dim=1)
        fused_peak = color_feat[0, :, c].flatten(1).argmax(dim=1)
        parts = []
        for k in range(NUM_PARTS):
            r, q = int(peak[k]) // grid, int(peak[k]) % grid
            shape_k, fused_k = proto_pooled[0, k, c].item(), color_pooled[0, k, c].item()
            match = min(1.0, fused_k / max(shape_k, 1e-6))
            spot = fused_peak[k] if match >= 0.1 else peak[k]
            _, frac = accepted_colours(net._colour_acceptance, k, c, max_n=0)
            parts.append({'part': k + 1,
                          'shape_texture': round(shape_k, 4),
                          'colour_match': round(match, 4),
                          'shape_plus_colour': round(fused_k, 4),
                          'location': [r, q], 'grid': grid,
                          'pixel_colour': pixels[:, int(spot) // grid, int(spot) % grid].tolist(),
                          'colour_space_accepted': round(frac, 3)})
        return parts

    c = ranked[0]
    other = ranked[1] if compare is None or compare == c else int(compare)
    parts = evidence(c)
    # Where each part's prototype of the predicted class fires: ShapeTexNet map, upsampled to the image size
    maps = F.interpolate(proto_feat[0, :, c].unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear',
                         align_corners=False)[0].cpu()
    return {
        'prediction': pretty(CLASSES[c]),
        'shapetexnet_prediction': pretty(CLASSES[int(torch.argmax(shape))]),
        'hue_shift': hue_shift,
        'top_k': top,
        'parts': parts,
        'parts_at_same_spot': shared_spots(parts),
        'comparison': {'class': pretty(CLASSES[other]), 'class_index': other,
                       'lucidppn_score': round(lucid[other].item(), 4),
                       'shapetexnet_score': round(shape[other].item(), 4),
                       'parts': evidence(other)},
    }, maps, resized
