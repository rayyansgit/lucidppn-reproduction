"""
Finds, for every species and every part, the training-image patch where that part's prototype fires most strongly:
the "that" in LucidPPN's "this part of your bird looks like *that* part of a training bird". Only training images of
the species itself are searched, so each patch shows part k as it looks on that species.

Run once from the repository root after training (needs CUB_subset/train from prepare_cub_subset.py):
    python deploy/build_prototypes.py
Writes deploy/prototypes/<Species>/part<k>.jpg (64x64 patches from the 224x224 input) and deploy/prototypes/index.json.
The app shows them next to the matching patch of the uploaded image; they are committed, so the demo works without
the dataset.
"""
import glob
import json
import os
import shutil
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lucid_inference import (CLASSES, DEFAULT_CHECKPOINT, NUM_PARTS, PROTOTYPE_DIR, ROOT, crop_patch,  # noqa: E402
                             load_model, preprocess)


@torch.no_grad()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = load_model(DEFAULT_CHECKPOINT, device)
    best = {}  # (species index, part) -> (activation, image path, row, col, grid)
    for c, species in enumerate(CLASSES):
        paths = sorted(glob.glob(os.path.join(ROOT, 'CUB_subset', 'train', species, '*.jpg')))
        if not paths:
            raise SystemExit(f'No training images for {species}: run prepare_cub_subset.py first')
        for path in paths:
            x, x_aug, _ = preprocess(Image.open(path))
            proto_feat = net(x.to(device), x_aug.to(device), None)[0]   # [1, parts, classes, grid, grid]
            grid = proto_feat.shape[-1]                                  # 28 for the authors' ResNet18
            maps = proto_feat[0, :, c].flatten(1)                        # this species' prototypes: [parts, grid^2]
            values, cells = maps.max(dim=1)
            for k in range(NUM_PARTS):
                if values[k] > best.get((c, k), (-1,))[0]:
                    best[(c, k)] = (values[k].item(), path, int(cells[k]) // grid, int(cells[k]) % grid, grid)
        print(f'{species}: {len(paths)} training images searched')

    shutil.rmtree(PROTOTYPE_DIR, ignore_errors=True)
    index = {}
    for (c, k), (value, path, row, col, grid) in sorted(best.items()):
        species = CLASSES[c]
        os.makedirs(os.path.join(PROTOTYPE_DIR, species), exist_ok=True)
        _, _, image224 = preprocess(Image.open(path))
        name = f'part{k + 1}.jpg'
        crop_patch(image224, row, col, grid).save(os.path.join(PROTOTYPE_DIR, species, name), quality=92)
        index.setdefault(species, {})[str(k + 1)] = {
            'file': f'{species}/{name}', 'source': os.path.relpath(path, ROOT).replace('\\', '/'),
            'activation': round(value, 4), 'location': [row, col], 'grid': grid}
    with open(os.path.join(PROTOTYPE_DIR, 'index.json'), 'w') as f:
        json.dump({'checkpoint': os.path.relpath(DEFAULT_CHECKPOINT, ROOT).replace('\\', '/'), 'prototypes': index},
                  f, indent=1)
    print(f'Saved {len(best)} prototype patches to {PROTOTYPE_DIR}')


if __name__ == '__main__':
    main()
