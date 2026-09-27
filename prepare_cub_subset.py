import os
from PIL import Image

ROOT = "CUB_200_2011"
OUT = "CUB_subset"
N_CLASSES = 8

with open(f"{ROOT}/images.txt") as f:
    images = dict(line.strip().split(" ", 1) for line in f)
with open(f"{ROOT}/image_class_labels.txt") as f:
    labels = dict(line.strip().split(" ") for line in f)
with open(f"{ROOT}/train_test_split.txt") as f:
    split = dict(line.strip().split(" ") for line in f)
with open(f"{ROOT}/bounding_boxes.txt") as f:
    bboxes = {}
    for line in f:
        parts = line.strip().split(" ")
        bboxes[parts[0]] = [float(x) for x in parts[1:]]
with open(f"{ROOT}/classes.txt") as f:
    class_names = dict(line.strip().split(" ", 1) for line in f)

chosen_classes = sorted(set(labels.values()), key=int)[:N_CLASSES]
print("Using classes:", [class_names[c] for c in chosen_classes])

count = 0
for img_id, rel_path in images.items():
    cls = labels[img_id]
    if cls not in chosen_classes:
        continue
    subset = "train" if split[img_id] == "1" else "test"
    class_name = class_names[cls].split(".", 1)[1]
    out_dir = f"{OUT}/{subset}/{class_name}"
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(f"{ROOT}/images/{rel_path}").convert("RGB")
    x, y, w, h = bboxes[img_id]
    cropped = img.crop((x, y, x + w, y + h))
    cropped.save(f"{out_dir}/{os.path.basename(rel_path)}")
    count += 1

print(f"Done. {count} images saved to {OUT}/")
