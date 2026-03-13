import json
from collections import defaultdict
from tabulate import tabulate
import os

# ==========================
# CONFIG
# ==========================
DATASET_ROOT = "/Users/colby/Desktop/military tracking custom/military_object_dataset"
COCO_JSON_PATH = os.path.join(DATASET_ROOT, "custom_subset", "custom_annotations.json")

# ==========================
# LOAD JSON
# ==========================
if not os.path.exists(COCO_JSON_PATH):
    raise FileNotFoundError(f"COCO JSON not found: {COCO_JSON_PATH}")

with open(COCO_JSON_PATH, "r") as f:
    coco = json.load(f)

# Build class ID -> name mapping
class_map = {c["id"]: c["name"] for c in coco["categories"]}

# ==========================
# COUNT ANNOTATIONS AND UNIQUE IMAGES
# ==========================
ann_counts = defaultdict(int)
images_per_class = defaultdict(set)

for ann in coco["annotations"]:
    cid = ann["category_id"]
    ann_counts[cid] += 1
    images_per_class[cid].add(ann["image_id"])

# ==========================
# BUILD TABLE
# ==========================
table = []
for cid, name in class_map.items():
    num_annotations = ann_counts.get(cid, 0)
    num_images = len(images_per_class.get(cid, []))
    table.append([cid, name, num_images, num_annotations])

# Sort by class ID
table.sort(key=lambda x: x[0])

print("\nClass-level statistics for COCO JSON:\n")
print(tabulate(table, headers=["class_id", "class_name", "num_images", "num_annotations"], tablefmt="github"))
