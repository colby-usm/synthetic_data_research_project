# view_coco_class_stats.py

import json
from collections import defaultdict
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================

JSON_PATH = input("Enter path to COCO annotations JSON [default: data/RefCOCO/refcoco+/annotations.json]: ").strip()
if not JSON_PATH:
    JSON_PATH = "data/RefCOCO/refcoco+/annotations.json"

# ==========================================================
# LOAD JSON
# ==========================================================

with open(JSON_PATH, "r") as f:
    coco = json.load(f)

# ==========================================================
# VIEW TOP-LEVEL KEYS
# ==========================================================

print("\nTop-level keys in JSON:")
for key in coco.keys():
    print("-", key)

# ==========================================================
# MAP CATEGORY ID -> NAME
# ==========================================================

cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}

# ==========================================================
# COUNT INSTANCES PER CLASS, SPLIT BY TRAIN/VAL
# ==========================================================

train_counts = defaultdict(int)
val_counts = defaultdict(int)

for ann in coco.get("annotations", []):
    cat_name = cat_id_to_name.get(ann["category_id"], f"id_{ann['category_id']}")
    split = ann.get("split", "train")  # default to train if missing
    if split.lower() == "train":
        train_counts[cat_name] += 1
    elif split.lower() == "val":
        val_counts[cat_name] += 1
    else:
        # count unknown splits as train by default
        train_counts[cat_name] += 1

# ==========================================================
# FUNCTION TO PRINT 5-NUMBER SUMMARY
# ==========================================================

def print_summary(title, counts_dict):
    values = np.array(list(counts_dict.values()))
    if len(values) > 0:
        summary = np.percentile(values, [0, 25, 50, 75, 100])
        print(f"\n{title} 5-number summary of instances per class:")
        print(f"Min: {summary[0]:.0f}")
        print(f"25%: {summary[1]:.0f}")
        print(f"Median: {summary[2]:.0f}")
        print(f"75%: {summary[3]:.0f}")
        print(f"Max: {summary[4]:.0f}")
    else:
        print(f"\n{title}: No data available")

# ==========================================================
# PRINT SUMMARIES
# ==========================================================

print_summary("Train", train_counts)
print_summary("Validation", val_counts)

# ==========================================================
# PRINT FULL COUNTS PER CLASS
# ==========================================================

def print_counts(title, counts_dict):
    print(f"\n{title} class counts (sorted descending):")
    for cls, count in sorted(counts_dict.items(), key=lambda x: -x[1]):
        print(f"{cls}: {count}")

print_counts("Train", train_counts)
print_counts("Validation", val_counts)
