import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# CONFIGURATION
# ==========================================================

REAL_PATH = "real_data/military_object_dataset/custom_subset/custom_annotations.json"
SYN_PATH = "synthetic_data_v1/annotations.json"
OUTPUT_FILE = "category_count_paired.pdf"

# Load canonical categories
CATEGORIES = [
    {"id": 2, "name": "military_tank"},
    {"id": 3, "name": "military_truck"}
]

# Create a mapping id -> name
CATEGORY_MAPPING = {cat["id"]: cat["name"] for cat in CATEGORIES}

# Colors for GT and Synthetic
GT_COLOR = "red"
SYN_COLOR = "green"

# ==========================================================
# HELPER FUNCTION
# ==========================================================

def count_categories(annotations_path):
    with open(annotations_path, "r") as f:
        data = json.load(f)
    counts = Counter()
    for ann in data["annotations"]:
        cat_name = CATEGORY_MAPPING.get(ann["category_id"])
        if cat_name:
            counts[cat_name] += 1
    return counts

# ==========================================================
# COUNT CATEGORIES
# ==========================================================

real_counts = count_categories(REAL_PATH)
syn_counts = count_categories(SYN_PATH)

# Use the canonical category order
all_categories = [cat["name"] for cat in CATEGORIES]

real_values = [real_counts.get(cat, 0) for cat in all_categories]
syn_values  = [syn_counts.get(cat, 0) for cat in all_categories]

# ==========================================================
# PLOT PAIRED BAR CHART
# ==========================================================

x = np.arange(len(all_categories))
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - width/2, real_values, width, label="Ground Truth", color=GT_COLOR, alpha=0.8)
plt.bar(x + width/2, syn_values, width, label="Synthetic", color=SYN_COLOR, alpha=0.8)

plt.xticks(x, all_categories, rotation=0)
plt.ylabel("Number of Annotations")
plt.title("Ground Truth vs Synthetic Category Counts")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=600)
print(f"Saved paired category count figure to {OUTPUT_FILE}")
plt.show()
