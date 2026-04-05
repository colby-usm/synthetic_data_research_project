import json
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================================================
# CONFIG: JSON paths for each dataset
# ==========================================================

REFCOCO_JSON = "data/ref_coco_train.json"            # Reference COCO training annotations
SYNTHETIC_JSON = "data/synthetic_annotations.json"  # Synthetic dataset annotations
GT_JSON = "data/gt_annotations.json"                # Ground truth dataset annotations

# Novel classes to highlight
NOVEL_CLASSES = ["novel_class1", "novel_class2"]

# Plotting settings
OUTPUT_FILE = "novel_classes_vs_refcoco.pdf"
FIGSIZE = (6, 6)
LOG_SCALE = False  # Set True if counts vary a lot

# ==========================================================
# FUNCTIONS
# ==========================================================

def get_class_counts(coco_path):
    """Return a dictionary mapping class name -> number of instances."""
    with open(coco_path, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    counts = defaultdict(int)
    for ann in coco["annotations"]:
        counts[cat_id_to_name[ann["category_id"]]] += 1
    return counts

# ==========================================================
# LOAD DATA
# ==========================================================

ref_counts = get_class_counts(REFCOCO_JSON)
synthetic_counts = get_class_counts(SYNTHETIC_JSON)
gt_counts = get_class_counts(GT_JSON)

# Values for plotting
ref_values = list(ref_counts.values())

# Get counts for novel classes (use 0 if missing)
novel_synthetic_values = [synthetic_counts.get(cls, 0) for cls in NOVEL_CLASSES]
novel_gt_values = [gt_counts.get(cls, 0) for cls in NOVEL_CLASSES]

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=FIGSIZE)

# Boxplot for RefCOCO
plt.boxplot(ref_values, vert=True, widths=0.6, patch_artist=True, boxprops=dict(facecolor='lightblue'))

# Scatter for Synthetic novel classes
plt.scatter([1]*len(NOVEL_CLASSES), novel_synthetic_values,
            color='green', label='Synthetic Novel Classes', zorder=5, s=60)

# Scatter for GT novel classes
plt.scatter([1]*len(NOVEL_CLASSES), novel_gt_values,
            color='red', label='GT Novel Classes', zorder=6, s=60, marker='x')

# Optional: annotate points with class names
for i, cls in enumerate(NOVEL_CLASSES):
    plt.text(1.05, novel_synthetic_values[i], cls, color='green', va='center')
    plt.text(1.05, novel_gt_values[i], cls, color='red', va='center')

# Axis labels, title, legend
plt.ylabel("Number of Instances per Class")
plt.xticks([1], ["RefCOCO"])
plt.title("Novel Classes vs RefCOCO Class Distribution")
plt.legend()

# Optional log scale
if LOG_SCALE:
    plt.yscale("log")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=600)
print(f"Saved figure to: {OUTPUT_FILE}")
plt.show()
