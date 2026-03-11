import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ==============================
# PATHS
# ==============================
REAL_PATH = "real_data/military_object_dataset/custom_subset/custom_annotations.json"
SYN_PATH = "synthetic_data_v1/annotations.json"
REFCOCO_JSON= "/Users/colby/Desktop/refcoco/refcoco/instances.json"
OUTPUT_FIG = "bbox_area_dataset_comparison.pdf"

# =========================================================
# FLAGS
# =========================================================
NORMALIZE_AREA = False

# =========================================================
# FUNCTION TO LOAD BBOX AREAS FROM COCO-STYLE JSON
# =========================================================
def load_coco_bbox_areas(path):
    with open(path, "r") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    areas = []

    for ann in data["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            print(f"Warning: image_id {ann['image_id']} not found. Skipping ann {ann['id']}")
            continue

        w, h = ann["bbox"][2], ann["bbox"][3]
        area = w * h

        if NORMALIZE_AREA:
            img_area = img["width"] * img["height"]
            area /= img_area

        areas.append(area)

    return np.array(areas)

# =========================================================
# FUNCTION TO PRINT 5-NUMBER SUMMARY STATISTICS
# =========================================================
def print_stats(name, data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    print(f"\n{'='*30}")
    print(f"  {name}")
    print(f"{'='*30}")
    print(f"  Count  : {len(data)}")
    print(f"  Min    : {np.min(data):,.2f} px²")
    print(f"  Q1     : {q1:,.2f} px²")
    print(f"  Median : {np.median(data):,.2f} px²")
    print(f"  Q3     : {q3:,.2f} px²")
    print(f"  Max    : {np.max(data):,.2f} px²")

# =========================================================
# LOAD DATASETS
# =========================================================
print("Loading Ground Truth dataset...")
real_areas = load_coco_bbox_areas(REAL_PATH)

print("Loading Synthetic dataset...")
syn_areas = load_coco_bbox_areas(SYN_PATH)

print("Loading RefCOCO dataset...")
refcoco_areas = load_coco_bbox_areas(REFCOCO_JSON)

# =========================================================
# PRINT 5-NUMBER SUMMARY STATISTICS
# =========================================================
print_stats("Ground Truth", real_areas)
print_stats("Synthetic", syn_areas)
print_stats("RefCOCO", refcoco_areas)

# =========================================================
# BOX-AND-WHISKER PLOT — Log10 Y-axis
# =========================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Reorder: RefCOCO, Synthetic, Ground Truth
data = [refcoco_areas, syn_areas, real_areas]
colors = ["skyblue", "salmon", "lightgreen"]
labels = ["RefCOCO", "Synthetic", "Ground Truth"]

bp = ax.boxplot(
    data,
    labels=labels,
    patch_artist=True
)

# Set individual box colors
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Log10 y-axis with ticks at 10, 100, 1000, 10000, ...
ax.set_yscale('log', base=10)

# Force major ticks at exact powers of 10 starting from 10
all_vals = np.concatenate(data)
min_exp = 1  # start at 10^1 = 10
max_exp = int(np.ceil(np.log10(np.max(all_vals)))) + 1
tick_vals = [10**e for e in range(min_exp, max_exp)]
ax.set_yticks(tick_vals)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{int(y):,}'))
ax.yaxis.set_minor_locator(mticker.NullLocator())  # remove minor ticks

ax.set_ylabel("Bounding Box Area (pixels²)")
ax.set_title("Bounding Box Area Distribution Across Datasets")
ax.grid(True, linestyle="--", alpha=0.5, which="major")

plt.tight_layout()
plt.savefig(OUTPUT_FIG)
print(f"\nSaved figure: {OUTPUT_FIG}")
