# plot_novel_classes_vs_refcoco.py

import json
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

# ==========================================================
# CONFIG
# ==========================================================

REFCOCO_JSON = "data/RefCOCO/refcoco/instances.json"
SYNTHETIC_JSON = "data/synthetic_data_v1/annotations.json"
GT_JSON = "data/real_data_v2/custom_subset/annotations.json"

NOVEL_CLASSES = ["military_tank", "military_truck"]

OUTPUT_FILE = "novel_classes_vs_refcoco.pdf"
FIGSIZE = (8,6)

# ==========================================================
# EXACT COLORS
# ==========================================================

RED = (235/255, 72/255, 63/255)
GREEN = (82/255, 151/255, 66/255)
BLUE = (66/255, 110/255, 180/255)

BAR_WIDTH = 0.28
OFFSET = 0.18

# ==========================================================
# FUNCTIONS
# ==========================================================

def get_class_counts_json(path):

    with open(path, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    counts = defaultdict(int)

    for ann in coco["annotations"]:
        name = cat_id_to_name[ann["category_id"]]
        counts[name] += 1

    return counts


# ==========================================================
# LOAD DATA
# ==========================================================

ref_counts = get_class_counts_json(REFCOCO_JSON)
synthetic_counts = get_class_counts_json(SYNTHETIC_JSON)
gt_counts = get_class_counts_json(GT_JSON)

ref_values = list(ref_counts.values())

tank_syn = synthetic_counts.get("military_tank", 0)
tank_gt = gt_counts.get("military_tank", 0)

truck_syn = synthetic_counts.get("military_truck", 0)
truck_gt = gt_counts.get("military_truck", 0)


# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=FIGSIZE)
ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

REF = 1
TANK = 2
TRUCK = 3

# ==========================================================
# REFCOCO BOXPLOT
# ==========================================================

# plt.boxplot(
#     ref_values,
#     positions=[REF],
#     widths=0.55,
#     patch_artist=True,
#     boxprops=dict(facecolor=BLUE, linewidth=1.5),
#     whiskerprops=dict(linewidth=1.5),
#     capprops=dict(linewidth=1.5),
#     medianprops=dict(color="black", linewidth=2),
#     showmeans=True,
#     meanprops=dict(marker='D', markerfacecolor='yellow', markeredgecolor='black')
# )

# ==========================================================
# MILITARY TANK BARS
# ==========================================================

bar1 = plt.bar(
    TANK - OFFSET,
    tank_syn,
    width=BAR_WIDTH,
    color=RED,
    edgecolor="black",
    linewidth=1.2
)

bar2 = plt.bar(
    TANK + OFFSET,
    tank_gt,
    width=BAR_WIDTH,
    color=GREEN,
    edgecolor="black",
    linewidth=1.2
)

# ==========================================================
# MILITARY TRUCK BARS
# ==========================================================

bar3 = plt.bar(
    TRUCK - OFFSET,
    truck_syn,
    width=BAR_WIDTH,
    color=RED,
    edgecolor="black",
    linewidth=1.2
)

bar4 = plt.bar(
    TRUCK + OFFSET,
    truck_gt,
    width=BAR_WIDTH,
    color=GREEN,
    edgecolor="black",
    linewidth=1.2
)

# ==========================================================
# ADD VALUE LABELS
# ==========================================================

def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height * 1.1,
            f"{int(height)}",
            ha='center',
            va='bottom',
            fontsize=10
        )

add_bar_labels(bar1)
add_bar_labels(bar2)
add_bar_labels(bar3)
add_bar_labels(bar4)

# ==========================================================
# AXES
# ==========================================================

plt.xticks(
    [REF, TANK, TRUCK],
    ["RefCOCO", "Military Tank", "Military Truck"]
)

plt.ylabel("Number of Instances per Class (log scale)")

plt.yscale("log")

plt.yticks(
    [10, 100, 1000, 10000],
    ["10", "100", "1,000", "10,000"]
)

ax.grid(True, axis="y", linestyle="--", alpha=0.35)

# ==========================================================
# LEGEND
# ==========================================================

legend_handles = [
    Patch(facecolor=BLUE, edgecolor='black', label="RefCOCO Class Distribution"),
    Patch(facecolor=RED, edgecolor='black', label="Synthetic"),
    Patch(facecolor=GREEN, edgecolor='black', label="Ground Truth"),
]

plt.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=3,
    frameon=False
)

plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=800)

print("Saved:", OUTPUT_FILE)

plt.show()
