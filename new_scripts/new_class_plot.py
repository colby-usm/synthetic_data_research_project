# plot_novel_classes.py

import json
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

# ==========================================================
# CONFIG
# ==========================================================

SYNTHETIC_JSON = "data/synthetic_data_v1/annotations.json"
REAL_JSON = "data/real_data_v2/custom_subset/annotations.json"

NOVEL_CLASSES = ["military_tank", "military_truck"]

OUTPUT_FILE = "ref_exps_novel_classes_comparison.pdf"
FIGSIZE = (8, 6)

# ==========================================================
# COLORS
# ==========================================================

RED = (235/255, 72/255, 63/255)     # Synthetic
GREEN = (82/255, 151/255, 66/255)   # Real

BAR_WIDTH = 0.35
OFFSET = 0.2

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

synthetic_counts = get_class_counts_json(SYNTHETIC_JSON)
real_counts = get_class_counts_json(REAL_JSON)

syn_values = [synthetic_counts.get(cls, 0) for cls in NOVEL_CLASSES]
real_values = [real_counts.get(cls, 0) for cls in NOVEL_CLASSES]

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=FIGSIZE)
ax = plt.gca()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

x_positions = range(len(NOVEL_CLASSES))

bars_syn = plt.bar(
    [x - OFFSET for x in x_positions],
    syn_values,
    width=BAR_WIDTH,
    color=RED,
    edgecolor="black",
    linewidth=1.2,
)

bars_real = plt.bar(
    [x + OFFSET for x in x_positions],
    real_values,
    width=BAR_WIDTH,
    color=GREEN,
    edgecolor="black",
    linewidth=1.2,
)

# ==========================================================
# LABELS
# ==========================================================

def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,
            f"{int(height)}",
            ha='center',
            va='bottom',
            fontsize=10
        )

add_bar_labels(bars_syn)
add_bar_labels(bars_real)

# ==========================================================
# AXES
# ==========================================================

plt.xticks(
    list(x_positions),
    [cls.replace("_", " ").title() for cls in NOVEL_CLASSES]
)

plt.ylabel("Number of Instances")
plt.title("Novel Class Referring Expression Instances")

# Linear scale with ticks every 50
plt.ylim(0, 1500)
plt.yticks(range(0, 1501, 100))

ax.grid(True, axis="y", linestyle="--", alpha=0.35)

# ==========================================================
# LEGEND UNDER THE PLOT
# ==========================================================

legend_handles = [
    Patch(facecolor=RED, edgecolor='black', label="Synthetic"),
    Patch(facecolor=GREEN, edgecolor='black', label="Real"),
]

plt.legend(
    handles=legend_handles,
    loc="lower center",          # move legend below the axes
    bbox_to_anchor=(0.5, -0.25), # adjust vertical position
    ncol=2,
    frameon=False
)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=800)

print("Saved:", OUTPUT_FILE)
plt.show()
