import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def extract_bbox_features(data):
    """
    Extract normalized bbox features:
    - width, height, area
    - aspect ratio
    """
    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]

    widths, heights, areas = [], [], []
    aspect_ratios = []

    for ann in annotations:
        img = images[ann["image_id"]]
        img_w, img_h = img["width"], img["height"]

        x, y, w, h = ann["bbox"]

        # Normalize
        w_n = w / img_w
        h_n = h / img_h
        area = w_n * h_n

        widths.append(w_n)
        heights.append(h_n)
        areas.append(area)
        aspect_ratios.append(w_n / (h_n + 1e-8))

    return {
        "width": np.array(widths),
        "height": np.array(heights),
        "area": np.array(areas),
        "aspect_ratio": np.array(aspect_ratios),
    }


def compute_summary_stats(arr):
    return {
        "min": np.min(arr),
        "q1": np.percentile(arr, 25),
        "median": np.median(arr),
        "q3": np.percentile(arr, 75),
        "max": np.max(arr),
        "mean": np.mean(arr),
        "std": np.std(arr),
    }


def print_stats(name, real, syn):
    print(f"\n=== {name} ===")
    real_stats = compute_summary_stats(real)
    syn_stats = compute_summary_stats(syn)

    print("Real:")
    for k, v in real_stats.items():
        print(f"  {k}: {v:.4f}")

    print("Synthetic:")
    for k, v in syn_stats.items():
        print(f"  {k}: {v:.4f}")


def create_boxplot(ax, real, syn, title):
    box = ax.boxplot(
        [real, syn],
        patch_artist=True,
        tick_labels=["Real", "Synthetic"],
    )

    colors = ["blue", "red"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


def main():
    # ==== PATHS ====
    REAL_JSON = Path("data/real_data/custom_subset/annotations.json")
    SYN_JSON = Path("data/synthetic_data/annotations.json")

    # ==== LOAD ====
    real_data = load_annotations(REAL_JSON)
    syn_data = load_annotations(SYN_JSON)

    # ==== EXTRACT FEATURES ====
    real_feats = extract_bbox_features(real_data)
    syn_feats = extract_bbox_features(syn_data)

    # ==== PRINT STATS ====
    for key in real_feats.keys():
        print_stats(key, real_feats[key], syn_feats[key])

    # ==== PLOTTING ====
    plot_keys = ["width", "height", "area", "aspect_ratio"]
    n_plots = len(plot_keys)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6))

    for i, key in enumerate(plot_keys):
        create_boxplot(
            axes[i],
            real_feats[key],
            syn_feats[key],
            key.replace("_", " ").title(),
        )

    plt.tight_layout()

    # Save high-res without affecting display size
    plt.savefig("bbox_domain_gap_row.png", dpi=600, bbox_inches='tight')
    print("\nSaved plot to bbox_domain_gap_row.png")

    plt.show()


if __name__ == "__main__":
    main()
