import json
import numpy as np
import matplotlib.pyplot as plt
import logging

# -----------------------------
# Configuration
# -----------------------------
ANNOTATIONS_PATH = "./data/synthetic_data_v1/annotations.json"
OUTPUT_FILE = "synthetic_heatmap"

GRID_SIZE = 512

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)

logger.info("Starting COCO heatmap generation")

# -----------------------------
# Load JSON
# -----------------------------
with open(ANNOTATIONS_PATH, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

logger.info(f"Loaded {len(images)} images")
logger.info(f"Loaded {len(annotations)} annotations")

# -----------------------------
# Image lookup
# -----------------------------
image_dims = {}

for img in images:
    image_dims[img["id"]] = (img["width"], img["height"])

logger.info(f"Built image dimension lookup for {len(image_dims)} images")

# -----------------------------
# Heatmap grid
# -----------------------------
heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

skipped = 0
bbox_areas = []

# -----------------------------
# Process annotations
# -----------------------------
for i, ann in enumerate(annotations):

    image_id = ann["image_id"]

    if image_id not in image_dims:
        skipped += 1
        continue

    img_w, img_h = image_dims[image_id]

    x, y, w, h = ann["bbox"]

    bbox_areas.append(w * h)

    # normalize bbox corners
    x0 = x / img_w
    y0 = y / img_h
    x1 = (x + w) / img_w
    y1 = (y + h) / img_h

    # clamp
    x0 = np.clip(x0, 0, 1)
    x1 = np.clip(x1, 0, 1)
    y0 = np.clip(y0, 0, 1)
    y1 = np.clip(y1, 0, 1)

    # convert to grid indices
    gx0 = int(x0 * GRID_SIZE)
    gx1 = int(x1 * GRID_SIZE)
    gy0 = int(y0 * GRID_SIZE)
    gy1 = int(y1 * GRID_SIZE)

    if gx0 >= gx1 or gy0 >= gy1:
        skipped += 1
        continue

    # logging first few for debugging
    if i < 5:
        logger.info(
            f"Annotation {i}: "
            f"bbox={ann['bbox']} "
            f"norm=({x0:.3f},{y0:.3f})-({x1:.3f},{y1:.3f}) "
            f"grid=({gx0}:{gx1},{gy0}:{gy1})"
        )

    # accumulate area
    heatmap[gy0:gy1, gx0:gx1] += 1


logger.info(f"Skipped {skipped} annotations")

# -----------------------------
# Statistics
# -----------------------------
bbox_areas = np.array(bbox_areas)

logger.info(f"BBox area statistics:")
logger.info(f"Mean area: {bbox_areas.mean():.2f}")
logger.info(f"Median area: {np.median(bbox_areas):.2f}")
logger.info(f"Min area: {bbox_areas.min():.2f}")
logger.info(f"Max area: {bbox_areas.max():.2f}")

logger.info(f"Total heatmap counts: {heatmap.sum()}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6,6))

plt.imshow(
    heatmap,
    cmap="inferno",
    origin="lower",
    extent=[0,1,0,1]
)

plt.colorbar(label="Annotation Density")

plt.xlabel("Normalized Image X")
plt.ylabel("Normalized Image Y")

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    format="pdf",
    dpi=600
)

logger.info(f"Saved heatmap to {OUTPUT_FILE}")
