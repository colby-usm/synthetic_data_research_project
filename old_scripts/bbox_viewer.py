"""
COCO Annotation Viewer
======================
Browse images with bounding box annotations overlaid.

Usage:
    python coco_annotation_viewer.py [OPTIONS]

Options:
    --start-image-id INT   Image ID to start viewing from (default: first image)

Controls:
    RIGHT / D      Next image
    LEFT  / A      Previous image
    Q              Quit

Example:
    python coco_annotation_viewer.py
    python coco_annotation_viewer.py --start-image-id 42
"""

import json
import os
import argparse
import cv2
import numpy as np

# ──────────────────────────────────────────────
# Configuration — fill these in before running
# ──────────────────────────────────────────────
IMAGE_DIR = "new_sim_data/"   # Path to the COCO annotations JSON file
ANNOTATIONS_IN = "./NEWESTremapped_bulk_output_annotations.json"   # Path to the directory containing the images


# ──────────────────────────────────────────────
# Colours (BGR for OpenCV)
# ──────────────────────────────────────────────
COLORS = [
    (0,   255,   0),   # green
    (0,   165, 255),   # orange
    (255,   0,   0),   # blue
    (0,   255, 255),   # yellow
    (255,   0, 255),   # magenta
    (0,   128, 255),   # sky blue
    (128,   0, 255),   # purple
    (0,   255, 128),   # mint
]
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def draw_annotations(image: np.ndarray,
                     anns: list,
                     label_map: dict) -> np.ndarray:
    frame = image.copy()
    h, w  = frame.shape[:2]

    for ann in anns:
        cat_id   = ann.get("category_id", 0)
        color    = COLORS[cat_id % len(COLORS)]
        bx, by, bw, bh = (int(v) for v in ann["bbox"])

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)

        label = label_map.get(cat_id, f"cat:{cat_id}")
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        ty = max(by - 5, th + 4)
        cv2.rectangle(frame, (bx, ty - th - 3), (bx + tw + 4, ty + 2), (30, 30, 30), -1)
        cv2.putText(frame, label, (bx + 2, ty), FONT, 0.55, color, 1, cv2.LINE_AA)

    # HUD
    cv2.putText(frame, "LEFT/A = prev    RIGHT/D = next    Q = quit",
                (10, h - 15), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def render_placeholder(msg: str, win_w: int = 960, win_h: int = 540) -> np.ndarray:
    frame = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize(msg, FONT, 0.7, 2)
    cv2.putText(frame, msg, ((win_w - tw) // 2, win_h // 2),
                FONT, 0.7, (100, 100, 100), 2, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────
# Main viewer
# ──────────────────────────────────────────────

def run_viewer(coco: dict, start_image_id: int | None) -> None:
    images     = coco.get("images", [])
    anns       = coco.get("annotations", [])
    categories = coco.get("categories", [])
    label_map  = {cat["id"]: cat["name"] for cat in categories}

    images_sorted = sorted(images, key=lambda img: img["id"])
    total         = len(images_sorted)

    if total == 0:
        print("[WARN] No images found in annotations file.")
        return

    # Build annotation lookup
    ann_by_image: dict[int, list] = {}
    for ann in anns:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Determine starting index
    idx = 0
    if start_image_id is not None:
        for i, img in enumerate(images_sorted):
            if img["id"] == start_image_id:
                idx = i
                break
        else:
            print(f"[WARN] image_id {start_image_id} not found — starting from the beginning.")

    WIN = "COCO Annotation Viewer"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 900)

    while True:
        img_info  = images_sorted[idx]
        img_id    = img_info["id"]
        file_name = img_info["file_name"]
        img_anns  = ann_by_image.get(img_id, [])

        print(f"[{idx + 1}/{total}] id={img_id}  file={file_name}  "
              f"annotations={len(img_anns)}")

        img_path = os.path.join(IMAGE_DIR, file_name)
        image    = cv2.imread(img_path)

        if image is None:
            frame = render_placeholder(f"Image not found: {file_name}")
        else:
            frame = draw_annotations(image, img_anns, label_map)

        # Progress counter top-right
        h, w = frame.shape[:2]
        progress = f"{idx + 1} / {total}"
        (pw, _), _ = cv2.getTextSize(progress, FONT, 0.55, 1)
        cv2.putText(frame, progress, (w - pw - 10, 25),
                    FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):          # Q or ESC
            break
        elif key in (ord('d'), 83, 39):    # D, right arrow
            idx = min(idx + 1, total - 1)
        elif key in (ord('a'), 81, 37):    # A, left arrow
            idx = max(idx - 1, 0)

    cv2.destroyAllWindows()
    print("[DONE]")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="COCO annotation viewer — browse images with bbox overlays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start-image-id",
        type=int,
        default=None,
        metavar="INT",
        help="Image ID to start viewing from (default: first image).",
    )
    args = parser.parse_args()

    if not ANNOTATIONS_IN:
        parser.error("Please set ANNOTATIONS_IN at the top of the script.")
    if not IMAGE_DIR:
        parser.error("Please set IMAGE_DIR at the top of the script.")
    if not os.path.isfile(ANNOTATIONS_IN):
        parser.error(f"Annotations file not found: {ANNOTATIONS_IN}")
    if not os.path.isdir(IMAGE_DIR):
        parser.error(f"Image directory not found: {IMAGE_DIR}")

    coco = load_json(ANNOTATIONS_IN)
    run_viewer(coco, args.start_image_id)


if __name__ == "__main__":
    main()
