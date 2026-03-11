"""
COCO Annotation Adder
=====================
Add new bounding box annotations to specific images in a COCO-style annotations file.

Usage:
    python coco_annotation_adder.py --image-name <filename> [OPTIONS]

Options:
    --image-name NAME      Filename of the image to annotate (required)
    --annotations PATH     Input annotations JSON file (overrides ANNOTATIONS_IN)
    --output PATH          Output annotations file (default: overwrites input)

Controls (in the editor window):
    Y              Start drawing a new annotation (click and drag)
    C              Confirm the drawn bbox — then enter class number in terminal
    R              Discard current draw and try again
    Q              Quit and save

Example:
    python coco_annotation_adder.py --image-name Jungle_1_CineCameraActor.png
    python coco_annotation_adder.py --image-name frame_042.jpg --output new_annotations.json
"""

import json
import os
import argparse
import copy
import cv2
import numpy as np

# ──────────────────────────────────────────────
# Configuration — fill these in before running
# ──────────────────────────────────────────────
ANNOTATIONS_IN = "./bulk_output_annotations.json"   # Path to the COCO annotations JSON file
IMAGE_DIR      = "./new_sim_data/"   # Path to the directory containing the images


# ──────────────────────────────────────────────
# Colours (BGR for OpenCV)
# ──────────────────────────────────────────────
COLORS = [
    (0,   255,   0),
    (0,   165, 255),
    (255,   0,   0),
    (0,   255, 255),
    (255,   0, 255),
    (0,   128, 255),
    (128,   0, 255),
    (0,   255, 128),
]
COLOR_DRAW    = (0, 165, 255)   # Orange — bbox being drawn
COLOR_NEW     = (0, 255, 255)   # Yellow — newly added this session
COLOR_EXIST   = (180, 180, 180) # Grey   — pre-existing annotations
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str, msg: str = "") -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    suffix = f"  ({msg})" if msg else ""
    print(f"[INFO] Saved → {path}{suffix}")


def next_annotation_id(annotations: list) -> int:
    return max((a["id"] for a in annotations), default=0) + 1


def print_categories(categories: list) -> None:
    print("\nAvailable classes:")
    for i, cat in enumerate(categories, start=1):
        print(f"  {i}. {cat['name']}  (id={cat['id']})")
    print()


def prompt_category(categories: list) -> dict | None:
    """Prompt the user to pick a class by number. Returns the category dict or None to cancel."""
    while True:
        try:
            raw = input(f"Enter class number (1–{len(categories)}) or 0 to cancel: ").strip()
            n = int(raw)
            if n == 0:
                return None
            if 1 <= n <= len(categories):
                return categories[n - 1]
            print(f"  Please enter a number between 1 and {len(categories)}.")
        except (ValueError, EOFError):
            print("  Invalid input — try again.")


# ──────────────────────────────────────────────
# Drawing state
# ──────────────────────────────────────────────

class DrawState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.drawing = False
        self.start   = None
        self.end     = None

    def to_bbox(self) -> list | None:
        if self.start is None or self.end is None:
            return None
        x1 = min(self.start[0], self.end[0])
        y1 = min(self.start[1], self.end[1])
        x2 = max(self.start[0], self.end[0])
        y2 = max(self.start[1], self.end[1])
        return [x1, y1, x2 - x1, y2 - y1]

    def is_valid(self, min_px: int = 3) -> bool:
        b = self.to_bbox()
        return b is not None and b[2] >= min_px and b[3] >= min_px


def make_mouse_callback(ds: DrawState, mode_ref: dict):
    def callback(event, x, y, flags, _):
        if not mode_ref["draw_mode"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            ds.drawing = True
            ds.start = (x, y)
            ds.end   = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and ds.drawing:
            ds.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            ds.drawing = False
            ds.end = (x, y)
    return callback


# ──────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────

def render_frame(image: np.ndarray,
                 existing_anns: list,
                 new_anns: list,
                 draw_mode: bool,
                 ds: DrawState,
                 label_map: dict,
                 status_text: str = "") -> np.ndarray:
    frame = image.copy()
    h, w  = frame.shape[:2]

    def draw_box(ann, color, thickness, tag=""):
        bx, by, bw, bh = (int(v) for v in ann["bbox"])
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, thickness)
        cat = label_map.get(ann["category_id"], f"cat:{ann['category_id']}")
        label = f"{tag}{cat}" if tag else cat
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        ty = max(by - 5, th + 4)
        cv2.rectangle(frame, (bx, ty - th - 3), (bx + tw + 4, ty + 2), (30, 30, 30), -1)
        cv2.putText(frame, label, (bx + 2, ty), FONT, 0.55, color, 1, cv2.LINE_AA)

    for ann in existing_anns:
        draw_box(ann, COLOR_EXIST, 1)

    for ann in new_anns:
        draw_box(ann, COLOR_NEW, 2, tag="NEW ")

    # Live draw preview
    if draw_mode and ds.start and ds.end:
        bbox = ds.to_bbox()
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_DRAW, 2)
            cv2.putText(frame, f"{bw}×{bh}", (x, max(y - 5, 10)),
                        FONT, 0.45, COLOR_DRAW, 1, cv2.LINE_AA)

    # HUD
    mode_str = "DRAW MODE — click & drag" if draw_mode else "VIEW MODE"
    cv2.putText(frame, mode_str, (10, h - 50), FONT, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Y=draw  C=confirm  R=retry  Q=quit+save",
                (10, h - 22), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Annotation counts top-left
    count_str = f"existing: {len(existing_anns)}   new this session: {len(new_anns)}"
    cv2.putText(frame, count_str, (10, 25), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if status_text:
        (tw, _), _ = cv2.getTextSize(status_text, FONT, 0.65, 2)
        cv2.putText(frame, status_text, ((w - tw) // 2, h - 80),
                    FONT, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_adder(coco: dict,
              image_name: str,
              output_path: str) -> None:

    images     = coco.get("images", [])
    anns       = coco.get("annotations", [])
    categories = coco.get("categories", [])
    label_map  = {cat["id"]: cat["name"] for cat in categories}

    if not categories:
        print("[ERROR] No categories found in annotations file.")
        return

    # Find the target image — match on file_name (basename only, tolerates subdirs)
    target_img = None
    for img in images:
        if os.path.basename(img["file_name"]) == os.path.basename(image_name):
            target_img = img
            break

    if target_img is None:
        print(f"[ERROR] Image '{image_name}' not found in annotations file.")
        print("  Available images (first 20):")
        for img in sorted(images, key=lambda x: x["id"])[:20]:
            print(f"    id={img['id']}  {img['file_name']}")
        return

    img_id    = target_img["id"]
    file_name = target_img["file_name"]
    img_path  = os.path.join(IMAGE_DIR, file_name)

    print(f"\n[IMAGE] id={img_id}  file={file_name}")

    if not os.path.isfile(img_path):
        print(f"[ERROR] Image file not found: {img_path}")
        return

    image = cv2.imread(img_path)
    if image is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return

    img_h, img_w = image.shape[:2]

    # Existing annotations for this image
    existing_anns = [a for a in anns if a["image_id"] == img_id]
    print(f"  Existing annotations: {len(existing_anns)}")

    # Print categories for reference (stays visible in terminal throughout)
    print_categories(categories)

    # New annotations added this session
    new_anns: list[dict] = []
    next_id = next_annotation_id(anns)

    # ── OpenCV window ─────────────────────────────────────────────────────────
    WIN = f"Annotation Adder — {os.path.basename(file_name)}"

    ds       = DrawState()
    mode_ref = {"draw_mode": False}

    draw_mode   = False
    status_text = ""
    pending_bbox: list | None = None   # bbox drawn but not yet class-assigned

    # macOS/Qt requires the window to be fully realized before setMouseCallback.
    # Strategy: create with WINDOW_AUTOSIZE, show the real first frame, pump the
    # event loop several times, then register the callback — same pattern as the
    # working editor script.
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    first_frame = render_frame(image, existing_anns, new_anns,
                               False, ds, label_map, "")
    cv2.imshow(WIN, first_frame)
    cv2.resizeWindow(WIN, 1280, 900)
    for _ in range(5):
        cv2.waitKey(1)
    cv2.setMouseCallback(WIN, make_mouse_callback(ds, mode_ref))

    print("Window open. Press Y to draw, Q to quit.")

    while True:
        frame = render_frame(image, existing_anns, new_anns,
                             draw_mode, ds, label_map, status_text)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(30) & 0xFF

        # Q — quit and save
        if key == ord('q'):
            break

        # Y — enter draw mode
        elif key == ord('y'):
            draw_mode = True
            mode_ref["draw_mode"] = True
            pending_bbox = None
            ds.reset()
            status_text = "Click & drag to draw bbox, then C to confirm"

        # R — reset draw
        elif key == ord('r') and draw_mode:
            ds.reset()
            pending_bbox = None
            status_text = "Cleared — draw again"

        # C — confirm bbox, then prompt for class in terminal
        elif key == ord('c') and draw_mode:
            if ds.is_valid():
                pending_bbox = ds.to_bbox()
                # Clamp to image bounds
                x, y, bw, bh = pending_bbox
                x  = max(0, min(int(x),  img_w - 1))
                y  = max(0, min(int(y),  img_h - 1))
                bw = max(1, min(int(bw), img_w - x))
                bh = max(1, min(int(bh), img_h - y))
                pending_bbox = [x, y, bw, bh]

                draw_mode = False
                mode_ref["draw_mode"] = False
                ds.reset()

                # Show a frozen frame while user types in terminal
                frozen = render_frame(image, existing_anns, new_anns,
                                      False, ds, label_map,
                                      "Check terminal — enter class number")
                # Draw the pending bbox in orange on the frozen frame
                px, py, pw, ph = pending_bbox
                cv2.rectangle(frozen, (px, py), (px + pw, py + ph), COLOR_DRAW, 2)
                cv2.imshow(WIN, frozen)
                cv2.waitKey(1)

                # ── Terminal class prompt ─────────────────────────────────────
                cat = prompt_category(categories)

                if cat is None:
                    print("  Cancelled — bbox discarded.")
                    pending_bbox = None
                    status_text  = "Cancelled — press Y to draw again"
                else:
                    new_ann = {
                        "id":           next_id,
                        "image_id":     img_id,
                        "category_id":  cat["id"],
                        "bbox":         [float(v) for v in pending_bbox],
                        "area":         float(pending_bbox[2] * pending_bbox[3]),
                        "segmentation": [[
                            float(px),        float(py),
                            float(px + pw),   float(py),
                            float(px + pw),   float(py + ph),
                            float(px),        float(py + ph),
                        ]],
                        "iscrowd": 0,
                    }
                    new_anns.append(new_ann)
                    next_id += 1
                    print(f"  [ADDED] id={new_ann['id']}  "
                          f"class='{cat['name']}'  bbox={new_ann['bbox']}")
                    status_text = f"Added: {cat['name']} — press Y for another"
                    pending_bbox = None

            else:
                status_text = "Bbox too small — draw again (R to clear)"

    # ── Save ──────────────────────────────────────────────────────────────────
    cv2.destroyAllWindows()

    if new_anns:
        coco["annotations"].extend(new_anns)
        save_json(coco, output_path,
                  f"{len(new_anns)} annotation(s) added to image id={img_id}")
        print(f"\n[DONE] Added {len(new_anns)} annotation(s).")
    else:
        print("\n[DONE] No annotations added — file unchanged.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add bounding box annotations to a specific image in a COCO dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--image-name",
        type=str,
        required=True,
        metavar="NAME",
        help="Filename of the image to annotate (e.g. Jungle_1_CineCameraActor.png).",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=None,
        metavar="PATH",
        help="Input annotations JSON file (overrides ANNOTATIONS_IN in script).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output file path (default: overwrites the input annotations file).",
    )
    args = parser.parse_args()

    # Resolve annotations path
    ann_path = args.annotations or ANNOTATIONS_IN
    if not ann_path:
        parser.error("Please set ANNOTATIONS_IN at the top of the script or pass --annotations.")
    if not os.path.isfile(ann_path):
        parser.error(f"Annotations file not found: {ann_path}")
    if not IMAGE_DIR:
        parser.error("Please set IMAGE_DIR at the top of the script.")
    if not os.path.isdir(IMAGE_DIR):
        parser.error(f"Image directory not found: {IMAGE_DIR}")

    output_path = args.output or ann_path

    coco = load_json(ann_path)
    run_adder(coco, args.image_name, output_path)


if __name__ == "__main__":
    main()
