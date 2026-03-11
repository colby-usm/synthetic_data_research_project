"""
review_outlier_anns.py
----------------------
Loads a COCO-style annotation JSON and displays every annotation whose
bounding-box area exceeds OUTLIER_THRESHOLD using OpenCV.

Controls
--------
  SPACE  – keep annotation, advance to next
  d      – delete this annotation from the JSON (in memory), advance to next
  q      – quit immediately (saves any deletions made so far)

On exit the (possibly modified) JSON is written back to SAVE_PATH.
"""

import json
import os
import copy
import cv2

# =========================================================
# CONFIGURATION
# =========================================================
IMAGE_DIR = "./custom_subset/images/"
ANNOTATIONS_JSON = "./custom_subset/annotations_fixed.json"


SAVE_PATH         = ANNOTATIONS_JSON   # overwrite in-place; change to a new path to keep the original
OUTLIER_THRESHOLD = 1_000_000          # px²

# =========================================================
# LOAD JSON
# =========================================================
print(f"Loading annotations from: {ANNOTATIONS_JSON}")
with open(ANNOTATIONS_JSON, "r") as f:
    coco = json.load(f)

images_meta = {img["id"]: img for img in coco["images"]}

# Collect outlier annotations
outliers = []
for ann in coco["annotations"]:
    img_meta = images_meta.get(ann["image_id"])
    if img_meta is None:
        continue
    area = ann["bbox"][2] * ann["bbox"][3]
    if area > OUTLIER_THRESHOLD:
        outliers.append((ann, img_meta, area))

print(f"Found {len(outliers)} annotation(s) with area > {OUTLIER_THRESHOLD:,} px²\n")

if not outliers:
    print("Nothing to review. Exiting.")
    exit(0)

print("Controls:  SPACE = keep & next  |  d = delete & next  |  q = quit")
print("-" * 60)

# =========================================================
# TRACK DELETIONS
# =========================================================
ids_to_delete = set()

# =========================================================
# REVIEW LOOP
# =========================================================
for idx, (ann, img_meta, area) in enumerate(outliers):
    fname    = img_meta.get("file_name", "")
    img_path = os.path.join(IMAGE_DIR, fname)


    print(f"[{idx + 1}/{len(outliers)}]  ann_id={ann['id']}  "
          f"image_id={ann['image_id']}  file={fname}")
    
    img_w = img_meta['width']
    img_h = img_meta['height']
    img_area = img_w * img_h
    
    print(f"  img_size : {img_w} x {img_h}")
    print(f"  img_area : {img_area:,} px²")  # <--- added
    print(f"  bbox     : x={ann['bbox'][0]:.1f}  y={ann['bbox'][1]:.1f}  "
          f"w={ann['bbox'][2]:.1f}  h={ann['bbox'][3]:.1f}")
    print(f"  area     : {area:,.0f} px²")


    # ── Load image ──────────────────────────────────────────
    if not os.path.exists(img_path):
        print(f"  WARNING: image not found at '{img_path}' — skipping display.\n")
        continue

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  WARNING: cv2 could not read '{img_path}' — skipping display.\n")
        continue

    # ── Draw bbox ───────────────────────────────────────────
    x1 = int(ann["bbox"][0])
    y1 = int(ann["bbox"][1])
    x2 = int(x1 + ann["bbox"][2])
    y2 = int(y1 + ann["bbox"][3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)

    # ── Draw label with background ──────────────────────────
    label      = f"ann={ann['id']}  area={area:,.0f} px²  [SPACE=keep  d=delete  q=quit]"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, frame.shape[1] / 1800)
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y = max(y1 - 10, th + 10)
    cv2.rectangle(
        frame,
        (x1, label_y - th - baseline - 4),
        (x1 + tw + 6, label_y + baseline),
        (0, 0, 200),
        cv2.FILLED,
    )
    cv2.putText(
        frame, label, (x1 + 3, label_y - 2),
        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )

    # ── Resize for display if very large ────────────────────
    MAX_DIM  = 1200
    h_img, w_img = frame.shape[:2]
    if max(h_img, w_img) > MAX_DIM:
        scale = MAX_DIM / max(h_img, w_img)
        frame = cv2.resize(frame, (int(w_img * scale), int(h_img * scale)))

    # ── Show and wait for keypress ───────────────────────────
    window_title = f"[{idx+1}/{len(outliers)}] ann_id={ann['id']}  ({os.path.basename(fname)})"
    cv2.imshow(window_title, frame)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(" "):                   # SPACE — keep
            print("  → kept\n")
            break
        elif key == ord("d"):                 # d — delete
            ids_to_delete.add(ann["id"])
            print("  → marked for deletion\n")
            break
        elif key == ord("q"):                 # q — quit
            print("  → quit early\n")
            cv2.destroyAllWindows()
            goto_save = True
            break
    else:
        goto_save = False

    cv2.destroyAllWindows()

    # Break out of the outer loop on quit
    if key == ord("q"):
        break

# =========================================================
# APPLY DELETIONS AND SAVE
# =========================================================
if ids_to_delete:
    original_count = len(coco["annotations"])
    coco["annotations"] = [
        a for a in coco["annotations"] if a["id"] not in ids_to_delete
    ]
    removed = original_count - len(coco["annotations"])
    print(f"Removed {removed} annotation(s): {sorted(ids_to_delete)}")

    with open(SAVE_PATH, "w") as f:
        json.dump(coco, f)
    print(f"Saved updated annotations to: {SAVE_PATH}")
else:
    print("No annotations deleted. JSON unchanged.")

print("Done.")
