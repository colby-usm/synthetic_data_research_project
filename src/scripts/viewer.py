# saver_no_ref.py

import os
import cv2
import json

# ==========================================================
# CONFIG
# ==========================================================

IMAGE_DIR = "images"
ANNOTATIONS_FILE = "annotations.json"
SAVE_DIR = "."  # save annotated images here

# ==========================================================
# LOAD JSON DATA
# ==========================================================

with open(ANNOTATIONS_FILE, "r") as f:
    annotations_data = json.load(f)

# ==========================================================
# BUILD MAPPINGS
# ==========================================================

# Map image_id -> list of annotations
image_to_anns = {}
for ann in annotations_data["annotations"]:
    image_id = ann["image_id"]
    if image_id not in image_to_anns:
        image_to_anns[image_id] = []
    image_to_anns[image_id].append(ann)

# Map image_id -> filename
id_to_filename = {img["id"]: img["file_name"] for img in annotations_data["images"]}

# ==========================================================
# ITERATE IMAGES AND ANNOTATIONS
# ==========================================================

image_ids = list(id_to_filename.keys())

for img_id in image_ids:
    img_path = os.path.join(IMAGE_DIR, id_to_filename[img_id])
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: could not read image {img_path}")
        continue

    anns = image_to_anns.get(img_id, [])
    if not anns:
        print(f"No annotations for image {img_path}")
        continue

    for ann in anns:
        # Make a copy for overlay
        img_copy = image.copy()
        bbox = ann["bbox"]  # COCO format: [x, y, width, height]
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print(f"\nImage: {id_to_filename[img_id]}")
        print(f"Annotation ID: {ann['id']}, BBox: {bbox}")

        # Display image
        cv2.imshow("Annotation Viewer", img_copy)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord("s"):
            # Save annotated image
            save_name = f"{id_to_filename[img_id].split('.')[0]}_ann.png"
            save_path = os.path.join(SAVE_DIR, save_name)
            cv2.imwrite(save_path, img_copy)
            print(f"Saved annotated image to {save_path}")
        elif key == 32:  # Space
            continue  # go to next annotation

cv2.destroyAllWindows()
print("Done iterating all images and annotations.")
