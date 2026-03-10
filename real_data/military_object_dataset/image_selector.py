import os
import cv2
import json
import random

# ==========================
# CONFIG
# ==========================
DATASET_ROOT = "./military tracking 2/military_object_dataset"
SPLIT = "train"  # train / val / test


CLASSES = [3]  # Only target these classes
CLASS_NAMES = {
    0: "camouflage_soldier",
    1: "weapon",
    2: "military_tank",
    3: "military_truck",
    4: "military_vehicle",
    5: "civilian",
    6: "soldier",
    7: "civilian_vehicle",
    8: "military_artillery",
    9: "trench",
    10: "military_aircraft",
    11: "military_warship",
}

IMAGES_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, SPLIT, "labels")

SUBSET_DIR = os.path.join(DATASET_ROOT, "custom_subset")
os.makedirs(os.path.join(SUBSET_DIR, "images"), exist_ok=True)
COCO_JSON_PATH = os.path.join(SUBSET_DIR, "custom_annotations.json")

# ==========================
# UTILS
# ==========================
def yolo_to_xywh(box, img_w, img_h):
    x_c, y_c, w, h = box
    x = (x_c - w / 2) * img_w
    y = (y_c - h / 2) * img_h
    w = w * img_w
    h = h * img_h
    return [x, y, w, h]

def image_contains_class(label_path, selected_classes):
    with open(label_path, "r") as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls in selected_classes:
            boxes.append([cls] + list(map(float, parts[1:5])))  # only take first 4 bbox values
    return boxes

# ==========================
# LOAD OR INIT COCO JSON
# ==========================
if os.path.exists(COCO_JSON_PATH):
    with open(COCO_JSON_PATH, "r") as f:
        coco_json = json.load(f)
    print(f"Loaded existing COCO JSON with {len(coco_json['images'])} images and {len(coco_json['annotations'])} annotations.")
    ann_id = max([a["id"] for a in coco_json["annotations"]], default=-1) + 1
    img_id = max([i["id"] for i in coco_json["images"]], default=-1) + 1

    existing_ids = [c["id"] for c in coco_json["categories"]]
    for cid in CLASSES:
        if cid not in existing_ids:
            coco_json["categories"].append({
                "id": cid,
                "name": CLASS_NAMES[cid]
            })

    # Track already saved image filenames to avoid duplicates
    already_saved = {i["file_name"] for i in coco_json["images"]}
else:
    coco_json = {
        "images": [],
        "annotations": [],
        "categories": [{"id": k, "name": CLASS_NAMES[k]} for k in CLASSES],
    }
    ann_id = 0
    img_id = 0
    already_saved = set()

saved_count = 0
image_files = sorted(os.listdir(IMAGES_DIR))
random.shuffle(image_files)

print("Press Y to save image + annotation, D to skip, ESC/q to quit")

for img_name in image_files:
    # Skip images already in the dataset
    if img_name in already_saved:
        continue

    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(LABELS_DIR, label_name)

    if not os.path.exists(label_path):
        continue

    boxes = image_contains_class(label_path, CLASSES)
    if not boxes:
        continue

    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # Draw boxes on a copy for display only
    img_display = img.copy()
    for box in boxes:
        cls = box[0]
        x_c, y_c, bw, bh = box[1:]
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_display, CLASS_NAMES[cls], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Subset Selector", img_display)
    key = cv2.waitKey(0)

    if key == ord("y") or key == ord("Y"):
        # Save the clean original image
        dest_img_path = os.path.join(SUBSET_DIR, "images", img_name)
        cv2.imwrite(dest_img_path, img)

        coco_json["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })

        for box in boxes:
            cls = box[0]
            x, y, bw, bh = yolo_to_xywh(box[1:], w, h)
            coco_json["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1
        saved_count += 1
        already_saved.add(img_name)
        print(f"Saved {saved_count} images so far.")

    elif key == ord("d") or key == ord("D"):
        continue

    elif key == 27 or key == ord("q"):
        break

cv2.destroyAllWindows()

with open(COCO_JSON_PATH, "w") as f:
    json.dump(coco_json, f, indent=4)

print(f"Done! Total saved images this session: {saved_count}")
print(f"COCO annotations saved to: {COCO_JSON_PATH}")
