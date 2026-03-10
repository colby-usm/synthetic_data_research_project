import os
import cv2

# ==========================
# CONFIG
# ==========================

DATASET_ROOT = "/Users/colby/Desktop/military tracking 2/military_object_dataset"
SPLIT = "train"  # change to train / val / test


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
CLASSES = [3]  # e.g. [2] for military_tank

IMAGES_DIR = os.path.join(DATASET_ROOT, SPLIT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, SPLIT, "labels")


# ==========================
# UTILS
# ==========================

def yolo_to_xyxy(box, img_w, img_h):
    """
    Convert YOLO normalized bbox to pixel xyxy
    """
    x_c, y_c, w, h = box

    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x_c - w / 2)
    y1 = int(y_c - h / 2)
    x2 = int(x_c + w / 2)
    y2 = int(y_c + h / 2)

    return x1, y1, x2, y2


def image_contains_class(label_path, selected_classes):
    """
    Check if label file contains any selected class
    """
    with open(label_path, "r") as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])
        if cls in selected_classes:
            boxes.append([cls] + list(map(float, parts[1:])))

    return boxes


# ==========================
# MAIN LOOP
# ==========================

image_files = sorted(os.listdir(IMAGES_DIR))

print("Press SPACE for next image | ESC or q to quit")

for img_name in image_files:
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

    # Draw boxes
    for box in boxes:
        cls = box[0]
        x1, y1, x2, y2 = yolo_to_xyxy(box[1:], w, h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"class {cls}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Filtered Viewer", img)

    key = cv2.waitKey(0)

    # SPACE → next image
    if key == 32:
        continue

    # ESC or q → exit
    if key == 27 or key == ord("q"):
        break

cv2.destroyAllWindows()
