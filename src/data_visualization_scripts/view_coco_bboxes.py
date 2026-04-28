import json
import os
import cv2
from collections import defaultdict

# ==============================
# CONFIG
# ==============================

ANNOTATION_PATH = "data/real_data/custom_subset/annotations.json"
IMAGE_DIR = "data/real_data/custom_subset/images"

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
THICKNESS = 2


# ==============================
# LOAD COCO
# ==============================

with open(ANNOTATION_PATH, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

cat_id_to_name = {c["id"]: c["name"] for c in categories}


# ==============================
# 1. CLASS COUNTS
# ==============================

print("\n==== CLASS COUNTS ====\n")

class_counts = defaultdict(int)

for ann in annotations:
    class_counts[ann["category_id"]] += 1

for cid, count in sorted(class_counts.items()):
    print(f"{cat_id_to_name[cid]:20s} : {count}")

print("\n======================\n")


# ==============================
# BUILD INDEXES
# ==============================

img_id_to_file = {img["id"]: img["file_name"] for img in images}

anns_by_image = defaultdict(list)

for ann in annotations:
    anns_by_image[ann["image_id"]].append(ann)


# ==============================
# 2. ORPHAN CHECK
# ==============================

print("\n==== ORPHAN CHECK ====\n")

# images in directory
dir_images = set(os.listdir(IMAGE_DIR))

# images referenced in json
json_images = set(img["file_name"] for img in images)

# orphaned images
orphan_images = dir_images - json_images

# missing images referenced by annotations
missing_images = json_images - dir_images

print(f"Images in directory: {len(dir_images)}")
print(f"Images in JSON:      {len(json_images)}")

if orphan_images:
    print("\nImages in directory NOT in JSON:")
    for img in orphan_images:
        print(img)

if missing_images:
    print("\nImages referenced in JSON but missing from directory:")
    for img in missing_images:
        print(img)

# images with no annotations
print("\nImages with NO annotations:")
for img in images:
    if img["id"] not in anns_by_image:
        print(img["file_name"])

print("\n=======================\n")


# ==============================
# 3. OPEN CV VIEWER
# ==============================

image_list = images
index = 0


def draw_annotations(image, anns):

    for ann in anns:

        x, y, w, h = ann["bbox"]
        cid = ann["category_id"]

        label = cat_id_to_name[cid]

        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))

        cv2.rectangle(image, pt1, pt2, BOX_COLOR, THICKNESS)

        cv2.putText(
            image,
            label,
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
        )

    return image


while True:

    img_meta = image_list[index]
    img_path = os.path.join(IMAGE_DIR, img_meta["file_name"])

    if not os.path.exists(img_path):
        print(f"Missing image: {img_path}")
        index = (index + 1) % len(image_list)
        continue

    img = cv2.imread(img_path)

    anns = anns_by_image.get(img_meta["id"], [])

    img = draw_annotations(img, anns)

    cv2.imshow("COCO Viewer", img)

    key = cv2.waitKeyEx(0)

    # SPACE → forward
    if key == 32:
        index = (index + 1) % len(image_list)

    # DELETE → backward (Mac)
    elif key == 127:
        index = (index - 1) % len(image_list)

    # Quit
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
