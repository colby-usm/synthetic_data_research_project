import os
import cv2
import json
import argparse

# ==========================
# CLI ARGUMENTS
# ==========================
parser = argparse.ArgumentParser(description="Dataset Deletion Tool")

parser.add_argument(
    "--image",
    type=int,
    default=None,
    help="Start from this image_id"
)

args = parser.parse_args()

# ==========================
# CONFIG
# ==========================
IMAGES_DIR = "data/synthetic_data_v1/images"
ANNOTATIONS_PATH = "data/synthetic_data_v1/annotations.json"

WINDOW_NAME = "Deletion Tool"

# ==========================
# LOAD DATA
# ==========================
with open(ANNOTATIONS_PATH, "r") as f:
    coco = json.load(f)

images = sorted(coco["images"], key=lambda x: x["id"])
annotations = coco["annotations"]


# ==========================
# HELPERS
# ==========================
def save_all():
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(coco, f, indent=4)


def delete_image_and_related(image_id, file_name):
    global coco

    print(f"Deleting image_id={image_id}")

    # remove image entry
    coco["images"] = [img for img in coco["images"] if img["id"] != image_id]

    # find annotations tied to image
    ann_ids_to_delete = [ann["id"] for ann in coco["annotations"] if ann["image_id"] == image_id]

    # remove annotations
    coco["annotations"] = [
        ann for ann in coco["annotations"]
        if ann["image_id"] != image_id
    ]


    # delete image file
    file_path = os.path.join(IMAGES_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print("Deleted file:", file_path)
    else:
        print("File not found:", file_path)

    save_all()


# ==========================
# FIND START INDEX
# ==========================
start_index = 0

if args.image is not None:
    for i, img in enumerate(images):
        if img["id"] == args.image:
            start_index = i
            break

# ==========================
# MAIN LOOP
# ==========================
print("\nDeletion Tool")
print("SPACE = next | d = delete | ESC = quit\n")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

idx = start_index

while idx < len(images):

    img_info = images[idx]
    image_id = img_info["id"]
    file_name = img_info["file_name"]

    print(f"Image {idx+1}/{len(images)} (id={image_id})")

    file_path = os.path.join(IMAGES_DIR, file_name)

    if not os.path.exists(file_path):
        print("Missing image:", file_path)
        idx += 1
        continue

    frame = cv2.imread(file_path)

    if frame is None:
        print("Failed to load:", file_path)
        idx += 1
        continue

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(0) & 0xFF

    # SPACE → next
    if key == 32:
        idx += 1
        continue

    # d → delete
    elif key == ord('d'):
        delete_image_and_related(image_id, file_name)

        # refresh local image list after deletion
        images = sorted(coco["images"], key=lambda x: x["id"])

        # do NOT increment idx (next image shifts into this position)
        continue

    # ESC → exit
    elif key == 27:
        break

    else:
        idx += 1


cv2.destroyAllWindows()

# ==========================
# SUMMARY
# ==========================
print("\nDone.")
print("Remaining images:", len(coco["images"]))
print("Remaining annotations:", len(coco["annotations"]))
