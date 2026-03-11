import json
import os
import cv2

# ==============================
# CONFIG
# ==============================
ANNOTATIONS_PATH = "./custom_subset/custom_annotations.json"
IMAGE_ROOT = "./custom_subset/images"   # directory containing images


def bbox_outside_image(bbox, width, height):
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return True

    if x < 0 or y < 0:
        return True

    if x + w > width:
        return True

    if y + h > height:
        return True

    return False


def draw_bbox(image, bbox, color=(0,0,255)):
    x, y, w, h = bbox
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)

    cv2.rectangle(image, (x1,y1), (x2,y2), color, 3)
    return image


def main():

    with open(ANNOTATIONS_PATH, "r") as f:
        coco = json.load(f)

    # map image_id -> image info
    images = {img["id"]: img for img in coco["images"]}

    bad_count = 0

    for ann in coco["annotations"]:

        image_id = ann["image_id"]
        bbox = ann["bbox"]

        img_info = images[image_id]

        width = img_info["width"]
        height = img_info["height"]

        if bbox_outside_image(bbox, width, height):

            bad_count += 1

            print("\n================================")
            print("INVALID BBOX FOUND")
            print("Annotation ID:", ann["id"])
            print("Image:", img_info["file_name"])
            print("BBox:", bbox)
            print("Image Size:", width, "x", height)

            img_path = os.path.join(IMAGE_ROOT, img_info["file_name"])

            if not os.path.exists(img_path):
                print("Image not found:", img_path)
                continue

            image = cv2.imread(img_path)

            if image is None:
                print("Failed to load image")
                continue

            image = draw_bbox(image, bbox)

            cv2.imshow("Invalid BBox", image)
            print("Press any key for next...")
            cv2.waitKey(0)

    print("\nDone.")
    print("Total invalid bboxes:", bad_count)


if __name__ == "__main__":
    main()
