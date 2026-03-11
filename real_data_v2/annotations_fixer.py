import json

# ==============================
# CONFIG
# ==============================
INPUT_JSON = "./custom_subset/custom_annotations.json"
OUTPUT_JSON = "./custom_subset/annotations_fixed.json"

ROUND_DECIMALS = 4


def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


def round_bbox(bbox):
    return [round(v, ROUND_DECIMALS) for v in bbox]


def fix_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox

    # convert to corners
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    # clamp to image bounds
    x1 = clamp(x1, 0, img_w)
    y1 = clamp(y1, 0, img_h)
    x2 = clamp(x2, 0, img_w)
    y2 = clamp(y2, 0, img_h)

    # recompute width/height
    new_w = max(0, x2 - x1)
    new_h = max(0, y2 - y1)

    fixed_bbox = [x1, y1, new_w, new_h]

    return round_bbox(fixed_bbox)


def main():
    with open(INPUT_JSON) as f:
        coco = json.load(f)

    # map image_id -> image info
    images = {img["id"]: img for img in coco["images"]}

    fixes = 0
    new_annotations = []

    for ann in coco["annotations"]:
        image_id = ann["image_id"]

        if image_id not in images:
            print(f"Removing orphan annotation {ann['id']} referencing missing image {image_id}")
            continue  # skip orphan annotations

        img_info = images[image_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        old_bbox = ann["bbox"]
        new_bbox = fix_bbox(old_bbox, img_w, img_h)

        if new_bbox != round_bbox(old_bbox):
            fixes += 1
            print("\nFixing annotation:", ann["id"])
            print("Image:", img_info["file_name"])
            print("Old bbox:", old_bbox)
            print("New bbox:", new_bbox)

        ann["bbox"] = new_bbox
        # recompute area
        ann["area"] = round(new_bbox[2] * new_bbox[3], ROUND_DECIMALS)

        new_annotations.append(ann)  # keep only valid annotations

    coco["annotations"] = new_annotations

    print("\nTotal boxes fixed:", fixes)
    print("Total annotations after cleaning:", len(coco["annotations"]))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco, f, indent=2)

    print("Saved cleaned dataset to:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
