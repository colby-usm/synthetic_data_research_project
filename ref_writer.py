import os
import cv2
import json

# ==========================
# CONFIG
# ==========================
DATASET_ROOT = "./military tracking 2/military_object_dataset"
SUBSET_DIR = os.path.join(DATASET_ROOT, "custom_subset")
IMAGES_DIR = os.path.join(SUBSET_DIR, "images")
COCO_JSON_PATH = os.path.join(SUBSET_DIR, "custom_annotations.json")
REF_JSON_PATH = os.path.join(SUBSET_DIR, "custom_refexps.json")

# ==========================
# LOAD COCO JSON
# ==========================
with open(COCO_JSON_PATH, "r") as f:
    coco = json.load(f)

image_map = {img["id"]: img for img in coco["images"]}

if os.path.exists(REF_JSON_PATH):
    with open(REF_JSON_PATH, "r") as f:
        ref_json = json.load(f)
    if not isinstance(ref_json, list):
        print("WARNING: existing ref file is not a list, reinitializing.")
        ref_json = []
else:
    ref_json = []

print("Instructions: type a referring expression for the highlighted object.")
print("Commands: 's' = skip, 'delete' = remove annotation+image, 'q' = quit.")

# ==========================
# UTILS
# ==========================
def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return int(x), int(y), int(x + w), int(y + h)

def get_next_ref_id(refs):
    if not refs:
        return 0
    return max(r["ref_id"] for r in refs) + 1

def get_next_sent_id(refs):
    max_sent_id = -1
    for r in refs:
        for s in r["sentences"]:
            if s["sent_id"] > max_sent_id:
                max_sent_id = s["sent_id"]
    return max_sent_id + 1

def find_existing_ref(refs, ann_id):
    for r in refs:
        if r["ann_id"] == ann_id:
            return r
    return None

def already_annotated(refs, ann_id):
    ref = find_existing_ref(refs, ann_id)
    return ref is not None and len(ref["sentences"]) > 0

def draw_frame(img, ann, all_anns, categories):
    img_copy = img.copy()
    ann_id = ann["id"]

    for other_ann in all_anns:
        if other_ann["image_id"] != ann["image_id"]:
            continue
        ox1, oy1, ox2, oy2 = xywh_to_xyxy(other_ann["bbox"])
        if other_ann["id"] == ann_id:
            cv2.rectangle(img_copy, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
            cat_name = next((c["name"] for c in categories if c["id"] == other_ann["category_id"]), "unknown")
            cv2.putText(img_copy, cat_name, (ox1, oy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(img_copy, (ox1, oy1), (ox2, oy2), (100, 100, 100), 1)

    return img_copy

def save_outputs():
    with open(REF_JSON_PATH, "w") as f:
        json.dump(ref_json, f, indent=4)
    with open(COCO_JSON_PATH, "w") as f:
        json.dump(coco, f, indent=4)

# ==========================
# MAIN LOOP
# ==========================
quit_flag = False

for ann in list(coco["annotations"]):
    if quit_flag:
        break

    ann_id = ann["id"]

    if already_annotated(ref_json, ann_id):
        continue

    img_info = image_map[ann["image_id"]]
    img_path = os.path.join(IMAGES_DIR, img_info["file_name"])
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not load image: {img_path}")
        continue

    cat_name = next((c["name"] for c in coco["categories"] if c["id"] == ann["category_id"]), "unknown")

    img_copy = draw_frame(img, ann, coco["annotations"], coco["categories"])
    cv2.imshow("Referring Expression Creator", img_copy)
    cv2.waitKey(1)

    expression = input(f"Annotation {ann_id} ({cat_name}): ").strip()
    cv2.destroyAllWindows()

    if expression.lower() in ["s", "skip"]:
        continue

    if expression.lower() in ["q", "quit"]:
        quit_flag = True
        break

    if expression.lower() == "delete":
        coco["annotations"] = [a for a in coco["annotations"] if a["id"] != ann_id]
        ref_json = [r for r in ref_json if r["ann_id"] != ann_id]
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Deleted image file: {img_path}")
        print(f"Deleted annotation {ann_id}")
        save_outputs()
        continue

    existing_ref = find_existing_ref(ref_json, ann_id)

    if existing_ref is not None:
        new_sent_id = get_next_sent_id(ref_json)
        existing_ref["sentences"].append({
            "sent_id": new_sent_id,
            "sent": expression
        })
    else:
        new_ref_id = get_next_ref_id(ref_json)
        new_sent_id = get_next_sent_id(ref_json)
        ref_json.append({
            "ref_id": new_ref_id,
            "ann_id": ann_id,
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "sentences": [
                {
                    "sent_id": new_sent_id,
                    "sent": expression
                }
            ]
        })

    save_outputs()

# ==========================
# FINAL SAVE + SUMMARY
# ==========================
save_outputs()

total_sents = sum(len(r["sentences"]) for r in ref_json)
print(f"\nDone. {len(ref_json)} refs, {total_sents} total sentences saved to {REF_JSON_PATH}")
print(f"Updated COCO annotations saved to {COCO_JSON_PATH}")
