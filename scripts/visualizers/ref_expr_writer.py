import os
import cv2
import json
import argparse

# ==========================
# CLI ARGUMENTS
# ==========================
parser = argparse.ArgumentParser(description="COCO Referring Expression Writer")

parser.add_argument(
    "--annotator",
    type=str,
    required=True,
    help="Annotator name (required)"
)

parser.add_argument(
    "--image",
    type=int,
    default=None,
    help="Start from this image_id"
)

args = parser.parse_args()

# normalize annotator name
ANNOTATOR = args.annotator.strip().lower().replace(" ", "_")

# ==========================
# CONFIG
# ==========================
IMAGES_DIR = "data/synthetic_data_v1/images"
ANNOTATIONS_PATH = "data/synthetic_data_v1/annotations.json"
REFEXP_PATH = "data/synthetic_data_v1/refexps.json"

WINDOW_NAME = "Referring Expression Writer"

# ==========================
# LOAD COCO DATA
# ==========================
with open(ANNOTATIONS_PATH, "r") as f:
    coco = json.load(f)

images = sorted(coco["images"], key=lambda x: x["id"])
annotations = coco["annotations"]
categories = coco["categories"]

image_map = {img["id"]: img for img in images}

# group annotations per image
anns_per_image = {}
for ann in annotations:
    anns_per_image.setdefault(ann["image_id"], []).append(ann)

# ==========================
# LOAD EXISTING REFEXPS
# ==========================
if os.path.exists(REFEXP_PATH):
    with open(REFEXP_PATH, "r") as f:
        refexps = json.load(f)

    if not isinstance(refexps, list):
        print("WARNING: corrupted refexps file. Resetting.")
        refexps = []
else:
    refexps = []

# ==========================
# ID HELPERS
# ==========================
def next_ref_id():
    if not refexps:
        return 0
    return max(r["ref_id"] for r in refexps) + 1


def next_sent_id(annotator):
    max_id = 0

    for r in refexps:
        for s in r["sentences"]:
            sid = s["sent_id"]

            if isinstance(sid, str) and sid.startswith(annotator + "_"):
                try:
                    num = int(sid.split("_")[1])
                    max_id = max(max_id, num)
                except:
                    pass

    return f"{annotator}_{max_id + 1}"


def find_ref_by_ann(ann_id):
    for r in refexps:
        if r["ann_id"] == ann_id:
            return r
    return None


# ==========================
# UTILS
# ==========================
def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return int(x), int(y), int(x + w), int(y + h)


def category_name(cid):
    for c in categories:
        if c["id"] == cid:
            return c["name"]
    return "unknown"


def draw_annotations(img, target_ann, all_anns):
    canvas = img.copy()

    for ann in all_anns:
        x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])

        if ann["id"] == target_ann["id"]:
            color = (0, 255, 0)
            thickness = 2
            label = category_name(ann["category_id"])

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            cv2.putText(
                canvas,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)

    return canvas


def save_refexps():
    with open(REFEXP_PATH, "w") as f:
        json.dump(refexps, f, indent=4)


# ==========================
# FIND START IMAGE
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
total_images = len(images)

print("\nReferring Expression Writer")
print(f"Annotator: {ANNOTATOR}")
print("ENTER = skip | q = quit\n")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

quit_flag = False

for img_idx in range(start_index, total_images):

    img_info = images[img_idx]
    image_id = img_info["id"]

    print(f"\nImage {img_idx+1}/{total_images} (id={image_id})")

    file_path = os.path.join(IMAGES_DIR, img_info["file_name"])

    if not os.path.exists(file_path):
        print("Missing image:", file_path)
        continue

    frame = cv2.imread(file_path)

    if frame is None:
        print("Failed to load:", file_path)
        continue

    anns = anns_per_image.get(image_id, [])

    for ann in anns:

        vis = draw_annotations(frame, ann, anns)

        cv2.imshow(WINDOW_NAME, vis)
        cv2.waitKey(1)

        cat = category_name(ann["category_id"])

        expr = input(
            f"Ann {ann['id']} ({cat}) -> "
        ).strip()

        if expr.lower() == "q":
            quit_flag = True
            break

        if expr == "":
            continue

        ref = find_ref_by_ann(ann["id"])

        sent_id = next_sent_id(ANNOTATOR)

        sentence_entry = {
            "sent_id": sent_id,
            "annotator": ANNOTATOR,
            "sent": expr
        }

        if ref:
            ref["sentences"].append(sentence_entry)

        else:
            refexps.append({
                "ref_id": next_ref_id(),
                "ann_id": ann["id"],
                "image_id": image_id,
                "category_id": ann["category_id"],
                "sentences": [sentence_entry]
            })

        save_refexps()

    if quit_flag:
        break

cv2.destroyAllWindows()

# ==========================
# SUMMARY
# ==========================
total_sentences = sum(len(r["sentences"]) for r in refexps)

print("\nDone.")
print("Refs:", len(refexps))
print("Sentences:", total_sentences)
print("Saved to:", REFEXP_PATH)

