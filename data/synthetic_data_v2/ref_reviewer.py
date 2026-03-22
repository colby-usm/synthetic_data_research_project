import os
import cv2
import json
import argparse

# ==========================
# CLI ARGUMENTS
# ==========================
parser = argparse.ArgumentParser(description="COCO Referring Expression Reviewer")

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
IMAGES_DIR = "images"
ANNOTATIONS_PATH = "annotations.json"
REFEXP_PATH = "refexps.json"

WINDOW_NAME = "Referring Expression Reviewer"

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
# HELPERS
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


def find_ref_by_ann(ann_id):
    for r in refexps:
        if r["ann_id"] == ann_id:
            return r
    return None


def save_refexps():
    with open(REFEXP_PATH, "w") as f:
        json.dump(refexps, f, indent=4)


def wait_for_key():
    """Block in OpenCV until a recognised key is pressed. Returns the character."""
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):   # Enter
            return "enter"
        ch = chr(key).lower()
        if ch in ("d", "r", "q"):
            return ch


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

print("\nReferring Expression Reviewer")
print("ENTER=keep | d=delete | r=rewrite (then type in terminal) | q=quit\n")

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

        ref = find_ref_by_ann(ann["id"])

        if not ref or not ref["sentences"]:
            continue

        cat = category_name(ann["category_id"])
        print(f"\nAnn {ann['id']} ({cat})")

        vis = draw_annotations(frame, ann, anns)
        cv2.imshow(WINDOW_NAME, vis)

        # iterate over sentences
        sent_idx = 0
        while sent_idx < len(ref["sentences"]):
            sentence = ref["sentences"][sent_idx]

            print(f"  [{sent_idx+1}/{len(ref['sentences'])}] "
                  f"[{sentence['annotator']}] \"{sentence['sent']}\"")
            print("  ENTER=keep | d=delete | r=rewrite | q=quit")

            action = wait_for_key()

            if action == "q":
                quit_flag = True
                break

            elif action == "enter":
                sent_idx += 1

            elif action == "d":
                ref["sentences"].pop(sent_idx)
                print("  Deleted.")
                save_refexps()
                # don't increment — next sentence slides into this index

            elif action == "r":
                # input() is fine here — the OpenCV window stays open,
                # we just need the user to click the terminal to type
                new_text = input("  New expression: ").strip()
                if new_text:
                    ref["sentences"][sent_idx]["sent"] = new_text
                    print("  Updated.")
                    save_refexps()
                else:
                    print("  Empty input, skipping rewrite.")
                sent_idx += 1

        # clean up refs with no sentences left
        if not ref["sentences"]:
            refexps.remove(ref)
            save_refexps()
            print(f"  Ref {ref['ref_id']} removed (no sentences left).")

        if quit_flag:
            break

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
