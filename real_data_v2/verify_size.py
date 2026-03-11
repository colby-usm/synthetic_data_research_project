import json

with open("./custom_subset//annotations_fixed.json") as f:
    coco = json.load(f)

images = {i["id"]: i for i in coco["images"]}

for ann in coco["annotations"]:
    img = images[ann["image_id"]]
    x,y,w,h = ann["bbox"]

    if x < 0 or y < 0 or x+w > img["width"] or y+h > img["height"]:
        print("BAD BBOX:", ann["id"])
