import pickle
from collections import defaultdict
from tabulate import tabulate
import numpy as np

# ---------------------------
# 1. Load the RefCOCO data
# ---------------------------
with open("refcoco/refs(unc).p", "rb") as f:
    data = pickle.load(f)

print(f"\nTotal annotations: {len(data)}")

# ---------------------------
# 2. Count annotations per split
# ---------------------------
split_counts = defaultdict(int)
for r in data:
    split_counts[r['split']] += 1

print("\nAnnotations per split:")
for split, count in split_counts.items():
    print(f"{split}: {count}")

# ---------------------------
# 3. Build class-level counts
# ---------------------------
class_counts = defaultdict(lambda: {
    'train': 0,
    'val': 0,
    'testA': 0,
    'testB': 0,
    'total': 0
})

for r in data:
    cid = r['category_id']
    split = r['split']

    if split in class_counts[cid]:
        class_counts[cid][split] += 1

    class_counts[cid]['total'] += 1

# ---------------------------
# 4. 5-Stat Overview Across Classes
# ---------------------------
totals = [counts['total'] for counts in class_counts.values()]

min_val = np.min(totals)
max_val = np.max(totals)
mean_val = np.mean(totals)
median_val = np.median(totals)
std_val = np.std(totals)

print("\n==============================")
print("Class-Level Distribution Overview")
print("==============================")
print(f"Number of classes: {len(totals)}")
print(f"Min instances per class:    {min_val}")
print(f"Max instances per class:    {max_val}")
print(f"Mean instances per class:   {mean_val:.2f}")
print(f"Median instances per class: {median_val}")
print(f"Std deviation:              {std_val:.2f}")

print("\nDistribution Buckets:")
print(f"Classes with < 100 instances:      {sum(t < 100 for t in totals)}")
print(f"Classes with 100–500 instances:    {sum(100 <= t <= 500 for t in totals)}")
print(f"Classes with > 500 instances:      {sum(t > 500 for t in totals)}")

# ---------------------------
# 5. Optional Detailed Table
# ---------------------------
user_input = input("\nDo you want full class-level table? (y/n): ").strip().lower()
if user_input == 'y':

    # COCO category mapping
    COCO_CATEGORY_MAP = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
        7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
        13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
        19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra",
        25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
        33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball",
        38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard",
        42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
        48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
        54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
        59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
        64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv",
        73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
        78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
        84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
        89: "hair drier", 90: "toothbrush"
    }

    table = []
    for cid, counts in sorted(class_counts.items()):
        name = COCO_CATEGORY_MAP.get(cid, "UNKNOWN")
        table.append([
            cid,
            name,
            counts['total'],
            counts['train'],
            counts['val'],
            counts['testA'],
            counts['testB']
        ])

    print("\nFull Class-Level Annotation Table:")
    print(tabulate(
        table,
        headers=["class_id", "name", "total", "train", "val", "testA", "testB"],
        tablefmt="github"
    ))
