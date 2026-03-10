import os
from collections import defaultdict

# ==========================
# CONFIG
# ==========================

DATASET_ROOT = "/Users/colby/Desktop/military tracking 2/military_object_dataset"

SPLITS = ["train", "val", "test"]

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

NUM_CLASSES = len(CLASS_NAMES)


# ==========================
# COUNTING
# ==========================

# Structure: counts[split][class_id] = count
counts = {
    split: defaultdict(int) for split in SPLITS
}

for split in SPLITS:
    labels_dir = os.path.join(DATASET_ROOT, split, "labels")

    if not os.path.exists(labels_dir):
        print(f"Warning: {labels_dir} not found")
        continue

    label_files = os.listdir(labels_dir)

    for file_name in label_files:
        if not file_name.endswith(".txt"):
            continue

        label_path = os.path.join(labels_dir, file_name)

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            counts[split][class_id] += 1


# ==========================
# PRINT TABLE
# ==========================

header = f"{'class_id':<10} {'class_name':<22} {'train':<10} {'val':<10} {'test':<10}"
print(header)
print("-" * len(header))

for class_id in range(NUM_CLASSES):
    train_count = counts["train"][class_id]
    val_count = counts["val"][class_id]
    test_count = counts["test"][class_id]

    print(f"{class_id:<10} "
          f"{CLASS_NAMES[class_id]:<22} "
          f"{train_count:<10} "
          f"{val_count:<10} "
          f"{test_count:<10}")
