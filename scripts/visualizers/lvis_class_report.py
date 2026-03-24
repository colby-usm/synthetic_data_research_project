import os
import ijson
from collections import defaultdict
from tqdm import tqdm

LVIS_ROOT = "/Users/colby/Desktop/LVIS"

FILES = {
    "train": "lvis_v1_train.json",
    "val": "lvis_v1_val.json",
    "test": "lvis_v1_image_info_test_challenge.json"
}


def get_file_path(split):
    return os.path.join(LVIS_ROOT, FILES[split])


# -------------------------------------------------
# STREAM HELPERS
# -------------------------------------------------
def stream_categories(json_path):
    id_to_name = {}
    name_to_id = {}

    with open(json_path, "rb") as f:
        for cat in ijson.items(f, "categories.item"):
            id_to_name[cat["id"]] = cat["name"]
            name_to_id[cat["name"]] = cat["id"]

    return id_to_name, name_to_id


def stream_annotation_counts(json_path, target_category_id=None):
    counts = defaultdict(int)
    with open(json_path, "rb") as f:
        for ann in tqdm(
            ijson.items(f, "annotations.item"),
            desc=f"Processing {os.path.basename(json_path)}",
            unit="ann"
        ):
            cat_id = ann["category_id"]
            if target_category_id is not None:
                if cat_id == target_category_id:
                    counts[cat_id] += 1
            else:
                counts[cat_id] += 1
    return counts


# -------------------------------------------------
# FREQUENCY BUCKET (LVIS v1 definition)
# -------------------------------------------------
def get_frequency_bucket(count):
    if count <= 10:
        return "rare"
    elif count <= 100:
        return "common"
    else:
        return "frequent"


# -------------------------------------------------
# CLASS QUERY ACROSS ALL SPLITS
# -------------------------------------------------
def query_class_across_splits(class_name):
    print(f'\nQuerying "{class_name}" across splits...\n')

    results = {}
    train_path = get_file_path("train")
    _, name_to_id = stream_categories(train_path)

    if class_name not in name_to_id:
        print("Class not found in LVIS categories.")
        return

    cat_id = name_to_id[class_name]

    for split in FILES.keys():
        json_path = get_file_path(split)
        try:
            counts = stream_annotation_counts(json_path, target_category_id=cat_id)
            results[split] = counts.get(cat_id, 0)
        except Exception:
            results[split] = 0

    print(f'"{class_name}"')
    for split in ["train", "val", "test"]:
        print(f"{split}: {results.get(split, 0)}")
    print()


# -------------------------------------------------
# FULL SPLIT REPORT
# -------------------------------------------------
def full_split_report(split):
    print(f"\nGenerating full report for: {split}\n")

    json_path = get_file_path(split)
    id_to_name, _ = stream_categories(json_path)
    counts = stream_annotation_counts(json_path)

    bucket_counts = defaultdict(int)

    for cat_id in tqdm(id_to_name, desc="Processing classes", unit="class"):
        count = counts.get(cat_id, 0)
        bucket = get_frequency_bucket(count)
        bucket_counts[bucket] += 1
        print(f"{id_to_name[cat_id]:30s} | {count:6d} | {bucket}")

    print("\n----- SUMMARY -----")
    print(f"Total classes: {len(id_to_name)}")
    print(f"Rare (≤10): {bucket_counts['rare']}")
    print(f"Common (11–100): {bucket_counts['common']}")
    print(f"Frequent (>100): {bucket_counts['frequent']}")
    print("-------------------\n")


# -------------------------------------------------
# QUERY BY FREQUENCY BUCKET (TABLE OUTPUT, SORTABLE)
# -------------------------------------------------
def query_by_frequency_bucket(bucket_name):
    from operator import itemgetter

    bucket_name = bucket_name.lower()
    if bucket_name not in ["rare", "common", "frequent"]:
        print("Invalid bucket. Choose from: rare, common, frequent")
        return

    print(f"\nClasses in '{bucket_name}' bucket across splits:\n")

    train_path = get_file_path("train")
    id_to_name, _ = stream_categories(train_path)

    # Precompute counts for all splits
    counts_per_split = {}
    for split in FILES.keys():
        counts_per_split[split] = stream_annotation_counts(get_file_path(split))

    # Identify category IDs in this bucket based on train split
    selected_cat_ids = [
        cat_id for cat_id, count in counts_per_split["train"].items()
        if get_frequency_bucket(count) == bucket_name
    ]

    # Build list of tuples: (class_name, train_count, val_count, test_count)
    data = []
    for cat_id in tqdm(selected_cat_ids, desc=f"Processing classes", unit="class"):
        counts = [
            counts_per_split[split].get(cat_id, 0)
            for split in ["train", "val", "test"]
        ]
        data.append((id_to_name[cat_id], *counts))

    # Ask user for sorting method
    print("\nSort options:")
    print("1) Alphabetically by class name")
    print("2) By train count")
    print("3) By val count")
    print("4) By test count")
    choice = input("Select sort option (1-4): ").strip()

    if choice == "1":
        data.sort(key=itemgetter(0))  # alphabetical
    elif choice in ["2", "3", "4"]:
        idx = int(choice)  # 2->train, 3->val, 4->test
        data.sort(key=itemgetter(idx), reverse=True)  # sort by count descending
    else:
        print("Invalid option, defaulting to alphabetical")
        data.sort(key=itemgetter(0))

    # Print table
    print(f"\n{'Class Name':30s} | {'Train':>6s} {'Val':>6s} {'Test':>6s}")
    print("-" * 54)
    for row in data:
        print(f"{row[0]:30s} | {row[1]:6d} {row[2]:6d} {row[3]:6d}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("\nLVIS CLI\n")
    print("1) Query class across all splits")
    print("2) Generate full report for one split")
    print("3) Query classes by frequency bucket (rare / common / frequent)\n")

    choice = input("Select option (1, 2, or 3): ").strip()

    if choice == "1":
        class_name = input("Enter class name: ").strip()
        query_class_across_splits(class_name)

    elif choice == "2":
        split = input("Choose split (train / val / test): ").strip().lower()
        if split not in FILES:
            print("Invalid split.")
            return
        full_split_report(split)

    elif choice == "3":
        bucket = input("Enter bucket name (rare / common / frequent): ").strip().lower()
        query_by_frequency_bucket(bucket)

    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
