import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================

MY_DATA_PATH = "real_data/military_object_dataset/custom_subset/custom_refexps.json"
REFCOCO_PICKLE_PATH = "/Users/colby/Desktop/RefCOCO/refcoco/refs(unc).p"

OUTPUT_FILE = "expression_length_distribution.pdf"

# Plot flags
ALPHA = 0.5
MY_COLOR = "green"
REFCOCO_COLOR = "blue"

# Histogram bins
MAX_EXPR_LEN = 15
BINS = np.arange(1, MAX_EXPR_LEN + 2) - 0.5


# ==========================================================
# LOAD MY DATASET (JSON)
# ==========================================================

def extract_lengths_json(path):

    with open(path, "r") as f:
        data = json.load(f)

    lengths = []

    for ref in data:
        for sent in ref.get("sentences", []):
            tokens = sent["sent"].strip().split()
            lengths.append(len(tokens))

    return np.array(lengths)


# ==========================================================
# LOAD REFCOCO PICKLE
# ==========================================================

def extract_lengths_refcoco(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    lengths = []

    for ref in data:
        for sent in ref["sentences"]:
            tokens = sent["tokens"]
            lengths.append(len(tokens))

    return np.array(lengths)


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading datasets...")

my_lengths = extract_lengths_json(MY_DATA_PATH)
refcoco_lengths = extract_lengths_refcoco(REFCOCO_PICKLE_PATH)


# ==========================================================
# METRICS
# ==========================================================

def print_metrics(name, lengths):

    print(f"\n{name} Dataset Metrics")
    print("-" * 40)

    print(f"Total expressions: {len(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths)}")
    print(f"Std dev: {np.std(lengths):.2f}")

    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")

print_metrics("My Dataset", my_lengths)
print_metrics("RefCOCO", refcoco_lengths)


# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(
    refcoco_lengths,
    bins=BINS,
    density=True,
    alpha=ALPHA,
    color=REFCOCO_COLOR,
    label="RefCOCO"
)

plt.hist(
    my_lengths,
    bins=BINS,
    density=True,
    alpha=ALPHA,
    color=MY_COLOR,
    label="Ground Truth Dataset"
)

# Mean indicators
plt.axvline(np.mean(refcoco_lengths), color=REFCOCO_COLOR, linestyle="dashed")
plt.axvline(np.mean(my_lengths), color=MY_COLOR, linestyle="dashed")

plt.xlabel("Expression Length (tokens)")
plt.ylabel("Normalized Frequency")
plt.title("Referring Expression Length Distribution")

plt.legend()

plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=600)

print(f"\nSaved figure to: {OUTPUT_FILE}")

plt.show()
