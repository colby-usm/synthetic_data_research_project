import json
from pathlib import Path
from transformers import AutoProcessor
import matplotlib.pyplot as plt
from collections import Counter


# ------------------------
# Load config and paths
# ------------------------
with open('cfg.json', 'r', encoding='utf-8') as file:
    train_cfg = json.load(file)

model_path_str = train_cfg.get("directories", {}).get("padt_7b_rec", "./padt_model")
MODEL_PATH = Path(model_path_str)

# Synthetic (REFERENCE)
SYN_DATA_ROOT = Path(train_cfg.get("directories", {}).get("synthetic_data", "synthetic_data"))
SYN_REFEXPS   = SYN_DATA_ROOT / "refexps.json"

# Real (TARGET)
REAL_DATA_ROOT = Path(train_cfg.get("eval", {}).get("real_data", "real_data"))
REAL_REFEXPS   = REAL_DATA_ROOT / "refexps.json"


# ------------------------
# Helpers
# ------------------------
def load_sentences(refexp_path):
    with open(refexp_path, "r") as f:
        data = json.load(f)

    sentences = []
    for item in data:
        for s in item.get("sentences", []):
            if s.get("sent"):
                sentences.append(s["sent"])
    return sentences


def build_vocab(tokenizer, texts):
    vocab = set()
    for t in texts:
        tokens = tokenizer.tokenize(t)
        tokens = [tok for tok in tokens if tok not in tokenizer.all_special_tokens]
        vocab.update(tokens)
    return vocab


def compute_stats(tokenizer, ref_vocab, target_texts):
    total_tokens = 0
    oov_tokens   = 0
    unk_tokens   = 0

    for text in target_texts:
        tokens = tokenizer.tokenize(text)
        tokens = [tok for tok in tokens if tok not in tokenizer.all_special_tokens]

        ids = tokenizer(text)["input_ids"]

        total_tokens += len(tokens)

        for tok in tokens:
            if tok not in ref_vocab:
                oov_tokens += 1

        if tokenizer.unk_token_id is not None:
            unk_tokens += sum(1 for i in ids if i == tokenizer.unk_token_id)

    return {
        "total_tokens": total_tokens,
        "oov_tokens": oov_tokens,
        "oov_rate": oov_tokens / total_tokens if total_tokens else 0.0,
        "unk_tokens": unk_tokens,
        "unk_rate": unk_tokens / total_tokens if total_tokens else 0.0,
        "avg_tokens_per_sentence": total_tokens / len(target_texts) if target_texts else 0.0,
    }


# ------------------------
# Load / download
# ------------------------
print("[INFO] Checking tokenizer availability...")

if MODEL_PATH.exists():
    print(f"[INFO] Loading processor from {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
else:
    print("[INFO] Downloading processor from Hugging Face...")
    HF_MODEL_ID = "PaDT-MLLM/PaDT_REC_7B"

    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(MODEL_PATH)

    print(f"[INFO] Processor saved to {MODEL_PATH}")

tokenizer = processor.tokenizer

print("[INFO] Tokenizer ready.")
print(f"[INFO] Tokenizer vocab size: {len(tokenizer)}")
print(f"[INFO] UNK token: {tokenizer.unk_token}")


# ------------------------
# Load datasets
# ------------------------
print("[INFO] Loading datasets...")

syn_texts  = load_sentences(SYN_REFEXPS)
real_texts = load_sentences(REAL_REFEXPS)

print(f"[INFO] Synthetic sentences: {len(syn_texts)}")
print(f"[INFO] Real sentences: {len(real_texts)}")


# ------------------------
# Build vocabularies
# ------------------------
print("[INFO] Building vocabularies...")

syn_vocab  = build_vocab(tokenizer, syn_texts)
real_vocab = build_vocab(tokenizer, real_texts)

print(f"[INFO] Synthetic vocab size: {len(syn_vocab)}")
print(f"[INFO] Real vocab size: {len(real_vocab)}")


# ------------------------
# Compute OOV (both directions)
# ------------------------
print("[INFO] Computing OOV statistics...")

syn_to_real = compute_stats(tokenizer, syn_vocab, real_texts)
real_to_syn = compute_stats(tokenizer, real_vocab, syn_texts)


# ------------------------
# Print results
# ------------------------
print("\n===== OOV RESULTS =====")

print("\n--- Synthetic → Real (domain shift) ---")
for k, v in syn_to_real.items():
    print(f"{k:30s}: {v:.6f}" if isinstance(v, float) else f"{k:30s}: {v}")

print("\n--- Real → Synthetic (coverage gap) ---")
for k, v in real_to_syn.items():
    print(f"{k:30s}: {v:.6f}" if isinstance(v, float) else f"{k:30s}: {v}")


# ------------------------
# Debug tokenization
# ------------------------
print("\n[DEBUG] Example tokenization (real data):")
for i in range(min(5, len(real_texts))):
    print(f"\nText: {real_texts[i]}")
    print(f"Tokens: {tokenizer.tokenize(real_texts[i])}")


# ------------------------
# Save results
# ------------------------
output = {
    "synthetic_to_real": syn_to_real,
    "real_to_synthetic": real_to_syn,
}

output_path = MODEL_PATH / "oov_stats.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n[INFO] Saved results to: {output_path}")






from matplotlib import pyplot as plt
from matplotlib_venn import venn2

# Compute sets
synthetic_only = syn_vocab - real_vocab
real_only      = real_vocab - syn_vocab
overlap        = syn_vocab & real_vocab

# Raw counts
syn_count = len(syn_vocab)
real_count = len(real_vocab)
overlap_count = len(overlap)

# Plot Venn diagram
plt.figure(figsize=(8,6))
v = venn2(subsets=(len(synthetic_only), len(real_only), overlap_count),
          set_labels=(f"Synthetic ({syn_count})", f"Real ({real_count})"))

# Set the counts inside the circles
v.get_label_by_id('10').set_text(len(synthetic_only))
v.get_label_by_id('01').set_text(len(real_only))
v.get_label_by_id('11').set_text(overlap_count)

plt.title("Vocabulary Overlap (Synthetic vs Real)")
plt.show()
