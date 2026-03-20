import json
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# ==========================================================
# CONFIGURATION
# ==========================================================

JSON_PATH = "./data/real_data_v2/custom_subset/refexps.json"
OUTPUT_FILE = "refexp_wordcloud.pdf"

# Words to remove from the word cloud (common stopwords or dataset-specific)
REMOVE_WORDS = [
    "the", "a", "an", "on", "in", "of", "and", "to", "is", "with", "that", "this", "these", "those"
]

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def preprocess_text(text):
    """
    Lowercase, remove punctuation, and filter out unwanted words
    """
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # split into words
    words = text.split()
    # filter out remove_words
    words = [w for w in words if w not in REMOVE_WORDS]
    return words

# ==========================================================
# LOAD JSON AND EXTRACT WORDS
# ==========================================================

with open(JSON_PATH, "r") as f:
    data = json.load(f)

all_words = []

for ref in data:
    for sent in ref.get("sentences", []):
        words = preprocess_text(sent["sent"])
        all_words.extend(words)

logger_info = f"Processed {len(data)} referring expressions, total words: {len(all_words)}"
print(logger_info)

# ==========================================================
# COUNT WORD FREQUENCIES
# ==========================================================

word_counts = Counter(all_words)
print(f"Top 10 words: {word_counts.most_common(10)}")

# ==========================================================
# GENERATE WORD CLOUD
# ==========================================================

wc = WordCloud(
    width=1200,
    height=800,
    background_color="white",
    colormap="inferno"
).generate_from_frequencies(word_counts)

# ==========================================================
# PLOT AND SAVE
# ==========================================================

plt.figure(figsize=(12, 8))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, format="pdf", dpi=600)
print(f"Word cloud saved to {OUTPUT_FILE}")
plt.show()
