# ============================================================
# 08_WordCloud + N-gram Analysis + Heatmap
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

df = pd.read_csv("reviews_with_vader.csv")
df["cleaned"] = df["cleaned"].fillna("")

pos_text = " ".join(df[df["Liked"] == 1]["cleaned"])
neg_text = " ".join(df[df["Liked"] == 0]["cleaned"])

# ── 1) WordClouds ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, text, title, colour in [
    (axes[0], pos_text, "Positive Reviews", "Greens"),
    (axes[1], neg_text, "Negative Reviews", "Reds"),
]:
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap=colour,
        max_words=100,
        stopwords=set(STOPWORDS),
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("08a_wordclouds.png", dpi=150)
plt.show()


# ── 2) Top Bigrams ─────────────────────────────────────────────
def get_top_ngrams(corpus, n=2, top_k=20):
    vec = CountVectorizer(ngram_range=(n, n), max_features=2000).fit(corpus)
    bag = vec.transform(corpus)
    freq = dict(zip(vec.get_feature_names_out(), np.asarray(bag.sum(axis=0)).ravel()))
    return pd.DataFrame(
        sorted(freq.items(), key=lambda x: -x[1])[:top_k], columns=["bigram", "count"]
    )


pos_bigrams = get_top_ngrams(df[df["Liked"] == 1]["cleaned"])
neg_bigrams = get_top_ngrams(df[df["Liked"] == 0]["cleaned"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, bdf, title, colour in [
    (axes[0], pos_bigrams, "Top Bigrams — Positive", "#7bc67a"),
    (axes[1], neg_bigrams, "Top Bigrams — Negative", "#e06b6b"),
]:
    ax.barh(bdf["bigram"][::-1], bdf["count"][::-1], color=colour, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Frequency")

plt.tight_layout()
plt.savefig("08b_ngrams.png", dpi=150)
plt.show()

# ── 3) VADER Category Heatmap ──────────────────────────────────
import seaborn as sns

# Bin compound score into categories
df["sentiment_cat"] = pd.cut(
    df["vader_compound"],
    bins=[-1, -0.5, -0.05, 0.05, 0.5, 1],
    labels=["Very Neg", "Negative", "Neutral", "Positive", "Very Pos"],
)

heatmap_data = df.groupby(["Liked", "sentiment_cat"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(9, 3))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="RdYlGn", ax=ax, linewidths=0.5)
ax.set_title("Review Distribution: VADER Category vs True Label")
ax.set_xlabel("VADER Sentiment Category")
ax.set_ylabel("True Label (0=Neg, 1=Pos)")
plt.tight_layout()
plt.savefig("08c_heatmap.png", dpi=150)
plt.show()
print(" Visualizations saved: 08a_wordclouds.png | 08b_ngrams.png | 08c_heatmap.png")
