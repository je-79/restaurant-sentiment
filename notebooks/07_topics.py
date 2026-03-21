# ============================================================
# 07_LDA Topic Modeling (Negative Reviews)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("reviews_with_vader.csv")
df["cleaned"] = df["cleaned"].fillna("")

# ── Separate by sentiment ─────────────────────────────────────
neg_reviews = df[df["Liked"] == 0]["cleaned"]
pos_reviews = df[df["Liked"] == 1]["cleaned"]


def run_lda(texts, n_topics=5, n_words=10, title=""):
    """Run LDA and print top words per topic."""
    cv = CountVectorizer(max_features=1000, min_df=2, max_df=0.9)
    X = cv.fit_transform(texts)
    feat = cv.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_decay=0.7,
        random_state=42,
        n_jobs=-1,
    )
    lda.fit(X)

    print(f"\n═══ {title} ═══")
    topic_words = {}
    for i, comp in enumerate(lda.components_):
        words = [feat[j] for j in comp.argsort()[: -n_words - 1 : -1]]
        topic_words[f"Topic {i+1}"] = words
        print(f"  Topic {i+1}: {', '.join(words)}")
    return topic_words, lda, cv


neg_topics, neg_lda, neg_cv = run_lda(
    neg_reviews, n_topics=5, title="NEGATIVE REVIEW TOPICS"
)
pos_topics, pos_lda, pos_cv = run_lda(
    pos_reviews, n_topics=5, title="POSITIVE REVIEW TOPICS"
)

# ── Visualise negative topic word weights ────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
colours = ["#e06b6b", "#e8914a", "#a78bda", "#5b9bd5", "#4ab8a0"]

feat_names = neg_cv.get_feature_names_out()
for i, (comp, ax, color) in enumerate(zip(neg_lda.components_, axes, colours)):
    top_idx = comp.argsort()[:-11:-1]
    top_words = feat_names[top_idx]
    top_vals = comp[top_idx]
    ax.barh(top_words[::-1], top_vals[::-1], color=color, alpha=0.85)
    ax.set_title(f"Neg Topic {i+1}", fontsize=10)
    ax.tick_params(labelsize=8)

plt.suptitle("Negative Review Topics — LDA Word Distributions", fontsize=12)
plt.tight_layout()
plt.savefig("07_lda_topics.png", dpi=150, bbox_inches="tight")
plt.show()
print(" LDA complete. Saved: 07_lda_topics.png")
