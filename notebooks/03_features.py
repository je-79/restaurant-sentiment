# ============================================================
# 03_Feature Engineering
# ============================================================
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("reviews_cleaned.csv")
df["cleaned"] = df["cleaned"].fillna("")

# ── A: TF-IDF with unigrams + bigrams ─────────────────────────
# max_features=5000 captures enough signal without noise
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  # unigrams and bigrams
    min_df=2,  # must appear in ≥2 reviews
    max_df=0.95,  # ignore terms in >95% of reviews
    sublinear_tf=True,  # log(1+tf) — dampens high freqs
    strip_accents="unicode",
)

X = vectorizer.fit_transform(df["cleaned"])
y = df["Liked"].values

# ── B: Manual features (sentiment proxies) ────────────────────
# These features augment TF-IDF with engineered signals
df["exclamation_count"] = df["Review"].str.count("!")
df["question_count"] = df["Review"].str.count(r"\?")
df["caps_ratio"] = df["Review"].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(x), 1)
)
df["word_count"] = df["cleaned"].str.split().str.len()

from scipy.sparse import hstack, csr_matrix

extra_feats = csr_matrix(
    df[["exclamation_count", "question_count", "caps_ratio", "word_count"]].values
)
X_combined = hstack([X, extra_feats])

# ── Train/Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Feature dims     : {X_train.shape[1]}")
print(f"Train balance    : {y_train.mean():.2%} positive")
print(f"Test balance     : {y_test.mean():.2%} positive")

# ── Save artifacts ────────────────────────────────────────────
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump((X_train, X_test, y_train, y_test), "train_test_split.pkl")
print("Features saved: tfidf_vectorizer.pkl | train_test_split.pkl")
