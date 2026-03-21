# ============================================================
# 06_VADER Sentiment + Hybrid Analysis
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv("reviews_cleaned.csv")

# ── Apply VADER ───────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()


def get_vader_scores(text):
    scores = sia.polarity_scores(str(text))
    return pd.Series(
        {
            "vader_neg": scores["neg"],
            "vader_neu": scores["neu"],
            "vader_pos": scores["pos"],
            "vader_compound": scores["compound"],
        }
    )


vader_df = df["Review"].apply(get_vader_scores)
df = pd.concat([df, vader_df], axis=1)

# VADER binary: compound ≥ 0.05 = positive (standard threshold)
df["vader_pred"] = (df["vader_compound"] >= 0.05).astype(int)
from sklearn.metrics import accuracy_score

vader_acc = accuracy_score(df["Liked"], df["vader_pred"])
print(f"\nVADER accuracy vs ground truth: {vader_acc:.4f}")

# ── VADER vs Label analysis ───────────────────────────────────
print("\n─── Avg VADER compound by label ───")
print(df.groupby("Liked")["vader_compound"].describe())

# Reviews where VADER and label disagree (interesting cases)
disagree = df[df["vader_pred"] != df["Liked"]]
print(f"\nVADER-label disagreements: {len(disagree)} ({len(disagree)/len(df):.1%})")
print("Sample disagreements (VADER missed):")
print(disagree["Review"].head(5).to_string())

# ── Visualisation ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Compound score distribution by class
for label, colour in [(1, "#7bc67a"), (0, "#e06b6b")]:
    axes[0].hist(
        df[df["Liked"] == label]["vader_compound"],
        bins=40,
        alpha=0.6,
        label=f'{"Positive" if label else "Negative"}',
        color=colour,
    )
axes[0].axvline(x=0.05, color="gray", linestyle="--", label="Threshold (0.05)")
axes[0].set_title("VADER Compound Score Distribution")
axes[0].set_xlabel("Compound Score")
axes[0].legend()

# Pos/Neg/Neu scores by class
vader_means = df.groupby("Liked")[["vader_pos", "vader_neg", "vader_neu"]].mean()
vader_means.plot(
    kind="bar",
    ax=axes[1],
    color=["#7bc67a", "#e06b6b", "#b0b0b0"],
    edgecolor="white",
    rot=0,
)
axes[1].set_title("VADER Score Components by Label")
axes[1].set_xlabel("Liked (0=No, 1=Yes)")
axes[1].legend(["Positive", "Negative", "Neutral"])

plt.tight_layout()
plt.savefig("06_vader.png", dpi=150, bbox_inches="tight")
plt.show()
df.to_csv("reviews_with_vader.csv", index=False)
print(" VADER done. Saved: 06_vader.png | reviews_with_vader.csv")
