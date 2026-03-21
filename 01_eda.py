# ==============================================
# 01_Exploratory Data Analysis
# ===============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Restaurant_Reviews.csv", sep=";")

print("\n═══ DATASET INFO ═══")
print(df.info())
print("\n─── First 5 rows ───")
print(df.head())
print("\n─── Shape ───", df.shape)
print("\n─── Missing values ───")
print(df.isnull().sum())
print("\n─── Class Distribution ───")
print(df["Liked"].value_counts())
print("\n─── Class Balance (%) ───")
print(df["Liked"].value_counts(normalize=True).mul(100).round(2))

df["review_length"] = df["Review"].str.len()
df["word_count"] = df["Review"].str.split().str.len()

print("\n─── Text Stats by Sentiment Class ───")
print(
    df.groupby("Liked")[["review_length", "word_count"]]
    .agg(["mean", "median", "std"])
    .round(2)
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
counts = df["Liked"].value_counts()
axes[0].bar(
    ["Negative (0)", "Positive (1)"],
    counts.values,
    color=["#e06b6b", "#7bc67a"],
    edgecolor="white",
    linewidth=1.5,
)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

for label, colour in [(1, "#7bc67a"), (0, "#e06b6b")]:
    axes[1].hist(
        df[df["Liked"] == label]["review_length"],
        bins=30,
        alpha=0.6,
        label=f'{"Positive" if label else "Negative"}',
        color=colour,
    )
axes[1].set_title("Review Length Distribution")
axes[1].set_xlabel("Characters")
axes[1].legend()

df.boxplot(column="word_count", by="Liked", ax=axes[2], boxprops=dict(color="#5b9bd5"))
axes[2].set_title("Word Count by Sentiment")
axes[2].set_xlabel("Liked (0=No, 1=Yes)")

plt.suptitle("")
plt.tight_layout()
plt.savefig("01_eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n EDA complete. Chart saved: 01_eda_overview.png")
