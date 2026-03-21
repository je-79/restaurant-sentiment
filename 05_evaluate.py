# ============================================================
# 05_Full Evaluation of Best Model
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings("ignore")

# ── Load artifacts ────────────────────────────────────────────
X_train, X_test, y_train, y_test = joblib.load("train_test_split.pkl")
trained_models = joblib.load("trained_models.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ── Choose best model: LinearSVC (calibrate for probabilities) ─
best_model = trained_models["LinearSVC"]
calibrated = CalibratedClassifierCV(best_model, cv=3)
calibrated.fit(X_train, y_train)

y_pred = calibrated.predict(X_test)
y_proba = calibrated.predict_proba(X_test)[:, 1]

print("\n═══ CLASSIFICATION REPORT ═══")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# ── Plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# 1) Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Negative", "Positive"]
).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix — LinearSVC")

# 2) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
axes[1].plot(fpr, tpr, color="#5b9bd5", lw=2, label=f"AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

# 3) Top 20 TF-IDF features for each class
feat_names = vectorizer.get_feature_names_out()
# Use underlying LinearSVC coefs
coef = trained_models["LinearSVC"].coef_[0]
top_pos_idx = coef.argsort()[-15:]
top_neg_idx = coef.argsort()[:15]

colors = ["#7bc67a"] * 15 + ["#e06b6b"] * 15
labels = list(feat_names[top_pos_idx]) + list(feat_names[top_neg_idx])
values = list(coef[top_pos_idx]) + list(coef[top_neg_idx])

sorted_pairs = sorted(zip(labels, values, colors), key=lambda x: x[1])
labs, vals, cols = zip(*sorted_pairs)
axes[2].barh(labs, vals, color=cols)
axes[2].axvline(x=0, color="black", linewidth=0.8)
axes[2].set_title("Top Predictive Words (SVC Coefficients)")
axes[2].set_xlabel("Coefficient Weight")

plt.tight_layout()
plt.savefig("05_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Save best model ───────────────────────────────────────────
joblib.dump(calibrated, "best_model_svc.pkl")
print("Evaluation complete. Saved: 05_evaluation.png | best_model_svc.pkl")
