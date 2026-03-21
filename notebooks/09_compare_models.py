# ============================================================
# 09_Visual Comparison of All Models
# ============================================================
import joblib
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")

# ── Load artifacts ────────────────────────────────────────────
X_train, X_test, y_train, y_test = joblib.load("train_test_split.pkl")
models = joblib.load("trained_models.pkl")

# ── Collect metrics ───────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
names = []
acc_test = []
auc_test = []
cv_means = []
cv_stds = []
calibrated_models = {}

print("═══ ALL MODELS PERFORMANCE ═══\n")
print(f'{"Model":<25} {"CV Acc":>8} {"±Std":>8} {"Test Acc":>10} {"ROC-AUC":>10}')
print("─" * 65)

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    # Test accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # ROC-AUC via calibration
    cal = CalibratedClassifierCV(model, cv=3)
    cal.fit(X_train, y_train)
    y_proba = cal.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    names.append(name)
    acc_test.append(acc)
    auc_test.append(auc)
    cv_means.append(cv_scores.mean())
    cv_stds.append(cv_scores.std())
    calibrated_models[name] = (cal, y_proba)

    print(
        f"{name:<25} {cv_scores.mean():>8.4f} {cv_scores.std():>8.4f} "
        f"{acc:>10.4f} {auc:>10.4f}"
    )

# ── Plot ──────────────────────────────────────────────────────
colours = ["#e06b6b", "#5b9bd5", "#7bc67a", "#a78bda", "#e8914a"]
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── 1) CV Accuracy comparison ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.barh(
    names,
    cv_means,
    xerr=cv_stds,
    color=colours,
    alpha=0.85,
    edgecolor="white",
    linewidth=1.2,
    capsize=4,
)
ax1.set_xlim(0.5, 1.0)
ax1.set_title("5-Fold CV Accuracy", fontsize=13, fontweight="bold")
ax1.set_xlabel("Accuracy")
ax1.axvline(x=max(cv_means), color="gray", linestyle="--", linewidth=0.8)
for bar, val in zip(bars, cv_means):
    ax1.text(
        val + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=10,
    )

# ── 2) Test Accuracy comparison ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(
    names, acc_test, color=colours, alpha=0.85, edgecolor="white", linewidth=1.2
)
ax2.set_xlim(0.5, 1.0)
ax2.set_title("Test Set Accuracy", fontsize=13, fontweight="bold")
ax2.set_xlabel("Accuracy")
ax2.axvline(x=max(acc_test), color="gray", linestyle="--", linewidth=0.8)
for bar, val in zip(bars2, acc_test):
    ax2.text(
        val + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=10,
    )

# ── 3) ROC-AUC comparison ─────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.barh(
    names, auc_test, color=colours, alpha=0.85, edgecolor="white", linewidth=1.2
)
ax3.set_xlim(0.5, 1.0)
ax3.set_title("ROC-AUC Score", fontsize=13, fontweight="bold")
ax3.set_xlabel("AUC")
ax3.axvline(x=max(auc_test), color="gray", linestyle="--", linewidth=0.8)
for bar, val in zip(bars3, auc_test):
    ax3.text(
        val + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=10,
    )

# ── 4) ROC Curves all models ──────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
for name, colour in zip(names, colours):
    cal, y_proba = calibrated_models[name]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax4.plot(fpr, tpr, color=colour, lw=2, label=f"{name} (AUC={auc:.3f})")
ax4.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax4.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.legend(loc="lower right", fontsize=9)
ax4.grid(alpha=0.3)

# ── 5) Metric summary table ───────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
table_data = [["Model", "CV Acc", "Test Acc", "AUC"]]
for n, cv_m, acc, auc in zip(names, cv_means, acc_test, auc_test):
    short = n.replace(" ", "\n") if len(n) > 12 else n
    table_data.append([short, f"{cv_m:.3f}", f"{acc:.3f}", f"{auc:.3f}"])

table = ax5.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0, 0.1, 1, 0.85],
)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Highlight best in each column
best_cv = cv_means.index(max(cv_means))
best_acc = acc_test.index(max(acc_test))
best_auc = auc_test.index(max(auc_test))

for col, best_row in [(1, best_cv), (2, best_acc), (3, best_auc)]:
    table[best_row + 1, col].set_facecolor("#c8f0c8")
    table[best_row + 1, col].set_text_props(fontweight="bold")

for col in range(4):
    table[0, col].set_facecolor("#2c2c2c")
    table[0, col].set_text_props(color="white", fontweight="bold")

ax5.set_title("Performance Summary\n(green = best)", fontsize=11, fontweight="bold")

plt.suptitle(
    "Restaurant Sentiment Analysis — Model Comparison",
    fontsize=16,
    fontweight="bold",
    y=1.01,
)
plt.savefig("09_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 09_model_comparison.png")
