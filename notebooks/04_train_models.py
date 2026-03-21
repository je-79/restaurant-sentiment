# ============================================================
# 04_Multi-Classifier Training + CV
# ============================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
import warnings

warnings.filterwarnings("ignore")

# ── Load features ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = joblib.load("train_test_split.pkl")

# ── Define models ─────────────────────────────────────────────
models = {
    "ComplementNB": ComplementNB(alpha=0.5),
    "Logistic Regression": LogisticRegression(
        C=1.5, max_iter=1000, solver="lbfgs", random_state=42
    ),
    "LinearSVC": LinearSVC(C=0.8, max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n═══ 5-FOLD CROSS-VALIDATION RESULTS ═══")
print(f"{'Model':<25} {'Mean Acc':>10} {'Std':>8}")
print("─" * 45)

results = {}
trained_models = {}
for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    results[name] = scores
    print(f"{name:<25} {scores.mean():.4f}   ±{scores.std():.4f}")
    model.fit(X_train, y_train)
    trained_models[name] = model

# ── Save all trained models ───────────────────────────────────
joblib.dump(trained_models, "trained_models.pkl")
print("\n All models trained and saved: trained_models.pkl")
