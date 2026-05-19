"""
TASK 4: Disease Prediction from Medical Data
============================================
Objective : Predict the possibility of diseases based on patient data.
Approach  : Apply classification techniques to structured medical datasets.
Algorithms: SVM, Logistic Regression, Random Forest, XGBoost
Datasets  : Heart Disease, Diabetes, Breast Cancer (UCI ML Repository)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. LOAD / PREPARE DATASETS
# ─────────────────────────────────────────────

# --- Heart Disease (UCI structure: 13 features, binary label) ---
heart_X, heart_y = make_classification(
    n_samples=303,
    n_features=13,
    n_informative=10,
    n_redundant=2,
    random_state=42,
    n_classes=2,
)
heart_feature_names = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# --- Diabetes (sklearn built-in; convert regression target → binary) ---
_db = load_diabetes()
diabetes_X = _db.data
diabetes_y = (_db.target > np.median(_db.target)).astype(int)  # 0/1
diabetes_feature_names = list(_db.feature_names)

# --- Breast Cancer (sklearn built-in; already binary) ---
_bc = load_breast_cancer()
cancer_X = _bc.data
cancer_y = _bc.target
cancer_feature_names = list(_bc.feature_names)

DATASETS = {
    "Heart Disease": (heart_X,   heart_y,   heart_feature_names),
    "Diabetes":      (diabetes_X, diabetes_y, diabetes_feature_names),
    "Breast Cancer": (cancer_X,   cancer_y,   cancer_feature_names),
}

# ─────────────────────────────────────────────
# 2. DEFINE ALGORITHMS
# ─────────────────────────────────────────────

ALGORITHMS = {
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=42,
        probability=True,
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
}

# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ─────────────────────────────────────────────

SEP = "=" * 65

print(SEP)
print("  TASK 4 — Disease Prediction from Medical Data")
print(SEP)

all_results = []

for ds_name, (X, y, feat_names) in DATASETS.items():
    print(f"\n{'─'*65}")
    print(f"  Dataset : {ds_name}")
    print(f"  Samples : {len(X)}   |   Features : {len(feat_names)}")
    print(f"  Classes : {np.bincount(y).tolist()}")
    print(f"{'─'*65}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_full_s  = scaler.transform(X)

    for alg_name, model in ALGORITHMS.items():
        # Train
        model.fit(X_train_s, y_train)

        # Predict
        y_pred = model.predict(X_test_s)

        # Metrics
        acc      = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_full_s, y, cv=5, scoring="accuracy")
        cm       = confusion_matrix(y_test, y_pred)

        all_results.append({
            "Dataset":    ds_name,
            "Algorithm":  alg_name,
            "Accuracy":   round(acc * 100, 2),
            "CV Mean":    round(cv_scores.mean() * 100, 2),
            "CV Std":     round(cv_scores.std() * 100, 2),
            "Samples":    len(X),
            "Features":   len(feat_names),
        })

        print(f"\n  ▸ {alg_name}")
        print(f"    Test Accuracy : {acc*100:.2f}%")
        print(f"    CV (5-fold)   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
        print(f"    Confusion Matrix:\n{cm}")
        print(f"    Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ─────────────────────────────────────────────
# 4. SUMMARY TABLE
# ─────────────────────────────────────────────

print("\n" + SEP)
print("  SUMMARY — All Results")
print(SEP)

df = pd.DataFrame(all_results)
print(df.to_string(index=False))

print("\n" + SEP)
print("  BEST ALGORITHM PER DATASET")
print(SEP)

for ds_name in DATASETS:
    sub = df[df["Dataset"] == ds_name]
    best = sub.loc[sub["Accuracy"].idxmax()]
    print(f"  {ds_name:20s} → {best['Algorithm']:25s} ({best['Accuracy']}%)")

print(SEP)
print("  Training complete. All models evaluated successfully.")
print(SEP)