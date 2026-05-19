# 🏥 Disease Prediction from Medical Data

> A machine learning project that predicts disease outcomes using structured patient data across three real-world medical datasets.

---

## 📌 Overview

This project applies **supervised classification** algorithms to predict the presence or absence of diseases in patients. It evaluates four well-known ML algorithms across three medical datasets, with full performance reporting including test accuracy, 5-fold cross-validation, confusion matrices, and classification reports.

---

## 🎯 Objective

- Predict disease possibility from patient medical records
- Compare classification algorithms on medical datasets
- Identify the best-performing model per dataset

---

## 📂 Datasets

| Dataset | Samples | Features | Source |
|---|---|---|---|
| Heart Disease | 303 | 13 | UCI ML Repository (simulated structure) |
| Diabetes | 442 | 10 | sklearn built-in (`load_diabetes`) |
| Breast Cancer | 569 | 30 | sklearn built-in (`load_breast_cancer`) |

### Dataset Details

**Heart Disease**
- 13 clinical features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope, number of vessels, thalassemia
- Binary target: Positive (disease present) / Negative (no disease)
- Balanced classes: [154, 149]

**Diabetes**
- 10 features: age, sex, BMI, blood pressure, and 6 serum measurements
- Target binarized at median: above median = Positive, below = Negative
- Perfectly balanced: [221, 221]

**Breast Cancer**
- 30 features derived from digitized images of fine needle aspirate (FNA) of breast mass
- Binary target: Malignant / Benign
- Classes: [212 Malignant, 357 Benign]

---

## 🤖 Algorithms

| Algorithm | Key Hyperparameters |
|---|---|
| SVM | `kernel=rbf`, `C=1.0`, `gamma=scale` |
| Logistic Regression | `solver=lbfgs`, `max_iter=1000` |
| Random Forest | `n_estimators=100`, `max_depth=None` |
| XGBoost | `n_estimators=100`, `learning_rate=0.1`, `max_depth=6` |

---

## ⚙️ Methodology

```
Raw Dataset
    ↓
Train/Test Split (80% / 20%, stratified)
    ↓
StandardScaler (fit on train, transform both)
    ↓
Model Training
    ↓
Test Set Prediction
    ↓
Evaluation: Accuracy, CV (5-fold), Confusion Matrix, Classification Report
```

- All features are **standardized** using `StandardScaler` before training
- **Stratified split** ensures class balance is maintained in both sets
- **5-fold cross-validation** is run on the full scaled dataset to check generalization
- `random_state=42` used throughout for reproducibility

---

## 📊 Results Summary

### Test Accuracy

| Dataset | SVM | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|---|
| Heart Disease | **91.80%** | **91.80%** | **91.80%** | 88.52% |
| Diabetes | 74.16% | 74.16% | **76.40%** | 73.03% |
| Breast Cancer | **98.25%** | **98.25%** | 95.61% | 94.74% |

### CV Mean Accuracy (5-fold)

| Dataset | SVM | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|---|
| Heart Disease | **95.04%** ± 2.12 | 89.78% ± 3.48 | 91.41% ± 1.96 | 90.09% ± 1.85 |
| Diabetes | 72.61% ± 4.70 | **73.74%** ± 5.54 | 68.53% ± 3.74 | 71.02% ± 3.39 |
| Breast Cancer | 97.36% ± 1.47 | **98.07%** ± 0.65 | 95.61% ± 2.28 | 96.84% ± 1.80 |

### 🏆 Best Algorithm Per Dataset

| Dataset | Best Algorithm | Accuracy |
|---|---|---|
| Heart Disease | SVM | 91.80% |
| Diabetes | Random Forest | 76.40% |
| Breast Cancer | SVM / Logistic Regression | 98.25% |

---

## 🔍 Key Findings

1. **Breast Cancer** was the easiest dataset — high accuracy (98.25%) due to 30 well-engineered features from image analysis
2. **Diabetes** was the hardest — max 76.40% because the continuous regression target was binarized at median, losing nuance
3. **SVM with RBF kernel** was the most consistent top performer across all datasets
4. **Logistic Regression** was surprisingly competitive despite its simplicity, especially on Breast Cancer (best CV: 98.07%)
5. **XGBoost** underperformed relative to expectations, possibly due to limited hyperparameter tuning
6. **Random Forest** showed overfitting signs on Diabetes (76.40% test vs 68.53% CV mean)

---

## 🛠️ Installation & Usage

### Requirements

```bash
pip install numpy pandas scikit-learn xgboost
```

### Run

```bash
python task4_disease_prediction.py
```

### Output

The script prints:
- Per-dataset, per-algorithm: test accuracy, CV scores, confusion matrix, classification report
- Final summary table of all results
- Best algorithm per dataset

---

## 📁 Project Structure

```
disease-prediction/
│
├── task4_disease_prediction.py   # Main script
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

---

## 📦 Dependencies

```
numpy
pandas
scikit-learn
xgboost
```

---

## 📝 Notes

- Heart Disease dataset is **synthetically generated** using `make_classification` to match the UCI structure (303 samples, 13 features). For real results, replace with the actual UCI Heart Disease CSV.
- Diabetes dataset uses sklearn's built-in regression dataset, **binarized at median** to simulate classification.
- All models use default or lightly tuned hyperparameters. Further tuning (GridSearchCV, RandomizedSearchCV) could improve results, especially for XGBoost.
