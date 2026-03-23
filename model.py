# Name : Bansi Ajagia
# Subject : Machine Learning
# Project : Cardiovascular Disease Prediction System

# ============================================================
# model.py — Training, saving, and loading the ML model
# ============================================================

import os, pickle, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (MODEL_PATH, METRICS_PATH, CONF_MATRIX_PATH,
                    TEST_SIZE, RANDOM_STATE)
from utils import load_raw_data, preprocess


# ── Manual rule-based baseline ───────────────────────────────
class RuleBasedClassifier:
    """
    Simple threshold-based classifier (no library ML used).
    Flags cardiovascular risk when systolic BP ≥ 140
    OR cholesterol is well above normal (3) OR age ≥ 55.
    """
    def fit(self, X, y):
        # Nothing to learn — pure rules
        return self

    def predict(self, X):
        # Columns: age_years(0) gender(1) height(2) weight(3)
        #          ap_hi(4) ap_lo(5) cholesterol(6) gluc(7)
        #          smoke(8) alco(9) active(10)
        age       = X[:, 0]
        ap_hi     = X[:, 4]
        chol      = X[:, 6]
        preds = ((ap_hi >= 140) | (chol == 3) | (age >= 55)).astype(int)
        return preds


def train_and_save():
    """Train LR, RF, and rule-based models; save best; return metrics dict."""
    df  = load_raw_data()
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}

    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_sc, y_train)
    results["Logistic Regression"] = _evaluate(lr, X_test_sc, y_test)

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    results["Random Forest"] = _evaluate(rf, X_test, y_test)

    # 3. Rule-Based (manual — no sklearn training)
    rb = RuleBasedClassifier().fit(X_train, y_train)
    results["Rule-Based"] = _evaluate(rb, X_test, y_test)

    # Pick best model by accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = lr if best_name == "Logistic Regression" else rf
    best_scaler = scaler if best_name == "Logistic Regression" else None

    # Save model + scaler
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "scaler": best_scaler,
                     "model_name": best_name}, f)

    # Save metrics
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        for name, m in results.items():
            f.write(f"=== {name} ===\n")
            for k, v in m.items():
                if k != "cm":
                    f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")

    # Save confusion matrix for best model
    best_res = results[best_name]
    _save_confusion_matrix(best_res["cm"], best_name)

    return results, best_name


def _evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1"       : f1_score(y_test, y_pred, zero_division=0),
        "cm"       : confusion_matrix(y_test, y_pred),
    }


def _save_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["No Disease", "Disease"],
           yticklabels=["No Disease", "Disease"],
           xlabel="Predicted", ylabel="Actual",
           title=f"Confusion Matrix — {title}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PATH, dpi=120)
    plt.close()


def load_model():
    """Load trained model bundle from disk."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_single(input_dict: dict) -> int:
    """
    Predict for a single patient dict.
    Returns 0 (No Disease) or 1 (High Risk).
    """
    bundle = load_model()
    model  = bundle["model"]
    scaler = bundle["scaler"]

    from config import FEATURE_COLS
    row = np.array([[input_dict[c] for c in FEATURE_COLS]])

    if scaler is not None:
        row = scaler.transform(row)

    return int(model.predict(row)[0])
