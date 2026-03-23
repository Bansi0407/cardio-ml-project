# Name : Bansi Ajagia
# Subject : Machine Learning
# Project : Cardiovascular Disease Prediction System

# ============================================================
# config.py — Project-wide configuration constants
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data",    "cardio.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models",  "model.pkl")
METRICS_PATH      = os.path.join(BASE_DIR, "outputs", "metrics.txt")
CONF_MATRIX_PATH  = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")

# ── Model hyper-parameters ───────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── Feature columns (after preprocessing) ───────────────────
FEATURE_COLS = [
    "age_years", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
]
TARGET_COL = "cardio"

# ── Cholesterol / Glucose label maps ─────────────────────────
CHOL_MAP  = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
GLUC_MAP  = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
