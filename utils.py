# Name : Bansi Ajagia
# Subject : Machine Learning
# Project : Cardiovascular Disease Prediction System

# ============================================================
# utils.py — Shared helper functions
# ============================================================

import pandas as pd
import numpy as np
from config import DATA_PATH, FEATURE_COLS, TARGET_COL


def load_raw_data() -> pd.DataFrame:
    """Load the raw cardio CSV and return a DataFrame."""
    df = pd.read_csv(DATA_PATH, sep=";")
    return df


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
      1. Convert age (days → years)
      2. Drop duplicates & obvious outliers
      3. Select features + target
      4. Return X (numpy array) and y (numpy array)
    """
    df = df.copy()

    # Age: days → years
    df["age_years"] = (df["age"] / 365).round(0).astype(int)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop physiologically impossible rows (blood pressure, height, weight)
    df = df[df["ap_hi"].between(60, 250)]
    df = df[df["ap_lo"].between(40, 200)]
    df = df[df["height"].between(100, 250)]
    df = df[df["weight"].between(30, 250)]

    # Gender: original is 1=female,2=male → keep as-is (numeric)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


def get_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the engineered feature columns as a DataFrame."""
    df = df.copy()
    df["age_years"] = (df["age"] / 365).round(0).astype(int)
    return df[FEATURE_COLS + [TARGET_COL]]
