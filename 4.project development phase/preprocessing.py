"""
Preprocessing utilities for Credit Card Approval Prediction.
Handles data cleaning, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ─── Column Definitions ────────────────────────────────────────────────
FEATURE_COLS = [
    'Gender', 'Age', 'Debt', 'Married', 'BankCustomer',
    'EducationLevel', 'Ethnicity', 'YearsEmployed',
    'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense',
    'Citizen', 'ZipCode', 'Income'
]

CATEGORICAL_COLS = [
    'Gender', 'Married', 'BankCustomer', 'EducationLevel',
    'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen'
]

NUMERICAL_COLS = [
    'Age', 'Debt', 'YearsEmployed', 'CreditScore', 'ZipCode', 'Income'
]

TARGET_COL = 'Approved'


class CreditCardPreprocessor:
    """
    Encapsulates all preprocessing steps so that the same
    transformations applied during training can be replayed
    at inference time in the Flask application.
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self._fitted = False

    # ── Fit ──────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame):
        """Learn encoding maps and scaler parameters from training data."""

        # Replace '?' with NaN
        df = df.replace('?', np.nan).copy()

        # Impute missing values
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                df[col].fillna(mode_val, inplace=True)

        for col in NUMERICAL_COLS:
            if col in df.columns:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

        # Label-encode categoricals
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Fit scaler on numerical columns
        self.scaler.fit(df[NUMERICAL_COLS].astype(float))

        # Store feature order
        all_features = CATEGORICAL_COLS + NUMERICAL_COLS
        self.feature_names = [c for c in all_features if c in df.columns]
        self._fitted = True
        return self

    # ── Transform ───────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations to new data."""

        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        df = df.replace('?', np.nan).copy()

        # Impute missing values (use training statistics)
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.label_encoders:
                mode_val = self.label_encoders[col].classes_[0]
                df[col] = df[col].fillna(mode_val)

        for col in NUMERICAL_COLS:
            if col in df.columns:
                median_val = 0  # fallback
                df[col] = df[col].fillna(median_val)

        # Label-encode using fitted encoders
        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen labels gracefully
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Scale numerical columns — ensure no NaN remain
        for col in NUMERICAL_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        df[NUMERICAL_COLS] = self.scaler.transform(df[NUMERICAL_COLS].astype(float))

        # Final NaN check — replace any remaining NaN with 0
        result = df[self.feature_names].fillna(0)

        return result

    # ── Fit-Transform shortcut ──────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    # ── Persistence ──────────────────────────────────────────────────
    def save(self, path: str = 'models/preprocessor.joblib'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = 'models/preprocessor.joblib'):
        return joblib.load(path)


def load_raw_data(data_dir: str = 'data') -> pd.DataFrame:
    """
    Load the UCI Credit Card Approval dataset (crx.data).
    The file is comma-separated with no header.
    """
    col_names = [
        'Gender', 'Age', 'Debt', 'Married', 'BankCustomer',
        'EducationLevel', 'Ethnicity', 'YearsEmployed',
        'PriorDefault', 'Employed', 'CreditScore', 'DriversLicense',
        'Citizen', 'ZipCode', 'Income', 'Approved'
    ]
    filepath = os.path.join(data_dir, 'crx.data')
    df = pd.read_csv(filepath, header=None, names=col_names, na_values='?')
    return df
