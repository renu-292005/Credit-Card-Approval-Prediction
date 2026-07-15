"""
Model training and evaluation pipeline for Credit Card Approval Prediction.
Trains multiple classifiers, evaluates them, and saves the best model.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import json

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import CreditCardPreprocessor, load_raw_data, TARGET_COL


def train_and_evaluate(data_dir: str = 'data', model_dir: str = 'models'):
    """
    Complete training pipeline:
      1. Load data
      2. Preprocess
      3. Handle class imbalance with SMOTE
      4. Train multiple models
      5. Evaluate and compare
      6. Save best model + preprocessor
    """

    os.makedirs(model_dir, exist_ok=True)

    # ── 1. Load Data ────────────────────────────────────────────────
    print("=" * 70)
    print("  CREDIT CARD APPROVAL PREDICTION — MODEL TRAINING PIPELINE")
    print("=" * 70)
    df = load_raw_data(data_dir)
    print(f"\n[INFO] Raw dataset shape : {df.shape}")
    print(f"[INFO] Target distribution:\n{df[TARGET_COL].value_counts()}")

    # ── 2. Preprocess ───────────────────────────────────────────────
    preprocessor = CreditCardPreprocessor()
    X = preprocessor.fit_transform(df)
    y = df[TARGET_COL].map({'+': 1, '-': 0})  # encode target

    print(f"\n[INFO] Feature matrix shape after preprocessing: {X.shape}")
    print(f"[INFO] Features used: {list(X.columns)}")

    # ── 3. Train-Test Split ─────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[INFO] Training samples : {X_train.shape[0]}")
    print(f"[INFO] Testing samples  : {X_test.shape[0]}")

    # ── 4. Handle Class Imbalance (SMOTE) ──────────────────────────
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"\n[INFO] After SMOTE — Training samples: {X_train_res.shape[0]}")
    print(f"[INFO] After SMOTE — Class distribution: {dict(pd.Series(y_train_res).value_counts())}")

    # ── 5. Define Models ────────────────────────────────────────────
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric='logloss', random_state=42
        ),
    }

    # ── 6. Train & Evaluate Each Model ─────────────────────────────
    results = {}
    best_f1 = 0
    best_model_name = None
    best_model = None

    print("\n" + "-" * 70)
    print("  MODEL TRAINING & EVALUATION RESULTS")
    print("-" * 70)

    for name, model in models.items():
        print(f"\n▶ Training: {name}")
        model.fit(X_train_res, y_train_res)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        roc  = roc_auc_score(y_test, y_prob)

        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='f1')

        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc, 4),
            'cv_f1_mean': round(cv_scores.mean(), 4),
            'cv_f1_std': round(cv_scores.std(), 4),
        }

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc:.4f}")
        print(f"  CV F1     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    # ── 7. Save Best Model & Preprocessor ──────────────────────────
    print("\n" + "=" * 70)
    print(f"  ★ BEST MODEL: {best_model_name}  (F1 = {best_f1:.4f})")
    print("=" * 70)

    model_path = os.path.join(model_dir, 'best_model.joblib')
    joblib.dump(best_model, model_path)
    preprocessor.save(os.path.join(model_dir, 'preprocessor.joblib'))

    # Save results
    with open(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── 8. Detailed Report for Best Model ───────────────────────────
    y_pred_best = best_model.predict(X_test)
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred_best, target_names=['Rejected', 'Approved']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)

    # ── 9. Feature Importance (if available) ────────────────────────
    if hasattr(best_model, 'feature_importances_'):
        importance = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        print("\nFeature Importances:")
        print(importance.to_string())

    print(f"\n[INFO] Model saved to    : {model_path}")
    print(f"[INFO] Preprocessor saved to : {os.path.join(model_dir, 'preprocessor.joblib')}")
    print(f"[INFO] Results saved to      : {os.path.join(model_dir, 'results.json')}")

    return best_model, preprocessor, results


if __name__ == '__main__':
    train_and_evaluate()
