# scripts/run_modeling.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

# -----------------------------
# Load processed data
# -----------------------------
def load_data(dataset_name):
    processed_dir = os.path.join('data', 'processed')
    if dataset_name == 'fraud':
        train_file = os.path.join(processed_dir, 'fraud_train.csv')
        test_file = os.path.join(processed_dir, 'fraud_test.csv')
        target = 'class'
    elif dataset_name == 'creditcard':
        train_file = os.path.join(processed_dir, 'creditcard_train.csv')
        test_file = os.path.join(processed_dir, 'creditcard_test.csv')
        target = 'Class'
    else:
        raise ValueError("dataset_name must be 'fraud' or 'creditcard'")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    return X_train, X_test, y_train, y_test

# -----------------------------
# Model evaluation metrics
# -----------------------------
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    f1 = f1_score(y, y_pred)
    
    precision, recall, _ = precision_recall_curve(y, y_proba)
    auc_pr = auc(recall, precision)
    
    cm = confusion_matrix(y, y_pred)
    
    return f1, auc_pr, cm

# -----------------------------
# Modeling pipeline
# -----------------------------
def run_modeling(dataset_name):
    print(f"\n=== Modeling for {dataset_name} dataset ===")

    X_train, X_test, y_train, y_test = load_data(dataset_name)

    # -----------------------------
    # Handle class imbalance with SMOTE
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Training data before SMOTE: {len(y_train)}, after SMOTE: {len(y_train_res)}")

    # -----------------------------
    # Baseline Model: Logistic Regression
    # -----------------------------
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_res, y_train_res)

    f1_train, aucpr_train, cm_train = evaluate_model(lr, X_train_res, y_train_res)
    f1_test, aucpr_test, cm_test = evaluate_model(lr, X_test, y_test)

    print("\n--- Logistic Regression ---")
    print(f"Train F1: {f1_train:.4f}, AUC-PR: {aucpr_train:.4f}")
    print(f"Test  F1: {f1_test:.4f}, AUC-PR: {aucpr_test:.4f}")
    print("Test Confusion Matrix:")
    print(cm_test)

    # -----------------------------
    # Ensemble Model: Random Forest
    # -----------------------------
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)

    f1_train, aucpr_train, cm_train = evaluate_model(rf, X_train_res, y_train_res)
    f1_test, aucpr_test, cm_test = evaluate_model(rf, X_test, y_test)

    print("\n--- Random Forest ---")
    print(f"Train F1: {f1_train:.4f}, AUC-PR: {aucpr_train:.4f}")
    print(f"Test  F1: {f1_test:.4f}, AUC-PR: {aucpr_test:.4f}")
    print("Test Confusion Matrix:")
    print(cm_test)

    # -----------------------------
    # Stratified K-Fold Cross-Validation
    # -----------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    aucpr_scores = []

    for train_idx, val_idx in skf.split(X_train_res, y_train_res):
        X_tr, X_val = X_train_res.iloc[train_idx], X_train_res.iloc[val_idx]
        y_tr, y_val = y_train_res.iloc[train_idx], y_train_res.iloc[val_idx]

        rf.fit(X_tr, y_tr)
        f1_cv, aucpr_cv, _ = evaluate_model(rf, X_val, y_val)
        f1_scores.append(f1_cv)
        aucpr_scores.append(aucpr_cv)

    print("\n--- Random Forest CV (5-fold) ---")
    print(f"F1 mean ± std: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"AUC-PR mean ± std: {np.mean(aucpr_scores):.4f} ± {np.std(aucpr_scores):.4f}")

    # -----------------------------
    # Save trained Random Forest model
    # -----------------------------
    os.makedirs('models', exist_ok=True)
    model_file = f"models/rf_{dataset_name}.pkl"  # e.g., rf_fraud.pkl or rf_creditcard.pkl
    joblib.dump(rf, model_file)
    print(f"Random Forest model saved to {model_file}")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    for dataset in ['fraud', 'creditcard']:
        run_modeling(dataset)
