# scripts/run_creditcard_preprocessing.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_creditcard_data():
    # -----------------------------
    # Paths
    # -----------------------------
    raw_path = os.path.join('data', 'raw', 'creditcard.csv')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    credit_data = pd.read_csv(raw_path)
    print(f"Loaded creditcard.csv with shape: {credit_data.shape}")

    # -----------------------------
    # Check missing values
    # -----------------------------
    missing = credit_data.isnull().sum()
    print("Missing values per column:")
    print(missing)

    # -----------------------------
    # Feature scaling
    # -----------------------------
    scaler = StandardScaler()
    credit_data[['Amount', 'Time']] = scaler.fit_transform(credit_data[['Amount', 'Time']])
    print("Scaled 'Amount' and 'Time' features.")

    # -----------------------------
    # Train/test split
    # -----------------------------
    X = credit_data.drop('Class', axis=1)
    y = credit_data['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_df = X_train.copy()
    train_df['Class'] = y_train

    test_df = X_test.copy()
    test_df['Class'] = y_test

    # -----------------------------
    # Save processed files
    # -----------------------------
    train_file = os.path.join(processed_dir, 'creditcard_train.csv')
    test_file = os.path.join(processed_dir, 'creditcard_test.csv')

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"✅ Credit card preprocessing complete. Train/test files saved in {processed_dir}")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    preprocess_creditcard_data()
