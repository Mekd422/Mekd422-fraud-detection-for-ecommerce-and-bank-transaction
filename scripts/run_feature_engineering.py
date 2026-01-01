import pandas as pd
from src.feature_engineering import preprocess_fraud_data
from sklearn.model_selection import train_test_split

def main():
    # Load raw dataset
    fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')

    # Process features
    processed_data = preprocess_fraud_data(fraud_data)

    # Split train/test
    X = processed_data.drop(
        ['class', 'user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address'],
        axis=1
    )
    y = processed_data['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Save processed data
    train_df = X_train.copy()
    train_df['class'] = y_train
    test_df = X_test.copy()
    test_df['class'] = y_test

    train_df.to_csv('data/processed/fraud_train.csv', index=False)
    test_df.to_csv('data/processed/fraud_test.csv', index=False)

    print("✅ Feature engineering complete. Processed files saved in data/processed/")

if __name__ == "__main__":
    main()
