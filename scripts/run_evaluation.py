import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def load_test_data(dataset_name):
    processed_dir = os.path.join('data', 'processed')
    
    if dataset_name == 'fraud':
        test_file = os.path.join(processed_dir, 'fraud_test.csv')
        target = 'class'
        model_file = os.path.join('models', 'rf_fraud.pkl')
    elif dataset_name == 'creditcard':
        test_file = os.path.join(processed_dir, 'creditcard_test.csv')
        target = 'Class'
        model_file = os.path.join('models', 'rf_creditcard.pkl')
    else:
        raise ValueError("dataset_name must be 'fraud' or 'creditcard'")
    
    test_df = pd.read_csv(test_file)
    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]
    
    return X_test, y_test, model_file

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    auc_pr = auc(recall, precision)
    
    cm = confusion_matrix(y_test, y_pred)
    
    return f1, auc_pr, cm, precision, recall

def plot_precision_recall(precision, recall, dataset_name):
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({dataset_name})')
    plt.legend()
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/pr_curve_{dataset_name}.png')
    plt.close()
    print(f"Precision-Recall curve saved to plots/pr_curve_{dataset_name}.png")

def plot_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(5,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({dataset_name})')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/cm_{dataset_name}.png')
    plt.close()
    print(f"Confusion matrix saved to plots/cm_{dataset_name}.png")

def run_evaluation(dataset_name):
    print(f"\n=== Evaluating {dataset_name} dataset ===")
    X_test, y_test, model_file = load_test_data(dataset_name)
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return
    
    model = joblib.load(model_file)
    f1, auc_pr, cm, precision, recall = evaluate_model(model, X_test, y_test)

    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-PR  : {auc_pr:.4f}")
    print("Confusion Matrix:")
    print(cm)

    plot_precision_recall(precision, recall, dataset_name)
    plot_confusion_matrix(cm, dataset_name)

# -----------------------------
# Run evaluation for both datasets
# -----------------------------
if __name__ == "__main__":
    for dataset in ['fraud', 'creditcard']:
        run_evaluation(dataset)
