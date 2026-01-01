import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------
# Load Test Data and Model
# -----------------------------
def load_data_model(dataset_name):
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

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found: {model_file}")

    model = joblib.load(model_file)
    return X_test, y_test, model

# -----------------------------
# Feature Importance
# -----------------------------
def plot_feature_importance(model, X_test, dataset_name, top_n=10):
    importance = model.feature_importances_
    features = X_test.columns
    feat_df = pd.DataFrame({'feature': features, 'importance': importance})
    feat_df = feat_df.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10,6))
    sns.barplot(x='importance', y='feature', data=feat_df, palette='viridis')
    plt.title(f'Top {top_n} Feature Importances ({dataset_name})')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/feature_importance_{dataset_name}.png')
    plt.close()
    print(f"Feature importance plot saved: plots/feature_importance_{dataset_name}.png")

# -----------------------------
# SHAP Analysis
# -----------------------------
def shap_analysis(model, X_test, y_test, dataset_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global summary plot
    os.makedirs('plots', exist_ok=True)
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.savefig(f'plots/shap_summary_{dataset_name}.png')
    plt.close()
    print(f"SHAP summary plot saved: plots/shap_summary_{dataset_name}.png")

    # Identify TP, FP, FN
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tp_idx = X_test[(y_test==1) & (y_pred==1)].index
    fp_idx = X_test[(y_test==0) & (y_pred==1)].index
    fn_idx = X_test[(y_test==1) & (y_pred==0)].index

    selected_indices = []
    if len(tp_idx) > 0: selected_indices.append(tp_idx[0])
    if len(fp_idx) > 0: selected_indices.append(fp_idx[0])
    if len(fn_idx) > 0: selected_indices.append(fn_idx[0])

    # Force plots for selected predictions
    for idx in selected_indices:
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][idx,:],
            X_test.iloc[idx,:],
            matplotlib=True,
            show=False
        )
        plt.savefig(f'plots/shap_force_{dataset_name}_{idx}.png')
        plt.close()
        print(f"SHAP force plot saved: plots/shap_force_{dataset_name}_{idx}.png")

# -----------------------------
# Run Explainability
# -----------------------------
def run_explainability(dataset_name):
    print(f"\n=== Explainability for {dataset_name} dataset ===")
    X_test, y_test, model = load_data_model(dataset_name)

    # Feature importance
    plot_feature_importance(model, X_test, dataset_name)

    # SHAP analysis
    shap_analysis(model, X_test, y_test, dataset_name)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    for dataset in ['fraud', 'creditcard']:
        run_explainability(dataset)
