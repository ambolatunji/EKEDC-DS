
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from prepare_feature import load_data, prepare_data
from tensorflow.keras.models import load_model as load_dl_model

MODEL_DIR = "models"
PLOTS_DIR = "plots"
METRICS_FILE = "evaluation_metrics.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)

metrics_log = []

def parse_model_filename(filename):
    filename = os.path.basename(filename).replace(".pkl", "").replace(".h5", "")
    parts = filename.split("_")
    known_algos = ["logistic", "randomforest", "xgboost", "ensemble", "mlp", "linear"]
    known_types = ["ml", "dl"]
    model_type = parts[-1] if parts[-1] in known_types else "ml"
    algorithm = parts[-2] if parts[-2] in known_algos else "unknown"
    target = "_".join(parts[:-3])
    return target, algorithm, model_type

def load_model(model_path, is_dl=False):
    return load_dl_model(model_path) if is_dl else pickle.load(open(model_path, 'rb'))

def evaluate_classification(model, X_test, y_test, target_name, model_path, is_dl=False):
    if is_dl:
        y_prob = model.predict(X_test, batch_size=1024).flatten()
        y_pred = (y_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    report = classification_report(y_test, y_pred, output_dict=True)
    acc = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    metrics_log.append({
        "target": target_name,
        "model_type": "DL" if is_dl else "ML",
        "model_file": os.path.basename(model_path),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "task": "classification"
    })

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {target_name}")
    plt.savefig(f"{PLOTS_DIR}/{target_name}_conf_matrix_eval.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve: {target_name}")
    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/{target_name}_roc_eval.png")
    plt.close()

    # Risk Distribution
    plt.hist(y_prob, bins=30, color="crimson", alpha=0.7)
    plt.axvline(0.8, color="orange", linestyle="--", label="High Risk >80%")
    plt.xlabel("Predicted Risk Score")
    plt.ylabel("Meter Count")
    plt.title(f"Risk Score Distribution: {target_name}")
    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/{target_name}_risk_distribution.png")
    plt.close()

def evaluate_regression(model, X_test, y_test, target_name, model_path, is_dl=False):
    y_pred = model.predict(X_test, batch_size=1024).flatten() if is_dl else model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics_log.append({
        "target": target_name,
        "model_type": "DL" if is_dl else "ML",
        "model_file": os.path.basename(model_path),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2,
        "task": "regression"
    })

    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted: {target_name}")
    plt.savefig(f"{PLOTS_DIR}/{target_name}_actual_vs_pred.png")
    plt.close()

def evaluate_pipeline(model_path, target, task_type='classification', is_dl=False, sample_size=None):
    target_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"🔍 Evaluating: {model_path}")
    model = load_model(model_path, is_dl)

    df = load_data()

    # 🔐 Load features used during training
    #filename = os.path.basename(model_path).replace(".pkl", "").replace(".h5", "")
    feature_file = os.path.join(MODEL_DIR, f"{target_name}_features.txt")

    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"⚠️ Feature file not found for {filename}")

    with open(feature_file, "r") as f:
        selected_features = [line.strip() for line in f.readlines()]

    # Restrict to selected features
    df = df[selected_features + [target]].dropna()

    # Optional sampling
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)

    X_train, X_test, y_train, y_test, _ = prepare_data(df, target, task_type, selected_features=selected_features)

    if task_type == 'classification':
        evaluate_classification(model, X_test, y_test, target, model_path, is_dl)
    else:
        evaluate_regression(model, X_test, y_test, target, model_path, is_dl)

def auto_discover_models():
    models = glob(os.path.join(MODEL_DIR, "*"))
    for model_path in models:
        is_dl = model_path.endswith(".h5")
        target, _, _ = parse_model_filename(model_path)
        task = "classification" if any(x in target for x in ["tamper", "fault", "band"]) else "regression"
        evaluate_pipeline(model_path, target, task_type=task, is_dl=is_dl)

    pd.DataFrame(metrics_log).to_csv(METRICS_FILE, index=False)
    print(f"📦 All evaluations saved to {METRICS_FILE}")

if __name__ == "__main__":
    auto_discover_models()
