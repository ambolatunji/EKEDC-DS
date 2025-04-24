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

def load_model(model_path, is_dl=False):
    if is_dl:
        return load_dl_model(model_path)
    else:
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def evaluate_classification(model, X_test, y_test, target_name, is_dl=False):
    if is_dl:
        y_prob = model.predict(X_test).flatten()
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
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "task": "classification"
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {target_name}")
    plt.savefig(f"{PLOTS_DIR}/{target_name}_conf_matrix_eval.png")
    plt.close()

    # ROC Curve
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

def evaluate_regression(model, X_test, y_test, target_name, is_dl=False):
    y_pred = model.predict(X_test).flatten() if is_dl else model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics_log.append({
        "target": target_name,
        "model_type": "DL" if is_dl else "ML",
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

def evaluate_pipeline(model_path, target, task_type='classification', is_dl=False):
    print(f"🔍 Evaluating: {model_path}")
    model = load_model(model_path, is_dl)
    df = load_data()
    X_train, X_test, y_train, y_test, _ = prepare_data(df, target, task_type)

    if task_type == 'classification':
        evaluate_classification(model, X_test, y_test, target, is_dl)
    else:
        evaluate_regression(model, X_test, y_test, target, is_dl)

def auto_discover_models():
    models = glob(os.path.join(MODEL_DIR, "*"))
    for model_path in models:
        fname = os.path.basename(model_path)
        is_dl = fname.endswith(".h5")
        parts = fname.split("_")
        target = "_".join(parts[:-2]) if "dl" in fname or "ml" in fname else parts[0]
        task = "classification" if "tamper" in fname or "fault" in fname or "band" in fname else "regression"
        evaluate_pipeline(model_path, target, task_type=task, is_dl=is_dl)

    pd.DataFrame(metrics_log).to_csv(METRICS_FILE, index=False)
    print(f"📦 All evaluations saved to {METRICS_FILE}")

if __name__ == "__main__":
    auto_discover_models()
