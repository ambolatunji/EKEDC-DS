from train_model_updated import train_pipeline

tasks = {
    "tamper_detected": "classification",
    "fault_detected": "classification",
    "energy_loss_kwh_sum": "regression",
    "band_compliance_max": "classification",
    "customer_risk_score_mean": "regression"
}

ml_algorithms = ["logistic", "randomforest", "xgboost", "ensemble"]
dl_algorithms = ["mlp"]  # Can be extended with more DL architectures

for target, task_type in tasks.items():
    for algo in ml_algorithms:
        train_pipeline(target=target, task_type=task_type, model_category="ml", algorithm=algo, tune=True)

    for algo in dl_algorithms:
        train_pipeline(target=target, task_type=task_type, model_category="dl", algorithm=algo, tune=False)
