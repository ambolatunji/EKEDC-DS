import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="modelling/daily_meter_summary.csv"):
    """
    Loads the final feature-engineered and aggregated dataset.
    Assumes this file is the end product of your preprocessing pipeline.
    """
    df = pd.read_csv(path, parse_dates=['date_'], dayfirst=True)
    return df

def prepare_data(df, target, task_type='classification', selected_features=None, split=True):
    """
    Prepares the data for training or prediction.

    Parameters:
    - df: DataFrame already loaded
    - target: str, name of the target variable
    - task_type: 'classification' or 'regression'
    - selected_features: list of columns to use as features (optional)
    - split: bool, whether to return train/test split

    Returns:
    - If split=True: (X_train, X_test, y_train, y_test, scaler)
    - If split=False: (X_scaled, y, scaler)
    """

    drop_cols = [
        'meter_id_', 'location_id_', 'feeder_id_', 'transformer_id_',
        'date_', 'anomaly_category', 'anomaly_summary_note'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    if selected_features:
        X = df[selected_features].copy()
    else:
        X = df.drop(columns=[target], errors='ignore')

    y = df[target]

    # Fill NaNs with column means
    X.fillna(X.mean(), inplace=True)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if not split:
        return X_scaled, y, scaler

    # Train/Test split
    if task_type == 'classification':
        stratify = y if y.nunique() == 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=stratify
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

    return X_train, X_test, y_train, y_test, scaler

# Optional test run (standalone)
if __name__ == "__main__":
    df = load_data()
    tasks = {
        "tamper_detected": "classification",
        "fault_detected": "classification",
        "energy_loss_kwh_sum": "regression",
        "band_compliance_max": "classification",
        "customer_risk_score_mean": "regression"
    }

    for target, task_type in tasks.items():
        print(f"\n🧪 Testing preparation for target: {target} | Type: {task_type}")
        X_train, X_test, y_train, y_test, _ = prepare_data(df, target, task_type=task_type)
        print(f"✔️ {target} → Train: {X_train.shape}, Test: {X_test.shape}")
