import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path='../daily_meter_summary.csv'):
    df = pd.read_csv(path, parse_dates=['date_'])
    return df

def prepare_data(df, target, task_type='classification'):
    features = df.copy()

    # Drop identifiers and dates
    features = features.drop(columns=[
        'meter_id_', 'location_id_', 'feeder_id_', 'transformer_id_', 'date_',
        'anomaly_category', 'anomaly_summary_note'
    ], errors='ignore')

    # Remove the target from X
    X = features.drop(columns=[target], errors='ignore')
    y = features[target]

    # Handle NaNs
    X.fillna(X.mean(), inplace=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    if task_type == 'classification':
        stratify = y if y.nunique() == 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=stratify)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler

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
        print(f"Preparing data for: {target}")
        X_train, X_test, y_train, y_test, _ = prepare_data(df, target, task_type)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
