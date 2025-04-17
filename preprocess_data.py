import pandas as pd
import numpy as np

def preprocess_data(input_path='meter_data.csv', output_path='processed_meter_data.csv'):
    # Load the dataset
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # Fill missing voltage/current values with zeros for single-phase meters
    for col in ['v1', 'v2', 'v3', 'i1', 'i2', 'i3']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Derive time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Energy loss detection (early stage feature)
    df['energy_loss_flag'] = (df['energy_consumed_kwh'] < df['expected_energy_kwh']).astype(int)

    # Output to new CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")

if __name__ == '__main__':
    preprocess_data()
