import pandas as pd
import numpy as np
import os
import sys
import gc
from tqdm import tqdm

def feature_engineering(input_path='meter_data.csv', output_path='engineered_meter_data.csv', chunk_size=500):
    print(f"\nProcessing {input_path} in chunks...")
    
    # Get total rows for progress bar
    total_rows = sum(1 for _ in open(input_path)) - 1
    chunks = pd.read_csv(input_path, chunksize=chunk_size, parse_dates=['timestamp'])
    
    # Write headers first
    first_chunk = next(chunks)
    processed_chunk = process_chunk(first_chunk)
    processed_chunk.to_csv(output_path, index=False)
    del processed_chunk
    gc.collect()
    
    # Process remaining chunks
    with tqdm(total=total_rows, desc="Processing") as pbar:
        for chunk in chunks:
            processed_chunk = process_chunk(chunk)
            processed_chunk.to_csv(output_path, mode='a', header=False, index=False)
            pbar.update(len(chunk))
            del processed_chunk
            gc.collect()
    
    print(f"✅ Feature engineered data saved to {output_path}")

def process_chunk(df):
    # TIME-BASED FEATURES
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['date'] = df['timestamp'].dt.date

    # SUPPLY FLAGS
    df['supply_voltage_flag'] = (df['v_agg'] > 200).astype(int)
    df['uptime_flag'] = ((df['supply_voltage_flag'] == 1) & (df['data_transmitted'] == 1)).astype(int)

    # POWER CALCULATION
    df['expected_energy_pf_calc'] = (df['v_agg'] * df['i_agg'] * df['power_factor']) / 1000

    # POWER FACTOR DEVIATION
    df['pf_delta'] = df.groupby('meter_id')['power_factor'].diff().fillna(0)

    # FAULT CATEGORIZATION
    def categorize_fault(row):
        if row['signal_strength_dbm'] < -90:
            return 'modem_fault'
        elif row['v_agg'] < 200:
            return 'voltage_drop'
        elif row['power_factor'] < 0.3:
            return 'burnt_meter'
        elif row['supply_voltage_flag'] == 1 and row['energy_consumed_kwh'] < 0.1:
            return 'relay_fault'
        return 'no_fault'
    df['fault_type'] = df.apply(lambda row: categorize_fault(row) if row['fault_flag'] else 'no_fault', axis=1)

    # FAULT/TAMPER TIMESTAMP AND TRANSITION DETECTION
    df['prev_status'] = df.groupby('meter_id')['meter_status'].shift()
    df['status_change'] = df['meter_status'] != df['prev_status']
    df['fault_start'] = ((df['fault_flag'] == 1) & (df.groupby('meter_id')['fault_flag'].shift().fillna(0) == 0)).astype(int)
    df['tamper_start'] = ((df['tamper_flag'] == 1) & (df.groupby('meter_id')['tamper_flag'].shift().fillna(0) == 0)).astype(int)
    df['fault_timestamp'] = df['timestamp'].where(df['fault_start'] == 1)
    df['tamper_timestamp'] = df['timestamp'].where(df['tamper_start'] == 1)

    # REVENUE AND ENERGY LOSS
    df['energy_loss_kwh'] = df['expected_energy_kwh'] - df['energy_consumed_kwh']
    df['revenue_loss'] = df['energy_loss_kwh'] * df['tariff_per_kwh']
    df['energy_loss_flag'] = (df['energy_loss_kwh'] > 0.5).astype(int)
    df['tamper_cost_estimate'] = df['revenue_loss'] * df['tamper_flag']

    # BAND COMPLIANCE
    band_hours = {'A': 20, 'B': 16, 'C': 12, 'D': 8, 'E': 4}
    df['supply_hour_counter'] = df.groupby(['meter_id', 'date'])['supply_voltage_flag'].transform('sum')
    df['band_compliance'] = df.apply(lambda row: int(row['supply_hour_counter'] >= band_hours.get(row['band'], 0)), axis=1)
    df['band_compliance_max'] = df.groupby(['meter_id', 'date'])['band_compliance'].transform('max')


    # INDUSTRIAL HIGH VEND FLAG
    df['high_vend_zone'] = ((df['location_id'].isin(['Lekki', 'Island', 'Ajah', 'Apapa', 'Ajele'])) & (df['band'] == 'A')).astype(int)
    df['is_industrial'] = df['high_vend_zone']

    # SUPPLY OUTAGE TAGGING
    df['supply_outage_flag'] = ((df['supply_voltage_flag'] == 0) & (df['data_transmitted'] == 0)).astype(int)

    # LOAD VS GENERATION MISMATCH
    df['load_generation_gap'] = df['expected_energy_pf_calc'] - df['energy_consumed_kwh']
    df['load_gap_flag'] = (df['load_generation_gap'].abs() > 0.3 * df['expected_energy_pf_calc']).astype(int)

    # SERVICE AVAILABILITY
    df['service_availability_ratio'] = df.groupby(['meter_id', 'date'])['uptime_flag'].transform('sum') / 24

    # UNRESOLVED FAULT DAYS
    df['unresolved_fault_days'] = df.groupby('meter_id')['fault_flag'].apply(lambda x: x.rolling(window=24, min_periods=1).sum()).reset_index(level=[0], drop=True)

    # LIVE VISUAL STREAM ANOMALY TAG
    def assign_anomaly_tag(row):
        tags = []
        if row['fault_flag']:
            tags.append('fault')
        if row['tamper_flag']:
            tags.append('tamper')
        if row['supply_outage_flag']:
            tags.append('outage')
        if row['load_gap_flag']:
            tags.append('mismatch')
        return ','.join(tags) if tags else 'normal'
    df['meter_anomaly_tag'] = df.apply(assign_anomaly_tag, axis=1)

    # CUSTOMER RISK SCORE
    df['customer_risk_score'] = (
        df['fault_flag'] * 1.5 +
        df['tamper_flag'] * 2 +
        df['supply_outage_flag'] +
        df['load_gap_flag']
    )
    
    return df

if __name__ == '__main__':
    feature_engineering()
    print("✅ Feature engineering completed.")