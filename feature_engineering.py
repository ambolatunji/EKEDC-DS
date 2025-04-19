import pandas as pd
import numpy as np
import os
import sys

def feature_engineering(input_path='meter_data.csv', output_path='engineered_meter_data.csv'):
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    # Sort to ensure correct groupby operations
    df.sort_values(by=['meter_id', 'timestamp'], inplace=True)

    # =========================== TIME-BASED FEATURES ===========================
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['date'] = df['timestamp'].dt.date

    # =========================== SUPPLY FLAGS ===========================
    df['supply_voltage_flag'] = (df['v_agg'] > 200).astype(int)
    df['uptime_flag'] = ((df['supply_voltage_flag'] == 1) & (df['data_transmitted'] == 1)).astype(int)

    # =========================== POWER-BASED EXPECTED ENERGY ===========================
    df['expected_energy_pf_calc'] = (df['v_agg'] * df['i_agg'] * df['power_factor']) / 1000

    # =========================== POWER FACTOR DEVIATION ===========================
    df['pf_delta'] = df.groupby('meter_id')['power_factor'].diff().fillna(0)

    # =========================== FAULT CATEGORIZATION ===========================
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

    # =========================== STATUS TRANSITIONS ===========================
    df['prev_status'] = df.groupby('meter_id')['meter_status'].shift()
    df['status_change'] = df['meter_status'] != df['prev_status']

    # =========================== FAULT/TAMPER TIMESTAMPS ===========================
    df['fault_start'] = ((df['fault_flag'] == 1) & (df.groupby('meter_id')['fault_flag'].shift().fillna(0) == 0)).astype(int)
    df['tamper_start'] = ((df['tamper_flag'] == 1) & (df.groupby('meter_id')['tamper_flag'].shift().fillna(0) == 0)).astype(int)
    
    df['fault_timestamp'] = df['timestamp'].where(df['fault_start'] == 1)
    df['tamper_timestamp'] = df['timestamp'].where(df['tamper_start'] == 1)

    # =========================== REVENUE LOSS CALCULATIONS ===========================
    df['energy_loss_kwh'] = df['expected_energy_kwh'] - df['energy_consumed_kwh']
    df['revenue_loss'] = df['energy_loss_kwh'] * df['tariff_per_kwh']
    df['energy_loss_flag'] = (df['energy_loss_kwh'] > 0.5).astype(int)

    # =========================== BAND COMPLIANCE / UPTIME CHECKS ===========================
    band_hours = {'A': 20, 'B': 16, 'C': 12, 'D': 8, 'E': 4}
    df['supply_hour_counter'] = df.groupby(['meter_id', 'date'])['supply_voltage_flag'].transform('sum')

    def check_band_compliance(row):
        min_hours = band_hours.get(row['band'], 0)
        return int(row['supply_hour_counter'] >= min_hours)

    df['band_compliance'] = df.apply(check_band_compliance, axis=1)

    # =========================== INDUSTRIAL ZONE HIGH VALUE FLAGS ===========================
    df['high_vend_zone'] = ((df['location_id'].isin(['Lekki', 'Island', 'Ajah', 'Apapa', 'Ajele'])) & (df['band'] == 'A')).astype(int)
    df['is_industrial'] = df['high_vend_zone']

    # =========================== SAVE ===========================
    df.to_csv(output_path, index=False)
    print(f"✅ Feature engineered data saved to {output_path}")

if __name__ == '__main__':
    feature_engineering()
