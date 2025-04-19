import pandas as pd

def aggregate_24hr(input_path='engineered_meter_data.csv', output_path='daily_meter_summary.csv'):
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    # Extract just the date for grouping
    df['date'] = df['timestamp'].dt.date

    # Define aggregation logic
    aggregation = {
        'energy_consumed_kwh': 'sum',
        'expected_energy_kwh': 'sum',
        'expected_energy_pf_calc': 'sum',
        'total_vended_amount': 'sum',
        'revenue_loss': 'sum',
        'energy_loss_kwh': 'sum',
        'energy_loss_flag': 'sum',
        'fault_flag': 'sum',
        'tamper_flag': 'sum',
        'uptime_flag': 'sum',
        'supply_voltage_flag': 'sum',
        'pf_delta': 'mean',
        'v_agg': ['mean', 'min', 'max'],
        'i_agg': ['mean', 'min', 'max'],
        'power_factor': ['mean', 'min'],
        'band_compliance': 'max',
        'is_industrial': 'max',
        'high_vend_zone': 'max',
        'tariff_per_kwh': 'mean'
    }

    # Group and aggregate per meter_id and date
    daily_df = df.groupby(['meter_id', 'date']).agg(aggregation)

    # Flatten multi-index columns
    daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns.values]
    daily_df.reset_index(inplace=True)

    # Add new flags/derived KPIs
    daily_df['fault_detected'] = (daily_df['fault_flag_sum'] > 0).astype(int)
    daily_df['tamper_detected'] = (daily_df['tamper_flag_sum'] > 0).astype(int)
    daily_df['energy_loss_ratio'] = daily_df['energy_loss_kwh_sum'] / daily_df['expected_energy_kwh_sum']
    daily_df['supply_hours'] = daily_df['supply_voltage_flag_sum']
    daily_df['uptime_hours'] = daily_df['uptime_flag_sum']
    daily_df['energy_efficiency_ratio'] = daily_df['energy_consumed_kwh_sum'] / daily_df['expected_energy_kwh_sum']

    # Export
    daily_df.to_csv(output_path, index=False)
    print(f"✅ Daily aggregation saved to {output_path}")

if __name__ == '__main__':
    aggregate_24hr()
