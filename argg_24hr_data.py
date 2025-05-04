import pandas as pd
from tqdm import tqdm

def aggregate_24hr(input_path='engineered_meter_data.csv', output_path='daily_meter_summary.csv', chunk_size=100000):
    print("🔄 Reading and aggregating in chunks...")

    chunk_iter = pd.read_csv(input_path, chunksize=chunk_size, parse_dates=['timestamp'])
    all_chunks = []

    for chunk in tqdm(chunk_iter, desc="⏳ Processing chunks"):
        chunk['date'] = chunk['timestamp'].dt.date

        aggregation = {
            'energy_consumed_kwh': 'sum',
            'expected_energy_kwh': 'sum',
            'expected_energy_pf_calc': 'sum',
            'total_vended_amount': 'sum',
            'revenue_loss': 'sum',
            'tamper_cost_estimate': 'sum',
            'energy_loss_kwh': 'sum',
            'energy_loss_flag': 'sum',
            'fault_flag': 'sum',
            'tamper_flag': 'sum',
            'uptime_flag': 'sum',
            'supply_voltage_flag': 'sum',
            'pf_delta': 'mean',
            'load_generation_gap': 'mean',
            'load_gap_flag': 'sum',
            'service_availability_ratio': 'mean',
            'unresolved_fault_days': 'mean',
            'customer_risk_score': 'mean',
            'power_factor': ['mean', 'min'],
            'v_agg': ['mean', 'min', 'max'],
            'i_agg': ['mean', 'min', 'max'],
            'tariff_per_kwh': 'mean',
            'band_compliance': 'max'
        }

        group_keys = ['meter_id', 'location_id', 'feeder_id', 'transformer_id', 'band', 'fault_type', 'is_industrial', 'date']
        grouped = chunk.groupby(group_keys).agg(aggregation).reset_index()
        all_chunks.append(grouped)

    print("🔁 Concatenating all results...")
    daily_df = pd.concat(all_chunks, ignore_index=True)

    # Flatten multi-index columns
    daily_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in daily_df.columns]

    # Derived KPIs
    daily_df['fault_detected'] = (daily_df['fault_flag_sum'] > 0).astype(int)
    daily_df['tamper_detected'] = (daily_df['tamper_flag_sum'] > 0).astype(int)
    daily_df['energy_loss_ratio'] = daily_df['energy_loss_kwh_sum'] / daily_df['expected_energy_kwh_sum']
    daily_df['supply_hours'] = daily_df['supply_voltage_flag_sum']
    daily_df['uptime_hours'] = daily_df['uptime_flag_sum']
    daily_df['energy_efficiency_ratio'] = daily_df['energy_consumed_kwh_sum'] / daily_df['expected_energy_kwh_sum']

    # Anomaly tagging
    def tag_anomaly(row):
        tags = []
        if row['fault_detected']: tags.append("fault")
        if row['tamper_detected']: tags.append("tamper")
        if row['energy_loss_flag_sum'] > 0: tags.append("loss")
        if row['load_gap_flag_sum'] > 0: tags.append("mismatch")
        return ",".join(tags) if tags else "normal"

    def severity_score(row):
        return (
            row['fault_detected'] * 3 +
            row['tamper_detected'] * 4 +
            row['energy_loss_flag_sum'] * 2 +
            row['load_gap_flag_sum'] * 2 +
            (1 if row['service_availability_ratio_mean'] < 0.5 else 0)
        )

    def anomaly_note(row):
        if row['anomaly_category'] == 'normal':
            return "No issue detected"
        else:
            return f"Detected: {row['anomaly_category']} | Risk={row['anomaly_severity_score']}"

    daily_df['anomaly_category'] = daily_df.apply(tag_anomaly, axis=1)
    daily_df['anomaly_severity_score'] = daily_df.apply(severity_score, axis=1)
    daily_df['anomaly_summary_note'] = daily_df.apply(anomaly_note, axis=1)

    daily_df.to_csv(output_path, index=False)
    print(f"✅ Daily summary with anomalies saved to {output_path}")

if __name__ == '__main__':
    aggregate_24hr()
