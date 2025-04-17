import pandas as pd
import numpy as np
import random
import os
import sys
from datetime import datetime, timedelta

# Config
NUM_METERS = 800000
NUM_DAYS = 30
READINGS_PER_DAY = 24

bands = {
    "A": 209,
    "B": 61,
    "C": 48,
    "D": 34,
    "E": 28,
}

phase_types = ["single", "three"]
locations = ["Lekki", "Ajah", "Island", "Ibeju", "Festac", "Agbara", "Ajele", "Ijora", "Apapa", "Mushin", "Ojo", "Orile"]
feeders = [f"FDR_{i:03}" for i in range(1, 400)]
transformers = [f"DMG_{i:04}" for i in range(1, 7800)]


def generate_meter_row(meter_id, timestamp):
    phase_type = random.choice(phase_types)
    band = random.choice(list(bands.keys()))
    tariff = bands[band]
    feeder_id = random.choice(feeders)
    transformer_id = random.choice(transformers)
    location_id = random.choice(locations)

    if phase_type == "three":
        v1, v2, v3 = np.random.normal(230, 10, 3)
        i1, i2, i3 = np.random.normal(15, 5, 3)
        v_agg = (v1 + v2 + v3) / 3
        i_agg = (i1 + i2 + i3) / 3
    else:
        v1 = v2 = v3 = np.nan
        i1 = i2 = i3 = np.nan
        v_agg = np.random.normal(230, 10)
        i_agg = np.random.normal(10, 3)

    signal_strength_dbm = np.random.uniform(-100, -30)
    power_factor = np.clip(np.random.normal(0.9, 0.05), 0.5, 1.0)
    total_vended_amount = round(np.random.choice([500, 1000, 2000, 5000]), 2)
    expected_energy_from_vend = round(total_vended_amount / tariff, 2)
    energy_consumed_kwh = expected_energy_from_vend * np.random.uniform(0.8, 1.1)
    expected_energy_kwh = expected_energy_from_vend
    reading_gap_minutes = random.choice([15, 30, 60])
    data_transmitted = np.random.choice([1, 0], p=[0.98, 0.02])
    tamper_flag = np.random.choice([1, 0], p=[0.03, 0.97])
    fault_flag = int(v_agg < 200 or signal_strength_dbm < -90)

    if fault_flag:
        meter_status = "faulty"
    elif tamper_flag:
        meter_status = "tampered"
    elif data_transmitted == 0:
        meter_status = "inactive"
    else:
        meter_status = "healthy"

    return {
        "meter_id": meter_id,
        "phase_type": phase_type,
        "timestamp": timestamp,
        "v1": v1, "v2": v2, "v3": v3,
        "i1": i1, "i2": i2, "i3": i3,
        "v_agg": v_agg, "i_agg": i_agg,
        "power_factor": power_factor,
        "energy_consumed_kwh": round(energy_consumed_kwh, 2),
        "expected_energy_kwh": expected_energy_kwh,
        "signal_strength_dbm": signal_strength_dbm,
        "data_transmitted": data_transmitted,
        "tamper_flag": tamper_flag,
        "fault_flag": fault_flag,
        "meter_status": meter_status,
        "reading_gap_minutes": reading_gap_minutes,
        "location_id": location_id,
        "feeder_id": feeder_id,
        "transformer_id": transformer_id,
        "band": band,
        "tariff_per_kwh": tariff,
        "total_vended_amount": total_vended_amount,
        "expected_energy_from_vend": expected_energy_from_vend
    }

def generate_dataset():
    all_data = []
    start_time = datetime.now() - timedelta(days=NUM_DAYS)

    for meter_num in range(1, NUM_METERS + 1):
        meter_id = f"EKEDC_{meter_num:06}"
        for day in range(NUM_DAYS):
            for interval in range(READINGS_PER_DAY):
                timestamp = start_time + timedelta(days=day, minutes=interval * 60)
                row = generate_meter_row(meter_id, timestamp)
                all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv("meter_data.csv", index=False)
    print("✅ Synthetic meter data generated and saved to CSV.")


if __name__ == "__main__":
    generate_dataset()
