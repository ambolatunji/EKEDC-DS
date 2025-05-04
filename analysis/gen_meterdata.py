import pandas as pd
import numpy as np
import random
import os
import sys
import gc
from datetime import datetime, timedelta

# Config
NUM_METERS = 2000
NUM_DAYS = 30
READINGS_PER_DAY = 24
CHUNK_SIZE = 500

bands = {
    "A": 209,
    "B": 61,
    "C": 48,
    "D": 34,
    "E": 28,
}

phase_types = ["single", "three"]
locations = ["Lekki", "Ajah", "Island", "Ibeju", "Festac", "Agbara", "Ajele", "Ijora", "Apapa", "Mushin", "Ojo", "Orile"]
industrial_locations = ["Lekki", "Island", "Apapa", "Ajele", "Ajah"]
feeders = [f"FDR_{i:03}" for i in range(1, 400)]
transformers = [f"DMG_{i:04}" for i in range(1, 7800)]

# Global mapping caches
meter_phase_types = {}
meter_static_info = {}
feeder_band_map = {}
meter_fault_history = {}
meter_tamper_history = {}

def get_meter_phase_type(meter_id):
    if meter_id not in meter_phase_types:
        meter_phase_types[meter_id] = random.choice(phase_types)
    return meter_phase_types[meter_id]

def generate_meter_row(meter_id, timestamp):
    # Consistent meter topology
    if meter_id not in meter_static_info:
        location_id = random.choice(locations)
        feeder_id = random.choice(feeders)
        transformer_id = random.choice(transformers)
        meter_static_info[meter_id] = (location_id, feeder_id, transformer_id)
    else:
        location_id, feeder_id, transformer_id = meter_static_info[meter_id]

    # Assign one band per feeder
    if feeder_id not in feeder_band_map:
        if location_id in industrial_locations:
            feeder_band_map[feeder_id] = "A"
        else:
            feeder_band_map[feeder_id] = random.choice(list(bands.keys()))
    band = feeder_band_map[feeder_id]
    tariff = bands[band]

    phase_type = get_meter_phase_type(meter_id)

    if phase_type == "three":
        v1, v2, v3 = np.random.normal(230, 10, 3).astype(np.float32)
        i1, i2, i3 = np.random.normal(15, 5, 3).astype(np.float32)
        v_agg = ((v1 + v2 + v3) / 3).astype(np.float32)
        i_agg = ((i1 + i2 + i3) / 3).astype(np.float32)
    else:
        v1 = v2 = v3 = np.nan
        i1 = i2 = i3 = np.nan
        v_agg = np.array(np.random.normal(230, 10), dtype=np.float32)
        i_agg = np.array(np.random.normal(10, 3), dtype=np.float32)

    signal_strength_dbm = np.array(np.random.uniform(-100, -30), dtype=np.float32)
    power_factor = np.array(np.clip(np.random.normal(0.9, 0.05), 0.5, 1.0), dtype=np.float32)

    if location_id in industrial_locations:
        total_vended_amount = round(np.random.uniform(100000, 5000000), 2)
    else:
        total_vended_amount = round(np.random.choice([500, 1000, 2000, 5000, 10000, 20000]), 2)

    expected_energy_from_vend = round(total_vended_amount / tariff, 2)
    energy_consumed_kwh = expected_energy_from_vend * np.random.uniform(0.8, 1.1)
    expected_energy_kwh = expected_energy_from_vend
    reading_gap_minutes = random.choice([15, 30, 60])
    data_transmitted = np.random.choice([1, 0], p=[0.98, 0.02])
    tamper_flag = np.random.choice([1, 0], p=[0.03, 0.97])

    # Fault detection logic
    fault_flag = int(v_agg < 200 or signal_strength_dbm < -90)
    fault_type = None
    fault_duration_hours = 0

    if fault_flag:
        if v_agg < 200 and signal_strength_dbm < -90:
            fault_type = "voltage + modem"
        elif v_agg < 200:
            fault_type = "low_voltage"
        elif signal_strength_dbm < -90:
            fault_type = "modem_disconnect"
        else:
            fault_type = "unknown"

        fault_duration_hours = np.random.randint(6, 24) if location_id in industrial_locations else np.random.randint(24, 72)

        meter_fault_history[meter_id] = {
            "fault_type": fault_type,
            "duration": fault_duration_hours,
            "start_time": timestamp
        }
    else:
        if meter_id in meter_fault_history:
            record = meter_fault_history[meter_id]
            elapsed = (timestamp - record["start_time"]).total_seconds() / 3600
            if elapsed > record["duration"]:
                del meter_fault_history[meter_id]
            else:
                fault_flag = 1
                fault_type = record["fault_type"]
                fault_duration_hours = record["duration"]

    # Tamper duration tracking
    tamper_duration_hours = 0
    if tamper_flag:
        tamper_duration_hours = np.random.randint(12, 48)
        meter_tamper_history[meter_id] = {
            "start_time": timestamp,
            "duration": tamper_duration_hours
        }
    else:
        if meter_id in meter_tamper_history:
            record = meter_tamper_history[meter_id]
            elapsed = (timestamp - record["start_time"]).total_seconds() / 3600
            if elapsed > record["duration"]:
                del meter_tamper_history[meter_id]
            else:
                tamper_flag = 1
                tamper_duration_hours = record["duration"]

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
        "tamper_duration_hours": tamper_duration_hours,
        "fault_flag": fault_flag,
        "fault_type": fault_type,
        "fault_duration_hours": fault_duration_hours,
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

def process_chunk(start_meter, end_meter, start_time):
    chunk_data = []
    for meter_num in range(start_meter, end_meter + 1):
        meter_id = f"EKEDC_{meter_num:06}"
        for day in range(NUM_DAYS):
            for interval in range(READINGS_PER_DAY):
                timestamp = start_time + timedelta(days=day, minutes=interval * 60)
                chunk_data.append(generate_meter_row(meter_id, timestamp))
        gc.collect()
    return chunk_data

def generate_dataset():
    start_time = datetime.now() - timedelta(days=NUM_DAYS)
    pd.DataFrame(columns=generate_meter_row("test", start_time).keys()).to_csv("meter_data.csv", index=False)

    for chunk_start in range(1, NUM_METERS + 1, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE - 1, NUM_METERS)
        print(f"Processing meters {chunk_start} to {chunk_end}...")
        chunk_data = process_chunk(chunk_start, chunk_end, start_time)
        chunk_df = pd.DataFrame(chunk_data)
        chunk_df.to_csv("meter_data.csv", mode='a', header=False, index=False)
        del chunk_data, chunk_df
        gc.collect()
        print(f"✅ Chunk {chunk_start}-{chunk_end} completed")
    print("✅ All synthetic meter data generated and saved to CSV.")

if __name__ == "__main__":
    generate_dataset()
    print("✅ Dataset generation complete.")