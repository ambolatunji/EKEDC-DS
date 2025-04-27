import streamlit as st
import pandas as pd
import pickle
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model as load_dl_model
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


st.title("🔮 Predict with Trained Models. Forecast Impact & Recovery")

def log_prediction_event(model_path, num_predictions, input_file="manual"):
    log_file = "audit_predict_log.csv"
    entry = {
         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": os.path.basename(model_path),
        "num_predictions": num_predictions,
        "input_source": input_file
    }
    log_df = pd.DataFrame([entry])
    if os.path.exists(log_file):
         log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
# ============== Feature Derivation Config ==============
FEATURE_DERIVATION_CONFIG = {
    'expected_energy_kwh_sum': {
        'calculation': lambda total_watt, time_usage: (total_watt * time_usage) / 1000,
        'inputs': {
            'total_watt': {'label': "Total Appliance Watt (W)", 'default': 1000},
            'time_usage': {'label': "Total Usage Time (hrs)", 'default': 24},
        },
    },
    'expected_energy_pf_calc_sum': {
        'calculation': lambda v_agg_mean, i_agg_mean, power_factor_mean: (v_agg_mean * i_agg_mean * power_factor_mean) / 1000,
        'inputs': {
            'v_agg_mean': {'label': "Average Voltage (V)", 'default': 230},
            'i_agg_mean': {'label': "Average Current (A)", 'default': 10},
            'power_factor_mean': {'label': "Power Factor Mean", 'default': 0.95},
        },
    },
    'energy_loss_kwh_sum': {
        'calculation': lambda expected, consumed: expected - consumed,
        'inputs': {
            'expected': {'label': "Expected Energy (kWh)", 'default': 10},
            'consumed': {'label': "Consumed Energy (kWh)", 'default': 8},
        },
    },
    'revenue_loss_sum': {
        'calculation': lambda energy_loss, tariff: energy_loss * tariff,
        'inputs': {
            'energy_loss': {'label': "Energy Loss (kWh)", 'default': 2},
            'tariff': {'label': "Tariff per kWh", 'default': 48},
        },
    },
    'energy_loss_ratio': {
        'calculation': lambda energy_loss, expected: (energy_loss / expected) if expected else 0,
        'inputs': {
            'energy_loss': {'label': "Energy Loss (kWh)", 'default': 2},
            'expected': {'label': "Expected Energy (kWh)", 'default': 10},
        },
    },
    'energy_efficiency_ratio': {
        'calculation': lambda consumed, expected: (consumed / expected) if expected else 0,
        'inputs': {
            'consumed': {'label': "Consumed Energy (kWh)", 'default': 8},
            'expected': {'label': "Expected Energy (kWh)", 'default': 10},
        },
    },
}

# ============== Load Daily Summary for Realistic Min/Max ==============
@st.cache_data
def load_summary_stats():
    if os.path.exists("../daily_meter_summary.csv"):
        df = pd.read_csv("../daily_meter_summary.csv")
        stats = {}
        for col in df.columns:
            stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean())
            }
        return stats
    return {}

# ============== Utility: Create Base Input Template ==============
def create_template(features):
    cols = []
    for feat in features:
        if feat in FEATURE_DERIVATION_CONFIG:
            cols.extend(FEATURE_DERIVATION_CONFIG[feat]['inputs'].keys())
    return list(set(cols))
        
def main():
    st.markdown("""
    ### Predict with Trained Models
    Use this app to make predictions using your trained models. You can input values manually or upload a CSV file for batch predictions.
    """)
    model_dir = "models"
    available_models = sorted(glob(f"{model_dir}/*"))

    if not available_models:
        st.warning("No models available. Please train first.")
        st.stop()

    model_path = st.selectbox("Select a Model", available_models)
    is_dl = model_path.endswith(".h5")

    # Load selected features used during training
    feature_file = model_path.replace(".pkl", "_features.txt").replace(".h5", "_features.txt")
    default_features = []
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            default_features = [line.strip() for line in f.readlines()]
    else:
        st.error("⚠️ Could not find features file for the selected model.")
        st.stop()

    daily_stats = load_summary_stats()

    @st.cache_resource
    def load_model(path, is_dl=False):
        return load_dl_model(path) if is_dl else pickle.load(open(path, "rb"))

    model = load_model(model_path, is_dl)

    # ------------------ Inferred Schema ------------------
    
    #st.markdown("### Feature Schema Inference")
    #st.code(", ".join(default_features), language="python")

    # ------------------ Input Mode ------------------
    mode = st.radio("Input Mode", ["Single", "Batch"], horizontal=True)

    if mode == "Single":
        st.subheader("📌 Input Values")
        values = {}
        for col in default_features:
            values[col] = st.number_input(col, value=0.0)
        
        df = pd.DataFrame([values])
        scaler = StandardScaler()
        X_scaled = StandardScaler().fit_transform(df)

        threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5) if not is_dl else 0.5

        if st.button("🔮 Predict Now"):
            
            y_prob = model.predict(X_scaled)
            y_pred = (y_prob > threshold).astype(int) if not isinstance(y_prob[0], np.int64) else y_prob

            prediction_value = y_pred.flatten()[0]
            risk_score = float(y_prob.flatten()[0]) * 100

            # Meaningful interpretation
            if prediction_value == 1:
                st.error(f"🚨 Anomaly Detected: Possible Tamper/Fault!")
                st.warning(f"⚡ Risk Score: {risk_score:.2f}%")
            else:
                st.success(f"✅ No Anomaly Detected: Operation Healthy.")
                st.info(f"🧠 Risk of Anomaly: {risk_score:.2f}%")

            # Alert if risk is high
            if risk_score > 80:
                st.error(f"❗ ALERT: Very High Risk Detected ({risk_score:.2f}%)")

            # Forecast revenue impact
            st.subheader("📈 Forecast: Revenue Loss / Recovery Scenario")

            base_loss = df['revenue_loss_sum'].values[0]
            base_energy = df['expected_energy_kwh_sum'].values[0]
            tariff = df['tariff_per_kwh_mean'].values[0]
            hours = 48

            timeline = pd.date_range(start=pd.Timestamp.now(), periods=hours, freq="H")

            # Two Scenarios
            if prediction_value == 1:
                loss_forecast = []
                for i in range(hours):
                    if i < 12:
                        loss_forecast.append(base_loss * (1 + 0.05*np.random.randn()))
                    else:
                        loss_forecast.append(base_loss * 0.1 * (1 + 0.02*np.random.randn()))
            else:
                loss_forecast = [base_energy * tariff * (1 + 0.02*np.random.randn()) for _ in range(hours)]

            lower_bound = [val * 0.8 for val in loss_forecast]
            upper_bound = [val * 1.2 for val in loss_forecast]

            forecast_df = pd.DataFrame({
                "hour": timeline,
                "projected_value": loss_forecast,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
             })

            # Plot
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(forecast_df["hour"], forecast_df["projected_value"], color="crimson", linewidth=2, label="Projected Impact")
            ax.axhline(np.mean(forecast_df["projected_value"]), color='blue', linestyle="--", label="Average")
            if prediction_value == 1:
                ax.axvline(forecast_df["hour"].iloc[12], color="green", linestyle="--", label="Projected Recovery")
            ax.set_xlabel("Timeline (Next 48 Hours)")
            ax.set_ylabel("₦ Value")
            ax.set_title("📈 Forecast Revenue / Loss Trend")
            ax.grid(True)
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Download CSV
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", csv, "forecast.csv", "text/csv")

            log_prediction_event(model_path, num_predictions=1, input_file="manual")


    elif mode == "Batch":
        st.subheader("📥 Upload CSV File")
        uploaded_file = st.file_uploader("Upload batch input file", type=["csv", "xlsx"])
        if uploaded_file and uploaded_file.name.endswith(".xlsx"):
            input_df = pd.read_excel(uploaded_file)
            st.write("📄 Preview:", input_df.head())
        
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.write("📄 Preview:", input_df.head())

            # Check if "actual" exists for comparison
            show_actual = "actual" in input_df.columns
            X = input_df[default_features].copy()
            scaled = StandardScaler().fit_transform(X)

            if st.button("🔮 Predict Batch"):
                pred = model.predict(scaled)
                input_df["prediction"] = pred.flatten()

                st.success("✅ Batch predictions complete.")
                st.dataframe(input_df.head())

                 # ✅ Log the prediction event safely
                log_prediction_event(model_path, num_predictions=len(input_df), input_file=uploaded_file.name)
                if show_actual:
                    st.markdown("### 📈 Predicted vs Actual")
                    plt.figure(figsize=(6, 4))
                    plt.scatter(input_df["actual"], input_df["prediction"], alpha=0.6)
                    plt.xlabel("Actual")
                    plt.ylabel("Prediction")
                    plt.title("Predicted vs Actual")
                    st.pyplot(plt.gcf())

                # Downloadable output
                csv = input_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", csv, "batch_predictions.csv", "text/csv")

if __name__ == '__main__':
    main()


