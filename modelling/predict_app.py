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
        dtype_mapping = {
        'meter_id': str,
        'location_id': str,
        'date': str,
        'energy_consumed_kwh_sum': float,
        'expected_energy_kwh_sum': float,
        'expected_energy_pf_calc_sum': float,
        'total_vended_amount_sum': float,
        'revenue_loss_sum': float,
        'energy_loss_kwh_sum': float,
        'energy_loss_flag_sum': int,
        'fault_flag_sum': int,
        'tamper_flag_sum': int,
        'uptime_flag_sum': int,
        'supply_voltage_flag_sum': int,
        'pf_delta_mean': float,
        'v_agg_mean': float,
        'v_agg_min': float,
        'v_agg_max': float,
        'i_agg_mean': float,
        'i_agg_min': float,
        'i_agg_max': float,
        'power_factor_mean': float,
        'power_factor_min': float,
        'band_compliance_max': int,
        'is_industrial_max': int,
        'high_vend_zone_max': int,
        'tariff_per_kwh_mean': float,
        'fault_detected': int,
        'tamper_detected': int,
        'energy_loss_ratio': float,
        'supply_hours': int,
        'uptime_hours': int,
        'energy_efficiency_ratio': float
        }

        if os.path.exists("../daily_meter_summary.csv"):
            df = pd.read_csv("../daily_meter_summary.csv", dtype=dtype_mapping)
            stats = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                stats[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean())
                }
            return stats
        return {}

# ============== Utility: Create Base Input Template ==============
def create_template(features):
    base_inputs = set()

    for feat in features:
        if feat in FEATURE_DERIVATION_CONFIG:
            base_inputs.update(FEATURE_DERIVATION_CONFIG[feat]['inputs'].keys())
        else:
            base_inputs.add(feat)  # Direct fields (like v_agg_mean etc.)

    return list(sorted(base_inputs))
        
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
    
    st.markdown("### Feature Schema Inference")
    st.code(", ".join(default_features), language="python")

    # ------------------ Input Mode ------------------
    mode = st.radio("Input Mode", ["Single", "Batch"], horizontal=True)

    if mode == "Single":
        st.subheader("📌 Input Values")
        values = {}
        for feature in default_features:
            if feature in FEATURE_DERIVATION_CONFIG:
                st.markdown(f"####{feature} (Auto-Calculated)")
                temp_inputs = {}
                for key, val in FEATURE_DERIVATION_CONFIG[feature]['inputs'].items():
                    temp_inputs[key] = st.number_input(val['label'], value=val['default'])
                calc_func = FEATURE_DERIVATION_CONFIG[feature]['calculation']
                try:
                    values[feature] = calc_func(**temp_inputs)
                except:
                    values[feature] = 0
            elif feature in daily_stats:
                stats = daily_stats[feature]
                values[feature] = st.slider(
                    f"{feature} (based on Daily Summary)",
                    min_value=round(stats["min"],2),
                    max_value=round(stats["max"],2),
                    value=round(stats["mean"],2)
                )
            else:
                values[feature] = st.number_input(f"{feature} (Unknown Feature, Manual Input)", value=0.0)

        st.divider()
        st.subheader("🔮 Predict Now!")

        df = pd.DataFrame([values])

        #for col in default_features:
          #  values[col] = st.number_input(col, value=0.0)
        
        st.dataframe(df)
        scaler = StandardScaler()
        X_scaled = StandardScaler().fit_transform(df)

        threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5) if not is_dl else 0.5

        if st.button("🔮 Predict Now"):
            
            y_prob = model.predict(X_scaled)
            y_pred = (y_prob > threshold).astype(int) if not isinstance(y_prob[0], np.int64) else y_prob

            prediction_value = y_pred.flatten()[0]
            risk_score = min(float(y_prob.flatten()[0]) * 100, 100)
            confidence_score = min(100-risk_score, 100)

            # Meaningful interpretation
            if prediction_value == 1:
                st.error(f"🚨 Anomaly Detected: Possible Tamper/Fault!")
                st.warning(f"⚡ Risk Score: {risk_score:.2f}%")
                st.info(f"🔰 Confidence Score: {confidence_score:.2f}%")
            else:
                st.success(f"✅ No Anomaly Detected: Operation Healthy.")
                st.info(f"🧠 Risk of Anomaly: {risk_score:.2f}%")
                st.success(f"🔰 Confidence Score: {confidence_score:.2f}%")

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

        template_cols = create_template(default_features)
        with st.expander(" Download Batch Template"):
            template = pd.DataFrame(columns=template_cols)
            csv = template.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV Template", csv, "batch_input_template.csv", "text/csv")
        uploaded_file = st.file_uploader("Upload batch input file", type=["csv", "xlsx"])
        if uploaded_file and uploaded_file.name.endswith(".xlsx"):
            input_df = pd.read_excel(uploaded_file)
            st.write("📄 Preview:", input_df.head())
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            st.write("📄 Preview:", input_df.head())

            processed_batch = []

            for idx, row in input_df.iterrows():
                single_data = {}
                for feat in default_features:
                    if feat in FEATURE_DERIVATION_CONFIG:
                        calc_inputs = {k: row.get(k, 0) for k in FEATURE_DERIVATION_CONFIG[feat]['inputs']}
                        calc_func = FEATURE_DERIVATION_CONFIG[feat]['calculation']
                        try:
                            single_data[feat] = calc_func(**calc_inputs)
                        except:
                            single_data[feat] = 0
                    else:
                        single_data[feat] = row.get(feat, 0)
                processed_batch.append(single_data)

            batch_input = pd.DataFrame(processed_batch)
            st.write("🧩 Processed Features:", batch_input.head())

            # Check if "actual" exists for comparison
            show_actual = "actual" in batch_input.columns
            #X = input_df[default_features].copy()
            scaled = StandardScaler().fit_transform(batch_input)

            if st.button("🔮 Predict Batch"):
                y_probs = model.predict(X_scaled).flatten()
                y_preds = (y_probs > 0.5).astype(int)

                batch_input['risk_score (%)'] = np.clip(y_probs * 100, 0, 100)
                batch_input['confidence_score (%)'] = 100 - batch_input['risk_score (%)']
                batch_input['prediction'] = y_preds

                st.success("✅ Batch Prediction Complete")
                st.dataframe(batch_input)

                today = pd.Timestamp.now().strftime("%Y-%m-%d")
                batch_input.to_csv(f"batch_predictions_{today}.csv", index=False)
                st.success(f"📥 Saved: batch_predictions_{today}.csv")
                # ----------------- Visualization of Batch Risk -----------------

                st.subheader("📊 Batch Risk Distribution")

                # Pie chart for high vs low risk
                high_risk = batch_input[batch_input['risk_score (%)'] > 80]
                low_risk = batch_input[batch_input['risk_score (%)'] <= 80]

                risk_counts = pd.DataFrame({
                    "Risk Category": ["High Risk (>80%)", "Low/Moderate Risk (<=80%)"],
                    "Count": [len(high_risk), len(low_risk)]
                })

                fig1, ax1 = plt.subplots()
                ax1.pie(risk_counts["Count"], labels=risk_counts["Risk Category"], autopct='%1.1f%%', startangle=90, colors=["red", "green"])
                ax1.axis('equal')
                ax1.set_title("📊 Risk Category Breakdown")
                st.pyplot(fig1)

                # Bar chart for top risky meters
                st.subheader("📈 Top Risky Meters (Batch)")
                top_risks = batch_input.sort_values(by="risk_score (%)", ascending=False).head(20)

                fig2, ax2 = plt.subplots(figsize=(10,5))
                ax2.bar(top_risks.index.astype(str), top_risks["risk_score (%)"], color="crimson")
                ax2.set_xlabel("Meter Index")
                ax2.set_ylabel("Risk Score (%)")
                ax2.set_title("📈 Top 20 Risk Scores in Batch")
                plt.xticks(rotation=90)
                st.pyplot(fig2)

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

if __name__ == '__main__':
    main()


