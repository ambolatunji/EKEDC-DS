import streamlit as st
import pandas as pd
import pickle
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model as load_dl_model
import matplotlib.pyplot as plt
import os
from datetime import datetime

st.title("🔮 Predict with Trained Models")

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

    @st.cache_resource
    def load_model(path, is_dl=False):
        return load_dl_model(path) if is_dl else pickle.load(open(path, "rb"))

    model = load_model(model_path, is_dl)

    # ------------------ Inferred Schema ------------------
    # Simulate or extract known schema
    default_features = [
        "energy_consumed_kwh_sum", "expected_energy_kwh_sum", "total_vended_amount_sum",
        "revenue_loss_sum", "fault_flag_sum", "tamper_flag_sum",
        "tariff_per_kwh_mean", "power_factor_mean", "customer_risk_score_mean"
    ]

    st.markdown("### Feature Schema Inference")
    st.code(", ".join(default_features), language="python")

    # ------------------ Input Mode ------------------
    mode = st.radio("Input Mode", ["Single", "Batch"], horizontal=True)

    if mode == "Single":
        st.subheader("📌 Input Values")
        values = {}
        for col in default_features:
            values[col] = st.number_input(col, value=0.0)
        
        df = pd.DataFrame([values])
        scaled = StandardScaler().fit_transform(df)

        threshold = st.slider("Prediction Threshold (for classification)", 0.0, 1.0, 0.5) if not is_dl else 0.5

        if st.button("🔮 Predict Now"):
            y_prob = model.predict(scaled)
            y_pred = (y_prob > threshold).astype(int) if not isinstance(y_prob[0], np.int64) else y_prob
            st.success(f"✅ Prediction: {y_pred.flatten()[0]:.4f}")

    elif mode == "Batch":
        st.subheader("📥 Upload CSV File")
        uploaded_file = st.file_uploader("Upload batch input CSV", type=["csv"])
        
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

    from datetime import datetime

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
        
    log_prediction_event(model_path, num_predictions=1, input_file="manual")
    # or
    log_prediction_event(model_path, num_predictions=len(input_df), input_file=uploaded_file.name)

if __name__ == '__main__':
    main()


