import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

def main():
    st.title("🏠 EKEDC AI Control Center Overview")

    st.header("🔮 Today's Live Risk Score")

    # Check latest batch prediction
    latest_pred_file = "batch_predictions.csv"
    if os.path.exists(latest_pred_file):
        df_pred = pd.read_csv(latest_pred_file)
        risk_score = (df_pred["prediction"] > 0.8).mean() * 100
        st.metric("🔥 High Risk Meters (%)", f"{risk_score:.2f}%")

        if risk_score > 50:
            st.error(f"⚠️ Warning: High Risk Level Detected ({risk_score:.2f}%)")
        elif risk_score > 20:
            st.warning(f"⚡ Moderate Risk Detected ({risk_score:.2f}%)")
        else:
            st.success(f"✅ Low Risk: Operations Healthy ({risk_score:.2f}%)")
    else:
        st.info("ℹ️ No batch predictions available yet.")

    st.divider()

    st.header("📈 Today's Forecast Summary")

    # Find forecast files
    forecast_files = sorted([f for f in os.listdir() if f.startswith("forecast_") and f.endswith(".csv")], reverse=True)

    if forecast_files:
        latest_forecast = forecast_files[0]
        st.success(f"Showing Latest Forecast: `{latest_forecast}`")
        df_forecast = pd.read_csv(latest_forecast)
        st.dataframe(df_forecast.head())

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_forecast["hour"], df_forecast["projected_value"], color="crimson", linewidth=2, label="Forecasted Value")
        
        if "lower_bound" in df_forecast.columns and "upper_bound" in df_forecast.columns:
            ax.fill_between(df_forecast["hour"], df_forecast["lower_bound"], df_forecast["upper_bound"],
                            color="pink", alpha=0.4, label="Confidence Band")

        ax.axhline(df_forecast["projected_value"].mean(), linestyle="--", color="green", label="Average Value")
        ax.set_xlabel("Timeline (Hours Ahead)")
        ax.set_ylabel("₦ Value")
        ax.set_title("📈 Forecasted Revenue / Loss with Recovery & Confidence")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("ℹ️ No forecasts found yet.")

    st.divider()

    st.header("🗂️ Recent Audit Log Summary")

    # Training Logs
    if os.path.exists("audit_train_log.csv"):
        df_train_log = pd.read_csv("audit_train_log.csv")
        st.subheader("🧠 Recent Model Trainings")
        st.dataframe(df_train_log.tail(5))
    else:
        st.info("No training logs available.")

    # Prediction Logs
    if os.path.exists("audit_predict_log.csv"):
        df_predict_log = pd.read_csv("audit_predict_log.csv")
        st.subheader("🔮 Recent Prediction Activities")
        st.dataframe(df_predict_log.tail(5))
    else:
        st.info("No prediction logs available.")

if __name__ == "__main__":
    main()
