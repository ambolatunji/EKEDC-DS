import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def main():
    st.markdown("""
    ### Admin & Audit Panel
    This panel allows you to monitor the status of your data and models.
    You can view logs, download files, and check the status of your training and predictions.
    """)
    st.title("🔐 Admin & Audit Panel")

    # ------------------- Optional Access Control -------------------
    ADMIN_PIN = "ekedc2024"

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        pin = st.text_input("🔑 Enter Admin PIN", type="password")
        if pin == ADMIN_PIN:
            st.success("✅ Access Granted")
            st.session_state.auth = True
        else:
            st.stop()

    # ------------------- File Logs -------------------
    st.markdown("### 🗂️ Data Uploads")

    if os.path.exists("daily_meter_summary.csv"):
        st.success("✅ File: `daily_meter_summary.csv`")
        st.code("✔️ Available for feature engineering")

    if os.path.exists("engineered_meter_data.csv"):
        st.success("✅ File: `engineered_meter_data.csv`")
        df = pd.read_csv("engineered_meter_data.csv")
        st.write(df.head())

    # ------------------- Training Logs -------------------
    st.markdown("### 🧠 Model Training Log")

    train_log_path = "audit_train_log.csv"
    if os.path.exists(train_log_path):
        df_train = pd.read_csv(train_log_path)
        st.dataframe(df_train)
        csv = df_train.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Training Log", csv, "training_log.csv", "text/csv")
    else:
        st.info("No training logs found.")

    # ------------------- Prediction Logs -------------------
    st.markdown("### 🔮 Prediction History")

    predict_log_path = "audit_predict_log.csv"
    if os.path.exists(predict_log_path):
        df_pred = pd.read_csv(predict_log_path)
        st.dataframe(df_pred)
        csv2 = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Prediction Log", csv2, "prediction_log.csv", "text/csv")
    else:
        st.info("No prediction logs found.")

    # --------------- Section 3: Risk Management ---------------
    st.header("⚡ Risk Monitoring & Alerts")

    if os.path.exists("batch_predictions.csv"):
        df_batch = pd.read_csv("batch_predictions.csv")
        
        if "prediction" in df_batch.columns:
            st.subheader("📈 Risk Distribution")
            fig, ax = plt.subplots()
            ax.hist(df_batch["prediction"], bins=30, color="crimson", alpha=0.7)
            ax.axvline(0.8, color="orange", linestyle="--", label="High Risk >80%")
            ax.set_xlabel("Predicted Risk Score")
            ax.set_ylabel("Meters")
            ax.set_title("Predicted Risk Distribution")
            ax.legend()
            st.pyplot(fig)

            high_risk_df = df_batch[df_batch["prediction"] > 0.8]
            st.subheader("🔥 High Risk Meters (Prediction > 80%)")
            st.dataframe(high_risk_df)

            csv_high = high_risk_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download High Risk Meters", csv_high, "high_risk_meters.csv", "text/csv")
    else:
        st.info("No batch prediction file found to calculate risks.")

# --------------- Section 4: Forecast Summary (Optional Future) ---------------
    st.header("📊 Forecast Summary (Optional Future)")
    st.markdown("This section is reserved for future forecast summaries.")
    st.header("📈 Forecast Impact Summary")

    forecast_file = "forecast.csv"

    if os.path.exists(forecast_file):
        df_forecast = pd.read_csv(forecast_file)
        
        st.subheader("🔮 Forecasted Revenue/Loss over Time")
        st.dataframe(df_forecast)

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df_forecast["hour"], df_forecast["projected_value"], label="Forecasted Value", color="purple")
        
        if "lower_bound" in df_forecast.columns and "upper_bound" in df_forecast.columns:
            ax.fill_between(df_forecast["hour"], df_forecast["lower_bound"], df_forecast["upper_bound"],
                            color="lavender", alpha=0.5, label="Confidence Band")
            
        ax.axhline(y=df_forecast["projected_value"].mean(), linestyle="--", color="green", label="Average Forecast")
        ax.set_xlabel("Timeline (Hours Ahead)")
        ax.set_ylabel("₦ Value")
        ax.set_title("Revenue Loss / Sales Forecast with Recovery")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Allow download
        forecast_download = df_forecast.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast CSV", forecast_download, "forecast_summary.csv", "text/csv")
    else:
        st.info("ℹ️ No forecast file found yet. Please run a prediction first.")

    # ------------------- End of Admin Panel -------------------
    st.markdown("### 🔒 End of Admin Panel")
    
if __name__ == "__main__":
    main()