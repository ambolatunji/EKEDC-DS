import streamlit as st
import pandas as pd
import os
from datetime import datetime

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
