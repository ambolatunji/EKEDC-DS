import streamlit as st
import pandas as pd
import os
from prepare_feature import prepare_data, load_data

st.title("📁 Data Preparation")
def main():
    st.markdown("""
    ### Data Preparation for Meter Analytics
    This app allows you to upload your daily meter summary file and run feature engineering.
    The processed data will be saved for model training and evaluation.
    """)
    st.markdown("""
    Upload your daily meter summary file (or raw file if logic is added).
    This file will be processed and saved as `engineered_meter_data.csv` for modeling.
    """)

    # ---------------- Upload CSV ----------------
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
        st.write("### 📄 File Preview", df.head())
        st.write(f"Shape: {df.shape}")

        if st.button("🧠 Run Feature Engineering"):
            try:
                df.to_csv("daily_meter_summary.csv", index=False)  # required by prepare_feature
                # Load and process
                df_loaded = load_data()
                st.success("📥 Data loaded successfully.")
                st.write(df_loaded.head())

                # Optionally: Preview engineered output
                output_path = "engineered_meter_data.csv"
                df_loaded.to_csv(output_path, index=False)
                st.success(f"✅ Feature engineered data saved to `{output_path}`.")

                # Download button
                csv = df_loaded.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Engineered Data", csv, "engineered_meter_data.csv", "text/csv")
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
    else:
        st.info("Upload a file to get started.")
if __name__ == '__main__':
    main()