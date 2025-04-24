import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

st.title("🥇 Compare Trained Models")

def main():

    metrics_file = "evaluation_metrics.csv"
    if not os.path.exists(metrics_file):
        st.warning("No evaluation metrics found. Please evaluate models first.")
        st.stop()

    df = pd.read_csv(metrics_file)

    # Display selection
    df["display_name"] = df["target"] + " | " + df["model_type"]
    selected = st.multiselect("Select Models to Compare", df["display_name"].tolist(), default=df["display_name"].tolist()[:2])

    if not selected or len(selected) < 2:
        st.info("Please select at least 2 models.")
        st.stop()

    compare_df = df[df["display_name"].isin(selected)]

    # ------------------ Summary Table ------------------
    st.markdown("### 📋 Model Comparison Summary")
    st.dataframe(compare_df.drop(columns=["display_name"]))

    # ------------------ Radar Chart ------------------
    st.markdown("### 🕸️ Metric Radar Chart")

    # Normalize all numeric values
    norm_df = compare_df.copy()
    numeric_cols = compare_df.select_dtypes(include='number').columns
    norm_df[numeric_cols] = (compare_df[numeric_cols] - compare_df[numeric_cols].min()) / (compare_df[numeric_cols].max() - compare_df[numeric_cols].min())

    fig = go.Figure()
    for _, row in norm_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[numeric_cols],
            theta=numeric_cols,
            fill='toself',
            name=row["display_name"]
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ Classification Metric Bar ------------------
    if "classification" in compare_df["task"].values:
        st.markdown("### 📊 Classification Metrics")
        metric = st.selectbox("Choose Metric", ["accuracy", "precision", "recall", "f1_score"])
        subset = compare_df[compare_df["task"] == "classification"].set_index("display_name")
        st.bar_chart(subset[metric])

    # ------------------ Regression Metric Bar ------------------
    if "regression" in compare_df["task"].values:
        st.markdown("### 📉 Regression Metrics")
        metric = st.selectbox("Choose Metric", ["mae", "mse", "rmse", "r2_score"], key="reg")
        subset = compare_df[compare_df["task"] == "regression"].set_index("display_name")
        st.bar_chart(subset[metric])

    # ------------------ Side-by-Side Confusion Matrices ------------------
    if st.toggle("📷 Show Confusion Matrices Side-by-Side"):
        st.markdown("### 🧾 Confusion Matrices")
        cols = st.columns(len(selected))
        for i, name in enumerate(selected):
            base = name.split(" | ")[0]
            plot_path = f"plots/{base}_conf_matrix_eval.png"
            if os.path.exists(plot_path):
                cols[i].image(Image.open(plot_path), caption=base)
            else:
                cols[i].warning(f"No plot found for {base}")

    # ------------------ Downloadable Table ------------------
    csv = compare_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Comparison CSV", csv, "model_comparison.csv", "text/csv")

if __name__ == '__main__':
    main()
