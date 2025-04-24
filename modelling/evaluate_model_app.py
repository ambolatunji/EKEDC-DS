import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from evaluate_model import evaluate_pipeline

st.title("📊 Evaluate Trained Models")

def main():
    # Load saved models
    model_dir = "models"
    all_models = sorted(glob(os.path.join(model_dir, "*")))

    if not all_models:
        st.warning("No trained models found. Please train models first.")
        st.stop()

    model_choice = st.selectbox("Select a Model to Evaluate", all_models)
    is_dl = model_choice.endswith(".h5")

    # Infer task type from filename
    target = os.path.basename(model_choice).split("_")[0]
    if any(x in model_choice for x in ["tamper", "fault", "band"]):
        task_type = "classification"
    else:
        task_type = "regression"

    if st.button("📈 Run Evaluation"):
        with st.spinner("Running evaluation..."):
            evaluate_pipeline(model_path=model_choice, target=target, task_type=task_type, is_dl=is_dl)
        st.success("✅ Evaluation complete!")

        # Show Plots
        plot_base = f"plots/{target}"
        if task_type == "classification":
            st.image(f"{plot_base}_conf_matrix_eval.png", caption="Confusion Matrix")
            st.image(f"{plot_base}_roc_eval.png", caption="ROC Curve")
        else:
            st.image(f"{plot_base}_actual_vs_pred.png", caption="Actual vs Predicted")

    # Metrics CSV
    if os.path.exists("evaluation_metrics.csv"):
        st.markdown("### 📋 Evaluation Summary")
        metrics_df = pd.read_csv("evaluation_metrics.csv")
        st.dataframe(metrics_df)

        csv = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Metrics CSV", csv, "evaluation_metrics.csv", "text/csv")

if __name__ == '__main__':
    main()
