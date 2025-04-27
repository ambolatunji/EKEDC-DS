import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from evaluate_model import evaluate_pipeline, parse_model_filename

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

    # ✅ Use unified filename parser
    target, algorithm, model_type = parse_model_filename(model_choice)

    if any(x in model_choice for x in ["tamper", "fault", "band"]):
        task_type = "classification"
    else:
        task_type = "regression"

    st.markdown(f"**Target:** `{target}` | **Algorithm:** `{algorithm}` | **Model Type:** `{model_type}`")
    # ✅ Sample size input
    sample_size = st.number_input(
        "Limit rows for evaluation (0 = full data)",
        min_value=0, max_value=50000, value=5000, step=500
    )

    if st.button("📈 Run Evaluation"):
        with st.spinner("Running evaluation..."):
            evaluate_pipeline(model_path=model_choice, target=target, task_type=task_type, is_dl=is_dl, sample_size=sample_size if sample_size > 0 else None)
        st.success("✅ Evaluation complete!")

        # Show Plots
        target_name = os.path.splitext(os.path.basename(model_choice))[0]
        plot_base = f"plots/{target_name}"
        if task_type == "classification":
            st.image(f"{plot_base}_conf_matrix_eval.png", caption="Confusion Matrix")
            st.image(f"{plot_base}_roc_eval.png", caption="ROC Curve")
        else:
            st.image(f"{plot_base}_actual_vs_pred.png", caption="Actual vs Predicted")

    # Metrics CSV
    if os.path.exists("evaluation_metrics.csv"):
        st.markdown("### 📋 Evaluation Summary")
        metrics_df = pd.read_csv("evaluation_metrics.csv")

        # Filter current model’s metrics
        model_id = os.path.basename(model_choice).replace(".pkl", "").replace(".h5", "")
        filtered = metrics_df[metrics_df["model_file"].str.contains(target_name)]
        st.dataframe(filtered)
        #st.dataframe(metrics_df)

        csv = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Metrics CSV", csv, "evaluation_metrics.csv", "text/csv")

if __name__ == '__main__':
    main()
