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

    model_files = [m for m in all_models if m.endswith(".pkl") or m.endswith(".h5")]

    if not model_files:
        st.warning("No trained models found. Please train models first.")
        st.stop()

    model_choice = st.selectbox("Select a Model to Evaluate", all_models)
    is_dl = model_choice.endswith(".h5")

    # ✅ Use unified filename parser
    target, algorithm, model_type = parse_model_filename(model_choice)

    # ✅ Use unified filename parser
    target, algorithm, model_type = parse_model_filename(model_choice)

    if any(x in model_choice for x in ["tamper", "fault", "band"]):
        task_type = "classification"
    else:
        task_type = "regression"

    # Map target names to actual column names if needed
    target_mapping = {
        "tamper_detected_xgboost_ml": "tamper_detected",
        "fault_detected_randomforest_ml": "fault_detected",
        "fault_detected_xgboost_ml": "fault_detected",
        "fault_detected_keras_dl": "fault_detected",
        "tamper_detected_randomforest_ml": "tamper_detected",
        "tamper_detected_keras_dl": "tamper_detected",

        "band_detected_xgboost_ml": "band_detected",
        "band_detected_randomforest_ml": "band_detected",
        "band_detected_keras_dl": "band_detected",

        # Adding more mappings as needed later na
    }
    
    # Use the mapping if available, otherwise keep original
    display_target = target
    actual_target = target_mapping.get(target, target)

    st.markdown(f"**Target:** `{display_target}` | **Algorithm:** `{algorithm}` | **Model Type:** `{model_type}`")
    if display_target != actual_target:
        st.info(f"Using actual database column name: `{actual_target}`")
    # ✅ Sample size input
    sample_size = st.number_input(
        "Limit rows for evaluation (0 = full data)",
        min_value=0, max_value=50000, value=5000, step=500
    )

    if st.button("📈 Run Evaluation"):
            try:
                with st.spinner("Running evaluation..."):
                    evaluate_pipeline(
                        model_path=model_choice, 
                        target=actual_target, 
                        task_type=task_type, 
                        is_dl=is_dl, 
                        sample_size=sample_size if sample_size > 0 else None
                    )
                st.success("✅ Evaluation complete!")

                # Show Plots
                model_basename = os.path.splitext(os.path.basename(model_choice))[0]
                plot_base = f"plots/{model_basename}"
                
                # Display appropriate plots based on task type
                if task_type == "classification":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(f"{plot_base}_conf_matrix_eval.png", caption="Confusion Matrix")
                    with col2:
                        st.image(f"{plot_base}_roc_eval.png", caption="ROC Curve")
                    st.image(f"{plot_base}_risk_distribution.png", caption="Risk Score Distribution")
                else:
                    st.image(f"{plot_base}_actual_vs_pred.png", caption="Actual vs Predicted")
                    
            except Exception as e:
                st.error(f"⚠️ Evaluation failed: {e}")
                st.exception(e)

    # Metrics CSV
    if os.path.exists("evaluation_metrics.csv"):
        st.markdown("### 📋 Evaluation Summary")
        metrics_df = pd.read_csv("evaluation_metrics.csv")

        # Filter current model's metrics
        filtered = metrics_df[metrics_df["model_file"].str.contains(os.path.basename(model_choice))]
        
        if not filtered.empty:
            st.dataframe(filtered)
        else:
            st.info("No metrics found for this model yet. Run an evaluation first.")

        st.markdown("#### All Evaluations")
        st.dataframe(metrics_df)
        
        csv = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Metrics CSV", csv, "evaluation_metrics.csv", "text/csv")

if __name__ == '__main__':
    main()
