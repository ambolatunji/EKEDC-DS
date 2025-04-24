import streamlit as st
import os
import pandas as pd # Added for Training History display

try:
    from train_model import train_pipeline
    from prepare_feature import load_data
except ImportError as e:
    st.error(f"Error importing required modules: {e}. Make sure 'train_model.py' and 'prepare_feature.py' are present.")
    st.stop() # Stop execution if imports fail
def main():
    st.title("🧠 Train Models")
    st.markdown("Train Machine Learning or Deep Learning models with optional hyperparameter tuning.")

    df = load_data()

    # ------------------- Task Definitions -------------------
    # Using the structure from your previous file
    task_options = {
        "Tamper Detection": ("tamper_detected", "classification"),
        "Fault Detection": ("fault_detected", "classification"),
        "Energy Loss Estimation": ("energy_loss_kwh_sum", "regression"),
        "SLA Violation Risk": ("band_compliance_max", "classification"),
        "Customer Risk Scoring": ("customer_risk_score_mean", "regression")
    }

    # Algorithm options (consistent with train_model.py)
    algorithm_options = {
        "ml": ["logistic", "randomforest", "xgboost", "ensemble"],
        "dl": ["mlp"], # Only MLP for now
        # RL is handled via NotImplementedError in train_pipeline now
    }

    # ------------------- Model Selection UI (Using Columns for better layout) -------------------
    col1, col2 = st.columns(2)

    with col1:
        task_name = st.selectbox("🎯 Choose a Task", list(task_options.keys()))
        # Derive target and task_type from selection
        target, task_type = task_options[task_name]

        model_category = st.radio(
            "🤖 Model Type",
            ["ml", "dl"], # Removed 'rl' as it's not implemented
            help="Choose between Machine Learning (ML) or Deep Learning (DL)."
            )

    with col2:
        # Dynamically populate algorithm options based on model_category
        if model_category in algorithm_options:
            algo_list = algorithm_options[model_category]
            algorithm = st.selectbox(
                f"🛠️ {'ML' if model_category == 'ml' else 'DL'} Algorithm",
                algo_list,
                help=f"Select the {model_category.upper()} algorithm to use."
            )
        else:
            # This case should ideally not be reached if radio options are correct
            st.warning("No algorithms defined for this model type.")
            algorithm = None

        # Only show tuning option if it makes sense (e.g., for ML)
        tune_disabled = model_category == 'dl' # Disable tuning for the simple DL example
        tune = st.toggle(
            "🔍 Enable Hyperparameter Tuning",
            value=False,
            disabled=tune_disabled,
            help="Tune hyperparameters using GridSearchCV (primarily for ML models)."
            )
        if tune_disabled and model_category == 'dl':
            st.caption("Tuning not available for this basic DL model.")


    # ------------------ Feature Selection Block ------------------
    with st.expander("🧬 Feature Selection & Correlation Matrix"):

        st.markdown(f"**Target Column:** `{target}`")
        exclude = ['meter_id_', 'location_id_', 'feeder_id_', 'transformer_id_', 'date_',
                'anomaly_category', 'anomaly_summary_note', target]
        candidate_features = df.drop(columns=exclude, errors='ignore').select_dtypes(include='number')

        # Correlation with target
        corr_target = candidate_features.corrwith(df[target])
        corr_df = pd.DataFrame({
            "Feature": corr_target.index,
            "Correlation with Target": corr_target.values
        }).sort_values(by="Correlation with Target", ascending=False)

        st.dataframe(corr_df)

        # User selection
        selected_features = st.multiselect(
            "✅ Select Features to Train On",
            options=candidate_features.columns.tolist(),
            default=candidate_features.columns.tolist()
        )
        st.markdown(f"**Selected Features:** `{', '.join(selected_features)}`")

    # ------------------- Training Trigger & History Display -------------------
    train_col, history_col = st.columns([2, 1]) # Give more space to the button area

    with train_col:
        st.markdown("---") # Separator line
        # Disable button if algorithm selection failed
        train_button_disabled = algorithm is None

        if st.button("🚀 Train Model", disabled=train_button_disabled, type="primary"):
            st.info("Button clicked! Initiating training process...") # Immediate feedback
            if not selected_features:
                st.warning("Please select at least one feature to train on.")
            else:
                # Show spinner while training runs
                with st.spinner(f"⏳ Training `{algorithm.upper()}` model for `{task_name}`. This may take a while..."):
                    try:
                        # Call the pipeline function and capture the return values
                        success, message = train_pipeline(
                            target=target,
                            task_type=task_type,
                            model_category=model_category,
                            algorithm=algorithm,
                            tune=tune,
                            selected_features=selected_features
                        )

                        # Display success or error message based on the outcome
                        if success:
                            st.success(message) # Display the success message from train_pipeline
                        else:
                            st.error(message) # Display the error message from train_pipeline

                    except FileNotFoundError as e:
                        st.error(f"Training failed: Could not find data file. {e}")
                    except ImportError as e:
                        st.error(f"Training failed: Import error. {e}")
                    except Exception as e:
                        # Catch any other unexpected errors during the call itself
                        st.error(f"An unexpected error occurred in the Streamlit app: {e}")
                        # Optionally log the full traceback here if needed for debugging
                        import traceback
                        st.error("Traceback:")
                        st.code(traceback.format_exc())

    # ------------------- Training History (Moved to a separate column) -------------------
    with history_col:
        st.markdown("##### Recent Training History")
        log_file = "audit_train_log.csv" # Defined in train_model.py
        if os.path.exists(log_file):
            try:
                log_df = pd.read_csv(log_file)
                log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
                # Select relevant columns for display
                display_cols = ['timestamp', 'target', 'algorithm', 'model_type', 'status', 'score']
                # Filter out columns that might not exist if the log format changes
                display_cols = [col for col in display_cols if col in log_df.columns]
                recent_logs = log_df.sort_values('timestamp', ascending=False).head(5)
                st.dataframe(
                    recent_logs[display_cols],
                    hide_index=True,
                    column_config={ # Optional: Improve formatting
                        "timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                        "score": st.column_config.NumberColumn("Score", format="%.3f"),
                    }
                )
            except Exception as e:
                st.warning(f"Could not display training history: {e}")
        else:
            st.info("No training history log found yet.")
if __name__ == '__main__':
    main()