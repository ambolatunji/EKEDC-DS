# train_model_updated.py
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor                
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import (
    classification_report, mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, confusion_matrix, roc_curve, auc
)
import seaborn as sns
from datetime import datetime
import traceback # For error logging

# Assuming prepare_feature.py exists in the same directory or PYTHONPATH
try:
    from prepare_feature import load_data, prepare_data
except ImportError:
    print("Error: prepare_feature.py not found. Please ensure it's accessible.")
    # Provide dummy functions if needed for basic script execution without data
    def load_data(): raise FileNotFoundError("Data file not found or prepare_feature not loaded.")
    def prepare_data(df, target, task_type): raise RuntimeError("prepare_data function not available.")


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Optional: Suppress TensorFlow/Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow/Keras not installed. Deep Learning models will not be available.")
    TF_AVAILABLE = False
    # Define dummy classes/functions if TF is not available to avoid runtime errors later
    class Sequential: pass
    class Dense: pass
    class Dropout: pass
    class Adam: pass
    def load_model(filepath): raise ImportError("TensorFlow/Keras not installed.")


MODEL_DIR = "models"
PLOTS_DIR = "plots"
LOG_FILE = "audit_train_log.csv"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Logging Function (Defined Outside Pipeline) ---
def log_training_event(target, algorithm, model_type, status, score=None, error_msg=None):
    """Logs the outcome of a training run."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": target,
        "model_type": model_type,
        "algorithm": algorithm,
        "status": status, # 'completed' or 'failed'
        "score": round(score, 4) if score is not None else None,
        "error": error_msg
    }
    log_df = pd.DataFrame([log_entry])
    try:
        if os.path.exists(LOG_FILE):
            log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)
        else:
            log_df.to_csv(LOG_FILE, index=False)
    except Exception as e:
        print(f"Error writing to log file {LOG_FILE}: {e}")


# --- Plotting Function ---
def plot_evaluation(y_true, y_pred, y_prob, target_name, task_type):
    """Generates and saves evaluation plots."""
    plt.style.use('seaborn-v0_8-darkgrid') # Use a modern style

    if task_type == 'classification':
        # Confusion Matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linecolor='black', linewidths=0.5)
            plt.title(f'Confusion Matrix: {target_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(f"{PLOTS_DIR}/{target_name}_confusion_matrix.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting confusion matrix for {target_name}: {e}")

        # ROC Curve
        try:
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(7, 5))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve: {target_name}')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{PLOTS_DIR}/{target_name}_roc_curve.png")
                plt.close()
            else:
                 print(f"Skipping ROC curve for {target_name}: No probabilities provided.")
        except Exception as e:
            print(f"Error plotting ROC curve for {target_name}: {e}")

    elif task_type == 'regression':
        # Actual vs Predicted Scatter Plot
        try:
            plt.figure(figsize=(7, 5))
            plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', s=50)
            # Add line y=x
            lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
            plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Actual vs Predicted: {target_name}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{PLOTS_DIR}/{target_name}_actual_vs_pred.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting actual vs predicted for {target_name}: {e}")


# --- ML Model Training ---
def train_ml_model(X_train, y_train, model_type='xgboost', tune=False):
    """Trains various ML models with optional hyperparameter tuning."""
    model = None
    # Define base models
    if model_type == 'logistic':
        base_model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
        param_grid = {'C': [0.01, 0.1, 1, 10]} if tune else None

    elif model_type == 'randomforest':
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]} if tune else None

    elif model_type == 'xgboost':
        base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1]} if tune else None

    elif model_type == 'ensemble':
        lr = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
        rf = RandomForestClassifier(random_state=42, n_estimators=50) # Keep ensemble simple
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=50)
        model = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)], voting='soft')
        # Tuning ensemble is complex, usually tune base estimators first
        param_grid = None
        tune = False # Disable tuning specifically for this simple ensemble example

    else:
        raise ValueError("Unsupported ML model type.")

    # Handle tuning or direct fitting
    if tune and param_grid and model_type != 'ensemble':
        print(f"⏳ Tuning {model_type}...")
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1) # Example scoring
        grid_search.fit(X_train, y_train)
        print(f"Best Params: {grid_search.best_params_}")
        model = grid_search.best_estimator_ # Use the best found model
    elif model is None: # If not ensemble and not tuning
        model = base_model
        model.fit(X_train, y_train)
    else: # Ensemble case (already defined 'model')
         model.fit(X_train, y_train)


    return model

# --- DL Model Building ---
def build_dense_model(input_dim, task_type='classification'):
    """Builds a simple Dense MLP model."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras is required for Deep Learning models.")

    model = Sequential(name=f"MLP_{task_type}")
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    if task_type == 'classification':
        model.add(Dense(1, activation='sigmoid')) # Binary classification
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    else: # Regression
        model.add(Dense(1, activation='linear')) # Regression output
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    return model

# --- Model Saving ---
def save_model(model, name, is_dl=False):
    """Saves the trained model to disk."""
    if is_dl and TF_AVAILABLE:
        path = os.path.join(MODEL_DIR, f"{name}.h5")
        model.save(path)
    elif not is_dl:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    elif is_dl and not TF_AVAILABLE:
         print(f"Skipping saving DL model {name}: TensorFlow not available.")
         return False # Indicate saving failed
    else:
         print(f"Unknown model type for saving: {name}")
         return False

    print(f"✅ Model saved to {path}")
    return True

# --- Main Training Pipeline ---
def train_pipeline(target, task_type='classification', model_category='ml', algorithm='xgboost', tune=False, selected_features=None, log_fn=print):
    """Orchestrates the full training, evaluation, and saving process."""
    from datetime import datetime
    start_time = datetime.now()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tune_flag = "tuned" if tune else "default"
    model_name_base = f"{target}_{algorithm}_{model_category}_{tune_flag}_{timestamp}"
    log_fn(f"\n🚀 [{start_time.strftime('%H:%M:%S')}] Starting training for: {target} | Task: {task_type} | Category: {model_category} | Algo: {algorithm} | Tuning: {tune}")

    score = None
    model = None
    status = 'failed' # Default status
    error_message = None

    try:
        # 1. Load and Prepare Data
        log_fn("💾 Loading data...")
        df = load_data() # Assumes load_data finds the file
        log_fn("⚙️ Preparing data...")
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            df, target, task_type=task_type, selected_features=selected_features
        )

        log_fn(f"Data Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

        input_dim = X_train.shape[1]

        # 2. Train Model
        log_fn(f"🧠 Training model ({model_category.upper()} - {algorithm})...")
        if model_category == 'ml':
            if task_type == 'classification':
                model = train_ml_model(X_train, y_train, model_type=algorithm, tune=tune)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            elif task_type == 'regression':
                log_fn(f"🧠 Training model (Regression - {algorithm})...")

                if algorithm == 'randomforest':
                    base_model = RandomForestRegressor()
                    if tune:
                        log_fn("⏳ Tuning RandomForestRegressor...")
                        param_grid = {"max_depth": [5, 10, 15]}
                        model = GridSearchCV(base_model, param_grid, cv=3)
                        model.fit(X_train, y_train)
                        log_fn(f"Best Params: {model.best_params_}")
                    else:
                        model = base_model.fit(X_train, y_train)

                elif algorithm == 'xgboost':
                    base_model = XGBRegressor()
                    if tune:
                        log_fn("⏳ Tuning XGBRegressor...")
                        param_grid = {"max_depth": [3, 6, 10]}
                        model = GridSearchCV(base_model, param_grid, cv=3)
                        model.fit(X_train, y_train)
                        log_fn(f"Best Params: {model.best_params_}")
                    else:
                        model = base_model.fit(X_train, y_train)

                elif algorithm == 'linear':
                    model = LinearRegression().fit(X_train, y_train)

                elif algorithm == 'ensemble':
                    log_fn("🤝 Building ensemble regressor...")
                    model1 = LinearRegression()
                    model2 = RandomForestRegressor()
                    model3 = XGBRegressor()
                    model = VotingRegressor(estimators=[
                        ("lr", model1),
                        ("rf", model2),
                        ("xgb", model3)
                    ])
                    model.fit(X_train, y_train)
                else:
                    raise ValueError(f"Unsupported regression algorithm: {algorithm}")

                y_pred = model.predict(X_test)
                y_prob = None # No probabilities for regression

        elif model_category == 'dl':
            if not TF_AVAILABLE:
                raise ImportError("Cannot train DL model: TensorFlow/Keras not installed.")
            model = build_dense_model(input_dim, task_type)
            log_fn(model.summary()) # Print model summary
            history = model.fit(X_train, y_train,
                                epochs=20, # Increased epochs
                                batch_size=64,
                                validation_split=0.1, # Use validation data
                                verbose=0, # Set to 1 or 2 for more verbose training output
                                callbacks=[]) # Add callbacks like EarlyStopping if desired
            log_fn("DL Model training finished.")
            y_pred_dl = model.predict(X_test)
            if task_type == 'classification':
                y_prob = y_pred_dl.flatten()
                y_pred = (y_prob > 0.5).astype(int) # Threshold probabilities
            else: # Regression
                y_prob = None # No probabilities for regression
                y_pred = y_pred_dl.flatten()


        elif model_category == 'rl':
            raise NotImplementedError("Reinforcement Learning training is not implemented yet.")

        else:
            raise ValueError(f"Unsupported model category: {model_category}")

        # 3. Evaluate Model
        log_fn("📊 Evaluating model...")
        if task_type == 'classification':
            report_str = classification_report(y_test, y_pred)
            log_fn(f"Classification Report:\n{report_str}")
            acc = accuracy_score(y_test, y_pred)
            score = auc(roc_curve(y_test, y_prob)[0], roc_curve(y_test, y_prob)[1]) if y_prob is not None else acc # Use AUC if available, else Accuracy
            log_fn(f"Score (AUC/Accuracy): {score:.4f}")
            plot_evaluation(y_test, y_pred, y_prob, model_name_base, task_type)

        elif task_type == 'regression':
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            score = r2 # Use R-squared as the score for regression
            log_fn(f"MAE: {mae:.4f}")
            log_fn(f"MSE: {mse:.4f}")
            log_fn(f"R-squared: {r2:.4f}")
            plot_evaluation(y_test, y_pred, None, model_name_base, task_type)
        feature_log_path = f"{MODEL_DIR}/{model_name_base}_features.txt"
        with open(feature_log_path, "w") as f:
            f.write("\n".join(selected_features or X_train.columns.tolist()))
        log_fn(f"Features used for training saved to {feature_log_path}")

        # 4. Save Model
        log_fn("💾 Saving model...")
        saved = save_model(model, model_name_base, is_dl=(model_category == 'dl'))
        if not saved:
             raise RuntimeError("Failed to save the model.")

        status = 'completed'
        final_message = f"✅ Successfully trained and evaluated model: {model_name_base}"
        log_fn(final_message)


    except FileNotFoundError as e:
        error_message = f"Data loading error: {e}"
        print(f"❌ ERROR: {error_message}")
        traceback.print_exc()
    except ImportError as e:
        error_message = f"Import error: {e}"
        print(f"❌ ERROR: {error_message}")
        traceback.print_exc()
    except NotImplementedError as e:
        error_message = str(e)
        print(f"❌ SKIPPED: {error_message}")
        status = 'skipped' # Special status for not implemented
    except Exception as e:
        error_message = f"An unexpected error occurred during training: {e}"
        print(f"❌ ERROR: {error_message}")
        traceback.print_exc() # Print full traceback for debugging

    finally:
        # 5. Log Outcome (always happens)
        end_time = datetime.now()
        duration = end_time - start_time
        log_fn(f"⏱️ Training duration: {duration}")
        if status != 'skipped': # Don't log skipped runs unless desired
            log_training_event(target, algorithm, model_category.upper(), status, score, error_message)

    # Return success status and message for Streamlit app
    success = (status == 'completed')
    message_out = error_message if not success and status != 'skipped' else final_message if success else "Training skipped."
    return success, message_out

# --- Main execution block (for direct script running) ---
if __name__ == "__main__":
    print("Running direct test of train_pipeline...")

    # Example 1: Classification ML
    train_pipeline(
        target="tamper_detected", # Example target
        task_type="classification",
        model_category="ml",
        algorithm="randomforest",
        tune=False # Set to True to test tuning
    )

    # Example 2: Regression ML
    train_pipeline(
        target="energy_loss_kwh_sum", # Example target
        task_type="regression",
        model_category="ml",
        algorithm="xgboost",
        tune=False
    )

    # Example 3: Classification DL (if TF is available)
    if TF_AVAILABLE:
        train_pipeline(
            target="fault_detected", # Example target
            task_type="classification",
            model_category="dl",
            algorithm="mlp", # 'mlp' is handled by build_dense_model
            tune=False # Tuning not implemented for this simple DL model here
        )
    else:
        print("\nSkipping DL test because TensorFlow/Keras is not available.")

    print("\nDirect tests completed.")

