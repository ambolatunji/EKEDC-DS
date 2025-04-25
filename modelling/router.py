# router.py

def get_routes():
    return {
        "🏠 Home": None,
        "📁 Data Preparation": ("data_preparation_app", "main"),
        "🧠 Train Models": ("train_models_app", "main"),
        "📊 Evaluate Models": ("evaluate_model_app", "main"),
        "🔮 Predict": ("predict_app", "main"),
        "🥇 Compare Models": ("compare_models_app", "main"),
        "🔐 Admin Panel": ("admin_app", "main"),
    }
