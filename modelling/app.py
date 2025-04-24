import streamlit as st
from streamlit_option_menu import option_menu

# ------------------ Page Config ------------------
st.set_page_config(page_title="EKEDC ML Dashboard", layout="wide")

# ------------------ CSS for Custom Styling ------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 40px;
        color: #FF4B4B;
        text-align: center;
        margin-top: 20px;
    }
    .sub-header {
        font-size: 20px;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ App Header ------------------
st.markdown('<p class="main-header">⚡ EKEDC Control Center</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">An intelligent dashboard for data-driven meter analytics and model deployment.</p>', unsafe_allow_html=True)


# ------------------ Sidebar Navigation ------------------
with st.sidebar:
    selected = option_menu(
        menu_title="EKEDC ML Suite",
        options=["🏠 Home", "📁 Data Preparation", "🧠 Train Models", "📊 Evaluate Models", "🔮 Predict", "🥇 Compare Models", "🔐 Admin Panel"],
        icons=["house", "upload", "activity", "bar-chart", "search", "trophy"],
        default_index=0,
        
    )

# ------------------ Dynamic Page Routing ------------------
if selected == "🏠 Home":
    st.success("Welcome to EKEDC's simu Control Center. Use the sidebar to begin.")

elif selected == "📁 Data Preparation":
    import data_preparation_app
    data_preparation_app.main()

elif selected == "🧠 Train Models":
    import train_models_app
    train_models_app.main()

elif selected == "📊 Evaluate Models":
    import evaluate_model_app
    evaluate_model_app.main()

elif selected == "🥇 Compare Models":
    import compare_models_app
    compare_models_app.main()

elif selected == "🔮 Predict":
    import predict_app
    predict_app.main()

elif selected == "🔐 Admin Panel":
    import admin_app
    admin_app.main()
