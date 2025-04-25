import streamlit as st
from streamlit_option_menu import option_menu
import importlib
#from router import get_routes
import os

# ------------------ Page Config ------------------
st.set_page_config(page_title="EKEDC ML Dashboard", layout="wide")

# ------------------ CSS Styling ------------------
st.markdown("""
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
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown('<p class="main-header">⚡ EKEDC Control Center</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">An intelligent dashboard for data-driven meter analytics and model deployment.</p>', unsafe_allow_html=True)

# ------------------ Define Routes ------------------
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

# ------------------ Sidebar Navigation ------------------
with st.sidebar:
    selected = option_menu(
        menu_title="EKEDC ML Suite",
        options=list(get_routes().keys()),
        icons=["house", "upload", "activity", "bar-chart", "search", "trophy", "lock"],
        default_index=0
    )

# ------------------ Dynamic Routing ------------------
route = get_routes().get(selected)

if route is None:
    st.success("Welcome to EKEDC's Control Center. Use the sidebar to begin.")
else:
    module_name, func_name = route
    try:
        module = importlib.import_module(module_name)
        getattr(module, func_name)()
    except Exception as e:
        st.error(f"⚠️ Failed to load `{selected}`: {e}")
