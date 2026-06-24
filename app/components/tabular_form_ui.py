import streamlit as st
import json

def load_feature_names(path: str = "models/feature_names.json"):
    with open(path, 'r') as f:
        return json.load(f)

def render_tabular_form():
    """
    فورم إدخال البيانات الرقمية
    """
    st.subheader("📋 Enter Clinical Measurements")
    
    feature_names = load_feature_names()
    
    # تقسيم الـ features لـ 3 groups
    mean_features = [f for f in feature_names if '_mean' in f]
    se_features = [f for f in feature_names if '_se' in f]
    worst_features = [f for f in feature_names if '_worst' in f]

    input_values = {}

    with st.expander("📊 Mean Values", expanded=True):
        cols = st.columns(3)
        for i, feature in enumerate(mean_features):
            with cols[i % 3]:
                input_values[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=0.0,
                    value=0.0,
                    format="%.4f",
                    key=f"mean_{feature}"
                )

    with st.expander("📏 Standard Error Values"):
        cols = st.columns(3)
        for i, feature in enumerate(se_features):
            with cols[i % 3]:
                input_values[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=0.0,
                    value=0.0,
                    format="%.4f",
                    key=f"se_{feature}"
                )

    with st.expander("⚠️ Worst Values"):
        cols = st.columns(3)
        for i, feature in enumerate(worst_features):
            with cols[i % 3]:
                input_values[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=0.0,
                    value=0.0,
                    format="%.4f",
                    key=f"worst_{feature}"
                )

    return input_values