import streamlit as st
import base64
from app.utils import show_metric_card, confidence_color

def render_tabular_results(result: dict, shap_img: str = None):
    """
    عرض نتيجة الـ tabular model
    """
    st.subheader("🔬 Tabular Analysis Results")

    col1, col2, col3 = st.columns(3)
    with col1:
        color = "#e74c3c" if result['prediction'] == 1 else "#2ecc71"
        show_metric_card("Diagnosis", result['label'], color=color)
    with col2:
        color = confidence_color(result['confidence'])
        show_metric_card("Confidence", f"{result['confidence']}%", color=color)
    with col3:
        show_metric_card("Benign Prob", f"{result['prob_benign']}%", color="#2ecc71")

    # Probability bars
    st.markdown("#### Probability Distribution")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🟢 Benign", f"{result['prob_benign']}%")
        st.progress(result['prob_benign'] / 100)
    with col2:
        st.metric("🔴 Malignant", f"{result['prob_malignant']}%")
        st.progress(result['prob_malignant'] / 100)

    # SHAP plot
    if shap_img:
        st.markdown("#### 🧠 SHAP Feature Explanation")
        st.image(
            base64.b64decode(shap_img),
            caption="Features most influencing this prediction",
            use_container_width=True
        )

def render_image_results(result: dict, gradcam_img: str = None):
    """
    عرض نتيجة الـ CNN model
    """
    st.subheader("🖼️ Image Analysis Results")

    col1, col2 = st.columns(2)
    with col1:
        color = "#e74c3c" if result['prediction'] == 'malignant' else "#2ecc71"
        show_metric_card("Diagnosis", result['label'], color=color)
    with col2:
        color = confidence_color(result['confidence'])
        show_metric_card("Confidence", f"{result['confidence']}%", color=color)

    # Probabilities
    st.markdown("#### Class Probabilities")
    for class_name, prob in result['probabilities'].items():
        emoji = {"benign": "🟢", "malignant": "🔴", "normal": "✅"}.get(class_name, "")
        st.write(f"{emoji} **{class_name.title()}**: {prob}%")
        st.progress(prob / 100)

    # Grad-CAM
    if gradcam_img:
        st.markdown("#### 🎯 Grad-CAM Visualization")
        st.image(
            base64.b64decode(gradcam_img),
            caption="Areas the model focused on",
            use_container_width=True
        )