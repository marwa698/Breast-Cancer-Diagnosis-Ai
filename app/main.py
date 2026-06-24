import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils import set_page_config, show_disclaimer
from app.components.tabular_form_ui import render_tabular_form
from app.components.image_upload_ui import render_image_upload
from app.components.results_panel import render_tabular_results, render_image_results
from app.components.combined_report import render_combined_report
from app.tabular.predictor import predict
from app.tabular.explainability import get_shap_explanation, plot_shap_bar
from app.imaging.image_predictor import predict_image
from app.imaging.gradcam import generate_gradcam

set_page_config()

# Header
st.title("🏥 Breast Cancer Diagnosis AI")
st.markdown("*AI-powered decision support system — Not a substitute for medical advice*")
show_disclaimer()

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔬 Tabular Analysis", 
    "🖼️ Image Analysis", 
    "📊 Combined Report"
])

# ======== Tab 1: Tabular ========
with tab1:
    input_values = render_tabular_form()
    
    if st.button("🔍 Analyze Clinical Data", type="primary", key="tabular_btn"):
        with st.spinner("Analyzing..."):
            try:
                result = predict(input_values)
                try:
                    shap_features = get_shap_explanation(input_values)
                    shap_img = plot_shap_bar(shap_features)
                except:
                    shap_img = None
                st.session_state['tabular_result'] = result
                render_tabular_results(result, shap_img)
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)

# ======== Tab 2: Image ========
with tab2:
    uploaded_file = render_image_upload()
    
    if uploaded_file and st.button("🔍 Analyze Image", type="primary", key="image_btn"):
        with st.spinner("Analyzing image..."):
            try:
                result = predict_image(uploaded_file)
                uploaded_file.seek(0)
                gradcam_img, _ = generate_gradcam(uploaded_file)
                
                st.session_state['image_result'] = result
                render_image_results(result, gradcam_img)
            except Exception as e:
                st.error(f"Error: {e}")

# ======== Tab 3: Combined ========
with tab3:
    if 'tabular_result' in st.session_state and 'image_result' in st.session_state:
        render_combined_report(
            st.session_state['tabular_result'],
            st.session_state['image_result']
        )
    else:
        st.info("👆 Please run both Tabular Analysis and Image Analysis first.")