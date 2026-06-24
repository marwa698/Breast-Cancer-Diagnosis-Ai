import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="Breast Cancer Diagnosis AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def show_disclaimer():
    st.warning("""
    ⚠️ **Medical Disclaimer**
    
    This tool is designed for **decision support only** and is **not a substitute** 
    for professional medical diagnosis. All results must be reviewed by a qualified 
    healthcare professional. The developers are not responsible for any clinical 
    decisions made based on this tool.
    """)

def show_metric_card(title: str, value: str, delta: str = None, color: str = "#1f77b4"):
    st.markdown(f"""
    <div style="background-color: {color}15; padding: 15px; border-radius: 10px; 
                border-left: 4px solid {color}; margin: 5px 0;">
        <h4 style="margin:0; color: {color};">{title}</h4>
        <h2 style="margin:5px 0;">{value}</h2>
        {f'<p style="margin:0; color: gray;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def confidence_color(confidence: float) -> str:
    if confidence >= 90:
        return "#2ecc71"
    elif confidence >= 70:
        return "#f39c12"
    else:
        return "#e74c3c"