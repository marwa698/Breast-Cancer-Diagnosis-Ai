import streamlit as st

def render_combined_report(tabular_result: dict, image_result: dict):
    """
    دمج نتيجة النظامين في تقرير واحد
    """
    st.markdown("---")
    st.subheader("📊 Combined Diagnostic Report")

    tab_pred = tabular_result['prediction']  # 0 or 1
    img_pred = image_result['prediction']    # benign/malignant/normal

    # تحويل image prediction لـ 0/1
    img_pred_binary = 1 if img_pred == 'malignant' else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔬 Tabular Model")
        color = "#e74c3c" if tab_pred == 1 else "#2ecc71"
        st.markdown(f"""
        <div style="background:{color}20; padding:15px; border-radius:10px; 
                    border-left:4px solid {color}; text-align:center;">
            <h3 style="color:{color};">{tabular_result['label']}</h3>
            <p>Confidence: {tabular_result['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🤝 Agreement")
        if tab_pred == img_pred_binary:
            st.markdown("""
            <div style="background:#2ecc7120; padding:15px; border-radius:10px;
                        border-left:4px solid #2ecc71; text-align:center;">
                <h3 style="color:#2ecc71;">✅ Both Agree</h3>
                <p>High confidence in result</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#e74c3c20; padding:15px; border-radius:10px;
                        border-left:4px solid #e74c3c; text-align:center;">
                <h3 style="color:#e74c3c;">⚠️ Disagreement</h3>
                <p>Please consult a specialist</p>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        st.markdown("#### 🖼️ Image Model")
        color = "#e74c3c" if img_pred == 'malignant' else "#2ecc71"
        st.markdown(f"""
        <div style="background:{color}20; padding:15px; border-radius:10px;
                    border-left:4px solid {color}; text-align:center;">
            <h3 style="color:{color};">{image_result['label']}</h3>
            <p>Confidence: {image_result['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Final recommendation
    st.markdown("---")
    if tab_pred == img_pred_binary:
        if tab_pred == 1:
            st.error("🚨 **Both models indicate MALIGNANT** — Immediate specialist consultation recommended.")
        else:
            st.success("✅ **Both models indicate BENIGN** — Continue routine monitoring.")
    else:
        st.warning("⚠️ **Models disagree** — Results are inconclusive. Please consult a qualified healthcare professional for further evaluation.")