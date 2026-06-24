import streamlit as st
from PIL import Image

def render_image_upload():
    """
    واجهة رفع صورة الـ ultrasound
    """
    st.subheader("🖼️ Upload Ultrasound Image")

    uploaded_file = st.file_uploader(
        "Upload a breast ultrasound image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.info(f"""
            **File:** {uploaded_file.name}  
            **Size:** {uploaded_file.size / 1024:.1f} KB  
            **Dimensions:** {img.size[0]} x {img.size[1]} px  
            **Mode:** {img.mode}
            """)
        return uploaded_file

    else:
        st.info("👆 Please upload a breast ultrasound image to get started.")
        return None