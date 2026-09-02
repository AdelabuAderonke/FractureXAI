import streamlit as st

st.set_page_config(page_title="FractureXAI", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #ffffff;
    }
    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #2a2f3a;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 20px;
        font-weight: 600;
        margin: 0;
        color: #e6e6e6;
    }
    .header-subtitle {
        font-size: 13px;
        color: #8b929e;
        margin: 0;
    }
    .research-badge {
        border: 1px solid #d4534b;
        color: #f0a7a3;
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 13px;
    }
    .info-card {
        background-color: #161b22;
        border: 1px solid #2a2f3a;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 16px;
    }
    .info-card-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #e6e6e6;
    }
    .info-card-text {
        font-size: 13px;
        color: #8b929e;
        line-height: 1.6;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #161b22;
        border: 2px dashed #2a2f3a;
        border-radius: 12px;
        min-height: 350px;
    }
     /* Center and color the "Drop an X-ray image..." label text */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        width: 100%;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #ffffff !important;
    }

    /* Make the browse button text black */
    [data-testid="stBaseButton-secondary"] {
        color: #000000 !important;
    }
    [data-testid="stBaseButton-secondary"] span {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="header-row">
    <div>
        <p class="header-title">FractureXAI</p>
        <p class="header-subtitle">AI bone fracture detection with Grad-CAM explainability</p>
    </div>
    <div class="research-badge"> Research demo — not for clinical use</div>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Drop an X-ray image here, or click to browse",
        type=["png", "jpg", "jpeg"],
        label_visibility="visible"
    )
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <p class="info-card-title">Findings</p>
        <p class="info-card-text">Upload an image and run the analysis to see detected fractures and confidence scores.</p>
    </div>
    <div class="info-card">
        <p class="info-card-title">AI explanation (XAI)</p>
        <p class="info-card-text">After analysis, an LLM-generated explanation will describe where the model looked and why it reached its conclusion.</p>
    </div>
    """, unsafe_allow_html=True)