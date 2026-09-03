import streamlit as st
from PIL import Image

from config import (
    APP_TITLE, APP_SUBTITLE, RESEARCH_DISCLAIMER, SCOPE_DISCLAIMER,
    MODEL_A_PATH, PROMPT_A, FRACTURE_CLASSES_A,
    PROMPT_B, FRACTURE_CLASSES_B, DETECTION_MODELS,
)
from model_utils import load_model, generate_response, postprocess

st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #ffffff; }
    .header-row { display: flex; justify-content: space-between; align-items: center;
        padding-bottom: 1.5rem; border-bottom: 1px solid #2a2f3a; margin-bottom: 2rem; }
    .header-title { font-size: 20px; font-weight: 600; margin: 0; color: #e6e6e6; }
    .header-subtitle { font-size: 13px; color: #8b929e; margin: 0; }
    .research-badge { border: 1px solid #d4534b; color: #f0a7a3; border-radius: 6px;
        padding: 6px 14px; font-size: 13px; }
    .info-card { background-color: #161b22; border: 1px solid #2a2f3a; border-radius: 10px;
        padding: 16px 18px; margin-bottom: 16px; }
    .info-card-title { font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #e6e6e6; }
    .info-card-text { font-size: 13px; color: #8b929e; line-height: 1.6; }
    [data-testid="stFileUploaderDropzone"] { background-color: #161b22; border: 2px dashed #2a2f3a;
        border-radius: 12px; min-height: 350px; }
    [data-testid="stFileUploaderDropzoneInstructions"] { display: flex; flex-direction: column;
        align-items: center; justify-content: center; text-align: center; width: 100%; }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small { color: #ffffff !important; }
    [data-testid="stBaseButton-secondary"] { color: #000000 !important; }
    [data-testid="stBaseButton-secondary"] span { color: #000000 !important;}

    .stRadio > label > div > p,
    div[data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
    }

    [data-testid="stFileUploaderFile"] {
        background-color: #ffffff;
    }
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] small {
        color: #000000 !important;
    }

    p, span, label, .stMarkdown {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-row">
    <div>
        <p class="header-title">{APP_TITLE}</p>
        <p class="header-subtitle">{APP_SUBTITLE}</p>
    </div>
    <div class="research-badge">{RESEARCH_DISCLAIMER}</div>
</div>
""", unsafe_allow_html=True)

# Mode selection
mode = st.radio(
    "What would you like to do?",
    ["Detect fracture (pediatric wrist X-ray)", "Classify fracture type (assumes fracture present)"],
    horizontal=True,
)

if mode.startswith("Detect"):
    detection_choice = st.radio("Detection model", list(DETECTION_MODELS.keys()), horizontal=True)
    st.info(SCOPE_DISCLAIMER)
else:
    st.warning(
        "This mode assumes a fracture is already present and identifies its type only. "
        "It cannot detect whether a fracture exists — use 'Detect fracture' first if unsure."
    )

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Drop an X-ray image here, or click to browse",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        if st.button("Run analysis"):
            if mode.startswith("Detect"):
                model_path = DETECTION_MODELS[detection_choice]
                prompt, classes = PROMPT_B, FRACTURE_CLASSES_B
            else:
                model_path = MODEL_A_PATH
                prompt, classes = PROMPT_A, FRACTURE_CLASSES_A

            model, processor = load_model(model_path)

            with st.spinner("Analyzing..."):
                raw_text = generate_response(image, prompt, model, processor)
                idx, label = postprocess(raw_text, classes)

            if idx is None:
                st.warning(f"Could not parse a clear result. Raw output: {label}")
            elif "No" in label:
                st.success(f"Result: {label}")
            else:
                st.error(f"Result: {label}")

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