
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Deepfake AI Detector",
    page_icon="🛡️",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_model_finetuned.keras"

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 5%, #29205a 0%, transparent 35%),
        radial-gradient(ellipse at 95% 30%, #182e50 0%, transparent 35%),
        #0b1020;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, p, label, span {
    font-family: 'Inter', sans-serif;
}

.hero {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 42px 20px 38px;
    border-radius: 25px;
    background: linear-gradient(115deg, #29205c, #41318c 55%, #174b68);
    border: 1px solid rgba(174, 157, 255, 0.25);
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
}

.hero:after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    border-radius: 50%;
    background: rgba(98, 220, 255, 0.10);
    top: -130px;
    right: -50px;
}

.hero-icon {
    font-size: 43px;
    margin-bottom: 10px;
}

.hero-tag {
    color: #a9eaff;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero h1 {
    color: #ffffff;
    font-size: 40px;
    font-weight: 900;
    letter-spacing: -1.5px;
    margin: 12px 0;
}

.hero p {
    color: #d4d5f4;
    font-size: 15px;
    margin: 0;
}

.section-title {
    color: #ffffff;
    font-size: 19px;
    font-weight: 800;
    margin: 15px 0;
}

.section-number {
    color: #8b7cff;
    font-weight: 900;
    margin-right: 8px;
}

.upload-hint {
    color: #9ca9c7;
    font-size: 13px;
    margin: 0 0 12px 2px;
}

/* Upload area */

[data-testid="stFileUploader"] {
    background: #171e35;
    border: 1.5px dashed #7867d9;
    border-radius: 18px;
    padding: 17px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #e3e7f5 !important;
}

[data-testid="stFileUploader"] button {
    background: #29234f !important;
    color: #ffffff !important;
    border: 1px solid #7867d9 !important;
    border-radius: 9px !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #45358b !important;
}

/* Analyze button */

div.stButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 13px;
    background: linear-gradient(90deg, #7457f5, #4c9df4) !important;
    color: #ffffff !important;
    border: 1px solid #8d7bff !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    box-shadow: 0 7px 22px rgba(104, 85, 240, 0.25);
    transition: all 0.2s ease;
}

div.stButton > button p,
div.stButton > button span {
    color: #ffffff !important;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #6243e8, #3286e8) !important;
    border-color: #b3a6ff !important;
    transform: translateY(-2px);
}

div.stButton > button:disabled {
    background: #34364f !important;
    border-color: #454966 !important;
    color: #a0a5bd !important;
    box-shadow: none;
}

/* Uploaded image */

div[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid #45466f;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

/* Result cards */

.result-card {
    padding: 27px 20px;
    text-align: center;
    border-radius: 20px;
    margin-top: 3px;
}

.real-card {
    background: linear-gradient(135deg, #103e39, #155b4c);
    border: 1px solid #36c99a;
}

.fake-card {
    background: linear-gradient(135deg, #481f3b, #64263e);
    border: 1px solid #f16d91;
}

.result-symbol {
    font-size: 42px;
    font-weight: 900;
    margin-bottom: 5px;
    color: white;
}

.result-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 3px;
    color: #d2d9e9;
}

.result-value {
    font-size: 43px;
    font-weight: 900;
    color: #ffffff;
    margin: 8px 0;
}

.result-description {
    color: #e0e7ef;
    font-size: 13px;
}

.confidence-card {
    background: #171e35;
    border: 1px solid #353d60;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-top: 15px;
}

.confidence-label {
    color: #a8b4d0;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.confidence-value {
    color: #ffffff;
    font-size: 29px;
    font-weight: 900;
    margin-top: 7px;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #7658f5, #42c6e8);
}

div[data-testid="stCaptionContainer"] p {
    color: #9ca9c7;
}

/* Waiting state - deliberately no fixed-height box */

.waiting {
    text-align: center;
    padding: 45px 15px;
    border-radius: 20px;
    background: linear-gradient(145deg, #171e35, #202044);
    border: 1px solid #353d60;
}

.waiting-icon {
    font-size: 42px;
}

.waiting h3 {
    color: #ffffff;
    font-size: 20px;
    margin: 12px 0 6px;
}

.waiting p {
    color: #a8b4d0;
    font-size: 13px;
}

/* Information cards */

.info-card {
    background: linear-gradient(145deg, #171e35, #202044);
    border: 1px solid #353d60;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
}

.info-icon {
    font-size: 23px;
    margin-bottom: 9px;
}

.info-card h4 {
    color: #a99aff;
    font-size: 14px;
    margin: 0 0 7px;
}

.info-card p {
    color: #b6c0d7;
    font-size: 12px;
    margin: 0;
}

.footer {
    text-align: center;
    color: #7885a4;
    font-size: 12px;
    padding-top: 35px;
}

hr {
    border-color: #303956;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------- PREPROCESS IMAGE ----------------

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((160, 160))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ---------------- HERO HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="hero-icon">🛡️</div>
    <div class="hero-tag">INTELLIGENT IMAGE AUTHENTICITY ANALYSIS</div>
    <h1>Deepfake Image Detector</h1>
    <p>Uncover AI-generated faces with the power of deep learning.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------- MAIN CONTENT ----------------

left, right = st.columns([1, 1], gap="large")


with left:

    st.markdown("""
    <div class="section-title">
        <span class="section-number">01</span> Upload Image
    </div>
    <p class="upload-hint">Select a facial image for authenticity analysis.</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Your uploaded image",
            use_container_width=True
        )

    else:
        st.markdown("""
        <p style="color:#7885a4; font-size:13px; margin-top:12px;">
            ✦ Your image preview will appear here after upload.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    analyze = st.button(
        "✦  ANALYZE IMAGE",
        disabled=uploaded_file is None
    )


with right:

    st.markdown("""
    <div class="section-title">
        <span class="section-number">02</span> Detection Result
    </div>
    <p class="upload-hint">Your AI analysis and confidence score.</p>
    """, unsafe_allow_html=True)

    if analyze and image is not None:

        try:
            with st.spinner("Running deep learning analysis..."):

                model = load_model()
                processed_image = preprocess_image(image)

                prediction = float(
                    model.predict(processed_image, verbose=0)[0][0]
                )

                if prediction >= 0.5:
                    result = "REAL"
                    confidence = prediction * 100
                    card_class = "real-card"
                    symbol = "✓"
                    description = "The model classifies this image as a real photograph."

                else:
                    result = "FAKE"
                    confidence = (1 - prediction) * 100
                    card_class = "fake-card"
                    symbol = "!"
                    description = "The model classifies this image as an AI-generated face."

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-symbol">{symbol}</div>
                <div class="result-label">DETECTION COMPLETE</div>
                <div class="result-value">{result}</div>
                <div class="result-description">{description}</div>
            </div>

            <div class="confidence-card">
                <div class="confidence-label">MODEL CONFIDENCE</div>
                <div class="confidence-value">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(confidence / 100)

            st.caption(
                "Confidence is the model's probability for its predicted "
                "class, not a guarantee of authenticity."
            )

        except Exception as e:
            st.error(f"Unable to analyze the image: {e}")

    else:

        st.markdown("""
        <div class="waiting">
            <div class="waiting-icon">🔍</div>
            <h3>Awaiting Analysis</h3>
            <p>Upload an image and select Analyze Image<br>
            to reveal the detection result.</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- ABOUT THE SYSTEM ----------------

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">
    <span class="section-number">03</span> About the Technology
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🧠</div>
        <h4>MobileNetV2</h4>
        <p>Transfer learning with a fine-tuned deep learning architecture.</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🎯</div>
        <h4>Binary Classification</h4>
        <p>Classifies facial images into REAL and FAKE categories.</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">📈</div>
        <h4>85.68% Test Accuracy</h4>
        <p>Measured on the held-out Kaggle 140K test dataset.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    AI-BASED DEEPFAKE IMAGE DETECTION SYSTEM<br><br>
    Built with Python · TensorFlow · MobileNetV2 · Streamlit
</div>
""", unsafe_allow_html=True)
