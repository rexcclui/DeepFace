import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# 1. Server Stability Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="How old am I?", layout="centered")

# Custom CSS for a clean, professional app look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border: 1px solid #dcdfe6; }
        .stButton>button { width: 100%; border-radius: 20px; height: 3em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("How old am I? 🎂")

# 2. TABS FOR INPUT
tab1, tab2 = st.tabs(["📸 Take a Selfie", "📁 Upload Image"])

source_img = None

with tab1:
    st.write("### Quick Selfie")
    cam_img = st.camera_input("Snap a photo to analyze automatically")
    if cam_img:
        source_img = cam_img

with tab2:
    st.write("### Select a Photo")
    st.info("💡 Hint: Use the 'Search' or 'Recent' folder on your phone to find files faster.")
    file_img = st.file_uploader("Upload from your device", type=["jpg", "jpeg", "png"])
    if file_img:
        source_img = file_img

# 3. ANALYSIS LOGIC
if source_img:
    # Load and optimize image size for the AI
    raw_img = Image.open(source_img).convert('RGB')
    raw_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    
    st.image(raw_img, width='stretch', caption="Image received!")
    
    # Auto-run for camera, manual button for file upload
    run_analysis = False
    if cam_img:
        run_analysis = True
    else:
        run_analysis = st.button("Analyze Uploaded Photo! 🚀", type="primary")

    if run_analysis:
        with st.spinner('AI is analyzing faces...'):
            try:
                # Lazy import to save startup RAM
                from deepface import DeepFace
                img_array = np.array(raw_img)
                
                # 4. MULTI-DETECTOR STRATEGY
                # We try 'fastmtcnn' first (better for groups), then fallback to 'opencv'
                try:
                    results = DeepFace.analyze(
                        img_path = img_array, 
                        actions = ['age'],
                        enforce_detection = False, 
                        detector_backend = 'fastmtcnn', 
                        align = True
                    )
                except:
                    results = DeepFace.analyze(
                        img_path = img_array, 
                        actions = ['