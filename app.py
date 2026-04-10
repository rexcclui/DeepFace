import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import os

# Server memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Age AI", layout="centered")
st.title("Age Detector AI 🎂")

# --- INPUT SECTION ---
st.write("### Choose your method:")
tab1, tab2 = st.tabs(["📸 Take Selfie", "📁 Upload Photo"])

source_img = None

with tab1:
    cam_file = st.camera_input("Smile for the AI!")
    if cam_file:
        source_img = cam_file

with tab2:
    uploaded_file = st.file_uploader("Select a photo from your gallery", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        source_img = uploaded_file

# --- PROCESSING SECTION ---
if source_img is not None:
    # Convert image for AI processing
    img = Image.open(source_img).convert('RGB')
    
    # Show a small preview so the user knows it's loaded
    st.image(img, caption="Image Loaded!", width=300)
    
    if st.button("Detect My Age! 🚀"):
        with st.spinner('Analyzing facial features...'):
            try:
                img_array = np.array(img)
                
                # DeepFace Analysis
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False,
                    detector_backend = 'opencv'
                )
                
                age = results[0]['age']
                
                # Big Result Display
                st.balloons()
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid #ff4b4b; border-radius: 10px;">
                    <h2 style="margin: 0;">Estimated Age</h2>
                    <h1 style="font-size: 72px; color: #ff4b4b; margin: 10px 0;">{age}</h1>
                    <p style="color: gray;">Years Old</p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error("The AI had a bit of trouble. Try a clearer photo with better lighting!")