import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# Keep the server quiet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Slim Age AI", layout="centered")
st.title("Age Detector AI 🎂")

# SIMPLE UI (No tabs or heavy elements for now)
uploaded_file = st.file_uploader("Upload a photo to begin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_container_width=True)
    
    if st.button("Check Age! 🚀"):
        with st.spinner('Waking up AI...'):
            try:
                # WE IMPORT HERE ONLY (Saves RAM on startup)
                from deepface import DeepFace
                
                img_array = np.array(img)
                
                # Analyze with the absolutely lightest model settings
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False,
                    detector_backend = 'opencv'
                )
                
                # Show results
                age = results[0]['age']
                st.metric("Estimated Age", f"{age} years old")
                
                # Clean up immediately
                del DeepFace
                gc.collect()
                
            except Exception as e:
                st.error("Server is too busy or image is too big.")