import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# 1. Block heavy logs to save tiny bits of RAM
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Lite Age AI", layout="centered")
st.title("Age Detector AI 🎂")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and immediately resize the image to be smaller
    # Large photos from modern phones are often the cause of the crash
    raw_img = Image.open(uploaded_file).convert('RGB')
    raw_img.thumbnail((800, 800)) # Resize to max 800px to save RAM
    
    st.image(raw_img, width='stretch') # Fixed the 2026 warning
    
    if st.button("Check Age! 🚀"):
        with st.spinner('Analyzing... (this uses a lot of RAM)'):
            try:
                # 2. Lazy Import
                from deepface import DeepFace
                
                img_array = np.array(raw_img)
                
                # 3. Use the absolute lightest settings
                # 'opencv' is the only one that reliably fits on free servers
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False,
                    detector_backend = 'opencv' 
                )
                
                age = results[0]['age']
                st.balloons()
                st.metric("Estimated Age", f"{age} years old")
                
                # 4. Immediate Cleanup
                del img_array
                gc.collect()
                
            except Exception as e:
                st.error("The server ran out of memory. Try a smaller or cropped photo!")
                gc.collect()

# Add a "Reset" button in the sidebar to clear stuck memory
if st.sidebar.button("Clear App Cache"):
    st.cache_data.clear()
    gc.collect()
    st.success("Memory cleared!")