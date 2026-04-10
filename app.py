import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import os
import gc # Garbage Collector to clear RAM

# 1. Force the server to be quiet and save memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

st.set_page_config(page_title="Safe Age AI", layout="centered")
st.title("Group Age Detector 📸")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_container_width=True)
    
    if st.button("Check All Ages! 🚀"):
        with st.spinner('Analyzing...'):
            try:
                img_array = np.array(img)
                
                # 2. Run analysis with the LIGHTEST possible settings
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False,
                    detector_backend = 'opencv' # Fastest and uses least RAM
                )

                # Loop through faces
                for i, face in enumerate(results):
                    age = face['age']
                    region = face['region']
                    
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            # Crop only if face is a decent size
                            if region['w'] > 20:
                                crop = img_array[region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
                                st.image(crop, use_container_width=True)
                        with col2:
                            st.metric(f"Person {i+1}", f"{age} yrs")
                
                # 3. THE MAGIC FIX: Clear memory manually after results show
                del img_array
                gc.collect() 
                
            except Exception as e:
                st.error("Memory full or face not found. Try a smaller image file!")

# Button to manually clear everything if it feels slow
if st.sidebar.button("Clear Memory Cache"):
    gc.collect()
    st.success("RAM Cleared!")