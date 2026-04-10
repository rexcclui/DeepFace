import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# 1. Server Memory Optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Multi-Person Age AI", layout="centered")
st.title("Group Age Detector 📸")

uploaded_file = st.file_uploader("Upload a photo with one or more people", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 2. TRIM RESOLUTION: Shrink to 800px max for stability
    raw_img = Image.open(uploaded_file).convert('RGB')
    raw_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    
    # Updated 2026 Streamlit syntax: width='stretch'
    st.image(raw_img, width='stretch', caption="Photo prepared for AI")
    
    if st.button("Analyze All Faces! 🚀"):
        with st.spinner('Checking faces... (This takes a few seconds)'):
            try:
                # 3. LAZY IMPORT: Keeps the app from crashing on load
                from deepface import DeepFace
                
                img_array = np.array(raw_img)
                
                # 4. RUN ANALYSIS
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False, 
                    detector_backend = 'opencv',
                    align = True
                )

                # 5. THE LOOP: This ensures we show every face found
                st.write(f"### Results: Found {len(results)} face(s)")
                
                for i, face in enumerate(results):
                    age = face['age']
                    region = face['region']
                    
                    # Create a visual "Card" for each person
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Crop the specific face from the photo
                            x, y, w, h = region['x'], region['y'], region['w'], region['h']
                            face_crop = img_array[y:y+h, x:x+w]
                            if face_crop.size > 0:
                                st.image(face_crop, width='stretch')
                            else:
                                st.write("📷")

                        with col2:
                            st.subheader(f"Person {i+1}")
                            st.metric("Estimated Age", f"{age} yrs")

                st.balloons()
                
                # 6. CLEAN UP: Clear RAM immediately
                del img_array
                gc.collect()
                
            except Exception as e:
                st.error("The AI had trouble reading this photo. Try one with clearer faces.")
                gc.collect()

# Sidebar memory tool
if st.sidebar.button("Reset Memory Cache"):
    gc.collect()
    st.success("RAM Cleared!")