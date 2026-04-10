import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# 1. Server stability settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set page title and favicon
st.set_page_config(page_title="How old am I?", layout="centered")

# CSS to hide the developer menu for a cleaner look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# THE NEW TITLE
st.title("How old am I? 🎂")
st.write("Upload a photo to see if the AI can guess your age!")

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # 2. FAST PREVIEW: Load and shrink for display only
        # This ensures the Analyze button shows up immediately
        display_img = Image.open(uploaded_file).convert('RGB')
        display_img.thumbnail((600, 600), Image.Resampling.LANCZOS)
        
        st.image(display_img, width='stretch', caption="Photo Loaded")
        
        # 3. THE ANALYZE BUTTON
        if st.button("Analyze My Age! 🚀", type="primary"):
            with st.spinner('AI is thinking...'):
                from deepface import DeepFace
                
                # Convert to array only when needed
                img_array = np.array(display_img)
                
                # Use SSD detector for better multi-face accuracy
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False, 
                    detector_backend = 'ssd', 
                    align = True
                )

                st.divider()
                st.subheader(f"Results: Found {len(results)} person(s)")
                
                # 4. LOOP THROUGH ALL DETECTED FACES
                for i, face in enumerate(results):
                    age = face['age']
                    region = face['region']
                    
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Crop the face for display
                            x, y, w, h = region['x'], region['y'], region['w'], region['h']
                            face_crop = img_array[y:y+h, x:x+w]
                            if face_crop.size > 0:
                                st.image(face_crop, width='stretch')
                        
                        with col2:
                            st.markdown(f"### Person {i+1}")
                            st.metric("Estimated Age", f"{age} yrs")

                st.balloons()
                
                # 5. CLEANUP
                del img_array
                gc.collect()

    except Exception as e:
        st.error("The AI had a moment of confusion. Please try a clearer photo!")
        gc.collect()

# Sidebar memory tool
if st.sidebar.button("Hard Reset RAM"):
    gc.collect()
    st.rerun()