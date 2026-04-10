import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# Server settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="How old am I?", layout="centered")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("How old am I? 🎂")
st.write("Upload a group photo or a selfie!")

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and show preview
        display_img = Image.open(uploaded_file).convert('RGB')
        # We use a larger size (1000px) so the AI can see smaller faces in the background
        display_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        
        st.image(display_img, width='stretch', caption="Photo Loaded")
        
        if st.button("Analyze All Ages! 🚀", type="primary"):
            with st.spinner('Scanning every face...'):
                from deepface import DeepFace
                
                img_array = np.array(display_img)
                
                # We switch to 'mediapipe'. It is the best balance of speed and 
                # its ability to find multiple faces without crashing the RAM.
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False, 
                    detector_backend = 'mediapipe', 
                    align = True
                )

                st.divider()
                # Sort results by their position (left to right) so the list makes sense
                results = sorted(results, key=lambda x: x['region']['x'])
                
                st.subheader(f"Found {len(results)} person(s)")
                
                for i, face in enumerate(results):
                    age = face['age']
                    r = face['region']
                    
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            # Extract face with a small margin (padding) so it's not too tight
                            pad = 20
                            y1, y2 = max(0, r['y']-pad), min(img_array.shape[0], r['y']+r['h']+pad)
                            x1, x2 = max(0, r['x']-pad), min(img_array.shape[1], r['x']+r['w']+pad)
                            
                            face_crop = img_array[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                st.image(face_crop, width='stretch')
                        
                        with col2:
                            st.write(f"**Person {i+1}**")
                            st.metric("Estimated Age", f"{age} yrs")

                st.balloons()
                del img_array
                gc.collect()

    except Exception as e:
        st.error("The AI is having trouble. Make sure faces are clear and not too far away.")
        gc.collect()

if st.sidebar.button("Clear Memory"):
    gc.collect()
    st.rerun()