import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# Stability Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="How old am I?", layout="centered")

# Custom UI cleanup
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("How old am I? 🎂")

uploaded_file = st.file_uploader("Upload a photo with one or more people", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Load and Resize (1000px allows AI to see background faces)
    raw_img = Image.open(uploaded_file).convert('RGB')
    raw_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    
    st.image(raw_img, width='stretch', caption="Ready for Scanning")
    
    if st.button("Analyze All Ages! 🚀", type="primary"):
        with st.spinner('AI is scanning for faces...'):
            try:
                from deepface import DeepFace
                img_array = np.array(raw_img)
                
                # 2. Use YOLOv8 - The best for multi-face detection in 2026
                # It finds faces even if they are small or slightly turned.
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False, 
                    detector_backend = 'yolov8', 
                    align = True
                )

                # 3. Handle Results
                if not results or len(results) == 0:
                    st.warning("No faces detected. Try a clearer photo!")
                else:
                    st.divider()
                    # Sort faces from left to right so the list matches the photo
                    results = sorted(results, key=lambda x: x['region']['x'])
                    st.subheader(f"Found {len(results)} person(s)")
                    
                    for i, face in enumerate(results):
                        age = face['age']
                        r = face['region']
                        
                        with st.container(border=True):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                # Crop with a tiny bit of breathing room
                                pad = 15
                                y1, y2 = max(0, r['y']-pad), min(img_array.shape[0], r['y']+r['h']+pad)
                                x1, x2 = max(0, r['x']-pad), min(img_array.shape[1], r['x']+r['w']+pad)
                                face_crop = img_array[y1:y2, x1:x2]
                                
                                if face_crop.size > 0:
                                    st.image(face_crop, width='stretch')
                            
                            with col2:
                                st.markdown(f"### Person {i+1}")
                                st.metric("Age Guess", f"{age} years")

                st.balloons()
                del img_array
                gc.collect()
                
            except Exception as e:
                st.error("The AI hit a snag.")
                # Show the real error so we can fix it if YOLOv8 fails
                st.info(f"Technical detail: {str(e)}")
                gc.collect()

# Reset button
if st.sidebar.button("Hard Reset"):
    gc.collect()
    st.rerun()