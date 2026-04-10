import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

st.set_page_config(page_title="Age AI", layout="centered")

st.title("Age Detector AI 🎂")

# Use st.camera_input for a better mobile experience
uploaded_file = st.camera_input("Take a selfie!")

# Fallback: if they prefer to upload a file
if not uploaded_file:
    uploaded_file = st.file_uploader("Or upload a photo", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and show the image immediately so you know it worked
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Got it! Looks good.", use_container_width=True)
    
    # Add a manual button to start the AI (prevents mobile timeouts)
    if st.button("Detect My Age! 🚀"):
        with st.spinner('AI is analyzing... (this takes 5-10 seconds)'):
            try:
                img_array = np.array(img)
                
                # We use a lighter model 'FastMtC' if possible, or just standard
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False, # Prevents crash if face is blurry
                    detector_backend = 'opencv' # Fastest for mobile servers
                )
                
                age = results[0]['age']
                st.balloons()
                st.metric(label="Estimated Age", value=f"{age} years old")
                
            except Exception as e:
                st.error("The AI had a brain freeze. Try a clearer photo in better light!")
                st.info("Technical tip: Make sure your face is clearly visible.")