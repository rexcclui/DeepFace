import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import os

# Prevent TensorFlow from using too much memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Age AI", layout="centered")
st.title("Age Detector AI 🎂")

# Use a better camera input for mobile
uploaded_file = st.camera_input("Take a selfie!")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    if st.button("Detect My Age! 🚀"):
        with st.spinner('Thinking...'):
            try:
                img_array = np.array(img)
                
                # Speed & Memory Optimization: 
                # We use 'opencv' backend which is the lightest and fastest.
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False,
                    detector_backend = 'opencv' 
                )
                
                age = results[0]['age']
                st.balloons()
                st.success(f"The AI thinks you are **{age}** years old!")
                
            except Exception as e:
                st.error("The app ran out of memory or couldn't find a face. Try again!")