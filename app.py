import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np

# Page Configuration for Mobile
st.set_page_config(page_title="Age AI", page_icon="🎂")

st.title("Age Detector AI 🎂")
st.write("Take a selfie or upload a photo to check your age!")

# File uploader (Works with mobile camera)
uploaded_file = st.file_uploader("Capture or Upload", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Analyzing this face...', use_container_width=True)
    
    with st.spinner('AI is thinking...'):
        try:
            # Convert to array for DeepFace
            img_array = np.array(img.convert('RGB'))
            
            # Analyze Age
            # We use 'enforce_detection=False' so the app doesn't crash if it misses a face
            results = DeepFace.analyze(img_array, actions=['age'], enforce_detection=False)
            
            age = results[0]['age']
            
            st.balloons()
            st.success(f"Estimated Age: **{age} years old**")
            
        except Exception as e:
            st.error("Oops! The AI couldn't find a face. Try a clearer photo.")