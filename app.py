import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# 1. Server stability settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Age AI Pro", layout="centered")

# Hide Streamlit UI elements for a cleaner look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("Group Age Detector 📸")
st.write("Upload a photo with multiple people to see the AI in action.")

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 2. Trim resolution for stability (800px is the sweet spot for memory vs. accuracy)
    raw_img = Image.open(uploaded_file).convert('RGB')
    raw_img.thumbnail((800, 800), Image.Resampling.LANCZOS)
    
    st.image(raw_img, width='stretch', caption="Image ready for analysis")
    
    if st.button("Analyze All Faces! 🚀"):
        with st.spinner('AI is scanning for faces...'):
            try:
                # 3. Lazy Import to keep startup memory low
                from deepface import DeepFace
                
                img_array = np.array(raw_img)
                
                # 4. RUN ANALYSIS 
                # We use 'ssd' because it is much more accurate for multiple faces than 'opencv'
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False, 
                    detector_backend = 'ssd', 
                    align = True
                )

                # 5. THE LOOP: This handles multiple people
                st