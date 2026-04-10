import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Group Age AI", layout="centered")
st.title("Group Age Detector 📸")

uploaded_file = st.file_uploader("Upload a group photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Resize for stability (Essential for free servers!)
    raw_img = Image.open(uploaded_file).convert('RGB')
    raw_img.thumbnail((600, 600), Image.Resampling.LANCZOS)
    
    st.image(raw_img, width='stretch', caption="Photo processed")
    
    if st.button("Analyze All Faces! 🚀"):
        with st.spinner('Checking every face...'):
            try:
                from deepface import DeepFace
                img_array = np.array(raw_img)
                
                # 2. Run analysis
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False,
                    detector_backend = 'opencv'
                )

                # 3. LOOP THROUGH THE RESULTS
                # DeepFace returns a list of dictionaries if multiple faces are found
                st.write(f"### Found {len(results)} person(s):")
                
                for i, person in enumerate(results):
                    age = person['age']
                    
                    # Create a nice box for each person
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Show the small face crop
                            r = person['region']
                            face_crop = img_array[r['y']:r['y']+r['h'], r['x']:r['x']+r['w']]
                            st.image(face_crop, width='stretch')
                            
                        with col2:
                            st.write(f"**Person {i+1}**")
                            st.metric(label="Age Guess", value=f"{age} yrs")

                st.balloons()
                
                # Cleanup
                del img_array
                gc.collect()
                
            except Exception as e:
                st.error("The AI got overwhelmed. Try a photo with fewer people or clearer faces.")
                gc.collect()