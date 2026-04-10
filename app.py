import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Multi-Face AI", layout="centered")
st.title("Face Finder AI 🎂📸")

st.write("### Choose a photo or take a group selfie")
tab1, tab2 = st.tabs(["📸 Take Selfie", "📁 Upload Photo"])

source_img = None
with tab1:
    cam_file = st.camera_input("Gather everyone!")
    if cam_file: source_img = cam_file

with tab2:
    uploaded_file = st.file_uploader("Select Photo", type=["jpg", "jpeg", "png"])
    if uploaded_file: source_img = uploaded_file

if source_img is not None:
    # Convert and show the FULL image immediately
    img = Image.open(source_img).convert('RGB')
    st.image(img, caption="Analyze this photo?", use_container_width=True)
    
    if st.button("Check All Ages! 🚀"):
        with st.spinner('Counting faces and checking ages...'):
            try:
                img_array = np.array(img)
                
                # Analyze faces (enforce_detection=False prevents crash, but means we might get empty results)
                results = DeepFace.analyze(
                    img_path = img_array, 
                    actions = ['age'],
                    enforce_detection = False,
                    detector_backend = 'opencv'
                )
                
                # Clean out any empty results if face was too blurry
                results = [r for r in results if r.get('region') is not None]
                num_faces = len(results)
                
                if num_faces == 0:
                    st.warning("The AI couldn't find any clear faces. Try a clearer photo!")
                else:
                    st.success(f"AI found **{num_faces}** face(s). Results below:")
                    st.write("---")
                
                # Loop through each face found
                for i, face in enumerate(results):
                    # Get Age and Face Region (Coordinates)
                    age = face['age']
                    region = face['region'] # Format: {'x': 10, 'y': 20, 'w': 100, 'h': 100}

                    # Create a Crop using the full image array
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    
                    # Cut out the face (safely, checking if crop is too small)
                    if w > 10 and h > 10:
                        face_crop_array = img_array[y:y+h, x:x+w]
                        face_crop_img = Image.fromarray(face_crop_array)
                    else:
                        st.write("*(Face too small to crop)*")
                        face_crop_img = None
                    
                    # Make a Result Card (Container) for this specific person
                    with st.container(border=True):
                        st.write(f"### Person {i+1}")
                        
                        col1, col2 = st.columns([1, 2]) # Image on left, Age on right
                        
                        with col1:
                            if face_crop_img:
                                # We set use_container_width=True so it fits nicely on a phone screen
                                st.image(face_crop_img, use_container_width=True)
                            else:
                                st.write("📷")

                        with col2:
                            st.metric(label="Estimated Age", value=f"{age} years old")
                        
                        st.write("---") # Simple separator line
                
                if num_faces > 1:
                    st.balloons()
                
            except Exception as e:
                st.error("Something went wrong with the AI. Try a photo where faces are clear.")
                # st.write(f"Technical error: {e}") # Uncomment to debug