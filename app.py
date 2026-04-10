import streamlit as st
from PIL import Image
import numpy as np
import os
import gc

# 1. Server Stability Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="How old am I?", layout="centered")

# Custom CSS for a clean, professional app look
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stMetric { background-color: #1e3a5f; padding: 15px; border-radius: 10px; border: 1px solid #2d5a8e; }
        .stMetric * { color: white !important; }
        .stButton>button { width: 100%; border-radius: 20px; height: 3em; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("How old am I? 🎂")

# 2. TABS FOR INPUT
tab1, tab2 = st.tabs(["📸 Take a Selfie", "📁 Upload Image"])

source_img = None

with tab1:
    st.write("### Quick Selfie")
    cam_img = st.camera_input("Snap a photo to analyze automatically")
    if cam_img:
        source_img = cam_img

with tab2:
    st.write("### Select a Photo")
    st.info("💡 Hint: Use the 'Search' or 'Recent' folder on your phone to find files faster.")
    file_img = st.file_uploader("Upload from your device", type=["jpg", "jpeg", "png"])
    if file_img:
        source_img = file_img

# 3. ANALYSIS LOGIC
if source_img:
    # Load and optimize image size for the AI
    raw_img = Image.open(source_img).convert('RGB')
    raw_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    
    st.image(raw_img, width='stretch', caption="Image received!")
    
    # Auto-run for camera, manual button for file upload
    run_analysis = False
    if cam_img:
        run_analysis = True
    else:
        run_analysis = st.button("Analyze Uploaded Photo! 🚀", type="primary")

    if run_analysis:
        with st.spinner('AI is analyzing faces...'):
            try:
                # Lazy import to save startup RAM
                from deepface import DeepFace
                img_array = np.array(raw_img)

                # 4. MULTI-DETECTOR STRATEGY
                # Try each backend in order; stop on first success
                results = None
                last_error = None
                for backend in ['opencv', 'ssd', 'mtcnn', 'fastmtcnn']:
                    try:
                        results = DeepFace.analyze(
                            img_path=img_array,
                            actions=['age'],
                            enforce_detection=True,
                            detector_backend=backend,
                            align=True
                        )
                        break  # success — stop trying
                    except ValueError:
                        # No face detected by this backend; try the next one
                        last_error = 'no_face'
                        continue
                    except Exception as e:
                        last_error = e
                        continue

                if not results:
                    if last_error == 'no_face':
                        st.warning("No face detected. Try facing the camera directly with good lighting.")
                    else:
                        st.error("The AI is having a moment. Please try a clearer photo or a different angle.")
                        with st.expander("Error details"):
                            st.caption(str(last_error))
                else:
                    st.divider()
                    # Pick the most prominent person (largest detected face area)
                    person = max(results, key=lambda x: x['region']['w'] * x['region']['h'])
                    age = person['age']
                    r = person['region']

                    # Create a card for the detected person
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            # Crop the face with some padding
                            pad = 25
                            y1, y2 = max(0, r['y']-pad), min(img_array.shape[0], r['y']+r['h']+pad)
                            x1, x2 = max(0, r['x']-pad), min(img_array.shape[1], r['x']+r['w']+pad)
                            face_crop = img_array[y1:y2, x1:x2]

                            if face_crop.size > 0:
                                st.image(face_crop, width='stretch')
                            else:
                                st.write("📷")

                        with col2:
                            st.metric("Age Guess", f"{age} yrs")

                    st.balloons()

                # Immediate Memory Cleanup
                del img_array
                gc.collect()

            except Exception as e:
                st.error("The AI is having a moment. Please try a clearer photo or a different angle.")
                with st.expander("Error details"):
                    st.caption(str(e))
                gc.collect()

# 5. SIDEBAR TOOLS
with st.sidebar:
    st.title("Settings")
    if st.button("Hard Reset App"):
        gc.collect()
        st.rerun()
    st.write("---")
    st.caption("Powered by DeepFace AI")