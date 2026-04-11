import streamlit as st
from PIL import Image
import numpy as np
import os
import gc
from io import BytesIO

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

# 2. Session state — persist results across Streamlit reruns
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None   # list of per-person dicts
if 'face_crops' not in st.session_state:
    st.session_state.face_crops = []           # pre-cropped face arrays
if 'show_balloons' not in st.session_state:
    st.session_state.show_balloons = False

# 3. TABS FOR INPUT
tab1, tab2, tab3 = st.tabs(["📸 Take a Selfie", "📁 Upload Image", "📋 Paste Image"])

source_img = None
is_camera = False
is_paste = False

with tab1:
    st.write("### Quick Selfie")
    cam_img = st.camera_input("Snap a photo to analyze automatically")
    if cam_img:
        source_img = cam_img
        is_camera = True

with tab2:
    st.write("### Select a Photo")
    st.info("💡 Hint: Use the 'Search' or 'Recent' folder on your phone to find files faster.")
    file_img = st.file_uploader("Upload from your device", type=["jpg", "jpeg", "png"])
    if file_img:
        source_img = file_img

with tab3:
    st.write("### Paste from Clipboard")
    st.info(
        "**On iPhone/iPad:** Open Photos → long-press a photo → **Copy Photo** → come back here → tap the button below.\n\n"
        "**On Android:** Open a photo in your gallery → tap **Share** → **Copy** → come back here → tap the button below."
    )
    try:
        from streamlit_paste_button import paste_image_button as pbutton
        paste_result = pbutton("📋 Paste Image from Clipboard", key="paste_btn")
        if paste_result.image_data is not None:
            buf = BytesIO()
            paste_result.image_data.save(buf, format='PNG')
            buf.seek(0)
            source_img = buf
            is_paste = True
    except ImportError:
        st.warning("Clipboard paste requires `streamlit-paste-button`. Run: `pip install streamlit-paste-button`")

# 4. ANALYSIS LOGIC
if source_img:
    raw_img = Image.open(source_img).convert('RGB')
    raw_img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)

    st.image(raw_img, width='stretch', caption="Image received!")

    run_analysis = False
    if is_camera or is_paste:
        run_analysis = True
    else:
        run_analysis = st.button("Analyze Uploaded Photo! 🚀", type="primary")

    if run_analysis:
        # Clear stale results from any previous photo
        st.session_state.analysis_results = None
        st.session_state.face_crops = []
        st.session_state.show_balloons = False

        with st.spinner('AI is analyzing faces...'):
            try:
                from deepface import DeepFace
                img_array = np.array(raw_img)

                # 5. MULTI-DETECTOR STRATEGY
                # Try every backend and keep whichever finds the most faces.
                # retinaface/mtcnn are best for group photos; opencv/ssd often
                # miss faces that are slightly off-angle or at different scales.
                best_results = None
                last_error = None
                for backend in ['retinaface', 'mtcnn', 'fastmtcnn', 'yunet', 'opencv', 'ssd']:
                    try:
                        r = DeepFace.analyze(
                            img_path=img_array,
                            actions=['age', 'emotion'],
                            enforce_detection=True,
                            detector_backend=backend,
                            align=False
                        )
                        if best_results is None or len(r) > len(best_results):
                            best_results = r
                        # stop early if we already found 2+ people
                        if best_results and len(best_results) >= 2:
                            break
                    except ValueError:
                        last_error = 'no_face'
                        continue
                    except Exception as e:
                        last_error = e
                        continue
                results = best_results

                if not results:
                    if last_error == 'no_face':
                        st.warning("No face detected. Try facing the camera directly with good lighting.")
                    else:
                        st.error("The AI is having a moment. Please try a clearer photo or a different angle.")
                        with st.expander("Error details"):
                            st.caption(str(last_error))
                else:
                    results = sorted(results, key=lambda x: x['region']['x'])

                    # Pre-compute per-person data and face crops, store in session state
                    persons = []
                    crops = []
                    pad = 25
                    h, w = img_array.shape[:2]
                    for person in results:
                        r = person['region']
                        face_conf = person.get('face_confidence', 0.5)
                        emotions = person.get('emotion', {})
                        positive = (emotions.get('happy', 0) + emotions.get('surprise', 0)) / 100
                        look_score = round(min(10.0, face_conf * 7 + positive * 3), 1)

                        y1, y2 = max(0, r['y'] - pad), min(h, r['y'] + r['h'] + pad)
                        x1, x2 = max(0, r['x'] - pad), min(w, r['x'] + r['w'] + pad)
                        crop = img_array[y1:y2, x1:x2]

                        persons.append({'age': person['age'], 'look_score': look_score})
                        crops.append(crop if crop.size > 0 else None)

                    st.session_state.analysis_results = persons
                    st.session_state.face_crops = crops
                    st.session_state.show_balloons = True

                del img_array
                gc.collect()

            except Exception as e:
                st.error("The AI is having a moment. Please try a clearer photo or a different angle.")
                with st.expander("Error details"):
                    st.caption(str(e))
                gc.collect()

# 6. DISPLAY RESULTS — rendered from session state so they survive reruns
if st.session_state.analysis_results:
    persons = st.session_state.analysis_results
    crops = st.session_state.face_crops

    st.divider()
    st.subheader(f"Found {len(persons)} person(s)")

    for i, (person, crop) in enumerate(zip(persons, crops)):
        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                if crop is not None:
                    st.image(crop, width='stretch')
                else:
                    st.write("📷")
            with col2:
                st.markdown(f"**Person {i+1}**")
                st.metric("Age Guess", f"{person['age']} yrs")
                st.metric("Look Score", f"{person['look_score']} / 10")

    if st.session_state.show_balloons:
        st.session_state.show_balloons = False
        st.balloons()

# 7. SIDEBAR TOOLS
with st.sidebar:
    st.title("Settings")
    if st.button("Hard Reset App"):
        st.session_state.analysis_results = None
        st.session_state.face_crops = []
        st.session_state.show_balloons = False
        gc.collect()
        st.rerun()
    st.write("---")
    st.caption("Powered by DeepFace AI")
