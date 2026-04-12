# Architecture & Technical Stack

## Table of Contents
1. [CV Description](#cv-description)
2. [System Overview](#system-overview)
3. [Repository Structure](#repository-structure)
4. [Technical Stack](#technical-stack)
5. [Application Architecture](#application-architecture)
6. [Core Analysis Pipeline](#core-analysis-pipeline)
7. [Data Flow](#data-flow)
8. [Advantages](#advantages)
9. [Limitations](#limitations)

---

## CV Description

> Built a privacy-first AI web application using Python, Streamlit, InsightFace, and DeepFace with TensorFlow that detects faces in real-time photos, estimates age with a dual-model strategy (InsightFace buffalo_sc primary, six-backend DeepFace fallback), and computes a custom Look Score — deployed via GitHub Codespaces with zero persistent infrastructure.

*(49 words — ready to paste into your CV)*

---

## System Overview

**"How old am I?"** is a single-page, browser-based AI web application that estimates a person's age and calculates a "Look Score" from a photo. It runs entirely as a Streamlit app inside a GitHub Codespaces Dev Container, requires no dedicated backend API, and stores no data beyond the active browser session.

```
User (Mobile / Desktop Browser)
        │
        ▼
┌─────────────────────────────────────────────┐
│     GitHub Codespaces (public port 8501)    │
│  ┌─────────────────────────────────────┐    │
│  │  Dev Container (Python 3.11/Debian) │    │
│  │                                     │    │
│  │  ┌─────────────────────────────┐    │    │
│  │  │   Streamlit Server (app.py) │    │    │
│  │  │                             │    │    │
│  │  │  ┌───────────────────────┐  │    │    │
│  │  │  │  PRIMARY: InsightFace │  │    │    │
│  │  │  │  buffalo_sc (ONNX)    │  │    │    │
│  │  │  └───────────┬───────────┘  │    │    │
│  │  │              │ fallback     │    │    │
│  │  │  ┌───────────▼───────────┐  │    │    │
│  │  │  │  FALLBACK: DeepFace   │  │    │    │
│  │  │  │  TensorFlow 2.13 CPU  │  │    │    │
│  │  │  │  OpenCV               │  │    │    │
│  │  │  └───────────────────────┘  │    │    │
│  │  └─────────────────────────────┘    │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

---

## Repository Structure

```
DeepFace/
├── app.py                      # Entire application — UI + analysis logic (~253 lines)
├── requirements.txt            # Python dependencies
├── packages.txt                # System (apt) dependencies
├── ARCHITECTURE.md             # This document
├── README.md                   # Minimal project title only
├── .streamlit/
│   └── config.toml             # Streamlit server & UI settings
└── .devcontainer/
    └── devcontainer.json       # Docker Dev Container definition
```

---

## Technical Stack

### Runtime Environment

| Layer | Technology | Version / Detail |
|---|---|---|
| Cloud Host | GitHub Codespaces | Dev Container, public port 8501 |
| Container Image | Microsoft Dev Container | `python:1-3.11-bookworm` |
| OS | Debian Bookworm | via Docker base image |
| Language | Python | 3.11 |

### Application Framework

| Component | Package | Version |
|---|---|---|
| Web framework | Streamlit | `>= 1.50.0` |
| Clipboard paste component | streamlit-paste-button | `>= 0.1.0` |

### AI / Machine Learning

| Component | Package | Version | Role |
|---|---|---|---|
| **Primary** face detection & age | InsightFace | `>= 0.7.3` | buffalo_sc model — better demographic accuracy |
| ONNX model runtime | onnxruntime | latest | Required by InsightFace backends |
| **Fallback** face analysis | DeepFace | latest | Age estimation, emotion detection (6-backend loop) |
| Deep learning runtime | tensorflow-cpu | `== 2.13.1` | Model inference for DeepFace (pinned for stability) |
| Keras compatibility | tf-keras | latest | Required by DeepFace with TF 2.13 |

### Computer Vision & Image Processing

| Component | Package | Version | Role |
|---|---|---|---|
| Image I/O & manipulation | Pillow | latest | Open, resize, convert images |
| Video / matrix ops | opencv-python-headless | latest | Face detection backends |

### System Libraries (apt)

| Package | Purpose |
|---|---|
| `libgl1` | OpenGL support required by OpenCV |
| `libglib2.0-0t64` | GLib runtime required by OpenCV on Debian Bookworm |

### Face Detection Strategy

The app uses a **dual-strategy approach**:

#### Strategy 1 — Primary: InsightFace (buffalo_sc)

| Property | Detail |
|---|---|
| Model | `buffalo_sc` (Scaled Chinese variant) |
| Runtime | ONNX via CPUExecutionProvider |
| Strengths | Better demographic accuracy; strong on diverse, younger, and Asian faces |
| Output | Bounding box, age integer, face confidence score |
| Init | `FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])` |

#### Strategy 2 — Fallback: DeepFace Multi-Detector Loop

Activated when InsightFace detects no faces. Six backends are attempted in priority order, keeping the result with the most detected faces (early exit if ≥ 2 faces found):

| Priority | Backend | Strength |
|---|---|---|
| 1 | RetinaFace | Best accuracy; handles group photos and angled faces |
| 2 | MTCNN | Strong multi-face detection |
| 3 | FastMTCNN | Faster MTCNN variant |
| 4 | YuNet | Lightweight neural network detector |
| 5 | OpenCV | Haar cascade fallback; very fast |
| 6 | SSD | SSD-based detection fallback |

### Streamlit Configuration

```toml
[client]
toolbarMode = "minimal"       # Hides most toolbar chrome
showErrorDetails = false      # Hides raw tracebacks from users

[ui]
hideSidebarNav = true         # No multi-page navigation shown
```

---

## Application Architecture

### Architectural Pattern

**Monolithic single-file MVC-lite** — the entire app lives in `app.py`. There is no separate:
- REST API or backend server
- Database or persistent storage
- Authentication layer
- Build pipeline or compiled assets

### Input Layer (3 tabs)

```
┌──────────────────────────────────────────────────────┐
│                    Input Tabs                        │
│                                                      │
│  📸 Take a Selfie  │  📁 Upload Image  │  📋 Paste   │
│                    │                  │             │
│  st.camera_input() │ st.file_uploader │ paste_image │
│  Auto-analyzes     │ Requires button  │ _button()   │
│                    │  click           │ Auto-analyzes│
└──────────────────────────────────────────────────────┘
```

### State Management

Streamlit reruns the entire script on every user interaction. Three session state keys persist results across reruns:

| Key | Type | Content |
|---|---|---|
| `analysis_results` | `List[Dict] \| None` | `[{age: int, look_score: float}, …]` |
| `face_crops` | `List[np.ndarray]` | Cropped face regions as pixel arrays |
| `show_balloons` | `bool` | One-shot flag for the celebration animation |

### Rendering Layer

```
Session State
      │
      ▼
┌─────────────────────────────────────────┐
│  Results Panel (rendered every rerun)   │
│                                         │
│  For each person:                       │
│  ┌───────────┬──────────────────────┐   │
│  │ Face crop │  Person N            │   │
│  │ (1/3 col) │  Age Guess: X yrs   │   │
│  │           │  Look Score: Y / 10 │   │
│  └───────────┴──────────────────────┘   │
└─────────────────────────────────────────┘
```

---

## Core Analysis Pipeline

```
Image Input (camera / file / clipboard)
        │
        ▼
PIL.Image.open() → convert('RGB')
        │
        ▼
thumbnail(1000×1000, LANCZOS)        ← resize for memory safety
        │
        ▼
np.array(raw_img)                    ← convert to NumPy
        │
        ▼
┌──────────────────────────────────────────────┐
│  STRATEGY 1 — InsightFace (Primary)          │
│                                              │
│  FaceAnalysis(name='buffalo_sc',             │
│               providers=['CPUExecutionProvider'])│
│                                              │
│  → returns: age, bbox, confidence per face  │
│  → emotion analysis: DeepFace on face crop  │
└──────────────────┬───────────────────────────┘
                   │ if 0 faces detected
                   ▼
┌──────────────────────────────────────────────┐
│  STRATEGY 2 — DeepFace Fallback Loop         │
│                                              │
│  for backend in [retinaface, mtcnn,          │
│                  fastmtcnn, yunet,           │
│                  opencv, ssd]:               │
│                                              │
│    DeepFace.analyze(                         │
│      actions=['age','emotion'],              │
│      enforce_detection=True,                 │
│      align=False                             │
│    )                                         │
│                                              │
│    Keep result with MOST faces               │
│    Stop early if ≥ 2 faces found             │
└──────────────────────────────────────────────┘
        │
        ▼
Sort detected faces left → right by region['x']
        │
        ▼
For each face:
  ├── Crop with 25px padding (boundary-clamped)
  └── Compute Look Score:
        face_conf = detection confidence  (default 0.5)
        positive  = (happy% + surprise%) / 100
        score     = min(10.0, face_conf×7 + positive×3)
        score     = round(score, 1)
        │
        ▼
Store in session_state → Render results → gc.collect()
```

### Look Score Formula

```
Look Score = min(10.0,  face_confidence × 7  +  positive_emotion × 3)

  face_confidence   → detection confidence  [0.0 – 1.0]  (weight 70%)
  positive_emotion  → (happy% + surprise%) ÷ 100  [0.0 – 1.0]  (weight 30%)
  result capped at 10.0, rounded to 1 decimal place
```

---

## Data Flow

```
User Photo
    │
    │ (HTTPS, Codespaces public port 8501)
    ▼
Streamlit Server
    │
    │ PIL decode → RGB conversion → 1000px resize
    ▼
NumPy Array (in-memory only)
    │
    ├──[Primary]─► InsightFace buffalo_sc (ONNX/CPU)
    │                  │
    │               age integer
    │               face bbox + confidence
    │                  │
    │              DeepFace.analyze() on face crop
    │                  └─► emotion dict {happy, surprise, …}
    │
    └──[Fallback]─► DeepFace multi-backend loop (TF 2.13 CPU)
                       │
                   age integer
                   emotion dict
                   face region {x, y, w, h}
    │
    │ Look Score calculation
    ▼
st.session_state  (browser session only — no disk, no DB)
    │
    ▼
Streamlit re-render → HTML/CSS/JS to browser
    │
    │ del img_array; gc.collect()
    ▼
Memory freed
```

**No data leaves the container.** Photos are never written to disk, sent to a third-party API, or persisted beyond the browser session.

---

## Advantages

### 1. Zero-Infrastructure Deployment
The entire stack runs inside a single GitHub Codespaces Dev Container. No database, no cloud storage bucket, no separate API server, and no CI/CD pipeline are required. A new contributor can have a running environment in under two minutes by opening the repo in Codespaces.

### 2. Privacy by Design
Images are processed entirely in-memory within the container. They are never written to disk, never sent to an external AI API, and are garbage-collected immediately after analysis. This is a strong privacy property compared to cloud-AI approaches (e.g. AWS Rekognition, Google Vision API).

### 3. Dual-Model Resilience
InsightFace (buffalo_sc) acts as the primary detector with superior demographic accuracy. When it finds no faces, a six-backend DeepFace fallback loop tolerates a wide range of photo conditions — angled faces, low contrast, partial occlusion — that would cause any single detector to fail silently.

### 4. Better Demographic Accuracy with InsightFace
The buffalo_sc model is trained on more diverse datasets than standard DeepFace age models and performs better on young faces, Asian demographics, and photos taken at angles. This replaces the older single-detector approach with a more robust primary strategy.

### 5. Instant Deployment to Any Device
Because it is a web app served over HTTPS on a public Codespaces port, users on any device (iOS, Android, desktop) can access it without installing an app. The centered layout and three-tab input (selfie, upload, paste) provide a mobile-friendly experience.

### 6. Self-Contained, Reproducible Environment
`devcontainer.json` defines the full environment: OS image, Python version, system packages, and Python packages. Any developer who clones the repo gets an identical environment, eliminating "works on my machine" issues.

### 7. Simple Codebase
~253 lines of Python. No build step, no transpilation, no separate frontend project. This makes the codebase easy to read, debug, and extend.

### 8. Session State Persistence
Results survive Streamlit reruns triggered by UI interactions (switching tabs, clicking sidebar buttons), so the user never loses their analysis while navigating the interface.

---

## Limitations

### 1. No Persistent Storage
All results are held in Streamlit's in-memory session state. Closing the browser tab, refreshing the page, or letting the Codespaces instance sleep destroys all results. Users cannot retrieve past analyses or build any history.

### 2. Single-User, Single-Session Architecture
Streamlit's execution model runs the entire script on every interaction for every connected client. There is no user authentication, no multi-tenancy, and no isolation between sessions beyond process-level memory separation. The app is not designed for concurrent use by many users.

### 3. CPU-Only Inference (Performance Ceiling)
Both InsightFace (ONNX/CPUExecutionProvider) and TensorFlow are pinned to CPU. Face analysis on a group photo — especially when the fallback loop iterates multiple backends — can take 15–45 seconds depending on the Codespaces machine size. There is no GPU acceleration path within the current infrastructure.

### 4. No Native Mobile Share Target
Users cannot long-press a photo in iOS Photos or Android Gallery and see this app in the system share sheet. Implementing the Web Share Target API would require serving a service worker from the root path, which Streamlit's built-in server does not support without a reverse proxy (e.g. nginx). The clipboard paste tab is a partial workaround.

### 5. Brittle Dependency Pinning
`tensorflow-cpu` is hard-pinned to `2.13.1` because later versions caused environment failures. `deepface`, `insightface`, and other packages have no upper bound. This creates a risk that a new release incompatible with existing packages silently breaks the container on next build.

### 6. No Input Validation Beyond File Type
The file uploader accepts only `.jpg`, `.jpeg`, and `.png` by extension, but does not verify image content, maximum file size, or minimum resolution. A very large image (e.g. 50 MP RAW-to-JPEG) would exhaust container memory before the `thumbnail()` resize executes.

### 7. Look Score Is Heuristic, Not Clinical
The Look Score formula (`face_confidence × 7 + positive_emotion × 3`) is an invented metric combining detection quality with emotional expression. It does not correspond to any published aesthetic or photographic quality standard and can mislead users who interpret it as objective.

### 8. Age Estimation Bias
Despite InsightFace's improved demographic coverage, all age estimation models carry dataset bias. Estimates for older adults and people from under-represented groups can still be systematically off by 5–15 years.

### 9. Codespaces Dependency
The app has no documented path to deploy outside of GitHub Codespaces (e.g. to Streamlit Community Cloud, Heroku, or a VPS). The `devcontainer.json` startup command disables CORS and XSRF protection, which is acceptable inside a private Codespace but would be a security concern on a public server.

### 10. No Automated Tests
There is no test suite. Regressions in face detection accuracy, score calculation, or UI behaviour are caught only by manual inspection.
