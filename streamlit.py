import streamlit as st
from ultralytics import YOLO
import cv2
from gtts import gTTS
import tempfile, os
from datetime import datetime
from collections import Counter
import pandas as pd
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Pre-Exam Monitoring 🚨", layout="wide", page_icon="📵")

# =====================================================
# MODELS
# =====================================================
CUSTOM_MODEL_PATH = r"C:\Users\dhanak\PycharmProjects\fyp\best (3).pt"
custom_model = YOLO(CUSTOM_MODEL_PATH)
yolo_general = YOLO("yolov8n.pt")

YOLO_RENAME = {"mobile": "cell phone", "phone": "cell phone"}
ALL_OBJECTS = ['bag', 'book', 'calculator', 'cell phone', 'notebook', 'notes', 'smartwatch']

# =====================================================
# SESSION STATE
# =====================================================
if "detected" not in st.session_state: st.session_state.detected = []
if "collected" not in st.session_state: st.session_state.collected = []
if "spoken" not in st.session_state: st.session_state.spoken = {}
if "history" not in st.session_state: st.session_state.history = []

# =====================================================
# SPEAK FUNCTION
# =====================================================
def speak_text(text):
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    tts = gTTS(text=text, lang='en')
    tts.save(path)
    os.system(f"start {path}")

# =====================================================
# AUTO SCREENSHOT FUNCTION
# =====================================================
def take_screenshot(frame, obj_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"screenshots/screenshot_{obj_name}_{ts}.png"
    os.makedirs("screenshots", exist_ok=True)
    cv2.imwrite(fname, frame)
    return fname

# =====================================================
# SIDEBAR CONFIGURATION
# =====================================================
st.sidebar.markdown("## ⚙️ Configuration")
confidence = st.sidebar.slider("Detection Confidence Threshold", 0.1, 1.0, 0.75)
selected_objects = st.sidebar.multiselect("Monitor Objects:", ALL_OBJECTS, default=['cell phone','calculator','bag'])

# =====================================================
# HEADER WITH GRADIENT
# =====================================================
st.markdown("""
<h1 style='background: linear-gradient(to right, #ff4b1f, #ff9068); -webkit-background-clip: text;
color: transparent; font-weight: 900;'>Pre-Exam Monitoring System 🚨</h1>
<p style='color:#555;'>Real-time intelligent proctoring interface</p>
""", unsafe_allow_html=True)

# =====================================================
# CAMERA SETUP
# =====================================================
start = st.toggle("🎥 Start Camera")
FRAME = st.empty()

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("🚫 Camera not found")
        return None
    return cap

# =====================================================
# DETECTION LOOP
# =====================================================
if start:
    cap = open_camera()
    if cap is None:
        st.stop()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        frame = cv2.resize(frame, (640, 480))
        detected_now = []

        results_custom = custom_model.predict(frame, conf=confidence)
        results_yolo = yolo_general.predict(frame, conf=confidence)

        for results in [results_custom, results_yolo]:
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        name = r.names[cls]
                        name = YOLO_RENAME.get(name, name)

                        # CONFIDENCE CHECK
                        if float(box.conf[0]) < confidence:
                            continue

                        detected_now.append(name)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # RED FLASH ALERT + PULSING
                        color = (0,0,255) if name in selected_objects else (255,0,0)
                        thickness = 3 if name in selected_objects else 2
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
                        cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                        # SPEAK + AUTO COLLECT + SCREENSHOT
                        if name in selected_objects:
                            if name not in st.session_state.spoken:
                                speak_text(f"Prohibited: {name} detected!")
                                st.session_state.spoken[name] = True
                            st.session_state.collected.append(name)
                            screenshot_path = take_screenshot(frame, name)
                            st.session_state.history.append({'name': name, 'screenshot': screenshot_path, 'time': datetime.now()})

        st.session_state.detected = list(set(detected_now))
        FRAME.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# =====================================================
# DASHBOARD SUMMARY
# =====================================================
st.markdown("## 📊 Examination Summary")
colA, colB, colC = st.columns(3)
with colA: st.metric("Total Items Collected", len(st.session_state.collected))
with colB: st.metric("Unique Item Types", len(set(st.session_state.collected)))
with colC: st.metric("Detected Now", len(st.session_state.detected))

# =====================================================
# HISTORY GALLERY
# =====================================================
if st.session_state.history:
    st.markdown("### 📸 Detection History")
    for item in reversed(st.session_state.history[-10:]):
        st.image(item['screenshot'], width=200, caption=f"{item['name']} @ {item['time'].strftime('%H:%M:%S')}")

# =====================================================
# DETECTION CHART
# =====================================================
if st.session_state.collected:
    df = pd.DataFrame(st.session_state.collected, columns=['Object'])
    count = df['Object'].value_counts()
    st.bar_chart(count)

# =====================================================
# ACTION BUTTONS
# =====================================================
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Ignore"):
        if st.session_state.detected: st.session_state.detected.pop()
with col2:
    if st.button("Collected"):
        if st.session_state.detected:
            st.session_state.collected.append(st.session_state.detected[0])
with col3:
    if st.button("Start New Examination"):
        st.session_state.collected = []
        st.session_state.detected = []
        st.session_state.spoken = {}
        st.session_state.history = []
