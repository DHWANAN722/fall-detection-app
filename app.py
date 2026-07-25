"""
AI-Powered Elderly Fall Detection System — Healthcare Monitoring Dashboard
FA-2 Step 7: Model Deployment using Streamlit

How it works:
  1. The CNN trained in the Colab notebook (fall_detection_model.keras)
     classifies each image/video frame into an activity class.
  2. MediaPipe Pose draws the 33-keypoint skeleton so caregivers can SEE
     the posture the AI is reacting to.
  3. If the predicted class is a fall, an emergency alert is raised and
     logged in the Analytics tab.
"""

import os
import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image

# MediaPipe is optional — the dashboard still works without pose overlays
try:
    import mediapipe as mp
    POSE_AVAILABLE = True
except Exception:
    POSE_AVAILABLE = False

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Elderly Fall Detection System",
                   page_icon="🏥", layout="wide")

IMG_SIZE = (128, 128)
MODEL_PATH = "fall_detection_model.keras"
CLASSES_PATH = "class_names.json"


# ----------------------------------------------------------------------
# Load model + class names (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_model_and_classes():
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def is_fall(label: str) -> bool:
    return "fall" in label.lower()


def predict(img_rgb: np.ndarray, model, class_names):
    """Classify one RGB frame. Returns (label, confidence, all probabilities)."""
    x = cv2.resize(img_rgb, IMG_SIZE).astype("float32") / 255.0
    probs = model.predict(x[None, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


def draw_pose(img_rgb: np.ndarray):
    """Draw the MediaPipe 33-keypoint skeleton. Returns (annotated_rgb, torso_angle)."""
    if not POSE_AVAILABLE:
        return img_rgb, None
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True,
                      min_detection_confidence=0.5) as pose:
        results = pose.process(img_rgb)
    annotated = img_rgb.copy()
    torso_angle = None
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Simple posture check: angle of the torso (shoulder→hip line)
        # relative to vertical. Near 90° means the body is horizontal —
        # supporting evidence for a fall.
        lm = results.pose_landmarks.landmark
        sh = np.array([(lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2])
        hp = np.array([(lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2])
        v = sh - hp
        torso_angle = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-6)))
    return annotated, torso_angle


def log_detection(source: str, label: str, conf: float):
    st.session_state.log.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Source": source,
        "Activity": label,
        "Confidence": round(conf * 100, 1),
        "Alert": "🚨 FALL" if is_fall(label) else "—",
    })


def fall_alert_banner(conf: float):
    st.error(f"🚨 **EMERGENCY ALERT — FALL DETECTED!** "
             f"(confidence {conf:.0%})")
    with st.expander("📞 Emergency response actions", expanded=True):
        st.markdown(
            "- **Caregiver notified** on duty station monitor\n"
            "- **SMS / call alert** would be sent to the emergency contact\n"
            "- **Location:** Room camera feed flagged for immediate review\n\n"
            "*(In a production system this would trigger a real SMS/pager "
            "API — here the alert is simulated on the dashboard.)*")


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "log" not in st.session_state:
    st.session_state.log = []

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 About this system")
    st.markdown(
        "**AI-Powered Elderly Fall Detection**\n\n"
        "- **Pose estimation:** MediaPipe Pose (33 body keypoints)\n"
        "- **Activity classifier:** Convolutional Neural Network (CNN)\n"
        "- **Purpose:** help caregivers and hospitals detect fall "
        "incidents and respond faster.")
    st.divider()
    conf_threshold = st.slider("Minimum confidence to record a detection",
                               0.0, 1.0, 0.5, 0.05)
    if not POSE_AVAILABLE:
        st.warning("MediaPipe not installed — pose overlays disabled.")
    st.caption("IBCP CRS: Artificial Intelligence — FA-2")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏥 AI-Powered Elderly Fall Detection System")
st.markdown("Real-time healthcare monitoring dashboard — upload an image or "
            "video to detect falls and classify activities.")

# Stop gracefully if the model files are missing
if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
    st.error(f"Model files not found. Please add **{MODEL_PATH}** and "
             f"**{CLASSES_PATH}** (exported from the training notebook) "
             "to the same folder/repository as this app.")
    st.stop()

model, class_names = load_model_and_classes()

tab_img, tab_vid, tab_stats = st.tabs(
    ["📷 Image Detection", "🎥 Video Detection", "📊 Monitoring Analytics"])

# ======================================================================
# TAB 1 — IMAGE DETECTION
# ======================================================================
with tab_img:
    up = st.file_uploader("Upload an image frame",
                          type=["jpg", "jpeg", "png", "bmp"], key="img_up")
    if up is not None:
        img_rgb = np.array(Image.open(up).convert("RGB"))
        label, conf, probs = predict(img_rgb, model, class_names)
        annotated, torso_angle = draw_pose(img_rgb)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Uploaded frame",
                     use_container_width=True)
        with c2:
            st.image(annotated, caption="Pose estimation (MediaPipe keypoints)",
                     use_container_width=True)

        if is_fall(label):
            fall_alert_banner(conf)
        else:
            st.success(f"✅ Normal activity detected: **{label.title()}** "
                       f"(confidence {conf:.0%})")

        if torso_angle is not None:
            posture = ("horizontal — consistent with a fall"
                       if torso_angle > 60 else "upright")
            st.info(f"🧍 Posture check: torso angle ≈ {torso_angle:.0f}° "
                    f"from vertical → body appears **{posture}**.")

        st.subheader("Prediction confidence per class")
        st.bar_chart(pd.DataFrame(
            {"Confidence": probs}, index=[c.title() for c in class_names]))

        if conf >= conf_threshold:
            log_detection("Image", label, conf)

# ======================================================================
# TAB 2 — VIDEO DETECTION
# ======================================================================
with tab_vid:
    upv = st.file_uploader("Upload a video clip",
                           type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    if upv is not None:
        with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(upv.name)[1]) as tmp:
            tmp.write(upv.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step = max(total // 40, 1)          # analyse up to ~40 frames
        st.write(f"Analysing ~{min(40, total)} frames "
                 f"(every {step}ᵗʰ frame of {total})...")

        results, first_fall_frame = [], None
        progress = st.progress(0.0)
        i = analysed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                label, conf, _ = predict(rgb, model, class_names)
                results.append({"Frame": i, "Activity": label.title(),
                                "Confidence": conf,
                                "Fall": 1 if is_fall(label) else 0})
                if is_fall(label) and first_fall_frame is None:
                    first_fall_frame = rgb
                analysed += 1
                progress.progress(min(analysed / max(total // step, 1), 1.0))
            i += 1
        cap.release()
        os.unlink(video_path)
        progress.empty()

        df = pd.DataFrame(results)
        falls = int(df["Fall"].sum())

        if falls > 0:
            worst = df[df["Fall"] == 1]["Confidence"].max()
            fall_alert_banner(worst)
            st.write(f"Falls detected in **{falls} of {len(df)}** "
                     "analysed frames.")
            if first_fall_frame is not None:
                annotated, _ = draw_pose(first_fall_frame)
                st.image(annotated, caption="First fall frame detected "
                         "(with pose keypoints)", width=420)
        else:
            st.success("✅ No falls detected in this video — "
                       "all activity appears normal.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Activities across the video")
            st.bar_chart(df["Activity"].value_counts())
        with c2:
            st.subheader("Fall detections over time")
            st.line_chart(df.set_index("Frame")["Fall"])

        # log the dominant activity of the video
        top = df["Activity"].mode()[0]
        top_conf = float(df[df["Activity"] == top]["Confidence"].mean())
        log_detection("Video", "fall" if falls else top, top_conf)

# ======================================================================
# TAB 3 — MONITORING ANALYTICS
# ======================================================================
with tab_stats:
    st.subheader("Session monitoring overview")
    log_df = pd.DataFrame(st.session_state.log)

    total_det = len(log_df)
    fall_count = int((log_df["Alert"] == "🚨 FALL").sum()) if total_det else 0
    normal_count = total_det - fall_count
    avg_conf = float(log_df["Confidence"].mean()) if total_det else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total activities detected", total_det)
    m2.metric("🚨 Falls detected", fall_count)
    m3.metric("✅ Normal activities", normal_count)
    m4.metric("Avg. confidence", f"{avg_conf:.1f}%")

    if total_det:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Activity distribution")
            st.bar_chart(log_df["Activity"].str.title().value_counts())
        with c2:
            st.subheader("Detection log")
            st.dataframe(log_df.iloc[::-1], use_container_width=True,
                         hide_index=True)
        if fall_count:
            st.error(f"⚠️ {fall_count} fall alert(s) recorded this session — "
                     "review the log above and confirm caregiver response.")
        if st.button("🗑️ Clear session log"):
            st.session_state.log = []
            st.rerun()
    else:
        st.info("No detections yet — analyse an image or video first.")

st.divider()
st.caption("Developed for IBCP Year 2 — CRS Artificial Intelligence, "
           "Machine Learning and Deep Learning (FA-2). "
           "This tool assists caregivers and does not replace "
           "professional medical supervision.")
