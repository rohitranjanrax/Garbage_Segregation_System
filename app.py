import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import urllib.request
import time
import os

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="AI Garbage Segregation System", layout="wide")
st.markdown("""
    <div style="padding:15px;">
        <center><h1>♻️ AI Garbage Segregation System</h1></center>
        <center><h3>YOLO + CNN | Upload, capture, or use webcam for detection</h3></center>
    </div>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("Models/best.pt")
        cnn_model = tf.keras.models.load_model("Models/cnn_model.h5")
        return yolo_model, cnn_model
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        st.stop()

yolo_model, cnn_model = load_models()
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# -------------------- HELPER: Preprocess for CNN --------------------
def preprocess(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------- IMAGE INPUT --------------------
option = st.selectbox(
    "Choose image input method:",
    ("Please Select", "Upload from Device", "Upload via URL", "Capture from Webcam", "Live Webcam (Real-time)")
)

image = None

# Upload from Device
if option == "Upload from Device":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file)

# Upload from URL
elif option == "Upload via URL":
    url = st.text_input("Enter image URL:")
    if url:
        try:
            image = Image.open(urllib.request.urlopen(url))
        except Exception:
            if st.button("Retry"):
                st.error("Invalid image URL!")

# Capture from Webcam
elif option == "Capture from Webcam":
    camera_file = st.camera_input("Capture image from webcam")
    if camera_file:
        image = Image.open(camera_file)

# -------------------- LIVE WEBCAM --------------------
elif option == "Live Webcam (Real-time)":
    st.warning("Press 'Start Live Detection' to begin. Press 'q' to quit.")
    if st.button("Start Live Detection"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Could not access webcam.")
                break

            # YOLO detection
            results = yolo_model.predict(frame, conf=0.4, verbose=False)
            annotated = results[0].plot()

            # Display annotated video
            stframe.image(annotated, channels="BGR", use_container_width=True)

            # Stop if 'q' pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# -------------------- SINGLE IMAGE PREDICTION --------------------
if image and option != "Live Webcam (Real-time)":
    st.image(image, width=350, caption="Uploaded / Captured Image")

    if st.button("🔍 Detect & Classify"):
        try:
            # --- YOLO DETECTION ---
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                image.save(temp.name)
                yolo_results = yolo_model.predict(source=temp.name, conf=0.4, save=False)
                yolo_img = yolo_results[0].plot()
                st.subheader("🟩 Object Detection (YOLO)")
                st.image(yolo_img, use_container_width=True)
                os.remove(temp.name)

            # --- CNN CLASSIFICATION ---
            st.subheader("🧠 Waste Type Classification (CNN)")
            img_ready = preprocess(image)
            preds = cnn_model.predict(img_ready)
            class_index = np.argmax(preds)
            confidence = np.max(preds)

            st.success(f"**Predicted Class:** {labels[class_index]} ({confidence*100:.2f}% confidence)")

        except Exception as e:
            st.error(f"Error during processing: {e}")

else:
    if option == "Please Select":
        st.info("👆 Select an input option to begin.")
