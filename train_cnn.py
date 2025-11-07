import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import cv2
import os

# -------------------- CONFIG --------------------
st.set_page_config(page_title="♻️ AI Garbage Segregation System", layout="centered")
st.title("♻️ AI Garbage Segregation System")
st.write("Upload an image or use webcam for real-time garbage detection and classification.")

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("Models/cnn_model.h5")
    yolo_model = YOLO("Models/best.pt")
    return cnn_model, yolo_model

cnn_model, yolo_model = load_models()

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# -------------------- IMAGE INPUT --------------------
option = st.selectbox("Choose how to provide the image:", ["Upload", "Webcam"])

img = None
if option == "Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
elif option == "Webcam":
    cam = st.camera_input("Capture image from webcam")
    if cam:
        img = Image.open(cam)

if img:
    st.image(img, caption="Input Image", use_container_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        img.save(temp_file.name)
        temp_path = temp_file.name

    # -------------------- YOLO DETECTION --------------------
    st.subheader("🔍 Object Detection (YOLO)")
    results = yolo_model.predict(source=temp_path, conf=0.4, save=False)
    yolo_img = results[0].plot()

    st.image(yolo_img, caption="Detected Objects", use_container_width=True)

    # Get bounding boxes from YOLO
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

    # -------------------- CNN CLASSIFICATION --------------------
    st.subheader("🧠 Waste Type Classification (CNN)")

    img_cv = cv2.imread(temp_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img_cv[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (128, 128))
        crop_array = np.expand_dims(crop_resized / 255.0, axis=0)

        pred = cnn_model.predict(crop_array)
        cls_idx = np.argmax(pred)
        conf = np.max(pred)

        st.write(f"🗑️ Object {i+1}: **{labels[cls_idx]}** ({conf*100:.2f}%)")

    # -------------------- CLEANUP --------------------
    os.remove(temp_path)
