import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os



st.set_page_config(page_title="YOLO Object Detection", layout="wide")

st.title("🔍 YOLO Object Detection App")
st.write("Upload gambar lalu model akan melakukan deteksi objek.")

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model_path = "yolo11n.pt"   # Ganti ke yolo11n.pt jika pakai model bawaan
    model = YOLO(model_path)
    return model

model = load_model()

# =========================
# Upload Image
# =========================
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to numpy
    img_np = np.array(img)

    # =========================
    # Run Inference
    # =========================
    with st.spinner("Running YOLO detection..."):
        results = model(img_np)
        result_img = results[0].plot()   # gives numpy array with bounding boxes

    st.subheader("📌 Detection Result")
    st.image(result_img, use_column_width=True)

    # Download Output
    result_pil = Image.fromarray(result_img)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    result_pil.save(temp_file.name)

    with open(temp_file.name, "rb") as f:
        st.download_button(
            label="Download Result Image",
            data=f,
            file_name="yolo_output.jpg",
            mime="image/jpeg"
        )

    os.remove(temp_file.name)
