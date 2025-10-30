import streamlit as st
import requests
from PIL import Image
import io

BACKEND_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(page_title="Deepfake Detector", page_icon="🧠")
st.title("🧠 Deepfake Detector (via Flask API)")
st.write("Upload an image to check if it's **real or deepfake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Convert to bytes
        img_bytes = io.BytesIO()
        image_data.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Send to Flask API
        files = {"file": img_bytes}
        response = requests.post(BACKEND_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            label = result["label"]
            confidence = result["confidence"]
            st.markdown(f"### **Prediction:** {label}")
            st.progress(confidence)
            st.markdown(f"**Confidence:** `{confidence:.3f}`")
        else:
            st.error("Error from backend: " + response.text)
