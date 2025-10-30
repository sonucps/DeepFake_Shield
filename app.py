from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os


app = Flask(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Load model once
MODEL_PATH = "../models/deepfake_detector_gpu.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    x = np.expand_dims(np.array(img) / 255.0, axis=0)
    return x

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    x = preprocess_image(img_bytes)
    pred = model.predict(x)[0][0]
    label = "Deepfake" if pred > 0.5 else "Real"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return jsonify({
        "label": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
