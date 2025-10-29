import tensorflow as tf
import os
import sys

MODEL_PATH = "models/deepfake_detector.h5"

def predict_image(image_path):
    """Predict if an image is real or fake"""
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found! Train first with: python train.py")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Show results
    if prediction > 0.5:
        print(f"🤖 FAKE (confidence: {prediction:.2%})")
    else:
        print(f"✅ REAL (confidence: {1-prediction:.2%})")
    
    return prediction

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py my_image.jpg")