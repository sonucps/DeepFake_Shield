import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np

# Configuration
DATA_DIR = "data"
REAL_DIR = os.path.join(DATA_DIR, "real")
FAKE_DIR = os.path.join(DATA_DIR, "fake")
MODEL_PATH = "models/deepfake_detector.h5"
BATCH_SIZE = 32
EPOCHS = 10

def check_data():
    """Check if we have data to train on"""
    if not os.path.exists(REAL_DIR):
        print(f"❌ Real images directory not found: {REAL_DIR}")
        return False
    if not os.path.exists(FAKE_DIR):
        print(f"❌ Fake images directory not found: {FAKE_DIR}")
        return False
    
    real_images = [f for f in os.listdir(REAL_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    fake_images = [f for f in os.listdir(FAKE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"✅ Found {len(real_images)} real images")
    print(f"✅ Found {len(fake_images)} fake images")
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("❌ Need at least one real and one fake image!")
        return False
    
    return True

def create_model():
    """Create a simple CNN model"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("🚀 Starting Deepfake Model Training...")
    
    # Check data
    if not check_data():
        print("\n📝 How to fix:")
        print("1. Create folders: data/real/ and data/fake/")
        print("2. Put real images in data/real/")
        print("3. Put fake images in data/fake/")
        return
    
    # Create data generators
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # 20% for validation
    )
    
    train_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    print(f"🎯 Classes: {train_generator.class_indices}")
    print(f"📊 Training samples: {train_generator.samples}")
    print(f"📊 Validation samples: {val_generator.samples}")
    
    # Create and train model
    model = create_model()
    print("🧠 Model architecture:")
    model.summary()
    
    print("🏋️‍♂️ Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save(MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    
    # Show final accuracy
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"🎉 Training completed!")
    print(f"   Final training accuracy: {final_acc:.2%}")
    print(f"   Final validation accuracy: {final_val_acc:.2%}")

if __name__ == "__main__":
    main()