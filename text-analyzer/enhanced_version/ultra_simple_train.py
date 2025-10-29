import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

print("🤖 ULTRA SIMPLE AI Text Detector Training")
print("=" * 50)

# Training data directly in code
human_texts = [
    "The weather is nice today for a walk in the park.",
    "I went to the store to buy groceries for dinner.",
    "My favorite hobby is reading books about history.",
    "The team worked hard to complete the project.",
    "Students need to study regularly for exams.",
    "We watched a great movie with amazing acting last night.",
    "Learning new skills takes consistent practice over time.",
    "The software update fixed several important security issues.",
    "People enjoy different foods based on their cultural background.",
    "Researchers found interesting results in their climate study."
]

ai_texts = [
    "In conclusion, the remarkable advancements have substantially transformed various aspects.",
    "Moreover, it is important to note that these developments offer unprecedented opportunities.",
    "Furthermore, the integration of sophisticated algorithms demonstrates exceptional capability.",
    "Additionally, research has shown that these methodologies provide significant advantages.",
    "Consequently, organizations leveraging these innovations achieve substantial benefits.",
    "The implementation of these strategies has resulted in considerable improvements across domains.",
    "Notably, the convergence of these technologies has created synergistic effects.",
    "Accordingly, the adoption of these approaches has led to measurable enhancements.",
    "Significantly, these innovations have fundamentally altered traditional paradigms.",
    "Thus, the utilization of these frameworks has enabled unprecedented scalability."
]

# Prepare data
texts = human_texts + ai_texts
labels = [0] * len(human_texts) + [1] * len(ai_texts)

print(f"Training on {len(texts)} samples:")
print(f"- Human texts: {len(human_texts)}")
print(f"- AI texts: {len(ai_texts)}")

# Train model
print("🔧 Training AI detection model...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
X = vectorizer.fit_transform(texts)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, labels)

# Test
predictions = model.predict(X)
print("\n📊 Training Results:")
print(classification_report(labels, predictions, target_names=['Human', 'AI']))

# Calculate accuracy
accuracy = (predictions == labels).mean() * 100
print(f"🎯 Overall Accuracy: {accuracy:.1f}%")

# Save model
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'ai_model.pkl')

print("\n✅ Model saved successfully!")
print("📁 Files created:")
print("   - vectorizer.pkl (text processing)")
print("   - ai_model.pkl (AI detection model)")

print("\n🎉 Training completed! Ready to detect AI-generated text.")
print("\n💡 Next: Create a detection script to use this model!")