import joblib
import pandas as pd
import numpy as np
import re
from collections import Counter

class EnhancedAITextDetector:
    def __init__(self):
        print("🛡️ Loading Enhanced Deepfake Shield AI Detector...")
        try:
            self.vectorizer = joblib.load('vectorizer.pkl')
            self.model = joblib.load('ai_model.pkl')
            print("✅ AI detection model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Enhanced pattern detection
        self.ai_indicators = [
            'in conclusion', 'moreover', 'furthermore', 'additionally',
            'as a result', 'therefore', 'consequently', 'however',
            'nevertheless', 'thus', 'hence', 'accordingly',
            'it is important to note', 'it should be noted that',
            'this suggests that', 'this indicates that',
            'research has shown', 'studies have found'
        ]
        
        self.quality_indicators = {
            'min_meaningful_words': 3,
            'max_gibberish_ratio': 0.7
        }

    def is_gibberish(self, text):
        """Check if text is mostly gibberish"""
        words = text.split()
        if len(words) < 3:
            return True
            
        # Check for meaningful word patterns
        meaningful_pattern = re.compile(r'^[a-zA-Z]{3,}$')
        meaningful_words = sum(1 for word in words if meaningful_pattern.match(word))
        
        return meaningful_words / len(words) < 0.5

    def calculate_text_quality(self, text):
        """Calculate text quality score"""
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Basic metrics
        word_count = len(words)
        sentence_count = len(sentences)
        
        if word_count == 0 or sentence_count == 0:
            return 0
        
        # Vocabulary richness
        unique_words = len(set(words))
        vocab_richness = unique_words / word_count
        
        # Sentence structure
        avg_sentence_length = word_count / sentence_count
        
        # Quality score (0-100)
        quality_score = (
            min(word_count, 20) * 2 +  # Word count contribution
            vocab_richness * 30 +      # Vocabulary richness
            min(avg_sentence_length, 15) * 2  # Sentence structure
        )
        
        return min(quality_score, 100)

    def analyze_text(self, text):
        """Enhanced text analysis with realistic performance metrics"""
        text = text.strip()
        
        # Quality validation
        if len(text) < 10:
            return {
                "error": "Text too short",
                "message": "Please enter at least 10 characters for meaningful analysis."
            }
        
        if self.is_gibberish(text):
            return {
                "error": "Low quality text", 
                "message": "Text appears to be gibberish or low quality. Please enter meaningful text.",
                "quality_score": self.calculate_text_quality(text)
            }
        
        # Get base prediction
        text_features = self.vectorizer.transform([text])
        prediction = self.model.predict(text_features)[0]
        probabilities = self.model.predict_proba(text_features)[0]
        
        # Add realistic noise and uncertainty
        base_human_prob = probabilities[0] * 100
        base_ai_prob = probabilities[1] * 100
        
        # Adjust based on text quality and patterns
        quality_score = self.calculate_text_quality(text)
        ai_pattern_count = sum(1 for phrase in self.ai_indicators if phrase in text.lower())
        
        # Realistic adjustments (not perfect)
        if ai_pattern_count > 0:
            base_ai_prob = min(base_ai_prob + (ai_pattern_count * 3), 95)
        
        if quality_score < 40:
            # Low quality texts are harder to classify
            base_ai_prob = max(base_ai_prob - 10, 10)
        
        # Normalize probabilities
        total = base_human_prob + base_ai_prob
        human_prob = (base_human_prob / total) * 100
        ai_prob = (base_ai_prob / total) * 100
        
        # Determine confidence based on probability gap
        prob_gap = abs(human_prob - ai_prob)
        if prob_gap > 40:
            confidence = "high"
        elif prob_gap > 20:
            confidence = "medium" 
        else:
            confidence = "low"
        
        # Realistic status determination
        if ai_prob >= 75:
            status = "🔴 High AI Probability"
            trust_score = max(20, 100 - ai_prob)  # Never 100% certain
        elif ai_prob >= 55:
            status = "🟡 Moderate AI Probability" 
            trust_score = 100 - ai_prob
        else:
            status = "🟢 Likely Human-Written"
            trust_score = min(95, human_prob)  # Never 100% certain
        
        return {
            'prediction': 'ai' if prediction == 1 else 'human',
            'human_probability': round(max(5, min(95, human_prob)), 1),  # Never 0% or 100%
            'ai_probability': round(max(5, min(95, ai_prob)), 1),       # Never 0% or 100%
            'confidence': confidence,
            'status': status,
            'trust_score': round(trust_score, 1),
            'quality_score': round(quality_score, 1),
            'ai_patterns_detected': ai_pattern_count,
            'word_count': len(text.split()),
            'text_quality': 'High' if quality_score > 70 else 'Medium' if quality_score > 40 else 'Low'
        }

# Create global instance with the correct name
enhanced_detector = EnhancedAITextDetector()

def main():
    print("🔍 Enhanced Deepfake Shield - AI Text Detection")
    print("=" * 60)
    
    # Test examples
    test_texts = [
        "The weather is really nice today for a walk outside with my friends.",
        "In conclusion, the substantial advancements have transformed multiple aspects considerably across various domains.",
        "v;ldks'vlks'v;",  # Gibberish
        "Moreover, it is imperative to acknowledge that these innovations facilitate unprecedented operational efficiencies.",
        "Hello world test",  # Too short
    ]
    
    print("\n🧪 Running Enhanced Test Analysis:")
    print("-" * 45)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        result = enhanced_detector.analyze_text(text)
        
        if 'error' in result:
            print(f"   ❌ {result['error']}: {result['message']}")
        else:
            print(f"   🎯 {result['status']}")
            print(f"   🤖 AI Probability: {result['ai_probability']}%")
            print(f"   👤 Human Probability: {result['human_probability']}%")
            print(f"   🛡️ Trust Score: {result['trust_score']}%")
            print(f"   📊 Quality: {result['text_quality']} ({result['quality_score']}/100)")
            print(f"   🔍 AI Patterns: {result['ai_patterns_detected']}")
            print(f"   💪 Confidence: {result['confidence'].upper()}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("💬 Enhanced Interactive Detection Mode")
    print("Type your text to analyze (or 'quit' to exit)")
    
    while True:
        user_text = input("\n📝 Enter text: ").strip()
        
        if user_text.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using Deepfake Shield!")
            break
            
        result = enhanced_detector.analyze_text(user_text)
        
        if 'error' in result:
            print(f"❌ {result['error'].upper()}")
            print(f"💡 {result['message']}")
            if 'quality_score' in result:
                print(f"📊 Quality Score: {result['quality_score']}/100")
            continue
        
        print(f"\n🎯 ENHANCED ANALYSIS RESULTS:")
        print(f"   {result['status']}")
        print(f"   🤖 AI Probability: {result['ai_probability']}%")
        print(f"   👤 Human Probability: {result['human_probability']}%")
        print(f"   🛡️ Trust Score: {result['trust_score']}%")
        print(f"   📊 Text Quality: {result['text_quality']} ({result['quality_score']}/100)")
        print(f"   🔍 AI Patterns Detected: {result['ai_patterns_detected']}")
        print(f"   📝 Word Count: {result['word_count']}")
        print(f"   💪 Confidence Level: {result['confidence'].upper()}")
        
        # Detailed interpretation
        print(f"\n💡 INTERPRETATION:")
        if result['ai_probability'] >= 75:
            print("   - Strong indicators of AI-generated content")
            print("   - Verify from reliable sources if important")
        elif result['ai_probability'] >= 55:
            print("   - Mixed signals detected")
            print("   - Exercise caution with this content")
        else:
            print("   - Content appears human-written")
            print("   - Natural language patterns detected")
        
        if result['quality_score'] < 40:
            print("   - Text quality is low, analysis confidence reduced")

if __name__ == "__main__":
    main()