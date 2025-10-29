import numpy as np
import re
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class TextAuthenticityAnalyzer:
    def __init__(self):
        print("Loading AI detection model...")
        # Use a model that's available and works offline
        self.ai_detector = pipeline(
            "text-classification",
            model="distilbert-base-uncased",  # Fallback model
            return_all_scores=True
        )
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        print("Models loaded successfully!")
    
    def calculate_text_metrics(self, text):
        """Calculate various text complexity and style metrics"""
        metrics = {}
        
        # Basic text statistics
        metrics['word_count'] = len(text.split())
        metrics['char_count'] = len(text)
        metrics['sentence_count'] = textstat.sentence_count(text)
        metrics['avg_sentence_length'] = metrics['word_count'] / max(metrics['sentence_count'], 1)
        
        # Readability scores
        metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        
        return metrics
    
    def detect_ai_patterns(self, text):
        """Analyze text for AI-generated patterns using heuristic methods"""
        text_lower = text.lower()
        
        # Heuristic 1: Check for common AI phrases
        ai_phrases = [
            'in conclusion', 'moreover', 'furthermore', 'additionally',
            'it is important to note', 'as a result', 'therefore'
        ]
        ai_phrase_count = sum(1 for phrase in ai_phrases if phrase in text_lower)
        
        # Heuristic 2: Sentence length variation
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 0]
        sentence_lengths = [len(s.split()) for s in sentences]
        
        if len(sentence_lengths) > 1:
            sentence_variance = np.std(sentence_lengths) / np.mean(sentence_lengths)
        else:
            sentence_variance = 0
        
        # Heuristic 3: Word repetition
        words = text_lower.split()
        unique_word_ratio = len(set(words)) / len(words) if words else 0
        
        # Heuristic 4: Passive voice detection (simple)
        passive_indicators = ['is', 'was', 'were', 'be', 'been'] + ['by']
        passive_count = sum(1 for word in words if word in passive_indicators)
        
        # Calculate AI probability based on heuristics
        ai_score = 0
        
        # More AI phrases = higher AI probability
        ai_score += min(ai_phrase_count * 0.1, 0.3)
        
        # Lower sentence variance = higher AI probability
        if sentence_variance < 0.3:
            ai_score += 0.2
        elif sentence_variance < 0.5:
            ai_score += 0.1
            
        # Lower unique word ratio = higher AI probability  
        if unique_word_ratio < 0.6:
            ai_score += 0.2
        elif unique_word_ratio < 0.7:
            ai_score += 0.1
            
        # More passive voice = higher AI probability
        ai_score += min(passive_count * 0.02, 0.2)
        
        ai_probability = min(ai_score, 0.9)  # Cap at 90%
        
        return {
            'ai_probability': ai_probability,
            'human_probability': 1 - ai_probability,
            'ai_phrases_detected': ai_phrase_count,
            'sentence_variance': sentence_variance,
            'unique_word_ratio': unique_word_ratio,
            'passive_voice_indicators': passive_count
        }
    
    def analyze_sentiment_patterns(self, text):
        """Analyze sentiment and emotional patterns"""
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Detect extreme sentiment (common in fake news)
        extreme_sentiment = abs(sentiment_scores['compound']) > 0.7
        
        # Detect emotional language patterns
        emotional_words = ['shocking', 'unbelievable', 'astounding', 'devastating', 'horrifying', 'amazing', 'incredible']
        emotional_count = sum(1 for word in emotional_words if word in text.lower())
        
        return {
            'sentiment_scores': sentiment_scores,
            'extreme_sentiment': extreme_sentiment,
            'emotional_word_count': emotional_count,
            'is_emotional': emotional_count > 2
        }
    
    def detect_clickbait_patterns(self, text):
        """Detect clickbait and sensationalist patterns"""
        clickbait_indicators = 0
        
        # Common clickbait phrases
        clickbait_phrases = [
            'you won\'t believe', 'shocked the world', 'what happened next',
            'going viral', 'the truth about', 'they don\'t want you to know',
            'this will blow your mind', 'secret revealed', 'what she did next'
        ]
        
        text_lower = text.lower()
        for phrase in clickbait_phrases:
            if phrase in text_lower:
                clickbait_indicators += 1
        
        # Excessive capitalization
        words = text.split()
        capitalized_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        excessive_caps = capitalized_words > len(words) * 0.1  # More than 10% caps
        
        # Excessive punctuation
        excl_marks = text.count('!')
        quest_marks = text.count('?')
        excessive_punctuation = (excl_marks + quest_marks) > len(words) * 0.05
        
        return {
            'clickbait_score': min(clickbait_indicators / 3, 1.0),  # Normalize to 0-1
            'excessive_capitalization': excessive_caps,
            'excessive_punctuation': excessive_punctuation,
            'clickbait_phrases_found': clickbait_indicators
        }
    
    def comprehensive_analysis(self, text):
        """Run comprehensive text authenticity analysis"""
        if len(text.strip()) < 10:
            return {"error": "Text too short for analysis"}
        
        print("Starting text analysis...")
        
        results = {
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'text_length': len(text)
        }
        
        # Run all analyses
        print("Running AI pattern detection...")
        results['ai_detection'] = self.detect_ai_patterns(text)
        
        print("Running sentiment analysis...")
        results['sentiment_analysis'] = self.analyze_sentiment_patterns(text)
        
        print("Running clickbait analysis...")
        results['clickbait_analysis'] = self.detect_clickbait_patterns(text)
        
        print("Running linguistic analysis...")
        results['linguistic_analysis'] = self.calculate_text_metrics(text)
        
        # Calculate overall trust score
        print("Calculating trust score...")
        trust_score = self.calculate_trust_score(results)
        results['overall_trust_score'] = trust_score
        
        print("Analysis complete!")
        return results
    
    def calculate_trust_score(self, analysis_results):
        """Calculate overall trust score from all analyses"""
        base_score = 50  # Start from neutral
        
        # AI detection impact (40% weight)
        ai_prob = analysis_results['ai_detection']['ai_probability']
        base_score -= (ai_prob - 0.5) * 40  # Adjust based on AI probability
        
        # Clickbait impact (20% weight)
        clickbait_score = analysis_results['clickbait_analysis']['clickbait_score']
        base_score -= clickbait_score * 20
        
        # Sentiment impact (15% weight)
        if analysis_results['sentiment_analysis']['extreme_sentiment']:
            base_score -= 10
        if analysis_results['sentiment_analysis']['is_emotional']:
            base_score -= 5
        
        # Ensure score stays within 0-100
        final_score = max(0, min(100, base_score))
        
        return round(final_score, 2)

# Global instance
text_analyzer = TextAuthenticityAnalyzer()