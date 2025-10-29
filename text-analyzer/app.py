import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from text_analyzer import text_analyzer
import time

# Configure the page
st.set_page_config(
    page_title="Deepfake Shield - Text Analyzer",
    page_icon="🛡️",
    layout="wide"
)

def main():
    st.title("🛡️ Deepfake Shield - Text Authenticity Analyzer")
    st.markdown("### Detect AI-Generated Text and Misinformation Patterns")
    
    # Sidebar for examples and info
    with st.sidebar:
        st.header("Quick Examples")
        
        example_texts = {
            "AI-Generated News": """In a remarkable development that has captivated the technological community, researchers have unveiled a groundbreaking advancement in artificial intelligence. This innovative approach promises to revolutionize how machines process natural language, offering unprecedented capabilities in understanding and generating human-like text. The implications for various industries are substantial, with potential applications ranging from customer service to creative writing. Moreover, the technology demonstrates exceptional proficiency in analyzing complex datasets and generating insightful conclusions that can drive strategic decision-making processes across multiple sectors.""",
            
            "Human-Written News": """Researchers at Stanford University announced today that they've made progress on a new AI model. The team, led by Dr. Sarah Chen, said the technology shows promise but still has limitations. "It's a step forward, but we're not claiming any revolution," Chen told reporters. The paper will be published next month in the Journal of AI Research. Some experts are skeptical about the claims, while others see potential in the approach. The system still makes errors in understanding context, according to early tests.""",
            
            "Clickbait/Sensational": """SHOCKING discovery will CHANGE everything! You WON'T BELIEVE what researchers found! This UNBELIEVABLE truth has been HIDDEN from the public! What happens next will BLOW YOUR MIND! Scientists are SPEECHLESS after this groundbreaking finding that will make you question EVERYTHING you know!""",
            
            "Normal Social Media Post": """Just saw an interesting talk about AI ethics at the conference today. The speaker made some good points about how we need to be careful with this technology. It's powerful but also has risks that we should think about more carefully.""",
        }
        
        selected_example = st.selectbox("Load Example:", list(example_texts.keys()))
        if st.button("Load Example Text"):
            st.session_state.input_text = example_texts[selected_example]
        
        st.markdown("---")
        st.info("""
        **What this analyzes:**
        - 🤖 AI-generated text patterns
        - 🎯 Clickbait and sensationalism  
        - 😊 Sentiment manipulation
        - 📊 Linguistic complexity
        - ✍️ Writing style consistency
        """)

    # Main input area
    st.subheader("Enter Text to Analyze")
    input_text = st.text_area(
        "Paste your text here:",
        height=200,
        value=st.session_state.get('input_text', ''),
        placeholder="Paste news article, social media post, or any text you want to analyze...",
        label_visibility="collapsed"
    )
    
    # Analysis button
    if st.button("🔍 Analyze Text Authenticity", type="primary", use_container_width=True) and input_text.strip():
        with st.spinner("Analyzing text patterns with AI..."):
            # Add a small delay to show the spinner
            time.sleep(1)
            results = text_analyzer.comprehensive_analysis(input_text)
        
        if "error" in results:
            st.error(f"Analysis error: {results['error']}")
            return
        
        display_results(results, input_text)
    elif not input_text.strip():
        st.warning("Please enter some text to analyze.")

def display_results(results, original_text):
    """Display the analysis results in an organized way"""
    
    # Overall trust score - Big display
    trust_score = results['overall_trust_score']
    
    st.markdown("---")
    
    # Create columns for the main score
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create gauge chart for trust score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = trust_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Trust Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Status message based on score
    if trust_score >= 70:
        st.success("✅ This text shows strong indicators of human authorship and authenticity.")
    elif trust_score >= 40:
        st.warning("⚠️ This text shows mixed signals. Exercise caution and verify from other sources.")
    else:
        st.error("🚨 This text shows strong indicators of AI generation or manipulation. Verify carefully!")
    
    # Detailed analysis in expandable sections
    with st.expander("🤖 AI Detection Analysis", expanded=True):
        ai_data = results['ai_detection']
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("AI Probability", f"{ai_data['ai_probability']*100:.1f}%")
            st.metric("Human Probability", f"{ai_data['human_probability']*100:.1f}%")
        
        with col2:
            # AI probability bar
            ai_percent = ai_data['ai_probability'] * 100
            st.progress(ai_percent / 100, text=f"AI Likelihood: {ai_percent:.1f}%")
            
            if ai_data['ai_probability'] > 0.7:
                st.error("High probability of AI-generated content")
            elif ai_data['ai_probability'] > 0.4:
                st.warning("Moderate signs of AI generation")
            else:
                st.success("Strong indicators of human authorship")
        
        # Show AI detection details
        st.write("**Detection Details:**")
        st.write(f"- AI phrases detected: {ai_data['ai_phrases_detected']}")
        st.write(f"- Sentence variance: {ai_data['sentence_variance']:.3f}")
        st.write(f"- Unique word ratio: {ai_data['unique_word_ratio']:.3f}")
        st.write(f"- Passive voice indicators: {ai_data['passive_voice_indicators']}")
    
    with st.expander("📊 Linguistic Analysis"):
        ling_data = results['linguistic_analysis']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", ling_data['word_count'])
            st.metric("Sentence Count", ling_data['sentence_count'])
        with col2:
            st.metric("Readability Score", f"{ling_data['flesch_reading_ease']:.1f}")
            st.metric("Grade Level", f"{ling_data['flesch_kincaid_grade']:.1f}")
        with col3:
            st.metric("Avg Sentence Length", f"{ling_data['avg_sentence_length']:.1f}")
            
        # Readability interpretation
        readability = ling_data['flesch_reading_ease']
        if readability > 80:
            st.info("📚 Very easy to read (common in AI-generated text)")
        elif readability > 60:
            st.success("📖 Standard readability level")
        else:
            st.warning("📕 Complex text (more likely human-written)")
    
    with st.expander("🎯 Clickbait & Sensationalism Analysis"):
        click_data = results['clickbait_analysis']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Clickbait Score", f"{click_data['clickbait_score']*100:.1f}%")
            st.metric("Sensational Phrases", click_data['clickbait_phrases_found'])
        
        with col2:
            if click_data['excessive_capitalization']:
                st.error("❌ Excessive capitalization detected")
            else:
                st.success("✅ Normal capitalization patterns")
                
            if click_data['excessive_punctuation']:
                st.error("❌ Excessive punctuation detected")
            else:
                st.success("✅ Normal punctuation patterns")
        
        if click_data['clickbait_score'] > 0.5:
            st.error("High clickbait indicators detected - content may be sensationalized")
        elif click_data['clickbait_score'] > 0.2:
            st.warning("Moderate clickbait indicators detected")
        else:
            st.success("Low clickbait indicators - content appears straightforward")
    
    with st.expander("😊 Sentiment Analysis"):
        sent_data = results['sentiment_analysis']
        
        sentiment_scores = sent_data['sentiment_scores']
        df_sentiment = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative', 'Neutral', 'Compound'],
            'Score': [sentiment_scores['pos'], sentiment_scores['neg'], 
                     sentiment_scores['neu'], sentiment_scores['compound']]
        })
        
        fig = px.bar(df_sentiment, x='Sentiment', y='Score', 
                    title="Sentiment Distribution",
                    color='Score', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        if sent_data['extreme_sentiment']:
            st.warning("⚠️ Extreme sentiment detected - common in manipulated content")
        else:
            st.success("✅ Balanced sentiment detected")
            
        if sent_data['is_emotional']:
            st.warning("⚠️ High emotional language - may indicate sensationalism")
        else:
            st.success("✅ Neutral emotional tone")
    
    with st.expander("📝 Original Text"):
        st.text_area("Analyzed Text", original_text, height=200, key="original_text_display")

if __name__ == "__main__":
    main()