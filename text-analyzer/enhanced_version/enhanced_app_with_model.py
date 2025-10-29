import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from detect_ai import ai_detector  # Import our trained model

st.set_page_config(
    page_title="Deepfake Shield - Enhanced",
    page_icon="🛡️",
    layout="wide"
)

def main():
    st.title("🛡️ Deepfake Shield - AI Text Detector")
    st.markdown("### Powered by Custom-Trained Machine Learning Model")
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        **Features:**
        - 🤖 AI-generated text detection
        - 🎯 Real-time analysis
        - 📊 Confidence scoring
        - 🛡️ Trust score calculation
        
        **Model Info:**
        - Accuracy: 100% (training)
        - Samples: 20 texts
        - Algorithm: Random Forest + TF-IDF
        """)
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text to Analyze")
        text_input = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Enter news article, social media post, or any text...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.subheader("Quick Examples")
        examples = {
            "🤖 AI Text": "In conclusion, the remarkable advancements have substantially transformed various aspects of modern society through innovative methodologies.",
            "👤 Human Text": "I went to the store to buy some groceries and then met my friends for coffee in the afternoon.",
            "🎯 Mixed Text": "The research findings indicate significant improvements, but we need more testing to be completely sure about the results."
        }
        
        for name, example in examples.items():
            if st.button(name):
                st.session_state.text_input = example
    
    # Set text from session state
    if 'text_input' in st.session_state:
        text_input = st.session_state.text_input
    
    # Analyze button
    if st.button("🔍 Analyze with AI Model", type="primary") and text_input:
        with st.spinner("Analyzing with trained AI model..."):
            results = ai_detector.analyze_text(text_input)
        
        display_results(results, text_input)
    elif not text_input:
        st.warning("Please enter some text to analyze.")

def display_results(results, original_text):
    """Display analysis results"""
    
    st.markdown("---")
    
    # Main metrics
    trust_score = results['trust_score']
    ai_prob = results['ai_probability']
    
    # Create columns for layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Trust score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=trust_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Trust Score"},
            gauge={
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
    
    # Status message
    st.subheader(results['status'])
    
    # Detailed metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "AI Probability", 
            f"{ai_prob}%",
            delta="High AI risk" if ai_prob > 70 else "Moderate AI risk" if ai_prob > 40 else "Low AI risk"
        )
    
    with col2:
        st.metric(
            "Human Probability", 
            f"{results['human_probability']}%",
            delta="Likely human" if results['human_probability'] > 70 else "Uncertain"
        )
    
    with col3:
        st.metric(
            "Confidence Level", 
            results['confidence'].title(),
            delta="High" if results['confidence'] == 'high' else "Medium" if results['confidence'] == 'medium' else "Low"
        )
    
    # Probability visualization
    st.subheader("📊 Probability Distribution")
    prob_data = {
        'Type': ['Human', 'AI'],
        'Probability': [results['human_probability'], results['ai_probability']]
    }
    df_prob = pd.DataFrame(prob_data)
    
    fig = px.bar(
        df_prob, 
        x='Type', 
        y='Probability',
        color='Type',
        color_discrete_map={'Human': '#00CC96', 'AI': '#EF553B'},
        text='Probability'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    with st.expander("💡 Interpretation Guide"):
        if ai_prob >= 80:
            st.error("""
            **High AI Probability (80-100%):**
            - This text shows strong indicators of AI generation
            - Common patterns: formal structure, repetitive phrasing, perfect grammar
            - Verify from additional sources if important
            """)
        elif ai_prob >= 60:
            st.warning("""
            **Medium AI Probability (60-79%):**
            - Mixed signals detected
            - Text may contain AI-generated elements
            - Exercise caution and cross-reference
            """)
        else:
            st.success("""
            **Low AI Probability (0-59%):**
            - Text appears human-written
            - Natural language patterns detected
            - Normal sentence variation and informal structures
            """)
    
    # Original text
    with st.expander("📝 Original Text"):
        st.text_area("Analyzed Text", original_text, height=150, key="result_display")

if __name__ == "__main__":
    main()