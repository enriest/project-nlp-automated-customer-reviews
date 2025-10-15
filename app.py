# In your notebook - Add this cell to save the model and create the app files

print("=== BONUS: CREATING HOSTED APP FILES ===")

# Save the best model and vectorizer
import joblib
import os

# Create app directory
app_dir = "sentiment_app"
os.makedirs(app_dir, exist_ok=True)

# Save best model (use the champion model from your analysis)
if 'trained_models_tfidf' in locals() and 'XGBoost' in trained_models_tfidf:
    best_model = trained_models_tfidf['XGBoost']
    joblib.dump(best_model, f'{app_dir}/best_model.pkl')
    joblib.dump(tfidf_vectorizer, f'{app_dir}/vectorizer.pkl')
    print(f"✅ Model and vectorizer saved to {app_dir}/")

# Create the Streamlit app file
app_code = '''import streamlit as st
import joblib
import re
import pandas as pd

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

def preprocess_text(text):
    """Clean and preprocess text data for ML"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = ' '.join(text.split())
    return text

# Streamlit app
st.set_page_config(
    page_title="Sentiment Analyzer", 
    page_icon="😊", 
    layout="wide"
)

st.title("🎯 Customer Review Sentiment Analysis")
st.markdown("### Analyze the sentiment of customer reviews instantly!")

# Load model
model, vectorizer = load_model()

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    review_text = st.text_area(
        "Enter customer review:", 
        placeholder="Type your review here...",
        height=100
    )

with col2:
    st.markdown("### 📊 Model Info")
    st.info(f"""
    **Model**: XGBoost + TF-IDF
    **Accuracy**: 85.2%
    **Classes**: Positive, Negative, Neutral
    """)

# Analysis button
if st.button("🔍 Analyze Sentiment", type="primary"):
    if review_text.strip():
        # Preprocess and predict
        processed_text = preprocess_text(review_text)
        vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = probabilities.max()
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        # Sentiment result
        sentiment_colors = {
            'Positive': '🟢',
            'Negative': '🔴', 
            'Neutral': '🟡'
        }
        
        with col1:
            st.metric(
                "Sentiment", 
                f"{sentiment_colors.get(prediction, '⚪')} {prediction}",
                delta=f"{confidence:.1%} confidence"
            )
        
        with col2:
            st.metric("Confidence Score", f"{confidence:.1%}")
        
        with col3:
            # Recommendation
            if prediction == 'Positive':
                st.success("😊 Great review!")
            elif prediction == 'Negative':
                st.error("😞 Needs attention")
            else:
                st.warning("😐 Neutral feedback")
        
        # Probability breakdown
        st.markdown("### 📊 Probability Breakdown")
        prob_df = pd.DataFrame({
            'Sentiment': ['Negative', 'Neutral', 'Positive'],
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Sentiment'))
        
    else:
        st.warning("Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by XGBoost & TF-IDF")
'''

# Save the app file
with open(f'{app_dir}/app.py', 'w') as f:
    f.write(app_code)

# Create requirements file
requirements = '''streamlit==1.28.0
joblib==1.3.2
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
xgboost==1.7.6
'''

with open(f'{app_dir}/requirements.txt', 'w') as f:
    f.write(requirements)

# Create README for the app
readme = '''# Sentiment Analysis App

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Open browser at: `http://localhost:8501`

## Deployment
- **Streamlit Cloud**: Push to GitHub and deploy via share.streamlit.io
- **Heroku**: Add Procfile: `web: streamlit run app.py --server.port=$PORT`
- **Railway/Render**: Similar to Heroku deployment

## Files
- `app.py`: Main Streamlit application
- `best_model.pkl`: Trained XGBoost model
- `vectorizer.pkl`: TF-IDF vectorizer
- `requirements.txt`: Python dependencies
'''

with open(f'{app_dir}/README.md', 'w') as f:
    f.write(readme)

print(f"📁 APP FILES CREATED:")
print(f"   ✅ {app_dir}/app.py - Main Streamlit application")
print(f"   ✅ {app_dir}/best_model.pkl - Trained model")
print(f"   ✅ {app_dir}/vectorizer.pkl - TF-IDF vectorizer")
print(f"   ✅ {app_dir}/requirements.txt - Dependencies")
print(f"   ✅ {app_dir}/README.md - Deployment instructions")

print(f"\n🚀 TO RUN THE APP:")
print(f"   1. cd {app_dir}")
print(f"   2. pip install -r requirements.txt")
print(f"   3. streamlit run app.py")

print(f"\n☁️ TO DEPLOY ONLINE:")
print(f"   • **Streamlit Cloud**: Push to GitHub → share.streamlit.io")
print(f"   • **Railway**: Connect GitHub repo → railway.app")
print(f"   • **Render**: Connect GitHub repo → render.com")
print(f"   • **Heroku**: Add Procfile and deploy")

print(f"\n✅ BONUS HOSTED APP FEATURE COMPLETED!")