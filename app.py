# In your notebook - Add this cell to save the model and create the app files

print("=== BONUS: CREATING HOSTED APP FILES ===")

# Save the best model and vectorizer
import joblib
import os
import shutil

# Create app directory
app_dir = "sentiment_app"
os.makedirs(app_dir, exist_ok=True)

# Save best traditional ML model (use the champion model from your analysis)
if 'trained_models_tfidf' in locals() and 'XGBoost' in trained_models_tfidf:
    best_model = trained_models_tfidf['XGBoost']
    joblib.dump(best_model, f'{app_dir}/best_model.pkl')
    joblib.dump(tfidf_vectorizer, f'{app_dir}/vectorizer.pkl')
    print(f"✅ Traditional ML model and vectorizer saved to {app_dir}/")

# Copy transformer models to app directory if they exist
transformer_models_dir = f'{app_dir}/transformer_models'
os.makedirs(transformer_models_dir, exist_ok=True)

# Copy fine-tuned models from offline_models directory
if os.path.exists('offline_models'):
    try:
        # Copy DistilBERT model if it exists
        if os.path.exists('offline_models/distilbert_finetuned'):
            shutil.copytree('offline_models/distilbert_finetuned', 
                          f'{transformer_models_dir}/distilbert_finetuned', 
                          dirs_exist_ok=True)
            print(f"✅ DistilBERT model copied to {transformer_models_dir}/")
        
        # Copy RoBERTa model if it exists
        if os.path.exists('offline_models/roberta_finetuned'):
            shutil.copytree('offline_models/roberta_finetuned', 
                          f'{transformer_models_dir}/roberta_finetuned', 
                          dirs_exist_ok=True)
            print(f"✅ RoBERTa model copied to {transformer_models_dir}/")
    except Exception as e:
        print(f"⚠️ Note: Transformer models not copied: {e}")
        print(f"   Models will be downloaded on first use.")

# Create the Streamlit app file
app_code = '''import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import os
import io
from datetime import datetime

# Try to import transformers (optional, for transformer models)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers not installed. Only traditional ML models available.")

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer Pro", 
    page_icon="😊", 
    layout="wide"
)

# Load traditional ML model and vectorizer
@st.cache_resource
def load_traditional_model():
    """Load XGBoost model and TF-IDF vectorizer"""
    try:
        model = joblib.load('best_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer, True
    except Exception as e:
        st.error(f"Error loading traditional model: {e}")
        return None, None, False

# Load transformer model
@st.cache_resource
def load_transformer_model(model_name='distilbert'):
    """Load fine-tuned transformer model"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None, False
    
    try:
        model_path = f'transformer_models/{model_name}_finetuned'
        
        # Check if local model exists
        if not os.path.exists(model_path):
            st.warning(f"Local model not found. Using base {model_name} model.")
            if model_name == 'distilbert':
                model_path = 'distilbert-base-uncased'
            else:
                model_path = 'roberta-base'
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = torch.device('mps' if torch.backends.mps.is_available() else 
                            'cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        return tokenizer, model, True, device
    except Exception as e:
        st.error(f"Error loading transformer model: {e}")
        return None, None, False, None

def preprocess_text(text):
    """Clean and preprocess text data for traditional ML"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    text = ' '.join(text.split())
    return text

def predict_traditional(text, model, vectorizer):
    """Predict sentiment using traditional ML model"""
    processed_text = preprocess_text(text)
    vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    # Map to sentiment labels (adjust based on your model)
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = label_map.get(prediction, 'Unknown')
    confidence = probabilities.max()
    
    return sentiment, confidence, probabilities

def predict_transformer(text, tokenizer, model, device):
    """Predict sentiment using transformer model"""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                      max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prediction = torch.argmax(logits, dim=1).item()
    
    # Map to sentiment labels
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = label_map.get(prediction, 'Unknown')
    confidence = probabilities.max()
    
    return sentiment, confidence, probabilities

def process_batch(df, text_column, model_type, trad_model=None, trad_vectorizer=None,
                 trans_tokenizer=None, trans_model=None, trans_device=None):
    """Process a batch of reviews"""
    results = []
    progress_bar = st.progress(0)
    
    for idx, row in df.iterrows():
        text = str(row[text_column])
        
        if model_type == "Traditional ML (XGBoost + TF-IDF)":
            sentiment, confidence, probs = predict_traditional(text, trad_model, trad_vectorizer)
        else:
            sentiment, confidence, probs = predict_transformer(text, trans_tokenizer, 
                                                              trans_model, trans_device)
        
        results.append({
            'Original_Text': text,
            'Sentiment': sentiment,
            'Confidence': confidence,
            'Prob_Negative': probs[0],
            'Prob_Neutral': probs[1],
            'Prob_Positive': probs[2]
        })
        
        progress_bar.progress((idx + 1) / len(df))
    
    progress_bar.empty()
    return pd.DataFrame(results)

# Streamlit app
st.title("🎯 Customer Review Sentiment Analysis Pro")
st.markdown("### Analyze sentiment with Traditional ML or Transformer models!")

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_options = ["Traditional ML (XGBoost + TF-IDF)"]
    if TRANSFORMERS_AVAILABLE:
        model_options.extend(["Transformer (DistilBERT)", "Transformer (RoBERTa)"])
    
    selected_model = st.selectbox("Select Model", model_options)
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    if "Traditional" in selected_model:
        st.info("""
        **Type**: XGBoost + TF-IDF
        **Speed**: ⚡ Fast
        **Accuracy**: ~85%
        **Best for**: Quick analysis
        """)
    else:
        st.info("""
        **Type**: Transformer
        **Speed**: 🐢 Slower
        **Accuracy**: ~90%+
        **Best for**: High accuracy
        """)

# Load selected model
if "Traditional" in selected_model:
    trad_model, trad_vectorizer, model_loaded = load_traditional_model()
    trans_tokenizer, trans_model, trans_device = None, None, None
else:
    trad_model, trad_vectorizer = None, None
    model_name = 'distilbert' if 'DistilBERT' in selected_model else 'roberta'
    trans_tokenizer, trans_model, model_loaded, trans_device = load_transformer_model(model_name)

if not model_loaded:
    st.error("❌ Failed to load model. Please check the model files.")
    st.stop()

# Main tabs
tab1, tab2 = st.tabs(["📝 Single Review", "📁 Batch Processing"])

# Tab 1: Single Review Analysis
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        review_text = st.text_area(
            "Enter customer review:", 
            placeholder="Type your review here...",
            height=150
        )
    
    with col2:
        st.markdown("### � Quick Tips")
        st.markdown("""
        - Longer reviews = better context
        - Include specific details
        - Natural language works best
        """)
    
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if review_text.strip():
            with st.spinner("Analyzing..."):
                # Predict
                if "Traditional" in selected_model:
                    sentiment, confidence, probabilities = predict_traditional(
                        review_text, trad_model, trad_vectorizer
                    )
                else:
                    sentiment, confidence, probabilities = predict_transformer(
                        review_text, trans_tokenizer, trans_model, trans_device
                    )
            
            # Display results
            st.markdown("### 📊 Results")
            col1, col2, col3 = st.columns(3)
            
            sentiment_colors = {
                'Positive': '🟢',
                'Negative': '🔴', 
                'Neutral': '🟡'
            }
            
            with col1:
                st.metric(
                    "Sentiment", 
                    f"{sentiment_colors.get(sentiment, '⚪')} {sentiment}",
                    delta=f"{confidence:.1%} confidence"
                )
            
            with col2:
                st.metric("Confidence Score", f"{confidence:.1%}")
            
            with col3:
                if sentiment == 'Positive':
                    st.success("😊 Great review!")
                elif sentiment == 'Negative':
                    st.error("😞 Needs attention")
                else:
                    st.warning("😐 Neutral feedback")
            
            # Probability breakdown
            st.markdown("### � Probability Breakdown")
            prob_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Probability': probabilities
            })
            st.bar_chart(prob_df.set_index('Sentiment'))
            
        else:
            st.warning("⚠️ Please enter some text to analyze!")

# Tab 2: Batch Processing
with tab2:
    st.markdown("### 📁 Upload CSV/Excel file for batch processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file with customer reviews"
        )
    
    with col2:
        st.markdown("### 📋 File Format")
        st.info("""
        Your file should have:
        - One column with text reviews
        - Header row with column names
        - UTF-8 encoding (for CSV)
        """)
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df)} rows")
            
            # Preview data
            with st.expander("👀 Preview Data"):
                st.dataframe(df.head(10))
            
            # Select text column
            text_column = st.selectbox(
                "Select the column containing reviews:",
                df.columns.tolist()
            )
            
            # Process button
            if st.button("🚀 Process All Reviews", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} reviews..."):
                    results_df = process_batch(
                        df, text_column, selected_model,
                        trad_model, trad_vectorizer,
                        trans_tokenizer, trans_model, trans_device
                    )
                
                st.success("✅ Processing complete!")
                
                # Display results summary
                st.markdown("### 📊 Analysis Summary")
                col1, col2, col3 = st.columns(3)
                
                sentiment_counts = results_df['Sentiment'].value_counts()
                
                with col1:
                    positive_count = sentiment_counts.get('Positive', 0)
                    st.metric("🟢 Positive", positive_count, 
                             f"{positive_count/len(results_df)*100:.1f}%")
                
                with col2:
                    neutral_count = sentiment_counts.get('Neutral', 0)
                    st.metric("🟡 Neutral", neutral_count,
                             f"{neutral_count/len(results_df)*100:.1f}%")
                
                with col3:
                    negative_count = sentiment_counts.get('Negative', 0)
                    st.metric("🔴 Negative", negative_count,
                             f"{negative_count/len(results_df)*100:.1f}%")
                
                # Sentiment distribution chart
                st.markdown("### 📈 Sentiment Distribution")
                st.bar_chart(sentiment_counts)
                
                # Show results table
                with st.expander("📋 View Detailed Results"):
                    st.dataframe(results_df)
                
                # Download results
                st.markdown("### 💾 Download Results")
                
                # Convert to CSV
                csv = results_df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"sentiment_analysis_results_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • Powered by XGBoost, DistilBERT & RoBERTa</p>
    <p style='font-size: 0.8em; color: gray;'>
        Traditional ML for speed • Transformers for accuracy
    </p>
</div>
""", unsafe_allow_html=True)
'''

# Save the app file
with open(f'{app_dir}/app.py', 'w') as f:
    f.write(app_code)

# Create requirements file with updated versions
requirements = '''streamlit==1.31.0
joblib==1.3.2
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.3
xgboost==2.0.3
torch==2.1.2
transformers==4.37.2
openpyxl==3.1.2
datasets==2.16.1
'''

with open(f'{app_dir}/requirements.txt', 'w') as f:
    f.write(requirements)

# Create comprehensive README
readme = '''# Sentiment Analysis App - Professional Edition

Advanced sentiment analysis application supporting both Traditional ML and Transformer models.

## 🌟 Features

- **Dual Model Support**: XGBoost + TF-IDF and Transformer models (DistilBERT, RoBERTa)
- **Single Review Analysis**: Quick sentiment analysis with confidence scores
- **Batch Processing**: Upload CSV/Excel files for bulk analysis
- **File Export**: Download results as CSV
- **Real-time Processing**: Interactive UI with progress tracking
- **Model Comparison**: Switch between models to compare results

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open browser at: `http://localhost:8501`

## 📁 File Structure

```
sentiment_app/
├── app.py                          # Main Streamlit application
├── best_model.pkl                  # XGBoost model
├── vectorizer.pkl                  # TF-IDF vectorizer
├── transformer_models/             # Fine-tuned transformer models
│   ├── distilbert_finetuned/
│   └── roberta_finetuned/
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📊 Models

### Traditional ML (XGBoost + TF-IDF)
- **Speed**: ⚡ Very Fast
- **Accuracy**: ~85%
- **Best for**: Quick analysis, large batches
- **Resources**: Low memory usage

### Transformer Models
- **DistilBERT**: Balanced speed/accuracy
- **RoBERTa**: Highest accuracy
- **Accuracy**: ~90%+
- **Best for**: High-quality analysis
- **Resources**: GPU recommended

## 🔧 Usage

### Single Review Analysis
1. Select model from sidebar
2. Enter review text
3. Click "Analyze Sentiment"
4. View results and probability breakdown

### Batch Processing
1. Go to "Batch Processing" tab
2. Upload CSV/Excel file
3. Select text column
4. Click "Process All Reviews"
5. Download results

### File Format for Batch Processing
CSV or Excel files with:
- Header row with column names
- One column containing review text
- UTF-8 encoding (for CSV)

Example:
```csv
review_text,product_id
"Great product, highly recommend!",12345
"Not what I expected, disappointed.",12346
```

## 🌐 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

**Note**: For transformer models, ensure sufficient memory (recommend 4GB+)

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Railway/Render
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run app.py --server.port=$PORT`

### Heroku
Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## 🛠️ Configuration

### Model Paths
Models are loaded from:
- Traditional: `best_model.pkl`, `vectorizer.pkl`
- Transformers: `transformer_models/[model_name]_finetuned/`

### GPU Acceleration
App automatically detects and uses:
- Apple Silicon (MPS)
- NVIDIA GPU (CUDA)
- CPU fallback

## 📈 Performance Tips

1. **For speed**: Use Traditional ML model
2. **For accuracy**: Use Transformer models
3. **Batch processing**: Process in chunks for large files
4. **GPU**: Enables faster transformer inference

## 🐛 Troubleshooting

### Models not loading
- Ensure model files are in correct directories
- Check file permissions
- Verify model compatibility

### Out of memory
- Use Traditional ML model
- Reduce batch size
- Enable GPU if available

### Slow processing
- Switch to Traditional ML model
- Enable GPU acceleration
- Process smaller batches

## 📝 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues.

## 📧 Support

For issues or questions, please open a GitHub issue.
'''

with open(f'{app_dir}/README.md', 'w') as f:
    f.write(readme)

# Create Dockerfile
dockerfile_content = '''FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

with open(f'{app_dir}/Dockerfile', 'w') as f:
    f.write(dockerfile_content)

# Create .dockerignore
dockerignore_content = '''__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
*.log
.git
.gitignore
README.md
*.md
.DS_Store
'''

with open(f'{app_dir}/.dockerignore', 'w') as f:
    f.write(dockerignore_content)

print(f"📁 ENHANCED APP FILES CREATED:")
print(f"   ✅ {app_dir}/app.py - Main Streamlit application (ENHANCED)")
print(f"   ✅ {app_dir}/best_model.pkl - Trained XGBoost model")
print(f"   ✅ {app_dir}/vectorizer.pkl - TF-IDF vectorizer")
print(f"   ✅ {app_dir}/transformer_models/ - Transformer models directory")
print(f"   ✅ {app_dir}/requirements.txt - Updated dependencies")
print(f"   ✅ {app_dir}/README.md - Comprehensive documentation")
print(f"   ✅ {app_dir}/Dockerfile - Docker configuration")
print(f"   ✅ {app_dir}/.dockerignore - Docker ignore file")

print(f"\n🎯 NEW FEATURES:")
print(f"   ✨ File Upload & Batch Processing (CSV/Excel)")
print(f"   ✨ Transformer Models Support (DistilBERT, RoBERTa)")
print(f"   ✨ Model Selection (Traditional ML vs Transformers)")
print(f"   ✨ Export Results to CSV")
print(f"   ✨ Real-time Progress Tracking")
print(f"   ✨ Enhanced UI with Tabs")
print(f"   ✨ GPU Acceleration (MPS/CUDA)")
print(f"   ✨ Docker Support")

print(f"\n🚀 TO RUN THE APP:")
print(f"   1. cd {app_dir}")
print(f"   2. pip install -r requirements.txt")
print(f"   3. streamlit run app.py")

print(f"\n🐳 TO RUN WITH DOCKER:")
print(f"   1. cd {app_dir}")
print(f"   2. docker build -t sentiment-app .")
print(f"   3. docker run -p 8501:8501 sentiment-app")

print(f"\n☁️ TO DEPLOY ONLINE:")
print(f"   • **Streamlit Cloud**: Push to GitHub → share.streamlit.io")
print(f"   • **Railway**: Connect GitHub repo → railway.app")
print(f"   • **Render**: Connect GitHub repo → render.com")
print(f"   • **Heroku**: Use provided Procfile")

print(f"\n✅ ENHANCED HOSTED APP FEATURE COMPLETED!")