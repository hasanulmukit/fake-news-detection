# app.py
import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# ---------------------------
# Page Configuration & Styling
# ---------------------------
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Custom CSS to enhance styling
custom_css = """
<style>
body {
    background-color: #f0f2f6;
}
h1, h2, h3, .stButton button {
    color: #333333;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stTextArea textarea {
    font-size: 16px;
}
div[data-testid="stMarkdownContainer"] > p {
    font-size: 16px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------
# Model Loading Functions (Cached)
# ---------------------------
@st.cache_resource
def load_tfidf_model():
    try:
        model = joblib.load("tfidf_model.pkl")
    except Exception as e:
        st.error("Error loading TF-IDF model: " + str(e))
        return None
    return model

@st.cache_resource
def load_bert_model():
    try:
        tokenizer, bert_model, clf = joblib.load("bert_model.pkl")
        bert_model.eval()  # set model to evaluation mode
    except Exception as e:
        st.error("Error loading BERT model: " + str(e))
        return None, None, None
    return tokenizer, bert_model, clf

# ---------------------------
# Helper Function: Embedding Extraction
# ---------------------------
def get_embedding(text, tokenizer, bert_model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Mean pooling over the token embeddings
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("Fake News Detection App")
st.subheader("Detect if a news article is **Fake** or **Real**")
st.write("Enter a news article below to predict whether it is **Fake** or **Real**.")

# Optional info message about the models
st.info("Select between two models: **TF-IDF** (faster) and **BERT/RoBERTa** (may capture deeper nuances).")

# Text input area with a placeholder
text_input = st.text_area("News Article Text", height=300, placeholder="Enter your news article text here...")

# Model selection radio buttons
model_choice = st.radio("Select the Model to use:", ("TF-IDF", "BERT/RoBERTa"))

# Predict button and prediction logic
if st.button("Predict"):
    if not text_input.strip():
        st.error("Please enter the text of a news article.")
    else:
        with st.spinner("Processing your input..."):
            if model_choice == "TF-IDF":
                model = load_tfidf_model()
                if model is not None:
                    prediction = model.predict([text_input])[0]
                else:
                    prediction = None
            else:
                tokenizer, bert_model, clf = load_bert_model()
                if None in (tokenizer, bert_model, clf):
                    prediction = None
                else:
                    embedding = get_embedding(text_input, tokenizer, bert_model)
                    prediction = clf.predict([embedding])[0]
            
            if prediction is None:
                st.error("Prediction could not be made due to a model loading error.")
            else:
                label = "Fake" if prediction == 1 else "Real"
                st.success(f"Prediction: **{label}**")
