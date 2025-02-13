# train_bert.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # progress bar (optional)

# Load the dataset
df = pd.read_csv("fake_news.csv")
X = df["text"]
y = df["label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained tokenizer and model (using RoBERTa)
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)
bert_model.eval()  # set to evaluation mode

# Function to compute the embedding for a given text
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Mean pooling over the token embeddings (ignoring padded tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().flatten()

# Compute embeddings for training data (use tqdm for progress tracking)
print("Extracting embeddings for training data...")
X_train_emb = np.array([get_embedding(text) for text in tqdm(X_train, desc="Train Embeddings")])
print("Extracting embeddings for test data...")
X_test_emb = np.array([get_embedding(text) for text in tqdm(X_test, desc="Test Embeddings")])

# Train a Logistic Regression classifier on the embeddings
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train)
accuracy = clf.score(X_test_emb, y_test)
print(f"BERT Embedding Model Test Accuracy: {accuracy:.4f}")

# Save the tokenizer, model, and classifier together
joblib.dump((tokenizer, bert_model, clf), "bert_model.pkl")
print("BERT model pipeline saved as bert_model.pkl")
