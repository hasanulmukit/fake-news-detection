# Fake News Detection App

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/fake-news-detection-app)
![GitHub contributors](https://img.shields.io/github/contributors/yourusername/fake-news-detection-app)
![GitHub stars](https://img.shields.io/github/stars/yourusername/fake-news-detection-app?style=social)
![GitHub license](https://img.shields.io/github/license/yourusername/fake-news-detection-app)

A web application that leverages modern Natural Language Processing (NLP) techniques to classify news articles as **Fake** or **Real**. This project provides two modeling pipelines:

- **TF-IDF + Logistic Regression:** A classical approach for text classification.
- **BERT/RoBERTa Embeddings + Logistic Regression:** A modern transformer-based approach for deeper contextual understanding.

Built using [Streamlit](https://streamlit.io/) for the web interface, [Scikit-learn](https://scikit-learn.org/) for classical machine learning, and [Hugging Face Transformers](https://huggingface.co/transformers/) for BERT/RoBERTa embeddings.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The Fake News Detection App is designed to help identify whether a news article is **Fake** or **Real**. With the rise of misinformation, this tool demonstrates how to combine traditional text processing with state-of-the-art NLP models. Two approaches are provided:

1. **TF-IDF + Logistic Regression:** A fast and interpretable baseline.
2. **BERT/RoBERTa Embeddings + Logistic Regression:** Utilizes pre-trained transformer models to extract contextual embeddings, offering potentially better accuracy at the cost of longer processing time.

---

## Features

- **User-Friendly Interface:** A simple Streamlit-based web app where users can input a news article and get an instant prediction.
- **Model Options:** Select between the fast TF-IDF model or the more sophisticated BERT/RoBERTa-based model.
- **Custom Styling:** Enhanced UI with custom CSS for a professional look.
- **Error Handling & Feedback:** Clear messages for input validation and model loading errors.
- **Caching for Efficiency:** Uses Streamlit’s caching to speed up model loading.

---

## Installation

### Prerequisites

- Python 3.7+
- pip

### Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detection-app.git
cd fake-news-detection-app
```

- Create and Activate a Virtual Environment (Optional but Recommended)

  ```bash

  ```

# For Windows

    python -m venv venv
    venv\Scripts\activate

# For macOS/Linux

    python3 -m venv venv
    source venv/bin/activate
    ```

### Install Required Packages

    ```bash
    pip install -r requirements.txt
    ```

---

### Usage

1. Training the Models

- TF-IDF Model:
  Run the script to train the TF-IDF + Logistic Regression model.

      ```bash
      python train_tfidf.py
      ```

  This script will load your dataset, train the model, and save it as tfidf_model.pkl.

- BERT/RoBERTa Model:
  Run the script to train the BERT-based classifier.

      ```bash
      python train_bert.py
      ```

  This script extracts embeddings in batches, trains a Logistic Regression classifier, and saves the pipeline as bert_model.pkl.

2. Running the Web App

- Start the Streamlit app by running:

      ```bash
      streamlit run app.py
      ```

  Then, open the provided local URL in your browser to interact with the app.

---

### Project Structure

- ├── app.py # Streamlit web app
- ├── train_tfidf.py # Script to train the TF-IDF + Logistic Regression model
- ├── train_bert.py # Script to train the BERT/RoBERTa based model
- ├── requirements.txt # Required Python packages
- ├── README.md # This file
- └── sample_news.csv # (Optional) A sample CSV with news articles for testing

---

### Dataset

This project uses a news dataset that should include at least the following columns:

- title
- text
- subject
- date
- label (0 for Real, 1 for Fake)
  You can merge your datasets as needed before training the models.

---

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

### License

This project is licensed under the MIT License - see the LICENSE file for details.
