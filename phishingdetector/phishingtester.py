import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import re
from joblib import load

# Load models and transformers
model = load('phishing_model.joblib')
tfidf_vectorizer = load('phishing_tfidf_vectorizer.joblib')
scaler = load('scaler.joblib')
sia = SentimentIntensityAnalyzer()

# Load URL vectorizer and classifier
url_vectorizer = load('url_tfidf_vectorizer.joblib')
url_classifier = load('URLclassificationmodel.joblib')

# Prepare stopwords set
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', 'urlplaceholder', text)
    text = re.sub(r'\S*@\S*\s?', 'emailplaceholder', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Prediction function for text-based phishing detection
def predict_phishing(text):
    text_clean = preprocess_text(text)
    features = tfidf_vectorizer.transform([text_clean]).toarray()
    sentiment = sia.polarity_scores(text_clean)['compound']
    features = np.hstack((features, [[sentiment]]))
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0]
    is_phishing = model.predict(features_scaled)[0]
    confidence = probability[1] if is_phishing else 1 - probability[1]
    confidence_percentage = f"{confidence * 100:.0f}%"
    return "Phishing" if is_phishing else "Not phishing", confidence_percentage

# Prediction function for URL-based phishing detection
def predict_url_phishing(url):
    url_features = url_vectorizer.transform([url])
    url_probability = url_classifier.predict_proba(url_features)[0]
    is_url_phishing = url_classifier.predict(url_features)[0]
    url_confidence = url_probability[1] if is_url_phishing else 1 - url_probability[1]
    url_confidence_percentage = f"{url_confidence * 100:.0f}%"
    return "Phishing URL" if is_url_phishing else "Not phishing URL", url_confidence_percentage

# Example usage
while True:
    print("Enter 'text' to test a text message or 'url' to test a URL (or 'quit' to exit):")
    input_type = input().lower()
    
    if input_type == 'text':
        print("Enter the text message to test:")
        text = input()
        result, confidence = predict_phishing(text)
        print(f"Result: {result}")
        print(f"Confidence: {confidence}")
    elif input_type == 'url':
        print("Enter the URL to test:")
        url = input()
        result, confidence = predict_url_phishing(url)
        print(f"Result: {result}")
        print(f"Confidence: {confidence}")
    elif input_type == 'quit':
        break
    else:
        print("Invalid input. Please enter 'text', 'url', or 'quit'.")
    print()