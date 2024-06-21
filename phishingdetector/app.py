from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from joblib import load

app = Flask(__name__)
CORS(app)

try:
    model = load('phishing_model.joblib')
    tfidf_vectorizer = load('phishing_tfidf_vectorizer.joblib')
    scaler = load('scaler.joblib')
    url_vectorizer = load('url_tfidf_vectorizer.joblib')
    url_classifier = load('URLclassificationmodel.joblib')
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Failed to load models or components: {e}")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://\S+', 'urlplaceholder', text)
    text = re.sub(r'\S*@\S*\s?', 'emailplaceholder', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words])

@app.route('/')
def home():
    return "Phishing Detection API is running!"

@app.route('/predict-email', methods=['POST'])
def predict_email():
    try:
        text_input = request.json['text']
        processed_text = preprocess_text(text_input)
        features = tfidf_vectorizer.transform([processed_text]).toarray()
        sentiment_analyzer = SentimentIntensityAnalyzer()
        sentiment = sentiment_analyzer.polarity_scores(processed_text)['compound']
        features = np.hstack((features, [[sentiment]]))
        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])
        probability = round(model.predict_proba(features_scaled)[0, 1] * 100, 2)
        return jsonify({'prediction': prediction, 'probability': probability})
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Error processing email prediction'}), 400

@app.route('/predict-url', methods=['POST'])
def predict_url():
    try:
        url_input = request.json['url']
        url_features = url_vectorizer.transform([url_input])
        url_prediction = int(url_classifier.predict(url_features)[0])
        url_probabilities = url_classifier.predict_proba(url_features)[0, 1] * 100
        url_probability = round(float(url_probabilities), 2)  # Convert numpy float32 to Python float and round
        return jsonify({'prediction': url_prediction, 'probability': url_probability})
    except Exception as e:
        print(f"An error occurred: {e}")  # Print the exception to the console for debugging
        return jsonify({'error': str(e), 'message': 'Error processing URL prediction'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Change to a different port like 5001

