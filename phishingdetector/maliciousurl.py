import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import time  # Import time library to track training duration

# Load data
print("Loading data...")
data = pd.read_csv("malicious_phish.csv")

# Data preprocessing
print("Preprocessing data...")
data['label'] = data['type'].astype('category').cat.codes

# Feature Engineering
print("Vectorizing data...")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
X = vectorizer.fit_transform(data['url'])

# Split data into features (X) and labels (y)
print("Splitting data...")
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBClassifier
classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Start training
print("Starting training...")
start_time = time.time()
classifier.fit(X_train, y_train)
end_time = time.time()

# Calculate and print the training time
training_time = end_time - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Predictions
print("Making predictions...")
y_pred = classifier.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
