import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Download system check for NLTK Stopwords
try:
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

# 2. Load the local CSV Dataset
try:
    data = pd.read_csv('Restaurant_Reviews.csv', delimiter=',')
    print(f"Dataset successfully loaded! Total rows: {data.shape[0]}")
except FileNotFoundError:
    print("Error: 'Restaurant_Reviews.csv' not found in D:\\pro1 folder!")
    exit()

# 3. Text Preprocessing Loop (Matches your exact notebook logic)
corpus = []
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

for i in range(0, len(data)):
    # Retain only alphabetic characters
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=str(data['Review'][i]))
    review = review.lower()
    review_words = review.split()
    
    # Filter stopwords and apply Porter Stemmer
    review = [ps.stem(word) for word in review_words if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)

data['cleaned_text'] = corpus

# 4. Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    data['cleaned_text'], data['Liked'], test_size=0.20, random_state=42
)

# 5. TF-IDF Vectorization (From your notebook requirements)
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Model Training using LinearSVC (Support Vector Machine)
model = LinearSVC(random_state=42)
model.fit(X_train_tfidf, y_train)

# 7. Print Model Performance Evaluation
y_pred = model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Live Prediction Testing (Using your notebook's logic)
def predict_sentiment(sample_review):
    cleaned = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    cleaned = cleaned.lower()
    cleaned_words = cleaned.split()
    cleaned = [ps.stem(word) for word in cleaned_words if not word in stop_words]
    cleaned_str = ' '.join(cleaned)
    
    vectorized = tfidf.transform([cleaned_str])
    prediction = model.predict(vectorized)[0]
    
    # Returns the exact statement strings requested in your notebook
    if prediction == 1:
        return "This is a Positive review"
    else:
        return "This is a Negative review"

print("-" * 50)
test_1 = "The food was very bad and disappointing."
print(f"Review: '{test_1}' -> Result: {predict_sentiment(test_1)}")

test_2 = "Wow... Highly recommended! The food was amazing."
print(f"Review: '{test_2}' -> Result: {predict_sentiment(test_2)}")
print("-" * 50)
