from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Define Flask app
app = Flask(__name__)

# Load pre-trained model (if applicable)
# Replace with your model loading logic if you have one
model = None  # Placeholder for a pre-trained model

# Function for sentiment analysis (replace with yours if needed)
def analyze_sentiment(text):
    # Preprocess text (remove stopwords, punctuation)
    stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ''.join([char for char in text if char not in string.punctuation])

    # Use VADER sentiment analyzer (or your trained model)
    analyser = SentimentIntensityAnalyzer()
    sentiment = analyser.polarity_scores(text)
    return sentiment

# Route for handling sentiment analysis requests
@app.route('/', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        user_input = request.form['text']
        sentiment = analyze_sentiment(user_input)

        # Process sentiment scores (e.g., classify into positive/negative/neutral)
        final_emotion = "neutral"  # Placeholder for classification
        if sentiment['pos'] > sentiment['neg'] and sentiment['pos'] > sentiment['neu']:
            final_emotion = "positive"
        elif sentiment['neg'] > sentiment['pos'] and sentiment['neg'] > sentiment['neu']:
            final_emotion = "negative"

        return render_template('index.html', sentiment=sentiment, final_emotion=final_emotion, user_input=user_input)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)