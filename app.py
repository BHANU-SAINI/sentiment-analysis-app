import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import praw

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Initialize Reddit client once
@st.cache_resource
def initialize_reddit():
    reddit = praw.Reddit(
        client_id="QECiIk7O5CMaWvXTaphrzw",
        client_secret="u5Pelzwf2qSmrXwraJTkWBICS71rng",
        user_agent="reddit-sentiment-app"
    )
    return reddit

# Initialize YouTube client once
@st.cache_resource
def initialize_youtube():
    from googleapiclient.discovery import build
    api_key = "AIzaSyA0uWwCrseupnda4I2dk4Q6HxUl9BZBuSE"
    youtube = build("youtube", "v3", developerKey=api_key)
    return youtube

# Preprocess text and predict sentiment using model (TextBlob polarity for info only)
def predict_sentiment_with_score(text, model, vectorizer, stop_words):
    # Preprocess text
    text_proc = re.sub('[^a-zA-Z]', ' ', text)
    text_proc = text_proc.lower()
    text_proc = text_proc.split()
    text_proc = [word for word in text_proc if word not in stop_words]
    text_proc = ' '.join(text_proc)

    # Vectorize
    vect_text = vectorizer.transform([text_proc])

    # Predict using model
    pred = model.predict(vect_text)[0]
    if pred == 1:
        sentiment_label = "Positive"
    else:
        sentiment_label = "Negative"

    # Get TextBlob polarity for reference
    from textblob import TextBlob
    polarity = TextBlob(text).sentiment.polarity

    return sentiment_label, polarity

def main():
    st.title("Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    reddit = initialize_reddit()
    youtube = initialize_youtube()

    option = st.selectbox("Choose an option", ["Input text", "Get posts from subreddit", "Get YouTube video comments"])

    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment, polarity = predict_sentiment_with_score(text_input, model, vectorizer, stop_words)
            st.write(f"**Sentiment (Model):** {sentiment}")
            st.write(f"**Polarity (TextBlob):** {polarity:.2f}")
            st.write("---")
            st.write(text_input)

    elif option == "Get posts from subreddit":
        subreddit_name = st.text_input("Enter subreddit name")
        if st.button("Fetch Posts"):
            try:
                subreddit = reddit.subreddit(subreddit_name)
                posts = []
                for submission in subreddit.hot(limit=5):
                    text_combined = (submission.title or "") + " " + (submission.selftext or "")
                    posts.append(text_combined.strip())
                if not posts:
                    st.write("No posts found.")
                else:
                    for post_text in posts:
                        sentiment, polarity = predict_sentiment_with_score(post_text, model, vectorizer, stop_words)
                        st.write(f"**Sentiment (Model):** {sentiment}")
                        st.write(f"**Polarity (TextBlob):** {polarity:.2f}")
                        st.write(post_text)
                        st.write("---")
            except Exception as e:
                st.error(f"Error fetching posts: {e}")

    elif option == "Get YouTube video comments":
        video_id = st.text_input("Enter YouTube Video ID (e.g. dQw4w9WgXcQ)")
        if st.button("Fetch Comments"):
            try:
