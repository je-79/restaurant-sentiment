# =========================================
# Build App with Streamlit
# ========================================


import streamlit as st
import joblib
import re
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

# For the Streamlit Cloud deployment, we need to download NLTK data at runtime
import os
import nltk

# Download NLTK data on Streamlit Cloud
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Sentiment Analyzer", page_icon="🍽", layout="centered"
)


# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load("best_model_svc.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    sia = SentimentIntensityAnalyzer()
    lem = WordNetLemmatizer()
    stops = set(stopwords.words("english")) - {"not", "no", "never"}
    return model, vectorizer, sia, lem, stops


model, vectorizer, sia, lem, stops = load_models()


# ── Preprocessing ─────────────────────────────────────────────
def clean(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lem.lemmatize(w) for w in word_tokenize(text) if w not in stops]
    return " ".join(tokens)


def predict(review):
    cleaned = clean(review)
    X_tfidf = vectorizer.transform([cleaned])
    extra = csr_matrix(
        [
            [
                review.count("!"),
                review.count("?"),
                sum(1 for c in review if c.isupper()) / max(len(review), 1),
                len(cleaned.split()),
            ]
        ]
    )
    X = hstack([X_tfidf, extra])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    vader = sia.polarity_scores(review)
    return pred, proba, vader


# ── UI ────────────────────────────────────────────────────────
st.title("🍽 Restaurant Sentiment Analyzer")
st.markdown(
    "Analyze customer reviews using Machine Learning + VADER sentiment scoring."
)
st.divider()

review = st.text_area(
    "Enter a customer review:",
    height=120,
    placeholder="e.g. The food was amazing but service was slow...",
)

if st.button("Analyze", type="primary"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        pred, proba, vader = predict(review)

        # ── Result ────────────────────────────────────────────
        st.divider()
        if pred == 1:
            st.success("✅ POSITIVE Review")
        else:
            st.error("❌ NEGATIVE Review")

        # ── Confidence scores ─────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Positive Confidence", f"{proba[1]:.1%}")
        with col2:
            st.metric("Negative Confidence", f"{proba[0]:.1%}")

        # ── VADER scores ──────────────────────────────────────
        st.divider()
        st.subheader("VADER Sentiment Scores")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Compound", f"{vader['compound']:.3f}")
        col2.metric("Positive", f"{vader['pos']:.2f}")
        col3.metric("Negative", f"{vader['neg']:.2f}")
        col4.metric("Neutral", f"{vader['neu']:.2f}")

        # ── Confidence bar ────────────────────────────────────
        st.divider()
        st.subheader("Confidence Breakdown")
        st.progress(float(proba[1]))
        st.caption(
            f"Model confidence: {proba[1]:.1%} positive | {proba[0]:.1%} negative"
        )

# ── Sample reviews ────────────────────────────────────────────
st.divider()
st.subheader("Try a sample review:")
samples = [
    "The food was absolutely amazing and staff were so friendly!",
    "Worst experience ever, disgusting food and rude staff.",
    "Good value but service was a bit slow.",
    "I will never come back here again.",
    "Best restaurant in town, highly recommended!",
]
for sample in samples:
    if st.button(sample, key=sample):
        st.rerun()
