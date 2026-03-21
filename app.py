# =========================================
# Build App with Streamlit
# ========================================


import os
import re
import string
import nltk
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from PIL import Image

# ── Download NLTK data first ──────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("vader_lexicon", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Import NLTK modules AFTER download ───────────────────────
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Sentiment Analyzer", page_icon="🍽", layout="centered"
)


# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    model = joblib.load(os.path.join(base, "best_model_svc.pkl"))
    vectorizer = joblib.load(os.path.join(base, "tfidf_vectorizer.pkl"))
    sia = SentimentIntensityAnalyzer()
    lem = WordNetLemmatizer()
    stops = set(stopwords.words("english")) - {"not", "no", "never"}
    return model, vectorizer, sia, lem, stops


model, vectorizer, sia, lem, stops = load_models()


# ── Train model ───────────────────────────────────────────────
@st.cache_resource
def train_and_load():
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "Restaurant_Reviews.csv"), sep=";")

    lem = WordNetLemmatizer()
    stops = set(stopwords.words("english")) - {"not", "no", "never"}

    def clean(text):
        text = str(text).lower()
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [lem.lemmatize(w) for w in word_tokenize(text) if w not in stops]
        return " ".join(tokens)

    df["cleaned"] = df["Review"].apply(clean)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X = vectorizer.fit_transform(df["cleaned"])
    y = df["Liked"].values.astype(int)

    df["exclamation_count"] = df["Review"].str.count("!")
    df["question_count"] = df["Review"].str.count(r"\?")
    df["caps_ratio"] = df["Review"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(x), 1)
    )
    df["word_count"] = df["cleaned"].str.split().str.len()

    extra = csr_matrix(
        df[["exclamation_count", "question_count", "caps_ratio", "word_count"]].values
    )
    X_combined = hstack([X, extra])

    svc = LinearSVC(C=0.8, max_iter=2000, random_state=42)
    model = CalibratedClassifierCV(svc, cv=3)
    model.fit(X_combined, y)

    sia = SentimentIntensityAnalyzer()
    return model, vectorizer, sia, lem, stops


with st.spinner("⏳ Loading model — first visit takes ~10 seconds..."):
    model, vectorizer, sia, lem, stops = train_and_load()


# ── Helper: load image ────────────────────────────────────────
def load_img(filename):
    base = os.path.dirname(__file__)
    path = os.path.join(base, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# ── Predict ───────────────────────────────────────────────────
def clean_input(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lem.lemmatize(w) for w in word_tokenize(text) if w not in stops]
    return " ".join(tokens)


def predict(review):
    cleaned = clean_input(review)
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


# ── Header ────────────────────────────────────────────────────
st.title("🍽 Restaurant Sentiment Analyzer")
st.markdown("NLP & Machine Learning pipeline for customer review analysis.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Predict", "📊 Model Performance", "📈 Data Insights", "🗂 Topic Analysis"]
)

# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyze a Customer Review")
    review = st.text_area(
        "Enter review:",
        height=120,
        placeholder="e.g. The food was amazing but service was slow...",
    )

    if st.button("Analyze", type="primary"):
        if review.strip() == "":
            st.warning("Please enter a review first.")
        else:
            pred, proba, vader = predict(review)
            st.divider()

            if pred == 1:
                st.success("✅ POSITIVE Review")
            else:
                st.error("❌ NEGATIVE Review")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positive Confidence", f"{proba[1]:.1%}")
            with col2:
                st.metric("Negative Confidence", f"{proba[0]:.1%}")

            st.divider()
            st.subheader("VADER Sentiment Scores")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Compound", f"{vader['compound']:.3f}")
            col2.metric("Positive", f"{vader['pos']:.2f}")
            col3.metric("Negative", f"{vader['neg']:.2f}")
            col4.metric("Neutral", f"{vader['neu']:.2f}")

            st.divider()
            st.subheader("Confidence Breakdown")
            st.progress(float(proba[1]))
            st.caption(
                f"Model confidence: {proba[1]:.1%} positive | {proba[0]:.1%} negative"
            )

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

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Comparison — All 5 Classifiers")
    img = load_img("09_model_comparison.png")
    if img:
        st.image(img, use_column_width=True)
    else:
        st.warning("Chart not found. Run 09_compare_models.py first.")

    st.divider()
    st.subheader("Best Model Deep Dive — LinearSVC")
    st.markdown("Confusion Matrix · ROC Curve · Top Predictive Words")
    img2 = load_img("05_evaluation.png")
    if img2:
        st.image(img2, use_column_width=True)
    else:
        st.warning("Chart not found. Run 05_evaluate.py first.")

    st.divider()
    st.subheader("Performance Summary")
    perf_data = {
        "Model": [
            "LinearSVC",
            "Logistic Regression",
            "Gradient Boosting",
            "ComplementNB",
            "Random Forest",
        ],
        "CV Accuracy": ["79.5%", "78.9%", "76.1%", "75.2%", "75.1%"],
        "Test Accuracy": ["85.0%", "79.0%", "76.5%", "75.5%", "74.0%"],
        "ROC-AUC": ["0.925", "0.871", "0.848", "0.830", "0.820"],
        "Recommended": ["⭐ Best", "Runner-up", "—", "—", "—"],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Dataset Overview")
    img3 = load_img("01_eda_overview.png")
    if img3:
        st.image(img3, use_column_width=True)
    else:
        st.warning("Chart not found. Run 01_eda.py first.")

    st.divider()
    st.subheader("Word Clouds — Positive vs Negative")
    img4 = load_img("08a_wordclouds.png")
    if img4:
        st.image(img4, use_column_width=True)
    else:
        st.warning("Chart not found. Run 08_visualize.py first.")

    st.divider()
    st.subheader("Top Bigrams by Sentiment")
    img5 = load_img("08b_ngrams.png")
    if img5:
        st.image(img5, use_column_width=True)
    else:
        st.warning("Chart not found. Run 08_visualize.py first.")

    st.divider()
    st.subheader("VADER Sentiment Heatmap")
    img6 = load_img("08c_heatmap.png")
    if img6:
        st.image(img6, use_column_width=True)
    else:
        st.warning("Chart not found. Run 08_visualize.py first.")

# ════════════════════════════════════════════════════════════
# TAB 4 — TOPIC ANALYSIS
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader("LDA Topic Modeling — Negative Reviews")
    st.markdown("Latent Dirichlet Allocation identifies 5 hidden complaint themes.")
    img7 = load_img("07_lda_topics.png")
    if img7:
        st.image(img7, use_column_width=True)
    else:
        st.warning("Chart not found. Run 07_topics.py first.")

    st.divider()
    st.subheader("Key Complaint Themes Identified")
    themes = {
        "Theme": [
            "Service Speed",
            "Food Quality",
            "Staff Attitude",
            "Value & Pricing",
            "Food Consistency",
        ],
        "Top Words": [
            "slow, wait, time, minutes, hour",
            "bland, dry, cold, tasteless, nasty",
            "rude, attitude, ignored, unprofessional",
            "overpriced, expensive, worth, cheap",
            "rubber, frozen, stale, undercooked",
        ],
        "Priority": ["🔴 Critical", "🔴 Critical", "🟡 High", "🟡 High", "🟠 Medium"],
    }
    st.dataframe(pd.DataFrame(themes), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Python · scikit-learn · NLTK · Streamlit | "
    "Dataset: 1,000 restaurant reviews · Model: LinearSVC · "
    "Accuracy: 85% · ROC-AUC: 0.93"
)
