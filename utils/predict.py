# ============================================================
# Predict sentiment of any new review
# ============================================================
import sys
import joblib
import re, string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix
import numpy as np

# ── Load artifacts ────────────────────────────────────────────
model = joblib.load("best_model_svc.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
sia = SentimentIntensityAnalyzer()
lem = WordNetLemmatizer()
stops = set(stopwords.words("english")) - {"not", "no", "never"}


def clean(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lem.lemmatize(w) for w in word_tokenize(text) if w not in stops]
    return " ".join(tokens)


review = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter review: ")
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

print(f"\n{'═'*52}")
print(f"  REVIEW  : {review}")
print(f"{'─'*52}")
print(f"  ML MODEL : {'✅ POSITIVE' if pred==1 else '❌ NEGATIVE'}")
print(f"  Confidence: Neg={proba[0]:.2%} | Pos={proba[1]:.2%}")
print(f"{'─'*52}")
print(
    f"  VADER    : compound={vader['compound']:.3f} | pos={vader['pos']:.2f} neg={vader['neg']:.2f}"
)
print(f"{'═'*52}\n")

# Usage e.g.: python predict.py "The food was amazing but service slow"
