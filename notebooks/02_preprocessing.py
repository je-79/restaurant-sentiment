# ============================================================
# 02_Text Preprocessing Pipeline
# ============================================================
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

df = pd.read_csv("Restaurant_Reviews.csv", sep=";")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

negation_words = {
    "not",
    "no",
    "never",
    "nor",
    "neither",
    "isn't",
    "wasn't",
    "weren't",
    "don't",
    "didn't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
}
stop_words -= negation_words


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)


df["cleaned"] = df["Review"].apply(clean_text)

print("\n═══ PREPROCESSING SAMPLE ═══")
for _, row in df.head(5).iterrows():
    print(f"[{row['Liked']}] ORIGINAL : {row['Review']}")
    print(f"    CLEANED  : {row['cleaned']}\n")

df.to_csv("reviews_cleaned.csv", index=False)
print(" Preprocessing done. Saved: reviews_cleaned.csv")
