# 🍽 Restaurant Sentiment Analysis

A complete NLP & Machine Learning pipeline to analyze customer restaurant reviews.

## Project Overview
- **Dataset**: 1,000 customer reviews with binary sentiment labels
- **Task**: Binary text classification (Positive/Negative)
- **Best Model**: LinearSVC — 85% accuracy, 0.93 ROC-AUC

## Live Demo
👉 [Try the app here](https://restaurant-sentiment-yuebakdfcqnw8kgubgc7si.streamlit.app/)

## Tech Stack
- Python 3.11
- scikit-learn, NLTK, pandas, numpy
- Streamlit (web app)
- VADER (lexicon sentiment)
- LDA (topic modeling)

## Project Structure
```
restaurant-sentiment/
├── app.py                    # Streamlit web app
├── requirements.txt          # Dependencies
├── runtime.txt               # Python specification for Streamlit Cloud
├── Restaurant_Reviews.csv    # Dataset
├── README.md                 
│
├── notebooks/                ← analysis scripts 
│   ├── 01_eda.py             # Exploratory data analysis
│   ├── 02_preprocessing.py   # Text cleaning pipeline
│   ├── 03_features.py        # TF-IDF feature engineering
│   ├── 04_train_models.py    # Model training & cross-validation
│   ├── 05_evaluate.py        # Model evaluation & diagnostics
│   ├── 06_vader.py           # VADER sentiment scoring
│   ├── 07_topics.py          # LDA topic modeling
│   └── 08_visualize.py       # Visualizations
│
└── utils/                    ← utility scripts 
    └── predict.py            # CLI predictor
```

## Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| LinearSVC | 84% | 0.93 |
| Logistic Regression | 84% | 0.90 |
| Gradient Boosting | 77% | 0.87 |

## Setup
```bash
conda create -n sentiment python=3.11 -y
conda activate sentiment
pip install -r requirements.txt
streamlit run app.py
```

## Business Recommendations
- Service speed is the top complaint driver
- Staff attitude directly impacts return visits
- Food consistency issues cluster around specific menu items
