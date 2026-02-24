import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def to_lowercase(text):
    return text.lower()

def remove_stopwords(text):
    words = text.split()
    filtered = [w for w in words if w not in STOPWORDS]
    return ' '.join(filtered)

def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return None                        # handle empty/null values

    text = remove_urls(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = remove_extra_spaces(text)
    # Note: we keep stopwords for FinBERT
    # stopword removal is only for analysis/visualization later
    return text

def clean_text_for_analysis(text):
    # extra cleaning for word frequency analysis (not for FinBERT)
    text = clean_text(text)
    if text:
        text = remove_stopwords(text)
    return text

def preprocess_news(ticker):
    path = f"data/{ticker}_news.csv"

    if not os.path.exists(path):
        print(f"File not found: {path}. Run data_collection.py first.")
        return

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} articles")

    # clean full_text for FinBERT
    df["cleaned_text"] = df["full_text"].apply(clean_text)

    # extra cleaned version for word analysis
    df["analysis_text"] = df["full_text"].apply(clean_text_for_analysis)

    # drop rows where cleaned text is empty
    before = len(df)
    df = df.dropna(subset=["cleaned_text"])
    df = df[df["cleaned_text"].str.len() > 10]
    after = len(df)
    print(f"Removed {before - after} empty/short rows")
    print(f"Clean articles ready: {after}")

    # save cleaned data
    df.to_csv(f"data/{ticker}_news_cleaned.csv", index=False)
    print(f"Saved → data/{ticker}_news_cleaned.csv")

    print("\nSample Cleaned Text:")
    print(df[["published_at", "cleaned_text"]].head())

    return df

if __name__ == "__main__":
    TICKER = "AAPL"
    preprocess_news(TICKER)