import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import time

print("Loading FinBERT model...")
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

LABELS = ["positive", "negative", "neutral"]

def get_finbert_sentiment(text, max_length=512):
    if not isinstance(text, str) or text.strip() == "":
        return None, None, None, None

    # tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(device)

    # get prediction
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    probs = probs.cpu().numpy()

    positive_score = float(probs[0])
    negative_score = float(probs[1])
    neutral_score  = float(probs[2])

    # compound style score → positive - negative (range -1 to +1)
    compound = positive_score - negative_score

    label = LABELS[probs.argmax()]

    return positive_score, negative_score, neutral_score, compound, label

def analyze_sentiment_finbert(ticker):
    path = f"data/{ticker}_news_cleaned.csv"

    if not os.path.exists(path):
        print(f"File not found: {path}. Run text_preprocessing.py first.")
        return

    df = pd.read_csv(path)
    print(f"Analyzing {len(df)} articles with FinBERT...")
    print("This may take a few minutes on CPU...\n")

    results = []
    for i, text in enumerate(df["cleaned_text"]):
        result = get_finbert_sentiment(text)
        results.append(result)

        # progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(df)} articles...")

    # unpack results
    df[["pos", "neg", "neu", "compound", "sentiment"]] = pd.DataFrame(
        results, columns=["pos", "neg", "neu", "compound", "sentiment"]
    )

    daily = df.groupby("published_at").agg(
        avg_compound=("compound", "mean"),
        positive_count=("sentiment", lambda x: (x == "positive").sum()),
        negative_count=("sentiment", lambda x: (x == "negative").sum()),
        neutral_count=("sentiment", lambda x: (x == "neutral").sum()),
        total_articles=("sentiment", "count")
    ).reset_index()

    daily.rename(columns={"published_at": "date"}, inplace=True)

    def classify(score):
        if score >= 0.05:   return "Positive"
        elif score <= -0.05: return "Negative"
        else:                return "Neutral"

    daily["daily_sentiment"] = daily["avg_compound"].apply(classify)

    df.to_csv(f"data/{ticker}_finbert_results.csv", index=False)
    daily.to_csv(f"data/{ticker}_finbert_daily.csv", index=False)

    print(f"\nSaved → data/{ticker}_finbert_results.csv")
    print(f"Saved → data/{ticker}_finbert_daily.csv")

    print("\n── FinBERT Sentiment Summary ──")
    print(df["sentiment"].value_counts())

    print("\n── Daily Sentiment ──")
    print(daily[["date", "avg_compound", "daily_sentiment", "total_articles"]])

    return df, daily

if __name__ == "__main__":
    TICKER = "AAPL"
    analyze_sentiment_finbert(TICKER)