import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

analyzer = SentimentIntensityAnalyzer()

def get_vader_scores(text):
    if not isinstance(text, str) or text.strip() == "":
        return None, None, None, None
    
    scores = analyzer.polarity_scores(text)
    return scores["pos"], scores["neg"], scores["neu"], scores["compound"]

def classify_sentiment(compound):
    if compound is None:
        return "Unknown"
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def analyze_sentiment(ticker):
    path = f"data/{ticker}_news_cleaned.csv"

    if not os.path.exists(path):
        print(f"File not found: {path}. Run text_preprocessing.py first.")
        return

    df = pd.read_csv(path)
    print(f"Analyzing sentiment for {len(df)} articles...")

    # apply vader to cleaned_text
    df[["pos", "neg", "neu", "compound"]] = df["cleaned_text"].apply(
        lambda x: pd.Series(get_vader_scores(x))
    )

    # classify
    df["sentiment"] = df["compound"].apply(classify_sentiment)


    daily = df.groupby("published_at").agg(
        avg_compound=("compound", "mean"),
        positive_count=("sentiment", lambda x: (x == "Positive").sum()),
        negative_count=("sentiment", lambda x: (x == "Negative").sum()),
        neutral_count=("sentiment", lambda x: (x == "Neutral").sum()),
        total_articles=("sentiment", "count")
    ).reset_index()

    daily.rename(columns={"published_at": "date"}, inplace=True)


    daily["daily_sentiment"] = daily["avg_compound"].apply(classify_sentiment)


    df.to_csv(f"data/{ticker}_vader_results.csv", index=False)
    daily.to_csv(f"data/{ticker}_vader_daily.csv", index=False)

    print(f"Saved → data/{ticker}_vader_results.csv")
    print(f"Saved → data/{ticker}_vader_daily.csv")

    print("\n── Sentiment Summary ──")
    print(df["sentiment"].value_counts())

    print("\n── Daily Sentiment ──")
    print(daily[["date", "avg_compound", "daily_sentiment", "total_articles"]])

    return df, daily


if __name__ == "__main__":
    TICKER = "AAPL"
    analyze_sentiment(TICKER)
