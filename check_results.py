import pandas as pd

df = pd.read_csv("data/AAPL_vader_results.csv")
daily = pd.read_csv("data/AAPL_vader_daily.csv")

print("── Sentiment Summary ──")
print(df["sentiment"].value_counts())

print("\n── Average Compound Score ──")
print(f"{df['compound'].mean():.4f}")

print("\n── Most Positive Articles ──")
print(df[df["sentiment"] == "Positive"][["cleaned_text", "compound"]]
      .sort_values("compound", ascending=False).head(3))

print("\n── Most Negative Articles ──")
print(df[df["sentiment"] == "Negative"][["cleaned_text", "compound"]]
      .sort_values("compound").head(3))

print("\n── Daily Sentiment ──")
print(daily[["date", "avg_compound", "daily_sentiment", "total_articles"]])