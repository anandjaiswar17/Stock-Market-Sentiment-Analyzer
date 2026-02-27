from newsapi import NewsApiClient
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()


newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

def fetch_news(ticker, company_name, days=7):
    end = datetime.today()
    start = end - timedelta(days=days)

    articles = newsapi.get_everything(
        q=f'"{ticker}" OR "Apple Inc" OR "Apple stock" OR "Tim Cook"',
        from_param=start.strftime('%Y-%m-%d'),
        to=end.strftime('%Y-%m-%d'),
        language="en",
        sort_by="publishedAt",
        page_size=100
    )

    posts = []
    for article in articles["articles"]:
        posts.append({
            "ticker": ticker,
            "source": article["source"]["name"],
            "title": article["title"],
            "description": article["description"],
            "published_at": article["publishedAt"][:10],  # keep only date
            "url": article["url"]
        })

    df = pd.DataFrame(posts)
    df["full_text"] = df["title"] + " " + df["description"].fillna("")
    df["full_text"] = df["full_text"].str.strip()
    return df

def fetch_stock_data(ticker, days=30):
    end = datetime.today()
    start = end - timedelta(days=days)

    stock = yf.download(ticker, start=start, end=end, progress=False)
    stock = stock[["Close", "Volume"]].reset_index()
    stock.columns = ["date", "close_price", "volume"]
    stock["date"] = stock["date"].dt.strftime('%Y-%m-%d')
    stock["ticker"] = ticker
    stock["daily_change_%"] = stock["close_price"].pct_change() * 100
    return stock

if __name__ == "__main__":
    TICKER = "AAPL"
    COMPANY_NAME = "Apple"  # used for better news search

    print(f"Fetching news articles for {TICKER}...")
    news_df = fetch_news(TICKER, COMPANY_NAME)
    print(f"  Found {len(news_df)} articles")

    print(f"Fetching stock price data for {TICKER}...")
    stock_df = fetch_stock_data(TICKER)
    print(f"  Found {len(stock_df)} trading days")

    # create data folder if not exists
    os.makedirs("data", exist_ok=True)

    news_df.to_csv(f"data/{TICKER}_news.csv", index=False)
    stock_df.to_csv(f"data/{TICKER}_stock.csv", index=False)

    print("\nSample News Articles:")
    print(news_df[["published_at", "source", "title"]].head())

    print("\nSample Stock Data:")
    print(stock_df.head())