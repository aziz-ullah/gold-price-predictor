"""News sentiment analysis for gold price prediction.

This module fetches gold-related news and calculates daily sentiment scores.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
SENTIMENT_DATA_DIR = BASE_DIR / "data" / "sentiment"


def fetch_news_articles(
    query: str = "gold price",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_articles: int = 50,
) -> List[Dict[str, str]]:
    """Fetch news articles using NewsAPI or web scraping.
    
    Parameters
    ----------
    query : str
        Search query for news articles.
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.
    max_articles : int
        Maximum number of articles to fetch.
    
    Returns
    -------
    List[Dict[str, str]]
        List of articles with 'title', 'description', 'date', 'source' keys.
    """
    articles = []
    
    # Try NewsAPI first (requires API key)
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": newsapi_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(max_articles, 100),
            }
            
            if start_date:
                params["from"] = start_date
            if end_date:
                params["to"] = end_date
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "articles" in data:
                for article in data["articles"][:max_articles]:
                    articles.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "date": article.get("publishedAt", ""),
                        "source": article.get("source", {}).get("name", ""),
                    })
            
            print(f"[*] Fetched {len(articles)} articles from NewsAPI")
            return articles
        except Exception as e:
            print(f"[!] NewsAPI error: {e}. Falling back to web scraping...")
    
    # Fallback: Scrape from investing.com gold news
    try:
        print("[*] Scraping news from investing.com...")
        url = "https://www.investing.com/commodities/gold/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find article elements (adjust selectors based on actual HTML structure)
        article_elements = soup.find_all("article", limit=max_articles)
        
        for elem in article_elements:
            title_elem = elem.find("a", class_="title")
            date_elem = elem.find("time") or elem.find("span", class_="date")
            
            if title_elem:
                title = title_elem.get_text(strip=True)
                description = ""
                desc_elem = elem.find("p") or elem.find("div", class_="summary")
                if desc_elem:
                    description = desc_elem.get_text(strip=True)
                
                date_str = ""
                if date_elem:
                    date_str = date_elem.get("datetime") or date_elem.get_text(strip=True)
                
                articles.append({
                    "title": title,
                    "description": description,
                    "date": date_str,
                    "source": "Investing.com",
                })
        
        print(f"[*] Scraped {len(articles)} articles from web")
    except Exception as e:
        print(f"[!] Web scraping error: {e}")
    
    return articles


def calculate_sentiment_score(text: str) -> float:
    """Calculate sentiment score for a text using simple keyword-based approach.
    
    For production, consider using VADER, TextBlob, or transformer-based models.
    
    Parameters
    ----------
    text : str
        Text to analyze.
    
    Returns
    -------
    float
        Sentiment score between -1 (negative) and 1 (positive).
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    
    # Positive keywords
    positive_keywords = [
        "surge", "rally", "gain", "rise", "up", "bullish", "strong",
        "increase", "higher", "boost", "soar", "jump", "climb",
        "momentum", "optimistic", "positive", "growth", "breakthrough",
    ]
    
    # Negative keywords
    negative_keywords = [
        "fall", "drop", "decline", "down", "bearish", "weak",
        "decrease", "lower", "plunge", "crash", "slump", "dip",
        "concern", "worry", "risk", "uncertainty", "volatility",
        "crisis", "recession", "inflation", "war", "conflict",
    ]
    
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)
    
    # Calculate score
    total_words = len(text.split())
    if total_words == 0:
        return 0.0
    
    # Normalize by text length
    score = (positive_count - negative_count) / max(total_words, 1)
    
    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, score * 10))


def analyze_daily_sentiment(
    articles: List[Dict[str, str]],
    date: str,
) -> Dict[str, float]:
    """Calculate daily sentiment metrics from articles.
    
    Parameters
    ----------
    articles : List[Dict[str, str]]
        List of articles for the day.
    date : str
        Date in YYYY-MM-DD format.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with sentiment metrics.
    """
    if not articles:
        return {
            "sentiment_score": 0.0,
            "sentiment_count": 0,
            "positive_ratio": 0.5,
        }
    
    scores = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        score = calculate_sentiment_score(text)
        scores.append(score)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    positive_count = sum(1 for s in scores if s > 0)
    positive_ratio = positive_count / len(scores) if scores else 0.5
    
    return {
        "sentiment_score": avg_score,
        "sentiment_count": len(scores),
        "positive_ratio": positive_ratio,
    }


def fetch_daily_sentiment_scores(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_to_file: bool = True,
) -> pd.DataFrame:
    """Fetch and calculate daily sentiment scores for gold-related news.
    
    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format. Defaults to 1 year ago.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to today.
    save_to_file : bool
        Whether to save results to CSV.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with Date and sentiment columns.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"[*] Fetching news sentiment from {start_date} to {end_date}...")
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    
    sentiment_data = []
    
    # Fetch articles for date range (in batches to avoid rate limits)
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    while current_date <= end_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        next_date = current_date + timedelta(days=7)  # Fetch weekly batches
        
        print(f"  - Fetching articles for {date_str}...")
        articles = fetch_news_articles(
            query="gold price OR gold market OR gold futures",
            start_date=date_str,
            end_date=next_date.strftime("%Y-%m-%d"),
            max_articles=20,
        )
        
        # Group articles by date
        daily_articles = {}
        for article in articles:
            article_date = article.get("date", "")
            if article_date:
                try:
                    # Parse date from various formats
                    if "T" in article_date:
                        article_dt = datetime.fromisoformat(article_date.replace("Z", "+00:00"))
                    else:
                        article_dt = datetime.strptime(article_date[:10], "%Y-%m-%d")
                    
                    date_key = article_dt.strftime("%Y-%m-%d")
                    if date_key not in daily_articles:
                        daily_articles[date_key] = []
                    daily_articles[date_key].append(article)
                except:
                    pass
        
        # Calculate sentiment for each day
        for date_key in daily_articles:
            sentiment = analyze_daily_sentiment(daily_articles[date_key], date_key)
            sentiment_data.append({
                "Date": date_key,
                **sentiment,
            })
        
        current_date = next_date
        time.sleep(1)  # Rate limiting
    
    # Create DataFrame
    if not sentiment_data:
        print("[!] Warning: No sentiment data generated!")
        return pd.DataFrame(columns=["Date", "sentiment_score", "sentiment_count", "positive_ratio"])
    
    df = pd.DataFrame(sentiment_data)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    # Fill missing dates with neutral sentiment
    full_date_range = pd.DataFrame({
        "Date": [d.date() for d in date_range],
    })
    df = full_date_range.merge(df, on="Date", how="left")
    df["sentiment_score"] = df["sentiment_score"].fillna(0.0)
    df["sentiment_count"] = df["sentiment_count"].fillna(0)
    df["positive_ratio"] = df["positive_ratio"].fillna(0.5)
    
    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)
    
    if save_to_file:
        SENTIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = SENTIMENT_DATA_DIR / "news_sentiment.csv"
        df.to_csv(output_path, index=False)
        print(f"[OK] Sentiment data saved to {output_path}")
        print(f"     Total rows: {len(df)}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch news sentiment scores")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    fetch_daily_sentiment_scores(
        start_date=args.start_date,
        end_date=args.end_date,
    )

