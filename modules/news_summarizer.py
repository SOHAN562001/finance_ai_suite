import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from transformers import pipeline
from bs4 import BeautifulSoup
from functools import lru_cache

# -------------------------------------------------------------
# 🔹 Helper: Text Cleaner
# -------------------------------------------------------------
def _clean_text(txt):
    if not txt:
        return ""
    return (
        str(txt)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("  ", " ")
        .strip()
    )


# -------------------------------------------------------------
# 🔹 Cached News Fetcher with Yahoo → Google fallback
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _fetch_news(query: str, max_items=10):
    """Fetch news from Yahoo Finance, fallback to Google News RSS if limited."""
    base = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "newsCount": max_items}

    # Try Yahoo Finance
    try:
        resp = requests.get(base, params=params, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        news = []
        for item in js.get("news", []):
            news.append({
                "title": _clean_text(item.get("title")),
                "publisher": item.get("publisher"),
                "link": item.get("link"),
                "pubDate": item.get("providerPublishTime"),
                "summary": _clean_text(item.get("summary")),
            })
        if news:
            return pd.DataFrame(news)
        raise ValueError("No Yahoo results")
    except Exception:
        # Fallback to Google News RSS
        st.info("⚠️ Yahoo Finance limit hit → fetching from Google News RSS 🌐")
        try:
            rss = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}+finance&hl=en-IN&gl=IN&ceid=IN:en"
            r = requests.get(rss, timeout=10)
            soup = BeautifulSoup(r.text, "xml")
            items = soup.find_all("item")[:max_items]
            news = []
            for item in items:
                news.append({
                    "title": item.title.text,
                    "publisher": "Google News",
                    "link": item.link.text,
                    "pubDate": item.pubDate.text,
                    "summary": item.description.text,
                })
            return pd.DataFrame(news)
        except Exception as e2:
            st.error(f"❌ News fetch failed from both sources: {e2}")
            return pd.DataFrame(columns=["title", "publisher", "link", "pubDate", "summary"])


# -------------------------------------------------------------
# 🔹 Cached Model Loaders
# -------------------------------------------------------------
@st.cache_resource
def _load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def _load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def _load_t5_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")


# -------------------------------------------------------------
# 🔹 Core Company Analysis
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def analyze_company_news(company_name, n_articles=5):
    news_df = _fetch_news(company_name, max_items=n_articles)
    if news_df.empty:
        return None

    sentiment_analyzer = _load_sentiment_model()
    summarizer = _load_summarizer()

    summaries, sentiments = [], []
    for _, row in news_df.iterrows():
        text = row["summary"] or row["title"]
        # Summarize
        try:
            summary = summarizer(text, max_length=60, min_length=25, do_sample=False)[0]["summary_text"]
        except Exception:
            summary = text
        # Sentiment
        try:
            sentiment = sentiment_analyzer(summary[:512])[0]["label"]
        except Exception:
            sentiment = "NEUTRAL"

        summaries.append(summary)
        sentiments.append(sentiment)

    news_df["summary_gen"] = summaries
    news_df["sentiment"] = sentiments
    return news_df


# -------------------------------------------------------------
# 🔹 Sentiment Chart
# -------------------------------------------------------------
def plot_sentiment_comparison(df1, df2, company1, company2):
    sentiments = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    counts1 = [sum(df1["sentiment"] == s) for s in sentiments]
    counts2 = [sum(df2["sentiment"] == s) for s in sentiments]

    fig, ax = plt.subplots()
    width = 0.35
    x = range(len(sentiments))
    ax.bar([i - width / 2 for i in x], counts1, width, label=company1)
    ax.bar([i + width / 2 for i in x], counts2, width, label=company2)

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title("Sentiment Comparison")
    ax.legend()
    st.pyplot(fig)


# -------------------------------------------------------------
# 🔹 Streamlit Runner
# -------------------------------------------------------------
def run_news_summarizer():
    st.header("📰 Company News Summarizer & Comparator")
    st.caption("Fetch, summarize, and compare financial news sentiment using AI (BERT + BART + FLAN-T5).")

    col1, col2 = st.columns(2)
    with col1:
        company1 = st.text_input("Enter first company name:", "Reliance Industries")
    with col2:
        company2 = st.text_input("Enter second company name:", "Tata Motors")

    n_articles = st.slider("Number of news articles per company:", 3, 25, 10)
    btn = st.button("🔍 Compare News & Sentiment", use_container_width=True)

    if btn:
        with st.spinner("Fetching and analyzing… please wait ⏳"):
            df1 = analyze_company_news(company1, n_articles)
            df2 = analyze_company_news(company2, n_articles)

        if df1 is None or df2 is None:
            st.error("No news found for either company. Try again with other names.")
            return

        # Plot Comparison
        plot_sentiment_comparison(df1, df2, company1, company2)

        # AI Insight
        flan = _load_t5_model()
        prompt = f"Compare investor sentiment between {company1} and {company2} based on news tone and summarize briefly."
        insight = flan(prompt, max_length=150)[0]["generated_text"]

        st.subheader("💬 AI Insight Summary")
        st.success(insight)

        # Display sample tables
        with st.expander(f"🧾 Summaries for {company1}"):
            st.dataframe(df1[["title", "summary_gen", "sentiment"]])
        with st.expander(f"🧾 Summaries for {company2}"):
            st.dataframe(df2[["title", "summary_gen", "sentiment"]])


# -------------------------------------------------------------
# 🔹 Run standalone
# -------------------------------------------------------------
if __name__ == "__main__":
    run_news_summarizer()
