import logging
import feedparser
import torch
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)
FINBERT_MODEL = "ProsusAI/finbert"
MAX_HEADLINES = 10
AGENT_VERSION = "1.1.0"
RSS_FETCH_TIMEOUT = 10
NEGATIVE_SIGNAL_THRESHOLD = 0.15
GOOGLE_NEWS_RSS_BASE = (
    "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
)
TICKER_TO_COMPANY_NAME = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "WIPRO.NS": "Wipro",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
}


def _empty_metrics(key_driver: str = "") -> dict:
    return {
        "headline_count": 0,
        "average_positive": 0.0,
        "average_negative": 0.0,
        "average_neutral": 0.0,
        "key_driver": key_driver,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "agent_version": AGENT_VERSION,
    }


def load_finbert_onnx_pipeline():
    logger.info("Loading FinBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    logger.info("FinBERT loaded successfully.")
    return (model, tokenizer)


def fetch_headlines(ticker: str) -> list[str]:
    company_name = TICKER_TO_COMPANY_NAME.get(
        ticker, ticker.replace(".NS", "").replace(".BSE", "")
    )
    rss_url = GOOGLE_NEWS_RSS_BASE.format(query=company_name.replace(" ", "+"))
    feed = feedparser.parse(
        rss_url, request_headers={"User-Agent": f"AgentGamma/{AGENT_VERSION}"}
    )
    if feed.bozo and (not feed.entries):
        raise RuntimeError(
            f"feedparser bozo error for '{ticker}': {feed.get('bozo_exception', 'unknown')}"
        )
    headlines: list[str] = []
    for entry in feed.entries[:MAX_HEADLINES]:
        title = entry.get("title", "").strip()
        if " - " in title:
            title = title.rsplit(" - ", 1)[0].strip()
        if title:
            headlines.append(title)
    if not headlines:
        logger.warning(
            "No headlines returned for ticker '%s' (url=%s)", ticker, rss_url
        )
    return headlines


def classify_headlines(headlines: list[str], sentiment_classifier) -> list[dict]:
    model, tokenizer = sentiment_classifier
    id2label = model.config.id2label
    classified_results: list[dict] = []
    for headline in headlines:
        inputs = tokenizer(
            headline, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        scores_by_label = {
            id2label[i]: round(probs[i].item(), 4) for i in range(len(id2label))
        }
        classified_results.append(
            {
                "headline": headline,
                "positive": scores_by_label.get("positive", 0.0),
                "negative": scores_by_label.get("negative", 0.0),
                "neutral": scores_by_label.get("neutral", 0.0),
            }
        )
    return classified_results


def aggregate_sentiment_score(classified_headlines: list[dict]) -> float:
    if not classified_headlines:
        return 5.0
    weighted_positive = 0.0
    weighted_negative = 0.0
    total_weight = 0.0
    for i, item in enumerate(classified_headlines):
        recency_weight = 1.0 / (i + 1)
        confidence_weight = 1.0 - item["neutral"]
        combined_weight = recency_weight * confidence_weight
        weighted_positive += item["positive"] * combined_weight
        weighted_negative += item["negative"] * combined_weight
        total_weight += combined_weight
    if total_weight == 0.0:
        return 5.0
    avg_positive = weighted_positive / total_weight
    avg_negative = weighted_negative / total_weight
    net_sentiment = avg_positive - avg_negative
    score = (net_sentiment + 1.0) / 2.0 * 10.0
    return round(min(10.0, max(0.0, score)), 2)


def build_key_driver(classified_headlines: list[dict]) -> str:
    if not classified_headlines:
        return "No headlines found for this ticker."
    most_positive = max(classified_headlines, key=lambda item: item["positive"])
    positive_part = f"Most positive: '{most_positive['headline'][:80]}' ({round(most_positive['positive'] * 100)}% confidence)."
    candidates = [
        h for h in classified_headlines if h["negative"] > NEGATIVE_SIGNAL_THRESHOLD
    ]
    if candidates:
        most_negative = max(candidates, key=lambda item: item["negative"])
        negative_part = f" Most negative: '{most_negative['headline'][:80]}' ({round(most_negative['negative'] * 100)}% confidence)."
    else:
        negative_part = " No significant negative signals detected."
    return positive_part + negative_part


def run_agent_gamma(ticker: str, sentiment_classifier) -> dict:
    try:
        headlines = fetch_headlines(ticker)
        classified = classify_headlines(headlines, sentiment_classifier)
        sentiment_score = aggregate_sentiment_score(classified)
        key_driver = build_key_driver(classified)
        headline_count = len(classified)
        avg_positive = (
            round(sum((item["positive"] for item in classified)) / headline_count, 4)
            if classified
            else 0.0
        )
        avg_negative = (
            round(sum((item["negative"] for item in classified)) / headline_count, 4)
            if classified
            else 0.0
        )
        avg_neutral = (
            round(sum((item["neutral"] for item in classified)) / headline_count, 4)
            if classified
            else 0.0
        )
        return {
            "agent_id": "gamma_sentiment",
            "status": "success",
            "normalized_score": sentiment_score,
            "raw_metrics": {
                "headline_count": headline_count,
                "average_positive": avg_positive,
                "average_negative": avg_negative,
                "average_neutral": avg_neutral,
                "key_driver": key_driver,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "agent_version": AGENT_VERSION,
            },
        }
    except Exception as error:
        logger.exception("Agent Gamma failed for ticker '%s': %s", ticker, error)
        metrics = _empty_metrics(f"Agent Gamma failed: {error}")
        return {
            "agent_id": "gamma_sentiment",
            "status": "fallback",
            "normalized_score": 5.0,
            "raw_metrics": metrics,
        }
