import asyncio
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from orchestrator import analyze
from agents.agent_gamma import TICKER_TO_COMPANY_NAME

app = FastAPI(
    title="Stock Analysis API",
    description="Multi-agent stock analysis backend — Fundamental, Technical, Sentiment.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class AnalyzeRequest(BaseModel):
    ticker: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/tickers")
def get_supported_tickers():
    tickers = [
        {"ticker": ticker, "company": company}
        for ticker, company in TICKER_TO_COMPANY_NAME.items()
    ]
    return {"supported_tickers": tickers}


@app.post("/analyze")
async def analyze_ticker(request: AnalyzeRequest):
    ticker = request.ticker.strip().upper()
    if ticker not in TICKER_TO_COMPANY_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Ticker '{ticker}' is not supported. Call GET /tickers for the full list.",
        )
    result = await analyze(ticker)
    if result.get("verdict") == "ERROR":
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {result.get('error', 'Unknown error')}",
        )
    return result
