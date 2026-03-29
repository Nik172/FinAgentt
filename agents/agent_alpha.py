import os
import json
import pdfplumber
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER = os.path.join(PROJECT_ROOT, "data")
PDF_SUMMARY_PAGES = 2
PDF_MAX_EXTRA_PAGES = 12
FINANCIAL_PAGE_KEYWORDS = [
    "profit",
    "loss",
    "revenue",
    "turnover",
    "earnings",
    "ebitda",
    "cash flow",
    "balance sheet",
    "equity",
    "liabilities",
    "assets",
    "debt",
    "interest",
    "tax",
    "margin",
    "crore",
    "lakhs",
    "₹",
]
SECTOR_BENCHMARKS = {
    "IT": {
        "debt_to_equity": (2.0, 0.0),
        "profit_margin": (0.0, 25.0),
        "operating_margin": (0.0, 30.0),
        "free_cash_flow_crores": (0.0, 4000.0),
        "roe": (0.0, 30.0),
        "revenue_growth_pct": (-5.0, 20.0),
        "interest_coverage": (1.0, 50.0),
        "eps_growth_pct": (-10.0, 25.0),
    },
    "BANKING": {
        "debt_to_equity": (15.0, 0.0),
        "profit_margin": (0.0, 25.0),
        "operating_margin": (0.0, 35.0),
        "free_cash_flow_crores": (0.0, 8000.0),
        "roe": (0.0, 18.0),
        "revenue_growth_pct": (-5.0, 18.0),
        "interest_coverage": (1.0, 5.0),
        "eps_growth_pct": (-10.0, 20.0),
    },
    "FMCG": {
        "debt_to_equity": (2.5, 0.0),
        "profit_margin": (0.0, 20.0),
        "operating_margin": (0.0, 25.0),
        "free_cash_flow_crores": (0.0, 5000.0),
        "roe": (0.0, 40.0),
        "revenue_growth_pct": (-2.0, 15.0),
        "interest_coverage": (1.0, 30.0),
        "eps_growth_pct": (-5.0, 20.0),
    },
    "INFRASTRUCTURE": {
        "debt_to_equity": (5.0, 0.0),
        "profit_margin": (0.0, 15.0),
        "operating_margin": (0.0, 20.0),
        "free_cash_flow_crores": (0.0, 6000.0),
        "roe": (0.0, 15.0),
        "revenue_growth_pct": (-5.0, 20.0),
        "interest_coverage": (1.0, 4.0),
        "eps_growth_pct": (-10.0, 20.0),
    },
    "DEFAULT": {
        "debt_to_equity": (3.0, 0.0),
        "profit_margin": (0.0, 30.0),
        "operating_margin": (0.0, 30.0),
        "free_cash_flow_crores": (0.0, 5000.0),
        "roe": (0.0, 25.0),
        "revenue_growth_pct": (-5.0, 20.0),
        "interest_coverage": (1.0, 15.0),
        "eps_growth_pct": (-10.0, 25.0),
    },
}
TICKER_TO_SECTOR = {
    "TCS": "IT",
    "INFY": "IT",
    "WIPRO": "IT",
    "HCLTECH": "IT",
    "TECHM": "IT",
    "LTIM": "IT",
    "MPHASIS": "IT",
    "COFORGE": "IT",
    "HDFCBANK": "BANKING",
    "ICICIBANK": "BANKING",
    "SBIN": "BANKING",
    "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING",
    "INDUSINDBK": "BANKING",
    "BAJFINANCE": "BANKING",
    "BAJAJFINSV": "BANKING",
    "HINDUNILVR": "FMCG",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG",
    "DABUR": "FMCG",
    "MARICO": "FMCG",
    "RELIANCE": "INFRASTRUCTURE",
    "NTPC": "INFRASTRUCTURE",
    "POWERGRID": "INFRASTRUCTURE",
    "ONGC": "INFRASTRUCTURE",
    "ADANIPORTS": "INFRASTRUCTURE",
    "LT": "INFRASTRUCTURE",
}
METRIC_WEIGHTS = {
    "debt_to_equity": 0.2,
    "profit_margin": 0.15,
    "operating_margin": 0.15,
    "free_cash_flow_crores": 0.15,
    "roe": 0.15,
    "revenue_growth_pct": 0.1,
    "interest_coverage": 0.05,
    "eps_growth_pct": 0.05,
}
SYSTEM_PROMPT = 'You are a quantitative financial analyst specialising in Indian equity markets.\nYou will receive structured text extracted from a quarterly or annual earnings PDF,\nincluding both raw text and formatted table data.\n\nExtract exactly these eight financial metrics and one key driver sentence.\nAll monetary values use Indian accounting: crores (1 crore = 10 million).\n\nRespond with ONLY a valid JSON object — no preamble, no explanation, no markdown fences.\n\nRequired keys:\n  - debt_to_equity: float. Total Debt / Total Shareholders Equity. Lower = safer. Typical 0.0–5.0.\n  - profit_margin: float as percentage (e.g. 18.5 = 18.5%). Net Profit / Revenue × 100. Can be negative.\n  - operating_margin: float as percentage. EBIT or Operating Profit / Revenue × 100. Can be negative.\n  - free_cash_flow_crores: float in crores INR. Operating Cash Flow minus CapEx. Can be negative.\n  - roe: float as percentage. Net Profit / Shareholders Equity × 100. Return on Equity.\n  - revenue_growth_pct: float as percentage. YoY revenue growth. Positive = growing.\n  - interest_coverage: float. EBIT / Interest Expense. Higher = safer. Null if company has no debt.\n  - eps_growth_pct: float as percentage. YoY EPS growth. Can be negative.\n  - key_driver: single sentence describing the primary financial strength or weakness seen in the data.\n  - extraction_confidence: one of "high", "medium", "low" — your confidence that the numbers are accurate.\n\nIf a metric genuinely cannot be found or calculated from the provided text, use null for that field.\nDo NOT guess or fabricate values. Prefer null over an uncertain estimate.\nFor Indian companies, watch for values in lakhs (divide by 100 to get crores) or in rupees (divide by 10,000,000 for crores).'


def _is_financial_page(text: str) -> bool:
    text_lower = text.lower()
    matches = sum((1 for kw in FINANCIAL_PAGE_KEYWORDS if kw in text_lower))
    return matches >= 3


def _format_table(table: list[list]) -> str:
    if not table:
        return ""
    formatted_rows = []
    for i, row in enumerate(table):
        cleaned = [str(cell).strip() if cell else "-" for cell in row]
        row_str = " | ".join(cleaned)
        if i == 0:
            formatted_rows.append(f"[HEADER] {row_str}")
            formatted_rows.append("-" * min(len(row_str), 80))
        else:
            formatted_rows.append(row_str)
    return "\n".join(formatted_rows)


def extract_text_from_pdf(pdf_path: str) -> str:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        summary_end = min(PDF_SUMMARY_PAGES, total_pages)
        for page in pdf.pages[:summary_end]:
            text = page.extract_text() or ""
            chunks.append(f"[PAGE {page.page_number} — SUMMARY]\n{text}")
            for table in page.extract_tables():
                chunks.append(
                    f"[TABLE — PAGE {page.page_number}]\n{_format_table(table)}"
                )
        financial_pages_found = 0
        for page in pdf.pages[summary_end:]:
            if financial_pages_found >= PDF_MAX_EXTRA_PAGES:
                break
            text = page.extract_text() or ""
            if _is_financial_page(text):
                chunks.append(f"[PAGE {page.page_number} — FINANCIAL DATA]\n{text}")
                for table in page.extract_tables():
                    chunks.append(
                        f"[TABLE — PAGE {page.page_number}]\n{_format_table(table)}"
                    )
                financial_pages_found += 1
    if not chunks:
        raise ValueError(f"No text could be extracted from {pdf_path}")
    combined = "\n\n".join(chunks)
    return combined[:12000]


def query_llm_for_metrics(raw_text: str) -> dict:
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        temperature=0.05,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Earnings report text:\n\n{raw_text}"},
        ],
    )
    raw_output = response.choices[0].message.content.strip()
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        raw_output = "\n".join(
            (line for line in lines if not line.strip().startswith("```"))
        ).strip()
    parsed = json.loads(raw_output)
    null_fields = [
        key
        for key in [
            "debt_to_equity",
            "profit_margin",
            "operating_margin",
            "free_cash_flow_crores",
            "roe",
            "revenue_growth_pct",
            "interest_coverage",
            "eps_growth_pct",
        ]
        if parsed.get(key) is None
    ]
    defaults = {
        "debt_to_equity": 1.0,
        "profit_margin": 0.0,
        "operating_margin": 0.0,
        "free_cash_flow_crores": 0.0,
        "roe": 0.0,
        "revenue_growth_pct": 0.0,
        "interest_coverage": 5.0,
        "eps_growth_pct": 0.0,
        "key_driver": "Insufficient data to determine key driver.",
        "extraction_confidence": "low",
    }
    for key, default_val in defaults.items():
        if parsed.get(key) is None:
            parsed[key] = default_val
    parsed["_null_fields"] = null_fields
    return parsed


def _detect_sector(ticker: str) -> str:
    base = ticker.upper().replace(".NS", "").replace(".BO", "")
    return TICKER_TO_SECTOR.get(base, "DEFAULT")


def _normalize(value: float, worst: float, best: float) -> float:
    if worst == best:
        return 5.0
    lo, hi = (min(worst, best), max(worst, best))
    clamped = max(lo, min(value, hi))
    raw = (clamped - worst) / (best - worst)
    return round(raw * 10.0, 2)


def compute_weighted_score(metrics: dict, sector: str) -> tuple[float, dict]:
    benchmarks = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["DEFAULT"])
    per_metric_scores = {}
    for metric, weight in METRIC_WEIGHTS.items():
        value = metrics.get(metric, 0.0) or 0.0
        worst, best = benchmarks[metric]
        per_metric_scores[metric] = {
            "raw_value": value,
            "score_0_to_10": _normalize(value, worst, best),
            "weight": weight,
        }
    weighted_sum = sum(
        (v["score_0_to_10"] * v["weight"] for v in per_metric_scores.values())
    )
    return (round(weighted_sum, 2), per_metric_scores)


def run_agent_alpha(ticker: str) -> dict:
    pdf_path = os.path.join(PDF_FOLDER, f"{ticker}_earnings.pdf")
    sector = _detect_sector(ticker)
    try:
        raw_text = extract_text_from_pdf(pdf_path)
        metrics = query_llm_for_metrics(raw_text)
        null_fields = metrics.pop("_null_fields", [])
        final_score, per_metric_scores = compute_weighted_score(metrics, sector)
        if len(null_fields) == 0:
            status = "success"
        elif len(null_fields) <= 3:
            status = "partial"
        else:
            status = "partial_heavy"
        return {
            "agent_id": "alpha_fundamental",
            "status": status,
            "normalized_score": final_score,
            "sector_detected": sector,
            "extraction_confidence": metrics.get("extraction_confidence", "low"),
            "raw_metrics": {
                "debt_to_equity": metrics["debt_to_equity"],
                "profit_margin": metrics["profit_margin"],
                "operating_margin": metrics["operating_margin"],
                "free_cash_flow_crores": metrics["free_cash_flow_crores"],
                "roe": metrics["roe"],
                "revenue_growth_pct": metrics["revenue_growth_pct"],
                "interest_coverage": metrics["interest_coverage"],
                "eps_growth_pct": metrics["eps_growth_pct"],
                "key_driver": metrics["key_driver"],
                "null_fields": null_fields,
            },
            "score_breakdown": per_metric_scores,
        }
    except FileNotFoundError:
        return {
            "agent_id": "alpha_fundamental",
            "status": "fallback",
            "normalized_score": 5.0,
            "sector_detected": sector,
            "extraction_confidence": "none",
            "raw_metrics": {
                "key_driver": f"PDF not found: {ticker}_earnings.pdf",
                "null_fields": ["all"],
            },
            "score_breakdown": {},
        }
    except Exception as error:
        return {
            "agent_id": "alpha_fundamental",
            "status": "fallback",
            "normalized_score": 5.0,
            "sector_detected": sector,
            "extraction_confidence": "none",
            "raw_metrics": {
                "key_driver": f"Agent Alpha failed: {str(error)}",
                "null_fields": ["all"],
            },
            "score_breakdown": {},
        }


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "HDFCBANK.NS"
    result = run_agent_alpha(ticker)
