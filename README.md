# FinAgent: Multi-Agent Stock Analysis System

FinAgent is a scalable, multi-agent AI system that performs parallel analysis on Indian equities (NSE). It combines Fundamental (LLM-based PDF extraction), Technical (Indicator confluence), and Sentiment (FinBERT ONNX) analysis to generate a composite score and a BUY/HOLD/SELL verdict.

## 📁 Project Structure

```text
FinAgentt/
├── agents/                     # The 3 core analysis agents
│   ├── __init__.py
│   ├── agent_alpha.py          # Fundamental Analysis (pdfplumber + Groq Llama 3 8B)
│   ├── agent_beta.py           # Technical Analysis (yfinance + 7 indicators)
│   └── agent_gamma.py          # Sentiment Analysis (Google News RSS + FinBERT pipeline)
│
├── data/                       # Local data storage
│   └── RELIANCE.NS_earnings.pdf # Example earnings report for Agent Alpha
│
├── .env                        # Environment variables (API keys)
├── main.py                     # FastAPI backend application & routes
├── master_node.py              # Decision engine for composite scoring and verdicts
├── orchestrator.py             # Parallel dispatcher for agents
├── requirements.txt            # Python dependencies
├── run_demo.bat                # 1-click script to start the backend and frontend
└── streamlit_app.py            # Premium dark-themed interactive demo UI
```

## 🚀 Setup Instructions

### 1. Prerequisites
- **Python 3.10+** (64-bit recommended)
- A **Groq API Key** (for Llama 3).

### 2. Environment Setup
1. Clone the repository and navigate to the project root.
2. Ensure your `.env` file contains your valid Groq API key:
   ```env
   GROQ_API_KEY=gsk_your_actual_key_here
   ```
3. Install the required dependencies:
   ```cmd
   pip install -r requirements.txt
   ```
   > **Note:** The first time you run the sentiment agent, it will download the HuggingFace `ProsusAI/finbert` model automatically. This requires an active internet connection.

### 3. How to Run

You can run the full system using the included batch script:

```cmd
run_demo.bat
```

**Alternatively, to run the components manually:**

1. **Start the FastAPI Backend:**
   Open a terminal and run:
   ```cmd
   uvicorn main:app --host 127.0.0.1 --port 8000
   ```
   The backend will start at `http://127.0.0.1:8000`. You can check the health endpoint at `http://127.0.0.1:8000/health`.

2. **Start the Streamlit Frontend:**
   Open a *second* terminal and run:
   ```cmd
   streamlit run streamlit_app.py
   ```
   This will automatically open the beautiful dark-themed FinAgent dashboard in your default web browser (usually at `http://localhost:8501`).

### 4. Using the Demo

- Select a ticker from the dropdown (e.g., `RELIANCE.NS`). 
  - *Note: Only `RELIANCE.NS` has an accompanying PDF in the `data/` folder for fundamental analysis. Other tickers will trigger a graceful "fallback" mode for Agent Alpha.*
- Click **"⚡ ANALYZE"**.
- The three agents will run in parallel, and the Master Node will synthesize their results into a final verdict, displayed alongside a radar chart and detailed breakdowns.

---
*Disclaimer: This tool is for demonstration and educational purposes only. It is NOT financial advice.*
