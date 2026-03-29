"""
FinAgent — Streamlit Demo Frontend
Connects to the FastAPI backend and displays the multi-agent stock analysis.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import time

# ── config ─────────────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8000"

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinAgent — Multi-Agent Stock Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500&display=swap');

    /* Dark theme foundation */
    .stApp {
        background: linear-gradient(180deg, #080b0f 0%, #0c1017 50%, #0e1318 100%);
    }

    /* Remove default streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Grid overlay effect */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(0,212,170,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,212,170,0.02) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }

    /* Header styling */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 42px;
        font-weight: 800;
        color: #e8edf2;
        letter-spacing: -1.5px;
        margin-bottom: 0;
        line-height: 1.1;
    }
    .main-header span {
        background: linear-gradient(135deg, #00d4aa 0%, #00a882 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #5a6a7a;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Verdict display */
    .verdict-container {
        text-align: center;
        padding: 32px 0;
    }
    .verdict-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #5a6a7a;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .verdict-text {
        font-size: 72px;
        font-weight: 900;
        letter-spacing: -3px;
        line-height: 1;
        margin-bottom: 8px;
    }
    .verdict-buy { color: #00d4aa; text-shadow: 0 0 60px rgba(0,212,170,0.4); }
    .verdict-sell { color: #ff5c5c; text-shadow: 0 0 60px rgba(255,92,92,0.4); }
    .verdict-hold { color: #f0b429; text-shadow: 0 0 60px rgba(240,180,41,0.4); }

    .confidence-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        color: #8a9aaa;
    }
    .confidence-value {
        font-weight: 600;
        color: #00d4aa;
    }

    /* Agent card styling */
    .agent-card {
        background: linear-gradient(135deg, #0e1318 0%, #141a22 100%);
        border: 1px solid #1e2830;
        border-radius: 16px;
        padding: 24px;
        transition: border-color 0.3s, transform 0.2s;
    }
    .agent-card:hover {
        border-color: #2a3540;
        transform: translateY(-2px);
    }

    .agent-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #5a6a7a;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .agent-name {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        font-weight: 700;
        color: #e8edf2;
        margin-bottom: 16px;
    }
    .agent-score {
        font-family: 'Inter', sans-serif;
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -2px;
        line-height: 1;
        margin-bottom: 8px;
    }
    .score-green { color: #00d4aa; }
    .score-gold { color: #f0b429; }
    .score-red { color: #ff5c5c; }

    .driver-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #5a6a7a;
        line-height: 1.6;
        margin-top: 12px;
    }

    /* Metric rows */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #1e2830;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-key {
        font-size: 12px;
        color: #5a6a7a;
    }
    .metric-val {
        font-size: 13px;
        font-weight: 500;
        color: #e8edf2;
    }

    /* Badge styling */
    .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-green { background: rgba(0,212,170,0.15); color: #00d4aa; }
    .badge-gold { background: rgba(240,180,41,0.15); color: #f0b429; }
    .badge-red { background: rgba(255,92,92,0.15); color: #ff5c5c; }

    /* Section title */
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #5a6a7a;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 16px;
        margin-top: 32px;
    }

    /* Weight pills */
    .weight-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        padding: 5px 12px;
        border-radius: 20px;
        border: 1px solid #2a3540;
        color: #5a6a7a;
        display: inline-block;
        margin-right: 8px;
    }

    /* Status pill */
    .status-online {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #00d4aa;
        border: 1px solid rgba(0,212,170,0.3);
        padding: 5px 14px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .status-online::before {
        content: '';
        width: 6px;
        height: 6px;
        background: #00d4aa;
        border-radius: 50%;
        display: inline-block;
    }

    /* Footer */
    .footer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #3a4a5a;
        text-align: center;
        margin-top: 48px;
        padding-top: 24px;
        border-top: 1px solid #1e2830;
    }

    /* Composite score */
    .composite-num {
        font-family: 'Inter', sans-serif;
        font-size: 64px;
        font-weight: 900;
        letter-spacing: -3px;
        line-height: 1;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom selectbox styling */
    .stSelectbox > div > div {
        background: #0e1318;
        border-color: #2a3540;
        color: #e8edf2;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa 0%, #00a882 100%);
        color: #000;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 14px;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        letter-spacing: 0.5px;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00a882 0%, #008866 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0,212,170,0.3);
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #1e2830 50%, transparent 100%);
        margin: 32px 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0e1318 0%, #141a22 100%);
        border: 1px solid #1e2830;
        border-radius: 12px;
        padding: 16px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── helper functions ───────────────────────────────────────────────────────────


def get_score_color_class(score: float) -> str:
    if score >= 6.5:
        return "score-green"
    elif score >= 4.0:
        return "score-gold"
    return "score-red"


def get_verdict_class(verdict: str) -> str:
    v = verdict.upper()
    if v == "BUY":
        return "verdict-buy"
    elif v == "SELL":
        return "verdict-sell"
    return "verdict-hold"


def get_badge_class(score: float) -> str:
    if score >= 6.5:
        return "badge-green"
    elif score >= 4.0:
        return "badge-gold"
    return "badge-red"


def check_backend() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── header ─────────────────────────────────────────────────────────────────────

col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.markdown(
        '<div class="main-header">Fin<span>Agent</span></div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Multi-Agent Stock Analysis System</div>',
        unsafe_allow_html=True,
    )
with col_status:
    backend_up = check_backend()
    if backend_up:
        st.markdown(
            '<div class="status-online">SYSTEM ONLINE</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="font-family: JetBrains Mono, monospace; font-size: 11px; '
            "color: #ff5c5c; border: 1px solid rgba(255,92,92,0.3); padding: 5px 14px; "
            'border-radius: 20px; display: inline-flex; align-items: center; gap: 6px;">'
            "⬤ BACKEND OFFLINE</div>",
            unsafe_allow_html=True,
        )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── sidebar: info ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Architecture")
    st.markdown("""
    **Agent Alpha** — Fundamental Analysis
    - PDF parsing with pdfplumber
    - Llama 3 8B via Groq API
    - 8 financial metrics

    **Agent Beta** — Technical Analysis
    - yfinance market data
    - 7 indicators (RSI, MACD, BB, SMA, Stochastic, ATR, Volume)
    - Weighted confluence scoring

    **Agent Gamma** — Sentiment Analysis
    - Google News RSS feed
    - FinBERT ONNX classification
    - Recency-weighted aggregation

    **Master Node** — Decision Engine
    - Weighted composite scoring
    - BUY / HOLD / SELL verdict
    - Confidence calculation
    """)

# ── ticker selection ───────────────────────────────────────────────────────────

# For now, only RELIANCE.NS has a PDF in data/
TICKERS = {
    "RELIANCE.NS": "Reliance Industries",
}

col_input, col_btn = st.columns([3, 1])

with col_input:
    selected_ticker = st.selectbox(
        "Select Ticker",
        options=list(TICKERS.keys()),
        format_func=lambda t: f"{t}  —  {TICKERS[t]}",
        label_visibility="collapsed",
    )

with col_btn:
    analyze_clicked = st.button("⚡ ANALYZE", use_container_width=True)


# ── analysis ───────────────────────────────────────────────────────────────────

if analyze_clicked:
    if not backend_up:
        st.error(
            "**Backend is offline.** Start the FastAPI server first:\n\n"
            "```\nuvicorn main:app --host 127.0.0.1 --port 8000\n```"
        )
        st.stop()

    # ── progress animation ─────────────────────────────────────────────────
    progress_bar = st.progress(0)
    status_text = st.empty()

    stages = [
        (0.15, "🔍 Agent Alpha: Extracting financial data from earnings PDF..."),
        (0.35, "📈 Agent Beta: Computing 7 technical indicators..."),
        (0.55, "📰 Agent Gamma: Analyzing news sentiment with FinBERT..."),
        (0.75, "🧠 Master Node: Computing weighted composite verdict..."),
        (0.90, "✨ Assembling final analysis report..."),
    ]

    # Start async request
    for progress_val, stage_msg in stages:
        status_text.markdown(
            f'<div style="font-family: JetBrains Mono, monospace; font-size: 12px; '
            f'color: #5a6a7a;">{stage_msg}</div>',
            unsafe_allow_html=True,
        )
        progress_bar.progress(progress_val)
        time.sleep(0.3)

    try:
        response = requests.post(
            f"{API_BASE}/analyze",
            json={"ticker": selected_ticker},
            timeout=120,
        )
        progress_bar.progress(1.0)
        time.sleep(0.2)
        progress_bar.empty()
        status_text.empty()

        if response.status_code != 200:
            st.error(
                f"Analysis failed: {response.json().get('detail', 'Unknown error')}"
            )
            st.stop()

        data = response.json()

    except requests.exceptions.Timeout:
        progress_bar.empty()
        status_text.empty()
        st.error(
            "Request timed out. The analysis is taking too long — please try again."
        )
        st.stop()
    except requests.exceptions.ConnectionError:
        progress_bar.empty()
        status_text.empty()
        st.error("Connection error. Is the backend running at `127.0.0.1:8000`?")
        st.stop()

    # ── store result in session state ──────────────────────────────────────
    st.session_state["result"] = data
    st.session_state["ticker"] = selected_ticker


# ── display results ────────────────────────────────────────────────────────────

if "result" in st.session_state:
    data = st.session_state["result"]
    ticker = st.session_state["ticker"]
    company = TICKERS.get(ticker, ticker)

    verdict = data["verdict"]
    confidence = data["confidence"]
    composite = data["composite_score"]
    agents = data["agents"]
    weights = data["weights"]
    quality = data.get("data_quality", "unknown")

    # ── Verdict Hero Section ───────────────────────────────────────────────
    col_ticker_info, col_verdict = st.columns([2, 1])

    with col_ticker_info:
        st.markdown(
            f'<div class="sub-header" style="margin-bottom: 8px;">NSE · EQUITY</div>'
            f'<div class="main-header" style="font-size: 36px;">{company}</div>'
            f'<div style="font-family: JetBrains Mono, monospace; font-size: 13px; '
            f'color: #5a6a7a; margin-top: 8px;">{ticker} &nbsp;·&nbsp; '
            f'<span class="badge {"badge-green" if quality == "full" else "badge-gold"}">'
            f'{"✓ FULL DATA" if quality == "full" else "⚠ PARTIAL DATA"}</span></div>',
            unsafe_allow_html=True,
        )

    with col_verdict:
        verdict_class = get_verdict_class(verdict)
        st.markdown(
            f"""
        <div class="verdict-container">
            <div class="verdict-label">VERDICT</div>
            <div class="verdict-text {verdict_class}">{verdict}</div>
            <div class="confidence-text">
                Confidence <span class="confidence-value">{confidence}%</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Agent Score Cards ──────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Agent Analysis</div>', unsafe_allow_html=True
    )

    col_alpha, col_beta, col_gamma = st.columns(3)

    agent_configs = [
        (
            col_alpha,
            "Agent Alpha",
            "Fundamental",
            "pdfplumber + Llama 3",
            agents["fundamental"]["score"],
            agents["fundamental"].get("key_driver", ""),
            agents["fundamental"]["status"],
        ),
        (
            col_beta,
            "Agent Beta",
            "Technical",
            "yfinance + 7 Indicators",
            agents["technical"]["score"],
            agents["technical"].get("key_driver", ""),
            agents["technical"]["status"],
        ),
        (
            col_gamma,
            "Agent Gamma",
            "Sentiment",
            "Google RSS + FinBERT ONNX",
            agents["sentiment"]["score"],
            agents["sentiment"].get("key_driver", ""),
            agents["sentiment"]["status"],
        ),
    ]

    for (
        col,
        agent_label,
        agent_name,
        agent_tech,
        score,
        driver,
        status,
    ) in agent_configs:
        with col:
            score_class = get_score_color_class(score)
            status_badge = (
                '<span class="badge badge-green">SUCCESS</span>'
                if status == "success"
                else (
                    '<span class="badge badge-gold">PARTIAL</span>'
                    if "partial" in status
                    else '<span class="badge badge-red">FALLBACK</span>'
                )
            )
            st.markdown(
                f"""
            <div class="agent-card">
                <div class="agent-title">{agent_label} {status_badge}</div>
                <div class="agent-name">{agent_name}</div>
                <div style="font-family: JetBrains Mono, monospace; font-size: 10px; color: #3a4a5a; margin-bottom: 16px;">{agent_tech}</div>
                <div class="agent-score {score_class}">{score}</div>
                <div style="height: 4px; background: #1e2830; border-radius: 2px; margin-bottom: 12px; overflow: hidden;">
                    <div style="height: 100%; width: {score * 10}%; background: {'#00d4aa' if score >= 6.5 else '#f0b429' if score >= 4.0 else '#ff5c5c'}; border-radius: 2px;"></div>
                </div>
                <div class="driver-text">{driver[:120] if driver else 'No data available.'}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Composite Score + Radar Chart ──────────────────────────────────────
    st.markdown(
        '<div class="section-title">Master Node Output</div>', unsafe_allow_html=True
    )

    col_composite, col_radar = st.columns([1, 1])

    with col_composite:
        composite_color = (
            "#00d4aa"
            if composite >= 6.5
            else "#f0b429" if composite >= 4.0 else "#ff5c5c"
        )
        threshold_diff = (
            round(composite - 6.5, 1)
            if verdict == "BUY"
            else round(4.0 - composite, 1) if verdict == "SELL" else 0
        )
        threshold_text = (
            f"Threshold exceeded by {abs(threshold_diff)} pts."
            if verdict != "HOLD"
            else "Score is in the HOLD range."
        )

        st.markdown(
            f"""
        <div class="agent-card" style="margin-bottom: 16px;">
            <div class="agent-title">Composite Score</div>
            <div style="display: flex; align-items: center; gap: 24px; margin-top: 16px;">
                <div class="composite-num" style="color: {composite_color}; text-shadow: 0 0 40px {composite_color}40;">{composite}</div>
                <div>
                    <div style="font-family: Inter, sans-serif; font-size: 18px; font-weight: 700; color: #e8edf2; margin-bottom: 6px;">
                        {'Strong Buy Signal' if verdict == 'BUY' else 'Sell Signal' if verdict == 'SELL' else 'Hold Position'}
                    </div>
                    <div style="font-family: JetBrains Mono, monospace; font-size: 12px; color: #5a6a7a; line-height: 1.6;">
                        {threshold_text}
                    </div>
                    <div style="margin-top: 12px;">
                        <span class="weight-pill">Fund <strong style="color: #e8edf2;">{int(weights.get('fundamental', 0.45) * 100)}%</strong></span>
                        <span class="weight-pill">Tech <strong style="color: #e8edf2;">{int(weights.get('technical', 0.25) * 100)}%</strong></span>
                        <span class="weight-pill">Sent <strong style="color: #e8edf2;">{int(weights.get('sentiment', 0.30) * 100)}%</strong></span>
                    </div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_radar:
        fig = go.Figure()

        categories = [
            "Fundamental",
            "Technical",
            "Sentiment",
            "Confidence",
            "Composite",
        ]
        values = [
            agents["fundamental"]["score"],
            agents["technical"]["score"],
            agents["sentiment"]["score"],
            confidence / 10,  # scale to 0-10
            composite,
        ]
        # Close the polygon
        values_plot = values + [values[0]]
        categories_plot = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_plot,
                theta=categories_plot,
                fill="toself",
                fillcolor="rgba(0, 212, 170, 0.12)",
                line=dict(color="rgba(0, 212, 170, 0.8)", width=2),
                marker=dict(
                    size=8, color="#00d4aa", line=dict(color="#080b0f", width=2)
                ),
                name=ticker,
            )
        )

        fig.update_layout(
            polar=dict(
                bgcolor="transparent",
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(family="JetBrains Mono", size=9, color="#5a6a7a"),
                    gridcolor="#1e2830",
                    linecolor="#1e2830",
                ),
                angularaxis=dict(
                    tickfont=dict(family="Inter", size=11, color="#8a9aaa"),
                    gridcolor="#1e2830",
                    linecolor="#1e2830",
                ),
            ),
            showlegend=False,
            paper_bgcolor="transparent",
            plot_bgcolor="transparent",
            margin=dict(l=60, r=60, t=40, b=40),
            height=350,
        )

        st.plotly_chart(fig, use_container_width=True, key="radar_chart")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── raw data section ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Raw Agent Data</div>', unsafe_allow_html=True
    )

    with st.expander("📋 Full Agent Alpha (Fundamental) Response", expanded=False):
        st.json(agents["fundamental"])

    with st.expander("📈 Full Agent Beta (Technical) Response", expanded=False):
        st.json(agents["technical"])

    with st.expander("📰 Full Agent Gamma (Sentiment) Response", expanded=False):
        st.json(agents["sentiment"])

    with st.expander("🧠 Full API Response", expanded=False):
        st.json(data)

    # ── footer ─────────────────────────────────────────────────────────────
    st.markdown(
        f"""
    <div class="footer">
        FinAgent v1.0 · Multi-Agent Stock Analysis System · <em>NOT FINANCIAL ADVICE</em>
    </div>
    """,
        unsafe_allow_html=True,
    )

else:
    # ── landing state ──────────────────────────────────────────────────────
    st.markdown(
        """
    <div style="text-align: center; padding: 80px 0 40px;">
        <div style="font-size: 80px; margin-bottom: 16px;">📊</div>
        <div style="font-family: Inter, sans-serif; font-size: 24px; font-weight: 700; color: #e8edf2; margin-bottom: 8px;">
            Select a Ticker & Hit Analyze
        </div>
        <div style="font-family: JetBrains Mono, monospace; font-size: 13px; color: #5a6a7a; line-height: 1.8;">
            Three AI agents will analyze the stock simultaneously —<br>
            Fundamental · Technical · Sentiment
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Architecture visual
    st.markdown(
        """
    <div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; margin: 32px 0;">
        <div class="agent-card" style="flex: 1; min-width: 200px; max-width: 280px; text-align: center;">
            <div style="font-size: 32px; margin-bottom: 12px;">📑</div>
            <div class="agent-name">Agent Alpha</div>
            <div class="driver-text">Extracts 8 financial metrics from earnings PDFs using Llama 3</div>
        </div>
        <div class="agent-card" style="flex: 1; min-width: 200px; max-width: 280px; text-align: center;">
            <div style="font-size: 32px; margin-bottom: 12px;">📈</div>
            <div class="agent-name">Agent Beta</div>
            <div class="driver-text">Computes 7 technical indicators with weighted confluence scoring</div>
        </div>
        <div class="agent-card" style="flex: 1; min-width: 200px; max-width: 280px; text-align: center;">
            <div style="font-size: 32px; margin-bottom: 12px;">📰</div>
            <div class="agent-name">Agent Gamma</div>
            <div class="driver-text">Classifies news headlines with FinBERT sentiment analysis</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="footer">
        FinAgent v1.0 · Multi-Agent Stock Analysis System · <em>NOT FINANCIAL ADVICE</em>
    </div>
    """,
        unsafe_allow_html=True,
    )
