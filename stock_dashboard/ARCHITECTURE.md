# Architecture & Technical Stack — Stock Research Dashboard

## Table of Contents
1. [CV / Use-Case Description](#cv--use-case-description)
2. [System Overview](#system-overview)
3. [Repository Structure](#repository-structure)
4. [Technical Stack](#technical-stack)
5. [Application Architecture](#application-architecture)
6. [Core Data Pipeline](#core-data-pipeline)
7. [Analysis Modules](#analysis-modules)
8. [Data Flow](#data-flow)
9. [Advantages](#advantages)
10. [Limitations](#limitations)

---

## CV / Use-Case Description

**30-word use case:**

> Retail-focused AI stock dashboard that unifies live prices, candlestick charts, RSI/MACD/Bollinger-Band technicals, VADER-scored news sentiment, and GPT-4o trade analysis — all in one Streamlit view, zero infrastructure required.

**49-word CV bullet:**

> Built an AI-powered stock research dashboard with Python and Streamlit that aggregates live yfinance prices, multi-source news with VADER sentiment scoring, and interactive Plotly technical charts (RSI, MACD, Bollinger Bands, Stochastic), then synthesises all signals into GPT-4o trade-ready insights — deployed via GitHub Codespaces with no backend infrastructure.

---

## System Overview

**"Stock Research Dashboard"** is a single-page Streamlit web application for active retail traders. It pulls live market data from Yahoo Finance, scores news sentiment locally with VADER, renders interactive technical-analysis charts, and calls OpenAI's GPT-4o API to produce structured, trade-ready insights — all from one Python file, with no database and no custom backend.

```
Retail Trader (Browser)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  GitHub Codespaces / any Python host            │
│  ┌───────────────────────────────────────────┐  │
│  │   Streamlit Server  app.py  (port 8501)   │  │
│  │                                           │  │
│  │  ┌────────────┐  ┌──────────────────────┐ │  │
│  │  │  yfinance  │  │   pandas-ta (local)  │ │  │
│  │  │  (prices + │  │   RSI · MACD · BB    │ │  │
│  │  │   news)    │  │   Stochastic · EMA   │ │  │
│  │  └────────────┘  └──────────────────────┘ │  │
│  │                                           │  │
│  │  ┌────────────┐  ┌──────────────────────┐ │  │
│  │  │   VADER    │  │  OpenAI GPT-4o API   │ │  │
│  │  │ (sentiment │  │  (trade-ready text   │ │  │
│  │  │  scoring)  │  │   analysis)          │ │  │
│  │  └────────────┘  └──────────────────────┘ │  │
│  │                                           │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │   Plotly (interactive charts)       │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
        │
        ▼ (external API calls only)
  Yahoo Finance API  ·  OpenAI API
```

---

## Repository Structure

```
stock_dashboard/
├── app.py                  # Complete application (~280 lines)
├── requirements.txt        # Python dependencies
├── ARCHITECTURE.md         # This document
└── .streamlit/
    └── config.toml         # Streamlit server & UI settings
```

---

## Technical Stack

### Runtime Environment

| Layer | Technology | Detail |
|---|---|---|
| Cloud host | GitHub Codespaces (or any Python env) | Port 8501, public forwarding |
| Language | Python | 3.11+ |

### Application Framework

| Component | Package | Version |
|---|---|---|
| Web framework | Streamlit | `>= 1.50.0` |

### Market Data

| Component | Package | Version | Role |
|---|---|---|---|
| Price, history, fundamentals, news | yfinance | `>= 0.2.40` | Wraps Yahoo Finance; free, no API key |

### AI & NLP

| Component | Package | Version | Role |
|---|---|---|---|
| Trade-ready text analysis | openai | `>= 1.30.0` | GPT-4o via Chat Completions API |
| News sentiment scoring | vaderSentiment | `>= 3.3.2` | Rule-based; runs entirely offline |

### Technical Analysis

| Component | Package | Version | Role |
|---|---|---|---|
| Indicator computation | pandas-ta | `>= 0.3.14b` | EMA, Bollinger Bands, RSI, MACD, Stochastic — all computed locally on pandas DataFrames |

### Charting

| Component | Package | Version | Role |
|---|---|---|---|
| Interactive charts | plotly | `>= 5.18.0` | Candlestick, OHLCV, RSI, MACD, Stochastic subplots |

### Indicators Computed

| Indicator | Parameters | Purpose |
|---|---|---|
| EMA | 20, 50 periods | Trend direction, crossover signals |
| Bollinger Bands | 20 period, 2σ | Volatility envelope, squeeze detection |
| RSI | 14 period | Momentum; overbought/oversold (70/30) |
| MACD | 12, 26, 9 | Trend momentum, signal-line crossovers |
| Stochastic | 14, 3, 3 | Short-term momentum reversal (80/20) |

### Streamlit Configuration

```toml
[client]
toolbarMode = "minimal"       # Minimal chrome
showErrorDetails = false      # Hides raw tracebacks

[ui]
hideSidebarNav = true         # Single-page app
```

---

## Application Architecture

### Pattern

**Monolithic single-file** — the entire app lives in `app.py`. No separate:
- Backend API or microservices
- Database or persistent storage
- Message queue or background workers
- Build pipeline

### Sidebar (controls)

```
┌─────────────────────────────────┐
│  Sidebar                        │
│  ─────────────────────────────  │
│  Ticker input     (text field)  │
│  Timeframe        (selectbox)   │
│  OpenAI API key   (password)    │
│  Refresh button                 │
└─────────────────────────────────┘
```

### Four-tab main panel

```
┌────────────────────────────────────────────────────────────────┐
│  Tab 1: Price & Volume                                         │
│    Candlestick (OHLC) + EMA20 + EMA50 + Bollinger Bands        │
│    Volume bar chart (colour-coded green/red)                   │
├────────────────────────────────────────────────────────────────┤
│  Tab 2: Technical Analysis                                     │
│    Summary metrics row (RSI status · EMA trend · price pos.)   │
│    RSI(14) chart with 70/30 bands                              │
│    MACD(12,26,9) chart with signal line + histogram            │
│    Stochastic chart with 80/20 bands                           │
├────────────────────────────────────────────────────────────────┤
│  Tab 3: News & Sentiment                                       │
│    Overall VADER sentiment label + score                       │
│    Per-headline: sentiment score + icon + link + publisher     │
├────────────────────────────────────────────────────────────────┤
│  Tab 4: GPT-4o Insights                                        │
│    Sends price + technicals + sentiment → GPT-4o               │
│    Returns: Trend Summary · Bull/Bear Case · Levels · Setup    │
└────────────────────────────────────────────────────────────────┘
```

### Caching Strategy

| Data | Cache TTL | Reason |
|---|---|---|
| Ticker info (fundamentals) | 60 s | Slow-moving; avoids hammering Yahoo Finance |
| OHLCV history | 60 s | Near-real-time refresh within a trading session |
| News | 300 s (5 min) | Less time-sensitive than price |
| VADER sentiment | 300 s | Pure function of headlines; no need to re-score |

### State Management

| Key | Type | Content |
|---|---|---|
| `avg_sentiment` | `float` | VADER compound score passed from Tab 3 → Tab 4 |
| `gpt_result` | `str` | GPT-4o markdown response, persists across tab switches |

---

## Core Data Pipeline

```
Ticker symbol (user input)
        │
        ▼
yfinance.Ticker(symbol)
  ├── .info          → fundamentals dict (mktcap, P/E, 52W range…)
  ├── .history()     → OHLCV DataFrame (period + interval)
  └── .news          → list of news dicts (title, link, publisher, timestamp)
        │
        ▼
compute_indicators(df)
  ├── df.ta.ema(20)          → EMA_20
  ├── df.ta.ema(50)          → EMA_50
  ├── df.ta.bbands(20)       → BBU_20_2.0, BBM_20_2.0, BBL_20_2.0
  ├── df.ta.rsi(14)          → RSI_14
  ├── df.ta.macd()           → MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
  └── df.ta.stoch()          → STOCHk_14_3_3, STOCHd_14_3_3
        │
        ├──[Tab 1]──► Plotly candlestick + volume chart
        │
        ├──[Tab 2]──► RSI / MACD / Stochastic sub-charts + summary metrics
        │
        ├──[Tab 3]──► VADER.polarity_scores(headline) → compound score per article
        │              avg_sentiment stored in session_state
        │
        └──[Tab 4]──► GPT-4o prompt (price + technicals + avg_sentiment)
                        → structured markdown analysis
```

---

## Analysis Modules

### 1. Price & Volume (Tab 1)

- **Candlestick chart**: green (#26a69a) for up-days, red (#ef5350) for down-days
- **EMA overlays**: amber (20-period), purple (50-period)
- **Bollinger Bands**: semi-transparent fill between upper/lower bands (20-period, 2σ)
- **Volume subplot**: colour-coded bars matching candlestick direction

### 2. Technical Analysis (Tab 2)

**RSI (Relative Strength Index)**
- Period: 14 bars
- Signals: ≥ 70 = Overbought, ≤ 30 = Oversold, 30–70 = Neutral

**MACD (Moving Average Convergence Divergence)**
- Fast EMA: 12 | Slow EMA: 26 | Signal: 9
- Histogram: green when MACD > Signal, red when MACD < Signal

**Stochastic Oscillator**
- %K: 14-period, %D: 3-period smoothing
- Signals: ≥ 80 = Overbought, ≤ 20 = Oversold

**EMA Trend**
- Bullish: EMA20 > EMA50 (golden cross territory)
- Bearish: EMA20 < EMA50 (death cross territory)

### 3. News & Sentiment (Tab 3)

- Source: yfinance built-in news (Yahoo Finance headlines)
- Scorer: VADER (Valence Aware Dictionary and sEntiment Reasoner) — rule-based, runs offline, no API key
- Score range: −1.0 (strongly negative) to +1.0 (strongly positive)
- Threshold: > 0.05 = Bullish, < −0.05 = Bearish, else Neutral
- Up to 15 recent headlines displayed with per-article score

### 4. GPT-4o Insights (Tab 4)

**Prompt context sent to GPT-4o:**
```
Symbol · Company · Sector
Price · Day change %
52W High / Low
Market Cap · P/E · EPS
RSI(14) · EMA20 · EMA50 · MACD
News sentiment (VADER avg)
```

**Structured output format:**
```
**Trend Summary**   — 1–2 sentences
**Bull Case**       — 2–3 bullet points
**Bear Case**       — 2–3 bullet points
**Key Levels**      — support / resistance
**Trade Setup**     — entry zone, stop loss, target (or reasons to wait)
```

- Model: `gpt-4o`
- Temperature: 0.3 (low, for consistent analytical output)
- Max tokens: 500

---

## Data Flow

```
Yahoo Finance (external)
    │ HTTPS
    ▼
yfinance library (in-process)
    │
    ├── OHLCV DataFrame ──────────────► pandas-ta (local compute)
    │                                       │
    │                                    EMA · BB · RSI · MACD · Stoch
    │                                       │
    │                                    Plotly charts → browser
    │
    ├── Fundamentals dict ────────────► Header metrics (Price, MCap, P/E…)
    │
    └── News list ────────────────────► VADER (local, offline)
                                            │
                                         Sentiment scores + labels → Tab 3
                                            │
                                         avg_sentiment → st.session_state
                                            │
                                         GPT-4o prompt ──► OpenAI API (HTTPS)
                                                               │
                                                            Markdown analysis → Tab 4

All intermediate data: in-memory only. Nothing written to disk.
```

**External API calls:**
| API | Data sent | Notes |
|---|---|---|
| Yahoo Finance | Ticker symbol only | Free, no key |
| OpenAI GPT-4o | Aggregated text summary (no raw images or PII) | Requires user-supplied key |

---

## Advantages

### 1. Zero-Infrastructure Deployment
Runs inside a single Codespaces container (or any Python environment with `pip install`). No database, no message queue, no separate API server. Start to running in under two minutes.

### 2. Unified Signal View
Price action, volume, three independent technical oscillators, news sentiment, and AI narrative are all visible in one browser tab. Traders no longer need to switch between TradingView, a news aggregator, and a chatbot.

### 3. Fully Offline Technical Analysis
All indicator computation (RSI, MACD, Bollinger Bands, Stochastic, EMA) runs locally via pandas-ta. No third-party technical-analysis API is required or billed.

### 4. Free Data with No API Keys (except GPT-4o)
yfinance wraps Yahoo Finance for free — prices, fundamentals, and news all require zero credentials. VADER sentiment runs entirely offline. Only GPT-4o insights require a user-supplied API key.

### 5. Intelligent Caching
`@st.cache_data` with per-data-type TTLs (60 s for prices, 300 s for news/sentiment) prevents redundant API calls during normal use while keeping data fresh enough for intraday trading.

### 6. Structured AI Prompting
The GPT-4o prompt is templated and low-temperature (0.3), producing consistent, scannable output (Trend → Bull/Bear → Levels → Setup) rather than free-form text that varies between runs.

### 7. Privacy-Conscious AI Integration
Only a compact, human-readable text summary is sent to OpenAI — no raw price arrays, no uploaded images, no account information. Users control their own API key.

---

## Limitations

### 1. Yahoo Finance Data Reliability
yfinance is an unofficial wrapper. Yahoo Finance occasionally changes its API, causing temporary outages or data gaps. There is no SLA, and some fundamental fields (e.g. analyst targets, detailed earnings) may be missing for smaller tickers.

### 2. Delayed / EOD Data for Some Intervals
For timeframes ≥ 1 Year (weekly bars), data is end-of-day. For 15-minute intraday bars (1 Week view), Yahoo Finance provides a 30-day maximum window. Real-time tick data is not available.

### 3. VADER Sentiment Is Shallow
VADER scores individual headline text using a lexicon, with no understanding of financial context, sarcasm, or ambiguous phrasing. A headline like "Apple misses estimates, stock falls less than feared" may score incorrectly.

### 4. GPT-4o Has No Live Market Access
The model's knowledge is static (training cutoff). It analyses only the structured summary provided — it cannot independently verify prices, check SEC filings, or browse current events.

### 5. No Persistent Storage or History
All results live in Streamlit session state. Closing or refreshing the browser tab destroys all analysis. There is no watchlist, portfolio tracker, or historical comparison feature.

### 6. Single-User Architecture
Streamlit re-runs the entire script on every user interaction. No multi-tenancy, no authentication. Not suitable for concurrent use by many traders without a dedicated deployment (e.g. multiple Codespaces instances or a production server with session isolation).

### 7. CPU-Only Compute
All local computation (pandas-ta, VADER) runs on CPU. For most tickers this is near-instant. For batch screening of hundreds of symbols, performance would degrade significantly.

### 8. No Automated Tests
No test suite. Regressions in indicator calculation, chart rendering, or GPT prompt format are caught only by manual inspection.
