import gc
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from openai import OpenAI
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import pandas_ta as ta  # noqa: F401  (imported for df.ta accessor)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Timeframe → (yfinance period, interval) ───────────────────────────────────
PERIODS = {
    "1 Week":   ("5d",  "15m"),
    "1 Month":  ("1mo", "1d"),
    "3 Months": ("3mo", "1d"),
    "6 Months": ("6mo", "1d"),
    "1 Year":   ("1y",  "1wk"),
    "2 Years":  ("2y",  "1wk"),
}


# ── Data fetchers (cached) ────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_info(symbol: str) -> dict:
    return yf.Ticker(symbol).info


@st.cache_data(ttl=60)
def fetch_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=300)
def fetch_news(symbol: str) -> list:
    return yf.Ticker(symbol).news or []


# ── Technical indicators ──────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    if n >= 20:
        df.ta.ema(length=20, append=True)
        df.ta.bbands(length=20, append=True)
    if n >= 50:
        df.ta.ema(length=50, append=True)
    if n >= 14:
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(append=True)
    if n >= 26:
        df.ta.macd(append=True)
    return df


def col_find(df: pd.DataFrame, prefix: str, exclude: list | None = None) -> str | None:
    """Return the first column whose name starts with `prefix` (optionally skipping exclusions)."""
    for c in df.columns:
        if c.startswith(prefix) and (not exclude or not any(c.startswith(e) for e in exclude)):
            return c
    return None


# ── Sentiment ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def sentiment_scores(headlines: tuple) -> list:
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(h)["compound"] for h in headlines]


# ── GPT-4o analysis ───────────────────────────────────────────────────────────
def gpt_analysis(symbol: str, info: dict, df: pd.DataFrame, avg_sent: float, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    rsi_c  = col_find(df, "RSI_")
    ema20  = col_find(df, "EMA_20")
    ema50  = col_find(df, "EMA_50")
    macd_c = col_find(df, "MACD_", exclude=["MACDh_", "MACDs_"])

    def safe(col: str | None) -> str:
        if col and pd.notna(df[col].iloc[-1]):
            return f"{df[col].iloc[-1]:.2f}"
        return "N/A"

    pct_change = (latest["Close"] - prev["Close"]) / prev["Close"] * 100

    ctx = (
        f"Symbol: {symbol} | Company: {info.get('longName', symbol)} | Sector: {info.get('sector', 'N/A')}\n"
        f"Price: ${latest['Close']:.2f} | Day change: {pct_change:+.2f}%\n"
        f"52W High: ${info.get('fiftyTwoWeekHigh', 'N/A')} | 52W Low: ${info.get('fiftyTwoWeekLow', 'N/A')}\n"
        f"Market Cap: ${info.get('marketCap', 0) / 1e9:.1f}B | "
        f"P/E: {info.get('trailingPE', 'N/A')} | EPS: {info.get('trailingEps', 'N/A')}\n"
        f"RSI(14): {safe(rsi_c)} | EMA20: {safe(ema20)} | EMA50: {safe(ema50)} | MACD: {safe(macd_c)}\n"
        f"News sentiment (VADER avg, −1 to +1): {avg_sent:+.2f}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert equity analyst delivering trade-ready insights to active retail traders. "
                    "Be concise, specific, and actionable. Focus on trend, momentum, key price levels, and risk."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyse this stock for an active retail trader:\n\n{ctx}\n\n"
                    "Structure your response exactly as:\n"
                    "**Trend Summary** (1–2 sentences)\n"
                    "**Bull Case** (2–3 bullets)\n"
                    "**Bear Case** (2–3 bullets)\n"
                    "**Key Levels** (support / resistance to watch)\n"
                    "**Trade Setup** (entry zone, stop loss, target — or reasons to stay flat)\n"
                    "Total: under 300 words."
                ),
            },
        ],
        max_tokens=500,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Research")
    symbol    = st.text_input("Ticker symbol", value="AAPL", max_chars=12).upper().strip()
    timeframe = st.selectbox("Timeframe", list(PERIODS.keys()), index=2)
    st.divider()
    openai_key = st.text_input(
        "OpenAI API key", type="password", placeholder="sk-…",
        help="Only required for the GPT-4o Insights tab.",
    )
    refresh = st.button("🔄 Refresh data", use_container_width=True)
    st.caption("Prices cached 60 s · News cached 5 min")

if not symbol:
    st.info("Enter a ticker symbol in the sidebar.")
    st.stop()

if refresh:
    st.cache_data.clear()

period, interval = PERIODS[timeframe]

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {symbol}…"):
    try:
        info   = fetch_info(symbol)
        df_raw = fetch_history(symbol, period, interval)
        news   = fetch_news(symbol)
    except Exception as e:
        st.error(f"Could not load **{symbol}**: {e}")
        st.stop()

if df_raw.empty:
    st.error(f"No price data for **{symbol}**. Check the ticker symbol.")
    st.stop()

df = compute_indicators(df_raw)

# ── Header metrics ────────────────────────────────────────────────────────────
name  = info.get("longName", symbol)
price = df["Close"].iloc[-1]
prev  = df["Close"].iloc[-2] if len(df) > 1 else price
pct   = (price - prev) / prev * 100

st.markdown(f"## {name} &nbsp; `{symbol}`")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Price", f"${price:.2f}", f"{pct:+.2f}%")

mktcap = info.get("marketCap")
c2.metric("Market Cap", f"${mktcap / 1e9:.1f}B" if mktcap else "—")

pe = info.get("trailingPE")
c3.metric("P/E Ratio", f"{pe:.1f}" if isinstance(pe, (int, float)) else "—")

high52 = info.get("fiftyTwoWeekHigh")
c4.metric("52W High", f"${high52:.2f}" if isinstance(high52, (int, float)) else "—")

low52 = info.get("fiftyTwoWeekLow")
c5.metric("52W Low", f"${low52:.2f}" if isinstance(low52, (int, float)) else "—")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(
    ["📊 Price & Volume", "📐 Technical Analysis", "📰 News & Sentiment", "🤖 GPT-4o Insights"]
)

# ══ Tab 1: Price & Volume ════════════════════════════════════════════════════
with t1:
    ema20 = col_find(df, "EMA_20")
    ema50 = col_find(df, "EMA_50")
    bbu   = col_find(df, "BBU_")
    bbl   = col_find(df, "BBL_")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350", name="OHLC",
    ), row=1, col=1)

    # EMA overlays
    if ema20 and df[ema20].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df[ema20], name="EMA 20",
                                 line=dict(color="#f0a500", width=1.5)), row=1, col=1)
    if ema50 and df[ema50].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df[ema50], name="EMA 50",
                                 line=dict(color="#7c4dff", width=1.5)), row=1, col=1)

    # Bollinger Bands
    if bbu and bbl and df[bbu].notna().any():
        fig.add_trace(go.Scatter(x=df.index, y=df[bbu], name="BB Upper",
                                 line=dict(color="rgba(100,149,237,0.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[bbl], name="BB Lower", fill="tonexty",
                                 fillcolor="rgba(100,149,237,0.07)",
                                 line=dict(color="rgba(100,149,237,0.4)", width=1)), row=1, col=1)

    # Volume bars
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
                         marker_color=vol_colors, opacity=0.7, name="Volume"), row=2, col=1)

    fig.update_layout(
        height=520, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

# ══ Tab 2: Technical Analysis ════════════════════════════════════════════════
with t2:
    rsi_c  = col_find(df, "RSI_")
    macd_c = col_find(df, "MACD_", exclude=["MACDh_", "MACDs_"])
    macds  = col_find(df, "MACDs_")
    macdh  = col_find(df, "MACDh_")
    stochk = col_find(df, "STOCHk_")
    stochd = col_find(df, "STOCHd_")
    ema20  = col_find(df, "EMA_20")
    ema50  = col_find(df, "EMA_50")

    rsi_val   = df[rsi_c].iloc[-1]  if rsi_c  and df[rsi_c].notna().any()  else None
    ema20_val = df[ema20].iloc[-1]  if ema20  and df[ema20].notna().any()  else None
    ema50_val = df[ema50].iloc[-1]  if ema50  and df[ema50].notna().any()  else None

    if rsi_val is not None:
        if rsi_val >= 70:   rsi_label = f"{rsi_val:.0f} — Overbought"
        elif rsi_val <= 30: rsi_label = f"{rsi_val:.0f} — Oversold"
        else:               rsi_label = f"{rsi_val:.0f} — Neutral"
    else:
        rsi_label = "N/A"

    ema_trend = (
        "Bullish (EMA20 > EMA50)" if ema20_val and ema50_val and ema20_val > ema50_val
        else "Bearish (EMA20 < EMA50)" if ema20_val and ema50_val
        else "N/A"
    )
    vs_ema = "Above EMA20 ▲" if ema20_val and price > ema20_val else "Below EMA20 ▼"

    ca, cb, cc = st.columns(3)
    ca.metric("RSI (14)", rsi_label)
    cb.metric("EMA Trend", ema_trend)
    cc.metric("Price vs EMA20", vs_ema)
    st.divider()

    if rsi_c and df[rsi_c].notna().any():
        fig_rsi = go.Figure([go.Scatter(
            x=df.index, y=df[rsi_c],
            line=dict(color="#f0a500", width=2), name="RSI 14",
        )])
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red",   annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        fig_rsi.update_layout(height=200, template="plotly_dark", title="RSI (14)",
                               showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
                               yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

    if macd_c and macdh and df[macd_c].notna().any():
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df[macd_c],
                                      line=dict(color="#26a69a", width=1.5), name="MACD"))
        if macds and df[macds].notna().any():
            fig_macd.add_trace(go.Scatter(x=df.index, y=df[macds],
                                          line=dict(color="#f0a500", width=1.5), name="Signal"))
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df[macdh].fillna(0)]
        fig_macd.add_trace(go.Bar(x=df.index, y=df[macdh],
                                   marker_color=hist_colors, name="Histogram"))
        fig_macd.update_layout(height=200, template="plotly_dark", title="MACD (12,26,9)",
                                margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_macd, use_container_width=True)

    if stochk and stochd and df[stochk].notna().any():
        fig_st = go.Figure([
            go.Scatter(x=df.index, y=df[stochk], line=dict(color="#7c4dff", width=1.5), name="%K"),
            go.Scatter(x=df.index, y=df[stochd], line=dict(color="#f0a500", width=1.5), name="%D"),
        ])
        fig_st.add_hline(y=80, line_dash="dash", line_color="red",   annotation_text="Overbought (80)")
        fig_st.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold (20)")
        fig_st.update_layout(height=200, template="plotly_dark", title="Stochastic Oscillator",
                              margin=dict(l=0, r=0, t=40, b=0), yaxis_range=[0, 100])
        st.plotly_chart(fig_st, use_container_width=True)

# ══ Tab 3: News & Sentiment ══════════════════════════════════════════════════
with t3:
    if not news:
        st.info("No recent news found for this ticker.")
    else:
        headlines = tuple(n.get("title", "") for n in news if n.get("title"))
        scores    = sentiment_scores(headlines)
        avg_sent  = sum(scores) / len(scores) if scores else 0.0
        st.session_state["avg_sentiment"] = avg_sent

        label = "Bullish" if avg_sent > 0.05 else ("Bearish" if avg_sent < -0.05 else "Neutral")
        clr   = "#26a69a" if avg_sent > 0.05 else ("#ef5350" if avg_sent < -0.05 else "#aaa")
        st.markdown(
            f"**Overall news sentiment:** "
            f"<span style='color:{clr};font-weight:bold'>{label} ({avg_sent:+.2f})</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        for item, score in zip(news[:15], scores):
            title    = item.get("title", "")
            link     = item.get("link", "#")
            pub      = item.get("publisher", "")
            ts       = item.get("providerPublishTime", 0)
            date_str = datetime.fromtimestamp(ts).strftime("%b %d %H:%M") if ts else ""
            sc       = "#26a69a" if score > 0.05 else ("#ef5350" if score < -0.05 else "#888")
            ic       = "▲" if score > 0.05 else ("▼" if score < -0.05 else "—")
            st.markdown(
                f"<span style='color:{sc};font-weight:bold'>{ic} {score:+.2f}</span> &nbsp;"
                f"[{title}]({link}) &nbsp;"
                f"<span style='color:#888;font-size:0.85em'>{pub} · {date_str}</span>",
                unsafe_allow_html=True,
            )

# ══ Tab 4: GPT-4o Insights ═══════════════════════════════════════════════════
with t4:
    if not openai_key:
        st.warning("Add your OpenAI API key in the sidebar to unlock GPT-4o analysis.")
    else:
        avg_sent = st.session_state.get("avg_sentiment", 0.0)
        if st.button("Generate GPT-4o Analysis", type="primary"):
            with st.spinner("Asking GPT-4o for trade-ready insights…"):
                try:
                    result = gpt_analysis(symbol, info, df, avg_sent, openai_key)
                    st.session_state["gpt_result"] = result
                except Exception as e:
                    st.error(f"GPT-4o error: {e}")
        if "gpt_result" in st.session_state:
            st.markdown(st.session_state["gpt_result"])

gc.collect()
