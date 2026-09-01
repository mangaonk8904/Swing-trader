import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas_ta as ta
from io import BytesIO
from datetime import date, datetime, timedelta

# --- Glassmorphism Plotly Theme ---
GLASS_PLOTLY = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(family="Fira Sans, sans-serif", color="#E2E8F0", size=13),
        title_font=dict(family="Fira Sans, sans-serif", color="#F8FAFC", size=16),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)", zerolinecolor="rgba(148,163,184,0.15)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)", zerolinecolor="rgba(148,163,184,0.15)"),
        legend=dict(bgcolor="rgba(15,23,42,0.5)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
        colorway=["#22C55E", "#EF4444", "#3B82F6", "#F59E0B", "#8B5CF6", "#EC4899", "#06B6D4", "#F97316"],
        margin=dict(t=40, b=20, l=20, r=20),
    )
)
pio.templates["glassmorphism"] = go.layout.Template(**GLASS_PLOTLY)
pio.templates.default = "glassmorphism"

from data.yahoo import (
    get_price_data,
    get_full_fundamentals,
    get_options_chain,
    get_all_options_summary,
)
from data.excel_io import read_revenue_data, read_institutional_data, get_available_sheets
from data.fintel import FintelClient
from data.seekingalpha import SeekingAlphaClient
from analysis.technicals import compute_technicals
from analysis.llm import chat as _llm_chat_raw
from analysis.institutional_flow import (
    apply_categories,
    build_narrative_prompt,
    classify_institution_names,
    parse_holders,
    summarize,
)
from analysis.chart_setup import (
    TF_SPECS,
    analyze_chart,
    fetch_timeframes,
    position_size,
)
from analysis.scoring import score_stock
from schemas import FundamentalData, InstitutionalData, SeekingAlphaData
from config import settings


def _secret(name: str) -> str:
    """Read a Streamlit Cloud secret without exploding when none are set."""
    try:
        return st.secrets.get(name, "")
    except Exception:  # pylint: disable=broad-exception-caught
        return ""


def llm_keys() -> tuple[str, str, str]:
    """(anthropic, openrouter, groq) keys — .env first, then Streamlit secrets."""
    return (
        settings.anthropic_api_key or _secret("ANTHROPIC_API_KEY"),
        settings.openrouter_api_key or _secret("OPENROUTER_API_KEY"),
        settings.groq_api_key or _secret("GROQ_API_KEY"),
    )


def llm_available() -> bool:
    return any(llm_keys())


NO_LLM_KEY_MESSAGE = (
    "No LLM key configured. Add ANTHROPIC_API_KEY (or OPENROUTER_API_KEY / "
    "GROQ_API_KEY) to your .env file or Streamlit secrets."
)


def llm_chat(prompt: str, **kwargs) -> tuple[str, str]:
    """One prompt to whichever provider is configured.

    Returns (answer, "provider/model") — both are resolved at call time, so a
    retired model ID or a provider switch cannot silently break the AI panels.
    """
    anthropic_key, openrouter_key, groq_key = llm_keys()
    return _llm_chat_raw(
        prompt,
        anthropic_key=anthropic_key,
        openrouter_key=openrouter_key,
        groq_key=groq_key,
        provider_preference=settings.llm_provider,
        model=settings.llm_model,
        **kwargs,
    )


st.set_page_config(page_title="Swing Trader", page_icon="📊", layout="wide")

# --- Glassmorphism Theme CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Fira Sans', sans-serif;
}
code, pre, .stCodeBlock {
    font-family: 'Fira Code', monospace;
}

/* Main background gradient */
.stApp {
    background: linear-gradient(135deg, #020617 0%, #0F172A 40%, #1E293B 100%);
}

/* Glass card effect for containers */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1rem;
}

/* Metric cards - glass effect */
[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    transition: all 0.2s ease;
}
[data-testid="stMetric"]:hover {
    background: rgba(30, 41, 59, 0.7);
    border-color: rgba(34, 197, 94, 0.3);
    box-shadow: 0 4px 20px rgba(34, 197, 94, 0.1);
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}
[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-family: 'Fira Code', monospace;
    font-weight: 600;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #F8FAFC;
    background: rgba(255, 255, 255, 0.05);
}
.stTabs [aria-selected="true"] {
    background: rgba(34, 197, 94, 0.15) !important;
    color: #22C55E !important;
    border-bottom-color: #22C55E !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #22C55E !important;
}

/* DataFrame / table styling */
[data-testid="stDataFrame"] {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    overflow: hidden;
}

/* Sidebar glass effect */
[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.85) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Expander styling */
[data-testid="stExpander"] {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
}

/* Input fields */
.stTextInput > div > div {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    color: #F8FAFC;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div:focus-within {
    border-color: rgba(34, 197, 94, 0.5);
    box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.15);
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: rgba(34, 197, 94, 0.15);
    color: #22C55E;
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
    cursor: pointer;
}
.stButton > button:hover {
    background: rgba(34, 197, 94, 0.25);
    border-color: rgba(34, 197, 94, 0.5);
    box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2);
}

/* Download button */
.stDownloadButton > button {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    transition: all 0.2s ease;
    cursor: pointer;
}
.stDownloadButton > button:hover {
    background: rgba(30, 41, 59, 0.7);
    border-color: rgba(255, 255, 255, 0.2);
}

/* Alert boxes */
.stAlert {
    background: rgba(15, 23, 42, 0.5);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 10px;
}

/* Success alert (bullish) */
[data-testid="stAlert"][data-baseweb*="positive"],
.element-container .stSuccess {
    border-left-color: #22C55E !important;
}

/* Warning alert */
[data-testid="stAlert"][data-baseweb*="warning"],
.element-container .stWarning {
    border-left-color: #F59E0B !important;
}

/* Error alert (bearish) */
[data-testid="stAlert"][data-baseweb*="negative"],
.element-container .stError {
    border-left-color: #EF4444 !important;
}

/* Headers */
h1 {
    font-family: 'Fira Sans', sans-serif !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    letter-spacing: -0.02em;
}
h2, h3 {
    font-family: 'Fira Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
}

/* Plotly chart container */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #22C55E !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.3);
}
::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(148, 163, 184, 0.5);
}

/* Number input */
.stNumberInput > div > div {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.4);
    border: 1px dashed rgba(255, 255, 255, 0.15);
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("Swing Trader Dashboard")

# --- Sidebar ---
st.sidebar.header("Data Sources")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Excel (Revenue + Institutional sheets)", type=["xlsx", "xls", "csv"])

# Manual ticker input
st.sidebar.markdown("---")
manual_tickers = st.sidebar.text_input("Enter tickers (comma-separated)", placeholder="AAPL, NVDA, MSFT")

# Parse uploaded data
fund_map: dict[str, FundamentalData] = {}
inst_map: dict[str, InstitutionalData] = {}

if uploaded_file:
    # Save to temp for openpyxl
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        sheets = get_available_sheets(tmp_path)
        st.sidebar.success(f"Sheets found: {', '.join(sheets)}")

        if "Revenue" in sheets:
            for f in read_revenue_data(tmp_path):
                fund_map[f.ticker] = f

        if "Institutional" in sheets:
            for i in read_institutional_data(tmp_path):
                inst_map[i.ticker] = i
    finally:
        os.unlink(tmp_path)

# Fintel integration
fintel = FintelClient()
sa_client = SeekingAlphaClient()

# Data source status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
st.sidebar.markdown(f"- yfinance: :green[Active]")
if fintel.enabled:
    st.sidebar.markdown(f"- Fintel.io: :green[Active]")
else:
    st.sidebar.markdown(f"- Fintel.io: :gray[No API key]")
if sa_client.enabled:
    st.sidebar.markdown(f"- Seeking Alpha: :green[Active]")
else:
    st.sidebar.markdown(f"- Seeking Alpha: :gray[No API key]")

# Combine tickers from file + manual input
file_tickers = sorted(set(list(fund_map.keys()) + list(inst_map.keys())))
manual_list = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()] if manual_tickers else []
all_tickers = sorted(set(file_tickers + manual_list))

# --- Tabs ---
tab_analysis, tab_chart, tab_fundamentals, tab_options, tab_filings, tab_watchlist = st.tabs(
    ["Analysis", "Chart Setup", "Fundamentals", "Options Flow", "Fintel Filings", "Watchlist Alerts"]
)


# ===================== HELPER FUNCTIONS =====================

def _merge_institutional(excel: InstitutionalData | None, fintel_data: InstitutionalData) -> InstitutionalData:
    if excel is None:
        return fintel_data
    return InstitutionalData(
        ticker=fintel_data.ticker,
        institutional_buyers=fintel_data.institutional_buyers if fintel_data.institutional_buyers is not None else excel.institutional_buyers,
        institutional_sellers=fintel_data.institutional_sellers if fintel_data.institutional_sellers is not None else excel.institutional_sellers,
        net_institutional=fintel_data.net_institutional if fintel_data.net_institutional is not None else excel.net_institutional,
        short_interest_pct=fintel_data.short_interest_pct if fintel_data.short_interest_pct is not None else excel.short_interest_pct,
        short_interest_change=fintel_data.short_interest_change if fintel_data.short_interest_change is not None else excel.short_interest_change,
    )


@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def fetch_and_score(tickers: tuple, fund_data: dict, inst_data: dict, fintel_enabled: bool, sa_enabled: bool):
    scores = []
    tech_data = {}
    price_data = {}
    sa_data = {}

    fintel_client = FintelClient() if fintel_enabled else None

    # Batch-fetch Seeking Alpha data for all tickers at once
    sa_all = {}
    if sa_enabled:
        try:
            sa_client = SeekingAlphaClient()
            sa_all = sa_client.get_ticker_data(list(tickers))
        except Exception:
            pass

    for ticker in tickers:
        tech = None
        try:
            df = get_price_data(ticker)
            price_data[ticker] = df
            tech = compute_technicals(ticker, df)
            tech_data[ticker] = tech
        except Exception:
            pass

        # Merge Fintel data over Excel data
        inst = inst_data.get(ticker)
        if fintel_client and fintel_client.enabled:
            fintel_inst = fintel_client.get_institutional_data(ticker)
            if fintel_inst:
                inst = _merge_institutional(inst, fintel_inst)

        # Build SeekingAlphaData if available
        sa = None
        sa_raw = sa_all.get(ticker)
        if sa_raw:
            sa = SeekingAlphaData(
                ticker=ticker,
                value=sa_raw.get("value", 0),
                growth=sa_raw.get("growth", 0),
                momentum=sa_raw.get("momentum", 0),
                profitability=sa_raw.get("profitability", 0),
                eps_revisions=sa_raw.get("eps_revisions", 0),
                analyst_count=sa_raw.get("analyst_count", 0),
                mean_score=sa_raw.get("mean_score", 0.0),
                rating=sa_raw.get("rating", "N/A"),
            )
            sa_data[ticker] = sa

        result = score_stock(
            tech=tech,
            fund=fund_data.get(ticker),
            inst=inst,
            sa=sa,
        )
        scores.append(result)

    return scores, tech_data, price_data, sa_data


def color_signal(val):
    colors = {
        "STRONG_BUY": "background-color: rgba(34,197,94,0.25); color: #22C55E; border-radius: 6px",
        "BUY": "background-color: rgba(34,197,94,0.15); color: #4ADE80; border-radius: 6px",
        "NEUTRAL": "background-color: rgba(245,158,11,0.15); color: #FBBF24; border-radius: 6px",
        "PASS": "background-color: rgba(239,68,68,0.15); color: #F87171; border-radius: 6px",
    }
    return colors.get(val, "")


def color_composite(val):
    if val >= 75:
        return "background-color: rgba(34,197,94,0.25); color: #22C55E"
    elif val >= 55:
        return "background-color: rgba(34,197,94,0.15); color: #4ADE80"
    elif val >= 40:
        return "background-color: rgba(245,158,11,0.15); color: #FBBF24"
    else:
        return "background-color: rgba(239,68,68,0.15); color: #F87171"


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    df.to_excel(output, index=False, sheet_name="Scores")
    return output.getvalue()


# ===================== ANALYSIS TAB =====================
with tab_analysis:
    if not all_tickers:
        st.info("Upload an Excel file or enter tickers in the sidebar to get started.")
    else:
        # Convert to hashable types for caching
        fund_dict = {k: v.model_dump() for k, v in fund_map.items()}
        inst_dict = {k: v.model_dump() for k, v in inst_map.items()}

        # Reconstruct for scoring (cache needs hashable inputs)
        fund_for_score = {k: FundamentalData(**v) for k, v in fund_dict.items()}
        inst_for_score = {k: InstitutionalData(**v) for k, v in inst_dict.items()}

        scores, tech_data, price_data, sa_data = fetch_and_score(tuple(all_tickers), fund_for_score, inst_for_score, fintel.enabled, sa_client.enabled)

        # --- Scores Table ---
        st.header("Swing Trade Scores")

        score_rows = []
        for s in sorted(scores, key=lambda x: x.composite_score, reverse=True):
            row = {
                "Ticker": s.ticker,
                "Signal": s.signal.value.upper(),
                "Composite": s.composite_score,
                "Technical": s.technical_score,
                "Fundamental": s.fundamental_score,
                "Institutional": s.institutional_score,
                "SA Score": s.sa_score,
                "Entry": s.entry_price,
                "Stop Loss": s.stop_loss,
                "Target": s.target_price,
                "Notes": s.notes,
            }
            # Add SA grade letters if available
            sa_info = sa_data.get(s.ticker)
            if sa_info:
                from data.seekingalpha import GRADE_MAP
                row["SA Rating"] = sa_info.rating
                row["Momentum"] = GRADE_MAP.get(sa_info.momentum, "")
                row["EPS Rev"] = GRADE_MAP.get(sa_info.eps_revisions, "")
                row["Growth"] = GRADE_MAP.get(sa_info.growth, "")
            score_rows.append(row)

        score_df = pd.DataFrame(score_rows)

        format_dict = {
            "Composite": "{:.1f}",
            "Technical": "{:.1f}",
            "Fundamental": "{:.1f}",
            "Institutional": "{:.1f}",
            "SA Score": "{:.1f}",
            "Entry": "${:,.2f}",
            "Stop Loss": "${:,.2f}",
            "Target": "${:,.2f}",
        }
        styled_df = score_df.style.map(color_signal, subset=["Signal"]).map(color_composite, subset=["Composite"]).format(format_dict, na_rep="N/A")

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # --- Excel Download ---
        st.download_button(
            label="Download Results (Excel)",
            data=to_excel_bytes(score_df),
            file_name=f"swing_trade_scores_{date.today()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # --- Charts for individual tickers ---
        st.header("Technical Charts")

        selected_ticker = st.selectbox("Select ticker for chart", all_tickers)

        if selected_ticker in price_data:
            df = price_data[selected_ticker]
            close = df["Close"]

            # Compute indicators for charting
            sma_20 = ta.sma(close, length=20)
            sma_50 = ta.sma(close, length=50)
            rsi = ta.rsi(close, length=14)
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)

            # Create subplots: candlestick, RSI, MACD
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=[f"{selected_ticker} Price", "RSI (14)", "MACD"],
                specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                name="Price",
            ), row=1, col=1, secondary_y=False)

            # SMA overlays
            if sma_20 is not None:
                fig.add_trace(go.Scatter(x=df.index, y=sma_20, name="SMA 20", line=dict(color="orange", width=1)), row=1, col=1, secondary_y=False)
            if sma_50 is not None:
                fig.add_trace(go.Scatter(x=df.index, y=sma_50, name="SMA 50", line=dict(color="blue", width=1)), row=1, col=1, secondary_y=False)

            # Volume on secondary y-axis so it doesn't overwhelm candlesticks
            colors = ["green" if c >= o else "red" for c, o in zip(df["Close"], df["Open"])]
            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.3), row=1, col=1, secondary_y=True)

            # RSI
            if rsi is not None:
                fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="purple", width=1.5)), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

            # MACD
            if macd_df is not None:
                macd_cols = macd_df.columns
                fig.add_trace(go.Scatter(x=df.index, y=macd_df[macd_cols[0]], name="MACD", line=dict(color="blue", width=1.5)), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=macd_df[macd_cols[2]], name="Signal", line=dict(color="red", width=1.5)), row=3, col=1)
                histogram = macd_df[macd_cols[1]]
                hist_colors = ["green" if v >= 0 else "red" for v in histogram]
                fig.add_trace(go.Bar(x=df.index, y=histogram, name="Histogram", marker_color=hist_colors, opacity=0.5), row=3, col=1)

            fig.update_layout(
                height=800,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True, showgrid=False)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Show fundamentals info
            if selected_ticker in tech_data:
                snap = tech_data[selected_ticker]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${snap.price:.2f}")
                col2.metric("RSI (14)", f"{snap.rsi_14:.1f}" if snap.rsi_14 else "N/A")
                col3.metric("MACD", snap.macd_signal or "N/A")
                col4.metric("ATR (14)", f"${snap.atr_14:.2f}" if snap.atr_14 else "N/A")

            # Show score for selected ticker
            selected_score = next((s for s in scores if s.ticker == selected_ticker), None)
            if selected_score:
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Composite Score", f"{selected_score.composite_score:.1f}")
                col2.metric("Signal", selected_score.signal.value.upper())
                if selected_score.entry_price and selected_score.stop_loss and selected_score.target_price:
                    col3.metric("Entry → Target", f"${selected_score.entry_price:.2f} → ${selected_score.target_price:.2f}")
                    col4.metric("Stop Loss", f"${selected_score.stop_loss:.2f}")

        else:
            st.warning(f"No price data available for {selected_ticker}")


# ===================== FUNDAMENTALS TAB =====================

def _fmt_large_number(val) -> str:
    """Format large numbers as $1.2B, $345M, etc."""
    if val is None:
        return "N/A"
    val = float(val)
    if abs(val) >= 1e12:
        return f"${val / 1e12:.2f}T"
    elif abs(val) >= 1e9:
        return f"${val / 1e9:.2f}B"
    elif abs(val) >= 1e6:
        return f"${val / 1e6:.1f}M"
    else:
        return f"${val:,.0f}"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt_ratio(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


@st.cache_data(ttl=600, show_spinner="Fetching fundamental data...")
def fetch_fundamentals(ticker: str):
    return get_full_fundamentals(ticker)


with tab_fundamentals:
    st.header("Fundamental Analysis")

    fund_ticker = st.text_input("Enter ticker symbol", placeholder="AAPL", key="fund_ticker")

    if fund_ticker:
        fund_ticker = fund_ticker.strip().upper()

        try:
            f = fetch_fundamentals(fund_ticker)
        except Exception as e:
            st.error(f"Error fetching fundamentals: {e}")
            f = None

        if f:
            # --- Company Header ---
            st.subheader(f"{f['name']} ({fund_ticker})")
            st.caption(f"{f.get('sector') or ''} — {f.get('industry') or ''}")

            # --- Price Context ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Price", f"${f['price']:,.2f}" if f.get("price") else "N/A")
            col2.metric("Market Cap", _fmt_large_number(f.get("market_cap")))
            high52 = f.get("fifty_two_week_high")
            low52 = f.get("fifty_two_week_low")
            col3.metric("52-Week High", f"${high52:,.2f}" if high52 else "N/A")
            col4.metric("52-Week Low", f"${low52:,.2f}" if low52 else "N/A")

            st.markdown("---")

            # --- Valuation ---
            st.subheader("Valuation")
            v1, v2, v3, v4, v5, v6 = st.columns(6)
            v1.metric("Trailing P/E", _fmt_ratio(f.get("trailing_pe")))
            v2.metric("Forward P/E", _fmt_ratio(f.get("forward_pe")))
            v3.metric("PEG Ratio", _fmt_ratio(f.get("peg_ratio")))
            v4.metric("Price/Book", _fmt_ratio(f.get("price_to_book")))
            v5.metric("EV/EBITDA", _fmt_ratio(f.get("ev_to_ebitda")))
            v6.metric("EV/Revenue", _fmt_ratio(f.get("ev_to_revenue")))

            st.markdown("---")

            # --- Earnings & Revenue ---
            st.subheader("Earnings & Revenue")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Total Revenue", _fmt_large_number(f.get("total_revenue")))
            e2.metric("Revenue Growth", _fmt_pct(f.get("revenue_growth")))
            e3.metric("EPS (TTM)", f"${f['eps_trailing']:.2f}" if f.get("eps_trailing") else "N/A")
            e4.metric("EPS (Forward)", f"${f['eps_forward']:.2f}" if f.get("eps_forward") else "N/A")

            e5, e6, e7, e8 = st.columns(4)
            e5.metric("EBITDA", _fmt_large_number(f.get("ebitda")))
            e6.metric("Earnings Growth", _fmt_pct(f.get("earnings_growth")))
            e7.metric("Quarterly Earnings Growth", _fmt_pct(f.get("earnings_quarterly_growth")))
            e8.metric("Revenue/Share", f"${f['revenue_per_share']:.2f}" if f.get("revenue_per_share") else "N/A")

            st.markdown("---")

            # --- Profitability ---
            st.subheader("Profitability")
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Gross Margin", _fmt_pct(f.get("gross_margins")))
            p2.metric("Operating Margin", _fmt_pct(f.get("operating_margins")))
            p3.metric("EBITDA Margin", _fmt_pct(f.get("ebitda_margins")))
            p4.metric("Profit Margin", _fmt_pct(f.get("profit_margins")))
            p5.metric("ROE", _fmt_pct(f.get("return_on_equity")))

            # Margin comparison bar chart
            margins = {
                "Gross": f.get("gross_margins"),
                "Operating": f.get("operating_margins"),
                "EBITDA": f.get("ebitda_margins"),
                "Net Profit": f.get("profit_margins"),
            }
            margin_names = [k for k, v in margins.items() if v is not None]
            margin_vals = [v * 100 for v in margins.values() if v is not None]

            if margin_vals:
                colors = ["#22C55E" if v > 20 else "#F59E0B" if v > 10 else "#EF4444" for v in margin_vals]
                fig_margins = go.Figure(go.Bar(
                    x=margin_names, y=margin_vals,
                    marker_color=colors, text=[f"{v:.1f}%" for v in margin_vals], textposition="outside",
                ))
                fig_margins.update_layout(title="Margin Comparison", yaxis_title="%", height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig_margins, use_container_width=True)

            st.markdown("---")

            # --- Balance Sheet & Cash Flow ---
            st.subheader("Balance Sheet & Cash Flow")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Cash", _fmt_large_number(f.get("total_cash")))
            b2.metric("Total Debt", _fmt_large_number(f.get("total_debt")))
            b3.metric("Debt/Equity", _fmt_ratio(f.get("debt_to_equity")))
            b4.metric("Current Ratio", _fmt_ratio(f.get("current_ratio")))

            b5, b6, b7, b8 = st.columns(4)
            b5.metric("Free Cash Flow", _fmt_large_number(f.get("free_cashflow")))
            b6.metric("Operating Cash Flow", _fmt_large_number(f.get("operating_cashflow")))
            b7.metric("Book Value/Share", f"${f['book_value']:.2f}" if f.get("book_value") else "N/A")
            b8.metric("ROA", _fmt_pct(f.get("return_on_assets")))

            # Cash vs Debt visual
            cash = f.get("total_cash") or 0
            debt = f.get("total_debt") or 0
            if cash or debt:
                fig_cd = go.Figure()
                fig_cd.add_trace(go.Bar(x=["Cash"], y=[cash], name="Cash", marker_color="#22C55E"))
                fig_cd.add_trace(go.Bar(x=["Debt"], y=[debt], name="Debt", marker_color="#EF4444"))
                fig_cd.update_layout(title="Cash vs Debt", height=300, margin=dict(t=40, b=20),
                                     yaxis_tickprefix="$", yaxis_tickformat=",")
                st.plotly_chart(fig_cd, use_container_width=True)

            st.markdown("---")

            # --- Dividends & Ownership ---
            st.subheader("Dividends & Ownership")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Dividend Yield", _fmt_pct(f.get("dividend_yield")))
            d2.metric("Payout Ratio", _fmt_pct(f.get("payout_ratio")))
            d3.metric("Insider Ownership", _fmt_pct(f.get("held_percent_insiders")))
            d4.metric("Institutional Ownership", _fmt_pct(f.get("held_percent_institutions")))

            d5, d6, _, _ = st.columns(4)
            d5.metric("Short % of Float", _fmt_pct(f.get("short_percent_of_float")))
            d6.metric("Short Ratio", _fmt_ratio(f.get("short_ratio")))

            # --- AI Fundamental Analysis ---
            st.markdown("---")
            if not llm_available():
                st.warning(NO_LLM_KEY_MESSAGE)
            elif st.button("Analyze Fundamentals", key="ai_fund_btn"):
                prompt = f"""You are a senior equity research analyst advising a swing trader (1-week to 1-month hold).

Analyze the fundamentals for {fund_ticker} ({f.get('name')}) — {f.get('sector')}, {f.get('industry')}:

VALUATION:
- Trailing P/E: {_fmt_ratio(f.get('trailing_pe'))} | Forward P/E: {_fmt_ratio(f.get('forward_pe'))}
- PEG Ratio: {_fmt_ratio(f.get('peg_ratio'))} | Price/Book: {_fmt_ratio(f.get('price_to_book'))}
- EV/EBITDA: {_fmt_ratio(f.get('ev_to_ebitda'))} | EV/Revenue: {_fmt_ratio(f.get('ev_to_revenue'))}

EARNINGS & REVENUE:
- Revenue: {_fmt_large_number(f.get('total_revenue'))} | Revenue Growth: {_fmt_pct(f.get('revenue_growth'))}
- EPS (TTM): ${f.get('eps_trailing') or 'N/A'} | EPS (Forward): ${f.get('eps_forward') or 'N/A'}
- Earnings Growth: {_fmt_pct(f.get('earnings_growth'))} | Quarterly Earnings Growth: {_fmt_pct(f.get('earnings_quarterly_growth'))}
- EBITDA: {_fmt_large_number(f.get('ebitda'))}

PROFITABILITY:
- Gross Margin: {_fmt_pct(f.get('gross_margins'))} | Operating Margin: {_fmt_pct(f.get('operating_margins'))}
- EBITDA Margin: {_fmt_pct(f.get('ebitda_margins'))} | Net Profit Margin: {_fmt_pct(f.get('profit_margins'))}
- ROE: {_fmt_pct(f.get('return_on_equity'))} | ROA: {_fmt_pct(f.get('return_on_assets'))}

BALANCE SHEET:
- Cash: {_fmt_large_number(f.get('total_cash'))} | Debt: {_fmt_large_number(f.get('total_debt'))}
- Debt/Equity: {_fmt_ratio(f.get('debt_to_equity'))} | Current Ratio: {_fmt_ratio(f.get('current_ratio'))}
- Free Cash Flow: {_fmt_large_number(f.get('free_cashflow'))}

OWNERSHIP:
- Insider: {_fmt_pct(f.get('held_percent_insiders'))} | Institutional: {_fmt_pct(f.get('held_percent_institutions'))}
- Short % of Float: {_fmt_pct(f.get('short_percent_of_float'))}

PRICE:
- Current: ${f.get('price') or 'N/A'} | 52W High: ${f.get('fifty_two_week_high') or 'N/A'} | 52W Low: ${f.get('fifty_two_week_low') or 'N/A'}

Provide a concise analysis covering:
1. **Verdict**: BUY, HOLD, or AVOID — state it clearly upfront
2. **Valuation**: Is the stock fairly valued, overvalued, or undervalued? Compare P/E, PEG, EV/EBITDA to typical ranges
3. **Growth**: Is revenue and earnings growth strong enough to justify the valuation?
4. **Profitability**: Are margins healthy? Is the company efficiently run?
5. **Financial Health**: Cash position vs debt — any red flags?
6. **Risk Factors**: What could go wrong? Short interest, valuation stretch, margin compression?
7. **Swing Trade Angle**: From a 1-4 week perspective, does the fundamental picture support a trade?

Be direct and actionable. No disclaimers."""

                with st.spinner("Analyzing fundamentals..."):
                    try:
                        answer, used = llm_chat(prompt, max_tokens=2500)
                        st.markdown(answer)
                        st.caption(f"Generated by {used}")
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")


# ===================== OPTIONS FLOW TAB =====================

@st.cache_data(ttl=300, show_spinner="Fetching options data...")
def fetch_options_summary(ticker: str):
    return get_all_options_summary(ticker)


@st.cache_data(ttl=300, show_spinner="Loading options chain...")
def fetch_options_chain(ticker: str, expiry: str):
    return get_options_chain(ticker, expiry)


with tab_options:
    st.header("Options Flow Analysis")

    options_ticker = st.text_input("Enter ticker symbol", placeholder="AAPL", key="options_ticker")

    if options_ticker:
        options_ticker = options_ticker.strip().upper()

        try:
            summary = fetch_options_summary(options_ticker)
        except ValueError as e:
            st.warning(str(e))
            summary = None
        except Exception as e:
            st.error(f"Error fetching options data: {e}")
            summary = None

        if summary:
            # --- Section 1: Top-Level Metrics ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${summary['current_price']:,.2f}" if summary['current_price'] else "N/A")
            col2.metric("P/C Volume Ratio", f"{summary['pc_volume_ratio']:.2f}")
            col3.metric("P/C OI Ratio", f"{summary['pc_oi_ratio']:.2f}")
            total_vol = summary['total_call_volume'] + summary['total_put_volume']
            col4.metric("Total Options Volume", f"{total_vol:,}")

            # --- Section 2: Sentiment Signal ---
            pcr = summary['pc_volume_ratio']
            if pcr > 1.5:
                st.error(f"**Extreme put activity (P/C {pcr:.2f})** — Heavy hedging or fear. Contrarian traders watch for reversals at these levels.")
            elif pcr > 1.0:
                st.warning(f"**Bearish / Hedging (P/C {pcr:.2f})** — Puts dominate. Possible downside protection or directional bearish bets.")
            elif pcr >= 0.7:
                st.info(f"**Neutral (P/C {pcr:.2f})** — Balanced call/put activity.")
            else:
                st.success(f"**Bullish (P/C {pcr:.2f})** — Calls dominate. Market expects upside.")

            # --- Section 3: Volume & OI by Expiration ---
            st.subheader("Volume & Open Interest by Expiration")
            by_expiry = summary['by_expiry']
            if by_expiry:
                exp_df = pd.DataFrame(by_expiry)
                col_vol, col_oi = st.columns(2)

                with col_vol:
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(x=exp_df["expiry"], y=exp_df["call_volume"], name="Call Volume", marker_color="#22C55E"))
                    fig_vol.add_trace(go.Bar(x=exp_df["expiry"], y=exp_df["put_volume"], name="Put Volume", marker_color="#EF4444"))
                    fig_vol.update_layout(barmode="group", title="Volume by Expiration", height=350, margin=dict(t=40, b=20))
                    st.plotly_chart(fig_vol, use_container_width=True)

                with col_oi:
                    fig_oi = go.Figure()
                    fig_oi.add_trace(go.Bar(x=exp_df["expiry"], y=exp_df["call_oi"], name="Call OI", marker_color="#22C55E"))
                    fig_oi.add_trace(go.Bar(x=exp_df["expiry"], y=exp_df["put_oi"], name="Put OI", marker_color="#EF4444"))
                    fig_oi.update_layout(barmode="group", title="Open Interest by Expiration", height=350, margin=dict(t=40, b=20))
                    st.plotly_chart(fig_oi, use_container_width=True)

            # --- Section 4: Single-Expiry Deep Dive ---
            st.subheader("Single Expiry Deep Dive")
            expiry_list = summary['expirations']
            selected_expiry = st.selectbox("Select expiration", expiry_list, key="options_expiry")

            if selected_expiry:
                try:
                    chain = fetch_options_chain(options_ticker, selected_expiry)
                except Exception as e:
                    st.error(f"Error loading chain: {e}")
                    chain = None

                if chain:
                    calls_df = chain["calls"]
                    puts_df = chain["puts"]
                    current_price = summary['current_price']

                    col_butterfly, col_iv = st.columns(2)

                    with col_butterfly:
                        # Butterfly volume chart: calls right, puts left
                        call_strikes = calls_df[calls_df["volume"].fillna(0) > 0]
                        put_strikes = puts_df[puts_df["volume"].fillna(0) > 0]

                        fig_bf = go.Figure()
                        fig_bf.add_trace(go.Bar(
                            y=call_strikes["strike"], x=call_strikes["volume"],
                            name="Calls", orientation="h", marker_color="#22C55E",
                        ))
                        fig_bf.add_trace(go.Bar(
                            y=put_strikes["strike"], x=-put_strikes["volume"],
                            name="Puts", orientation="h", marker_color="#EF4444",
                        ))
                        if current_price:
                            fig_bf.add_hline(y=current_price, line_dash="dash", line_color="#94A3B8",
                                             annotation_text=f"Price ${current_price:.2f}")
                        fig_bf.update_layout(
                            title="Volume by Strike", barmode="overlay", height=500,
                            xaxis_title="Volume (puts negative)", yaxis_title="Strike",
                            margin=dict(t=40, b=20),
                        )
                        st.plotly_chart(fig_bf, use_container_width=True)

                    with col_iv:
                        # IV smile/skew
                        calls_iv = calls_df[calls_df["impliedVolatility"].fillna(0) > 0]
                        puts_iv = puts_df[puts_df["impliedVolatility"].fillna(0) > 0]

                        fig_iv = go.Figure()
                        fig_iv.add_trace(go.Scatter(
                            x=calls_iv["strike"], y=calls_iv["impliedVolatility"] * 100,
                            name="Call IV", mode="lines+markers", line=dict(color="#22C55E"),
                        ))
                        fig_iv.add_trace(go.Scatter(
                            x=puts_iv["strike"], y=puts_iv["impliedVolatility"] * 100,
                            name="Put IV", mode="lines+markers", line=dict(color="#EF4444"),
                        ))
                        if current_price:
                            fig_iv.add_vline(x=current_price, line_dash="dash", line_color="#94A3B8",
                                             annotation_text=f"${current_price:.2f}")
                        fig_iv.update_layout(
                            title="Implied Volatility Skew", height=500,
                            xaxis_title="Strike", yaxis_title="IV (%)",
                            margin=dict(t=40, b=20),
                        )
                        st.plotly_chart(fig_iv, use_container_width=True)

                    # --- Section 5: Unusual Activity ---
                    st.subheader("Unusual Activity")
                    st.caption("Contracts where volume > 2x open interest — suggests new positions being opened")

                    unusual_rows = []
                    for side, df in [("CALL", calls_df), ("PUT", puts_df)]:
                        for _, row in df.iterrows():
                            vol = row.get("volume") or 0
                            oi = row.get("openInterest") or 0
                            if vol > 0 and oi > 0 and vol > 2 * oi:
                                unusual_rows.append({
                                    "Type": side,
                                    "Strike": row["strike"],
                                    "Volume": int(vol),
                                    "Open Interest": int(oi),
                                    "Vol/OI": round(vol / oi, 1),
                                    "IV %": round((row.get("impliedVolatility") or 0) * 100, 1),
                                    "Last Price": row.get("lastPrice") or 0,
                                    "ITM": "Yes" if row.get("inTheMoney") else "No",
                                })

                    if unusual_rows:
                        unusual_df = pd.DataFrame(unusual_rows).sort_values("Vol/OI", ascending=False)
                        st.dataframe(
                            unusual_df.style.format({
                                "Strike": "${:,.2f}",
                                "Last Price": "${:,.2f}",
                                "IV %": "{:.1f}%",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No unusual activity detected for this expiration.")

                    # --- Section 6: AI Analysis ---
                    st.subheader("AI Analysis")

                    # Resolve Groq key: config (.env) first, then Streamlit secrets
                    if not llm_available():
                        st.warning(NO_LLM_KEY_MESSAGE)
                    elif st.button("Analyze Options Activity", key="ai_analyze_btn"):
                        # Build context for the LLM
                        total_cv = int(calls_df["volume"].fillna(0).sum())
                        total_pv = int(puts_df["volume"].fillna(0).sum())
                        total_coi = int(calls_df["openInterest"].fillna(0).sum())
                        total_poi = int(puts_df["openInterest"].fillna(0).sum())
                        exp_pcr = round(total_pv / total_cv, 2) if total_cv > 0 else 0

                        # Top volume strikes
                        top_calls = calls_df.nlargest(5, "volume")[["strike", "volume", "openInterest", "impliedVolatility"]].to_string(index=False)
                        top_puts = puts_df.nlargest(5, "volume")[["strike", "volume", "openInterest", "impliedVolatility"]].to_string(index=False)

                        # Unusual activity summary
                        unusual_summary = ""
                        if unusual_rows:
                            unusual_summary = f"\nUnusual activity (volume > 2x OI):\n"
                            for u in unusual_rows[:10]:
                                unusual_summary += f"  {u['Type']} ${u['Strike']}: Vol={u['Volume']}, OI={u['Open Interest']}, Vol/OI={u['Vol/OI']}, IV={u['IV %']}%\n"

                        prompt = f"""You are an expert options analyst advising a swing trader (1-week to 1-month holding period).

Analyze this options data for {options_ticker} (expiry: {selected_expiry}, current price: ${summary['current_price']:.2f}):

OVERALL SUMMARY (across {len(summary['expirations'])} expirations):
- Total Call Volume: {summary['total_call_volume']:,} | Total Put Volume: {summary['total_put_volume']:,}
- Put/Call Volume Ratio: {summary['pc_volume_ratio']:.2f}
- Total Call OI: {summary['total_call_oi']:,} | Total Put OI: {summary['total_put_oi']:,}
- Put/Call OI Ratio: {summary['pc_oi_ratio']:.2f}

THIS EXPIRATION ({selected_expiry}):
- Call Volume: {total_cv:,} | Put Volume: {total_pv:,} | P/C Ratio: {exp_pcr}
- Call OI: {total_coi:,} | Put OI: {total_poi:,}

Top 5 Call Strikes by Volume:
{top_calls}

Top 5 Put Strikes by Volume:
{top_puts}
{unusual_summary}
Provide a concise analysis covering:
1. **Sentiment**: What is the options market telling us — bullish, bearish, or mixed? Why?
2. **Key Levels**: Which strike prices have significant positioning? What do they suggest as support/resistance?
3. **Unusual Activity**: Any notable signals from volume spikes or unusual Vol/OI ratios?
4. **Swing Trade Signal**: Based on this options data, what's the actionable takeaway for a 1-4 week swing trade?

Keep it direct and actionable. No disclaimers."""

                        with st.spinner("Analyzing options flow..."):
                            try:
                                analysis, _used = llm_chat(prompt, max_tokens=2000)
                                st.markdown(analysis)
                            except Exception as e:
                                st.error(f"AI analysis failed: {e}")

                    # --- Section 7: Raw Chain ---
                    with st.expander("View Full Calls Chain"):
                        st.dataframe(calls_df, use_container_width=True, hide_index=True)
                    with st.expander("View Full Puts Chain"):
                        st.dataframe(puts_df, use_container_width=True, hide_index=True)


# ===================== FINTEL FILINGS TAB =====================
# ===================== FINTEL FILINGS TAB =====================

def _llm_error(exc: Exception) -> str:
    """Turn an SDK exception into something that names the actual problem.

    Status code first — the class name alone sent us chasing a "broken feature"
    when the real causes were a retired model and an exhausted credit budget.
    """
    status = getattr(exc, "status_code", None)
    by_status = {
        401: "the provider rejected the API key",
        402: "out of credits — top up at openrouter.ai/settings/credits",
        403: "the provider blocked this request",
        404: "no such model on this account",
        429: "rate limited — retry shortly",
    }
    if status in by_status:
        return f"failed: {by_status[status]}"
    if status and status >= 500:
        return f"failed: provider error ({status}), retry shortly"

    name = type(exc).__name__
    friendly = {
        "NotFoundError": "no such model on this account",
        "AuthenticationError": "the provider rejected the API key",
        "PermissionDeniedError": "the provider blocked this request",
        "RateLimitError": "rate limited — retry shortly",
        "NoProviderConfigured": "no LLM key configured",
        "NoUsableModel": "the provider offers no usable chat model",
    }
    return f"failed: {friendly.get(name, name)}"


@st.cache_data(ttl=3600, show_spinner=False)
def classify_institutions(names: tuple) -> tuple[dict, str]:
    """Label each institution by type. Returns (labels, status_note).

    Falls back to name heuristics whenever the model is unavailable — the
    summary must still render, just with less confident labels.
    """
    if not llm_available():
        return {}, "no-key"
    try:
        return classify_institution_names(list(names), llm_chat), "ok"
    except Exception as exc:
        return {}, _llm_error(exc)


@st.cache_data(ttl=3600, show_spinner=False)
def institutional_narrative(prompt: str) -> tuple[str, str]:
    """Written interpretation of the computed flow. Returns (text, status)."""
    if not llm_available():
        return "", "no-key"
    try:
        text, _used = llm_chat(prompt, max_tokens=2000)
        return text, "ok"
    except Exception as exc:
        return "", _llm_error(exc)


CATEGORY_BADGE = {
    "hedge fund": "🟣 hedge fund",
    "index/passive": "⚪ index/passive",
    "mutual fund": "🔵 mutual fund",
    "bank/broker": "🟠 bank/broker",
    "pension/sovereign": "🟡 pension/sovereign",
    "other": "⚫ other",
}


def _holder_table(moves: list) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "Institution": m.name,
                "Type": CATEGORY_BADGE.get(m.category.value, m.category.value),
                "Shares Δ": m.shares_change,
                "% Δ": round(m.shares_pct_change, 2) if m.shares_pct_change is not None else None,
                "Value Δ ($)": m.value_change,
                "Own %": round(m.ownership_pct, 2) if m.ownership_pct is not None else None,
                "New": "✳️" if m.is_new_position else "",
                "As of": m.as_of,
            }
            for m in moves
        ]
    )
    # Missing values must render blank, not as the string "None".
    for col in ("Shares Δ", "% Δ", "Value Δ ($)", "Own %"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


with tab_filings:
    st.header("Fintel Filings Lookup")

    filing_ticker = st.text_input("Enter ticker symbol", placeholder="AAPL", key="filing_ticker")

    if filing_ticker:
        filing_ticker = filing_ticker.strip().upper()

        if not fintel.enabled:
            st.warning("Fintel API key not configured. Add FINTEL_API_KEY to your .env file or Streamlit secrets.")
        else:
            # --- 13F Institutional Ownership (full width) ---
            st.subheader(f"13F Institutional Ownership — {filing_ticker}")
            inst_ownership = []
            try:
                with st.spinner("Fetching 13F institutional data..."):
                    inst_ownership = fintel.get_institutional_ownership(filing_ticker)
            except Exception as e:
                st.error(f"Error fetching institutional ownership: {e}")

            if inst_ownership:
                rows = []
                for h in inst_ownership[:100]:
                    row = {}
                    row["Institution"] = h.get("name") or ""
                    row["Form"] = h.get("formType") or ""
                    row["Filing Date"] = h.get("fileDate") or ""
                    row["Effective Date"] = h.get("effectiveDate") or ""
                    row["Shares"] = h.get("shares") or ""
                    row["Shares Change"] = h.get("sharesChange") or ""
                    row["Shares % Change"] = h.get("sharesPercentChange") or ""
                    row["Ownership %"] = h.get("ownershipPercent") or ""
                    row["Value ($)"] = h.get("value") or ""
                    row["Value Change ($)"] = h.get("valueChange") or ""
                    row["URL"] = h.get("url") or ""
                    rows.append(row)

                df_inst = pd.DataFrame(rows)
                # Drop empty columns
                df_inst = df_inst.loc[:, df_inst.ne("").any()]
                if "URL" in df_inst.columns and df_inst["URL"].any():
                    st.dataframe(
                        df_inst,
                        column_config={"URL": st.column_config.LinkColumn("Link", display_text="View")},
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.dataframe(df_inst.drop(columns=["URL"], errors="ignore"), use_container_width=True, hide_index=True)
            else:
                st.info(f"No 13F institutional ownership data found for {filing_ticker}")

            st.markdown("---")

            # --- SEC Filings & Insider Trades side by side ---
            col_sec, col_insider = st.columns(2)

            with col_sec:
                st.subheader(f"SEC Filings — {filing_ticker}")
                sec_filings = []
                try:
                    with st.spinner("Fetching SEC filings..."):
                        sec_filings = fintel.get_sec_filings(filing_ticker)
                except Exception as e:
                    st.error(f"Error fetching SEC filings: {e}")

                if sec_filings:
                    rows = []
                    for f in sec_filings[:50]:
                        row = {}
                        row["Date"] = f.get("filingDate") or f.get("date") or f.get("filed") or ""
                        row["Type"] = f.get("formType") or f.get("type") or f.get("form") or ""
                        row["Description"] = f.get("description") or f.get("title") or f.get("name") or ""
                        row["URL"] = f.get("filingUrl") or f.get("url") or f.get("link") or ""
                        rows.append(row)

                    df_filings = pd.DataFrame(rows)
                    if df_filings["URL"].any():
                        st.dataframe(
                            df_filings,
                            column_config={"URL": st.column_config.LinkColumn("Link", display_text="View")},
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.dataframe(df_filings.drop(columns=["URL"], errors="ignore"), use_container_width=True, hide_index=True)
                else:
                    st.info(f"No SEC filings found for {filing_ticker}")

            with col_insider:
                st.subheader(f"Insider Trades — {filing_ticker}")
                insider_trades = []
                try:
                    with st.spinner("Fetching insider trades..."):
                        insider_trades = fintel.get_insider_trades(filing_ticker)
                except Exception as e:
                    st.error(f"Error fetching insider trades: {e}")

                if insider_trades:
                    rows = []
                    for t in insider_trades[:50]:
                        row = {}
                        row["Date"] = t.get("filingDate") or t.get("date") or t.get("transactionDate") or ""
                        row["Insider"] = t.get("ownerName") or t.get("name") or t.get("insider") or ""
                        row["Title"] = t.get("ownerTitle") or t.get("title") or t.get("relationship") or ""
                        row["Type"] = t.get("transactionType") or t.get("type") or t.get("acquiredDisposed") or ""
                        row["Shares"] = t.get("sharesTraded") or t.get("shares") or t.get("amount") or ""
                        row["Price"] = t.get("pricePerShare") or t.get("price") or ""
                        row["URL"] = t.get("filingUrl") or t.get("url") or t.get("link") or ""
                        rows.append(row)

                    df_insider = pd.DataFrame(rows)
                    if df_insider["URL"].any():
                        st.dataframe(
                            df_insider,
                            column_config={"URL": st.column_config.LinkColumn("Link", display_text="View")},
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.dataframe(df_insider.drop(columns=["URL"], errors="ignore"), use_container_width=True, hide_index=True)
                else:
                    st.info(f"No insider trades found for {filing_ticker}")

            # --- Institutional flow summary ---
            st.markdown("---")
            st.subheader(f"Institutional Flow Summary — {filing_ticker}")

            if not inst_ownership:
                st.info("No 13F/NPORT holdings available, so there is nothing to summarize.")
            else:
                moves = parse_holders(inst_ownership)
                labels, class_status = classify_institutions(tuple(m.name for m in moves))
                moves = apply_categories(moves, labels)
                flow = summarize(filing_ticker, moves)

                if class_status == "no-key":
                    st.caption(
                        "Institution types inferred from names — add ANTHROPIC_API_KEY for "
                        "model-assisted classification."
                    )
                elif class_status != "ok":
                    st.caption(f"Model classification unavailable ({class_status}); using name heuristics.")

                tone = (
                    "#22C55E" if flow.sentiment_score >= 15
                    else "#EF4444" if flow.sentiment_score <= -15
                    else "#F59E0B"
                )
                st.markdown(
                    f"<div style='padding:0.6rem 0.9rem;border-radius:10px;margin:0.3rem 0 0.9rem;"
                    f"background:rgba(59,130,246,0.10);border:1px solid {tone}55;'>"
                    f"<span style='color:{tone};font-weight:600;font-size:1.05rem'>"
                    f"{flow.sentiment_label}</span>"
                    f"<span style='color:#94A3B8'> · score {flow.sentiment_score:+.0f} on −100…+100 · "
                    f"{flow.buyers} of {flow.holders} holders added</span></div>",
                    unsafe_allow_html=True,
                )

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Buy breadth", f"{flow.buy_breadth:.0f}%",
                          help="Share of holders that moved which added rather than reduced")
                k2.metric("Net shares", f"{flow.net_shares:+,.0f}")
                k3.metric("Net value ($)", _fmt_large_number(flow.net_value))
                k4.metric("New positions", f"{flow.new_positions}")

                c_buy, c_sell = st.columns(2)
                with c_buy:
                    st.markdown("**Largest adds**")
                    if flow.top_buyers:
                        st.dataframe(_holder_table(flow.top_buyers), hide_index=True,
                                     use_container_width=True)
                    else:
                        st.caption("No holder increased its position.")
                with c_sell:
                    st.markdown("**Largest reductions**")
                    if flow.top_sellers:
                        st.dataframe(_holder_table(flow.top_sellers), hide_index=True,
                                     use_container_width=True)
                    else:
                        st.caption("No holder reduced its position.")

                if flow.hedge_funds:
                    st.markdown("**Hedge funds in this set**")
                    st.dataframe(_holder_table(flow.hedge_funds), hide_index=True,
                                 use_container_width=True)
                else:
                    st.info(
                        "No holder here was classified as a hedge fund. Fintel returns the largest "
                        "holders, which for a large cap are index, bank and mutual fund managers — "
                        "hedge funds show up more often on smaller names."
                    )

                if flow.by_category:
                    st.markdown("**Net share change by institution type**")
                    cat_df = pd.DataFrame(
                        [{"Type": CATEGORY_BADGE.get(k, k), "Net shares": v}
                         for k, v in sorted(flow.by_category.items(), key=lambda kv: -abs(kv[1]))]
                    )
                    st.dataframe(cat_df, hide_index=True, use_container_width=True)

                narrative, narr_status = institutional_narrative(build_narrative_prompt(flow))
                if narrative:
                    st.markdown("**What this means**")
                    st.markdown(narrative)
                elif narr_status == "no-key":
                    st.caption(
                        "Add ANTHROPIC_API_KEY to get a written interpretation of these numbers."
                    )
                else:
                    st.caption(f"Written summary unavailable ({narr_status}). The figures above are unaffected.")

                for c in flow.caveats:
                    st.warning(c)
                st.caption(
                    "13F and NPORT filings are submitted up to 45 days after quarter end, so this is "
                    "a lagging record of past positioning — not current activity. Descriptive data, "
                    "not investment advice."
                )


# ===================== WATCHLIST ALERTS TAB =====================

# Default watchlist tickers
DEFAULT_WATCHLIST = "AAPL, NVDA, MSFT, GOOGL, AMZN, META, TSLA"

# Initialize session state for watchlist
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []


@st.cache_data(ttl=600, show_spinner=False)
def fetch_watchlist_filings(tickers: tuple) -> dict[str, list[dict]]:
    """Fetch 13F institutional ownership for all watchlist tickers."""
    client = FintelClient()
    if not client.enabled:
        return {}
    results = {}
    for ticker in tickers:
        try:
            data = client.get_institutional_ownership(ticker)
            if data:
                results[ticker] = data
        except Exception:
            pass
    return results


with tab_watchlist:
    st.header("Watchlist — New Institutional Filing Alerts")

    if not fintel.enabled:
        st.warning("Fintel API key not configured. Add FINTEL_API_KEY to your .env file or Streamlit secrets.")
    else:
        # --- Watchlist management ---
        col_input, col_window = st.columns([3, 1])
        with col_input:
            watchlist_input = st.text_input(
                "Watchlist tickers (comma-separated)",
                value=DEFAULT_WATCHLIST,
                key="watchlist_input",
            )
        with col_window:
            alert_days = st.number_input("Alert window (days)", min_value=1, max_value=365, value=90, key="alert_days")

        watchlist_tickers = sorted(set(t.strip().upper() for t in watchlist_input.split(",") if t.strip()))
        cutoff_date = date.today() - timedelta(days=alert_days)

        if not watchlist_tickers:
            st.info("Enter tickers above to monitor for new institutional filings.")
        else:
            st.caption(f"Monitoring **{len(watchlist_tickers)}** tickers — showing filings since **{cutoff_date}**")

            with st.spinner(f"Scanning {len(watchlist_tickers)} tickers for new filings..."):
                all_filings = fetch_watchlist_filings(tuple(watchlist_tickers))

            # Build alerts: filings newer than cutoff
            alert_rows = []
            for ticker in watchlist_tickers:
                filings = all_filings.get(ticker, [])
                for f in filings:
                    file_date_str = f.get("fileDate") or ""
                    if not file_date_str:
                        continue
                    try:
                        file_date = datetime.strptime(file_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    if file_date >= cutoff_date:
                        alert_rows.append({
                            "Ticker": ticker,
                            "Institution": f.get("name") or "",
                            "Form": f.get("formType") or "",
                            "Filing Date": file_date_str,
                            "Effective Date": f.get("effectiveDate") or "",
                            "Shares": f.get("shares") or "",
                            "Shares Change": f.get("sharesChange") or "",
                            "Shares % Chg": f.get("sharesPercentChange") or "",
                            "Ownership %": f.get("ownershipPercent") or "",
                            "Value ($)": f.get("value") or "",
                            "URL": f.get("url") or "",
                        })

            # --- Alert summary ---
            tickers_with_alerts = sorted(set(r["Ticker"] for r in alert_rows))

            if alert_rows:
                st.success(f"**{len(alert_rows)}** new institutional filings across **{len(tickers_with_alerts)}** tickers in the last {alert_days} days")

                # Per-ticker expandable sections
                for ticker in tickers_with_alerts:
                    ticker_rows = [r for r in alert_rows if r["Ticker"] == ticker]
                    with st.expander(f"**{ticker}** — {len(ticker_rows)} new filing(s)", expanded=True):
                        df_alert = pd.DataFrame(ticker_rows).drop(columns=["Ticker"])
                        df_alert = df_alert.loc[:, df_alert.ne("").any()]
                        if "URL" in df_alert.columns and df_alert["URL"].any():
                            st.dataframe(
                                df_alert,
                                column_config={"URL": st.column_config.LinkColumn("Link", display_text="View")},
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.dataframe(df_alert.drop(columns=["URL"], errors="ignore"), use_container_width=True, hide_index=True)

                # Full combined table
                st.markdown("---")
                st.subheader("All Recent Filings")
                df_all = pd.DataFrame(alert_rows)
                df_all = df_all.loc[:, df_all.ne("").any()]
                if "URL" in df_all.columns and df_all["URL"].any():
                    st.dataframe(
                        df_all,
                        column_config={"URL": st.column_config.LinkColumn("Link", display_text="View")},
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.dataframe(df_all.drop(columns=["URL"], errors="ignore"), use_container_width=True, hide_index=True)
            else:
                st.info(f"No new institutional filings found in the last {alert_days} days for your watchlist.")

            # Show tickers with no data
            no_data_tickers = [t for t in watchlist_tickers if t not in all_filings]
            if no_data_tickers:
                st.caption(f"No Fintel data available for: {', '.join(no_data_tickers)}")


# ===================== CHART SETUP TAB =====================

def _trend_arrow(score: float) -> str:
    if score >= 25:
        return "▲"
    if score <= -25:
        return "▼"
    return "▬"


ZONE_STYLE = {
    "bullish_ob": ("rgba(34,197,94,0.16)", "#22C55E", "Bullish OB"),
    "bearish_ob": ("rgba(239,68,68,0.16)", "#EF4444", "Bearish OB"),
    "bullish_fvg": ("rgba(6,182,212,0.12)", "#06B6D4", "Bullish FVG"),
    "bearish_fvg": ("rgba(249,115,22,0.12)", "#F97316", "Bearish FVG"),
}


@st.cache_data(ttl=900, show_spinner=False)
def run_chart_analysis(ticker: str, tfs: tuple, entry_tf: str):
    """Fetch every timeframe once, then read the chart. Cached for 15 minutes."""
    frames, warnings = fetch_timeframes(ticker, list(tfs))
    analysis = analyze_chart(ticker, list(tfs), entry_tf, frames=frames, warnings=warnings)
    return analysis, frames


def build_setup_chart(df: pd.DataFrame, analysis, tf_key: str, bars: int = 180) -> go.Figure:
    """Candles + EMAs + volume, with the zones and the trade plan drawn on top."""
    view = df.tail(bars)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"],
            low=view["Low"], close=view["Close"], name="Price",
            increasing_line_color="#22C55E", increasing_fillcolor="#22C55E",
            decreasing_line_color="#EF4444", decreasing_fillcolor="#EF4444",
        ),
        row=1, col=1,
    )

    for span, color in ((20, "#F59E0B"), (50, "#3B82F6"), (200, "#8B5CF6")):
        if len(df) >= span:
            ema = df["Close"].ewm(span=span, adjust=False).mean().tail(bars)
            fig.add_trace(
                go.Scatter(x=view.index, y=ema, name=f"EMA{span}", mode="lines",
                           line=dict(color=color, width=1.2)),
                row=1, col=1,
            )

    vol_colors = ["#22C55E" if c >= o else "#EF4444"
                  for c, o in zip(view["Close"], view["Open"])]
    fig.add_trace(
        go.Bar(x=view.index, y=view["Volume"], name="Volume",
               marker_color=vol_colors, opacity=0.45, showlegend=False),
        row=2, col=1,
    )

    x0, x1 = view.index[0], view.index[-1]
    lo, hi = float(view["Low"].min()), float(view["High"].max())

    def in_view(*prices) -> bool:
        pad = (hi - lo) * 0.35
        return all(lo - pad <= p <= hi + pad for p in prices)

    # Only draw the structure that's near enough to matter — a chart covered in
    # boxes hides the one zone the plan is built on.
    spot = float(view["Close"].iloc[-1])
    tf_atr = float((view["High"] - view["Low"]).tail(14).mean()) or spot * 0.02
    near = max(3.5 * tf_atr, (hi - lo) * 0.12)

    for z in analysis.zones:
        if z.timeframe != tf_key or not in_view(z.top, z.bottom):
            continue
        if min(abs(z.top - spot), abs(z.bottom - spot)) > near:
            continue
        fill, line, _ = ZONE_STYLE.get(z.kind, ("rgba(148,163,184,0.1)", "#94A3B8", z.kind))
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=z.bottom, y1=z.top,
            fillcolor=fill, line=dict(color=line, width=0.6, dash="dot"),
            layer="below", row=1, col=1,
        )

    s = analysis.setup
    if s.entry_low and s.entry_high and in_view(s.entry_low, s.entry_high):
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=s.entry_low, y1=s.entry_high,
            fillcolor="rgba(59,130,246,0.22)", line=dict(color="#3B82F6", width=1.4),
            layer="below", row=1, col=1,
        )
        fig.add_annotation(x=x1, y=s.entry_high, text=f"Entry {s.entry_low:,.2f}–{s.entry_high:,.2f}",
                           showarrow=False, xanchor="right", yanchor="bottom",
                           font=dict(color="#93C5FD", size=11), row=1, col=1)

    plan_lines = [
        (s.stop, "#EF4444", f"Stop {s.stop:,.2f}" if s.stop else ""),
        (s.target_1, "#22C55E", f"T1 {s.target_1:,.2f} ({s.rr_target_1:.2f}R)" if s.target_1 else ""),
        (s.target_2, "#16A34A", f"T2 {s.target_2:,.2f} ({s.rr_target_2:.2f}R)" if s.target_2 else ""),
        (s.target_3, "#15803D", f"T3 {s.target_3:,.2f}" if s.target_3 else ""),
    ]
    for value, color, label in plan_lines:
        if value is None or not in_view(value):
            continue
        fig.add_hline(y=value, line=dict(color=color, width=1.2, dash="dash"),
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(color=color, size=11), row=1, col=1)

    # Levels already inside the entry band are redundant — the band shows them.
    in_zone = (
        (lambda v: s.entry_low <= v <= s.entry_high)
        if s.entry_low and s.entry_high else (lambda v: False)
    )
    nearest_levels = [lv for lv in analysis.levels if not in_zone(lv.price)]
    nearest_levels = sorted(nearest_levels, key=lambda lv: abs(lv.price - spot))[:5]
    for lv in nearest_levels:
        if not in_view(lv.price):
            continue
        fig.add_hline(y=lv.price, line=dict(color="rgba(148,163,184,0.35)", width=0.8, dash="dot"),
                      annotation_text=f"{lv.price:,.2f} ×{lv.touches}",
                      annotation_position="left",
                      annotation_font=dict(color="#94A3B8", size=9), row=1, col=1)

    fig.update_layout(
        height=620,
        title=f"{analysis.ticker} — {TF_SPECS[tf_key].label}",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        margin=dict(t=50, b=20, l=20, r=90),
    )
    # Hide the gaps that make intraday equity charts look broken.
    if tf_key in ("15m", "1h", "4h"):
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
    elif tf_key == "1d":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return fig


with tab_chart:
    st.subheader("Chart Setup")
    st.caption(
        "Reads price structure across timeframes — swings, breaks of structure, order "
        "blocks, imbalances and level clusters — then proposes one entry, stop and "
        "target ladder. Structural read of price data, not investment advice."
    )

    c1, c2, c3, c4 = st.columns([2, 3, 1.4, 1.4])
    with c1:
        default_ticker = all_tickers[0] if all_tickers else "NVDA"
        chart_ticker = st.text_input("Ticker", value=default_ticker, key="chart_ticker").upper().strip()
    with c2:
        chart_tfs = st.multiselect(
            "Timeframes", options=list(TF_SPECS.keys()),
            default=["1wk", "1d", "4h", "1h"],
            format_func=lambda k: TF_SPECS[k].label, key="chart_tfs",
        )
    with c3:
        entry_tf = st.selectbox(
            "Entry timeframe", options=chart_tfs or ["1d"],
            index=(chart_tfs.index("1d") if "1d" in chart_tfs else 0),
            format_func=lambda k: TF_SPECS[k].label, key="chart_entry_tf",
        )
    with c4:
        account_size = st.number_input("Account ($)", min_value=0, value=0, step=1000, key="chart_account")

    if not chart_ticker:
        st.info("Enter a ticker to analyze.")
    elif not chart_tfs:
        st.warning("Pick at least one timeframe.")
    else:
        with st.spinner(f"Reading {chart_ticker} across {len(chart_tfs)} timeframes…"):
            try:
                analysis, frames = run_chart_analysis(chart_ticker, tuple(chart_tfs), entry_tf)
            except Exception as exc:
                analysis, frames = None, None
                st.error(f"Could not analyze {chart_ticker}: {exc}")

        if analysis:
            setup = analysis.setup
            bias_color = {"long": "#22C55E", "short": "#EF4444"}.get(analysis.bias.value, "#F59E0B")
            status_text = {
                "at_entry": "Price is in the entry zone now",
                "approaching": "Within 1 ATR of the entry zone",
                "wait": "Valid setup — wait for the pullback",
                "no_setup": "No actionable setup",
            }[setup.status.value]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Price ($)", f"{analysis.price:,.2f}", help=f"As of {analysis.as_of}")
            m2.metric("Bias", f"{analysis.bias_score:+.0f}", analysis.bias_label, delta_color="off")
            m3.metric("TF alignment", f"{analysis.alignment_pct:.0f}%",
                      help="Share of timeframe weight agreeing with the bias")
            m4.metric("Setup grade", setup.grade, f"{setup.confidence:.0f}/100", delta_color="off")

            distance = (
                f" · entry is {setup.distance_to_entry_pct:+.1f}% from spot"
                if setup.distance_to_entry_pct is not None else ""
            )
            st.markdown(
                f"<div style='padding:0.55rem 0.9rem;border-radius:10px;margin:0.4rem 0 0.8rem;"
                f"background:rgba(59,130,246,0.12);border:1px solid {bias_color}44;'>"
                f"<span style='color:{bias_color};font-weight:600'>{status_text}</span>"
                f"<span style='color:#94A3B8'>{distance}</span></div>",
                unsafe_allow_html=True,
            )

            for w in analysis.warnings:
                st.warning(w)

            chart_tf = st.radio(
                "Chart", options=[k for k in chart_tfs if k in frames],
                format_func=lambda k: TF_SPECS[k].label,
                index=[k for k in chart_tfs if k in frames].index(entry_tf) if entry_tf in frames else 0,
                horizontal=True, key="chart_view_tf",
            )
            st.plotly_chart(
                build_setup_chart(frames[chart_tf], analysis, chart_tf),
                use_container_width=True,
            )

            left, right = st.columns([1.15, 1])
            with left:
                st.markdown(f"### {setup.setup_type or 'No setup'}")
                if setup.entry_ref is None:
                    st.info(setup.invalidation or "No actionable entry right now.")
                else:
                    risk_pct_price = setup.risk_per_share / setup.entry_ref * 100
                    plan = pd.DataFrame(
                        [
                            ("Entry zone", f"{setup.entry_low:,.2f} – {setup.entry_high:,.2f}", ""),
                            ("Work the order at", f"{setup.entry_ref:,.2f}", f"{setup.distance_to_entry_pct:+.1f}% from spot"),
                            ("Stop", f"{setup.stop:,.2f}", f"risk {setup.risk_per_share:,.2f}/sh ({risk_pct_price:.1f}%)"),
                            ("Target 1", f"{setup.target_1:,.2f}", f"{setup.rr_target_1:.2f}R"),
                            ("Target 2", f"{setup.target_2:,.2f}", f"{setup.rr_target_2:.2f}R"),
                            ("Target 3", f"{setup.target_3:,.2f}", ""),
                        ],
                        columns=["", "Price", "Note"],
                    )
                    st.dataframe(plan, hide_index=True, use_container_width=True)

                    if account_size > 0:
                        risk_pct = st.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25, key="chart_risk_pct")
                        shares = position_size(setup.risk_per_share, account_size, risk_pct)
                        st.success(
                            f"**{shares:,} shares** — ${shares * setup.entry_ref:,.0f} notional, "
                            f"risking ${account_size * risk_pct / 100:,.0f} if the stop hits."
                        )

                    st.markdown(f"**Invalidation** — {setup.invalidation}")
                    for n in setup.notes:
                        st.warning(n)

            with right:
                if setup.confluences:
                    st.markdown("### Why this zone")
                    for c in setup.confluences:
                        st.markdown(f"- {c}")
                if setup.triggers:
                    st.markdown("### Wait for")
                    for t in setup.triggers:
                        st.markdown(f"- {t}")

            st.markdown("### Timeframe read")
            tf_df = pd.DataFrame(
                [
                    {
                        "TF": r.label,
                        "Trend": f"{_trend_arrow(r.trend_score)} {r.trend}",
                        "Score": r.trend_score,
                        "Structure": r.structure, "Last event": r.last_event or "—",
                        "EMA stack": r.ema_stack, "RSI": r.rsi_14, "ADX": r.adx_14,
                        "ATR %": r.atr_pct, "Rel vol": r.rel_volume,
                        "Swing high": r.swing_high, "Swing low": r.swing_low,
                    }
                    for r in analysis.timeframes
                ]
            )
            st.dataframe(
                tf_df, hide_index=True, use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", help="-100 (bearish) to +100 (bullish)",
                        min_value=-100, max_value=100, format="%.0f",
                    )
                },
            )

            with st.expander("Zones and levels the read is built on"):
                zc, lc = st.columns(2)
                with zc:
                    st.markdown("**Unmitigated zones**")
                    if analysis.zones:
                        st.dataframe(
                            pd.DataFrame([
                                {"TF": TF_SPECS[z.timeframe].label, "Kind": z.kind.replace("_", " "),
                                 "Low": z.bottom, "High": z.top, "Created": z.created_at,
                                 "Tested": z.tested, "Impulse (ATR)": z.strength}
                                for z in analysis.zones
                            ]), hide_index=True, use_container_width=True,
                        )
                    else:
                        st.caption("No unmitigated zones in range.")
                with lc:
                    st.markdown("**Level clusters**")
                    st.dataframe(
                        pd.DataFrame([
                            {"Price": lv.price, "Kind": lv.kind, "Touches": lv.touches,
                             "Timeframes": ", ".join(TF_SPECS[t].label for t in lv.timeframes),
                             "Last touch": lv.last_touch}
                            for lv in analysis.levels
                        ]), hide_index=True, use_container_width=True,
                    )
