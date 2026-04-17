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

from data.yahoo import get_price_data, get_basic_fundamentals, get_full_fundamentals, get_options_expirations, get_options_chain, get_all_options_summary
from data.excel_io import read_revenue_data, read_institutional_data, get_available_sheets
from data.fintel import FintelClient
from data.seekingalpha import SeekingAlphaClient
from analysis.technicals import compute_technicals
from analysis.scoring import score_stock
from schemas import FundamentalData, InstitutionalData, SeekingAlphaData, StockScore
from config import settings

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
    import tempfile, os
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
tab_analysis, tab_fundamentals, tab_options, tab_filings, tab_watchlist = st.tabs(["Analysis", "Fundamentals", "Options Flow", "Fintel Filings", "Watchlist Alerts"])


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
            _groq_key_fund = settings.groq_api_key or st.secrets.get("GROQ_API_KEY", "")
            if not _groq_key_fund:
                st.warning("Groq API key not configured. Add GROQ_API_KEY to your .env file or Streamlit secrets.")
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

                with st.spinner("Analyzing fundamentals with Llama..."):
                    try:
                        from groq import Groq
                        client = Groq(api_key=_groq_key_fund)
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                            max_tokens=1200,
                        )
                        st.markdown(response.choices[0].message.content)
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
                    _groq_key = settings.groq_api_key or st.secrets.get("GROQ_API_KEY", "")
                    if not _groq_key:
                        st.warning("Groq API key not configured. Add GROQ_API_KEY to your .env file or Streamlit secrets.")
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

                        with st.spinner("Analyzing with Llama..."):
                            try:
                                from groq import Groq
                                client = Groq(api_key=_groq_key)
                                response = client.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.3,
                                    max_tokens=1000,
                                )
                                analysis = response.choices[0].message.content
                                st.markdown(analysis)
                            except Exception as e:
                                st.error(f"AI analysis failed: {e}")

                    # --- Section 7: Raw Chain ---
                    with st.expander("View Full Calls Chain"):
                        st.dataframe(calls_df, use_container_width=True, hide_index=True)
                    with st.expander("View Full Puts Chain"):
                        st.dataframe(puts_df, use_container_width=True, hide_index=True)


# ===================== FINTEL FILINGS TAB =====================
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
