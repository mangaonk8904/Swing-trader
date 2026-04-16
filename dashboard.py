import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from io import BytesIO
from datetime import date

from data.yahoo import get_price_data, get_basic_fundamentals
from data.excel_io import read_revenue_data, read_institutional_data, get_available_sheets
from data.fintel import FintelClient
from analysis.technicals import compute_technicals
from analysis.scoring import score_stock
from models import FundamentalData, InstitutionalData, StockScore

st.set_page_config(page_title="Swing Trader", page_icon="📊", layout="wide")
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

# Data source status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
st.sidebar.markdown(f"- yfinance: :green[Active]")
if fintel.enabled:
    st.sidebar.markdown(f"- Fintel.io: :green[Active]")
else:
    st.sidebar.markdown(f"- Fintel.io: :gray[No API key]")
st.sidebar.markdown(f"- Seeking Alpha: :gray[Phase 6]")

# Combine tickers from file + manual input
file_tickers = sorted(set(list(fund_map.keys()) + list(inst_map.keys())))
manual_list = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()] if manual_tickers else []
all_tickers = sorted(set(file_tickers + manual_list))

# --- Tabs ---
tab_analysis, tab_filings = st.tabs(["Analysis", "Fintel Filings"])


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
def fetch_and_score(tickers: tuple, fund_data: dict, inst_data: dict, fintel_enabled: bool):
    scores = []
    tech_data = {}
    price_data = {}

    fintel_client = FintelClient() if fintel_enabled else None

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

        result = score_stock(
            tech=tech,
            fund=fund_data.get(ticker),
            inst=inst,
        )
        scores.append(result)

    return scores, tech_data, price_data


def color_signal(val):
    colors = {
        "STRONG_BUY": "background-color: #1a7a1a; color: white",
        "BUY": "background-color: #2ecc71; color: white",
        "NEUTRAL": "background-color: #f39c12; color: white",
        "PASS": "background-color: #e74c3c; color: white",
    }
    return colors.get(val, "")


def color_composite(val):
    if val >= 75:
        return "background-color: #1a7a1a; color: white"
    elif val >= 55:
        return "background-color: #2ecc71; color: white"
    elif val >= 40:
        return "background-color: #f39c12; color: white"
    else:
        return "background-color: #e74c3c; color: white"


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

        scores, tech_data, price_data = fetch_and_score(tuple(all_tickers), fund_for_score, inst_for_score, fintel.enabled)

        # --- Scores Table ---
        st.header("Swing Trade Scores")

        score_rows = []
        for s in sorted(scores, key=lambda x: x.composite_score, reverse=True):
            score_rows.append({
                "Ticker": s.ticker,
                "Signal": s.signal.value.upper(),
                "Composite": s.composite_score,
                "Technical": s.technical_score,
                "Fundamental": s.fundamental_score,
                "Institutional": s.institutional_score,
                "Entry": s.entry_price,
                "Stop Loss": s.stop_loss,
                "Target": s.target_price,
                "Notes": s.notes,
            })

        score_df = pd.DataFrame(score_rows)

        styled_df = score_df.style.map(color_signal, subset=["Signal"]).map(color_composite, subset=["Composite"]).format({
            "Composite": "{:.1f}",
            "Technical": "{:.1f}",
            "Fundamental": "{:.1f}",
            "Institutional": "{:.1f}",
            "Entry": "${:,.2f}",
            "Stop Loss": "${:,.2f}",
            "Target": "${:,.2f}",
        }, na_rep="N/A")

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


# ===================== FINTEL FILINGS TAB =====================
with tab_filings:
    st.header("Fintel Filings Lookup")

    filing_ticker = st.text_input("Enter ticker symbol", placeholder="AAPL", key="filing_ticker")

    if filing_ticker:
        filing_ticker = filing_ticker.strip().upper()

        if not fintel.enabled:
            st.warning("Fintel API key not configured. Add FINTEL_API_KEY to your .env file or Streamlit secrets.")
        else:
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
