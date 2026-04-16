import pandas as pd
from pathlib import Path
from models import FundamentalData, InstitutionalData, StockScore


def read_revenue_data(file_path: str | Path) -> list[FundamentalData]:
    """Read revenue data from the 'Revenue' sheet of an Excel file.

    Expected columns: Ticker, Revenue_Current, Revenue_Prior
    """
    path = Path(file_path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="Revenue")

    results = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not ticker:
            continue

        current = _safe_float(row.get("Revenue_Current"))
        prior = _safe_float(row.get("Revenue_Prior"))

        growth = None
        if current is not None and prior is not None and prior != 0:
            growth = ((current - prior) / abs(prior)) * 100

        results.append(FundamentalData(
            ticker=ticker,
            revenue_current=current,
            revenue_prior=prior,
            revenue_growth_pct=growth,
        ))

    return results


def read_institutional_data(file_path: str | Path) -> list[InstitutionalData]:
    """Read 13F institutional data from the 'Institutional' sheet of an Excel file.

    Expected columns: Ticker, Buyers, Sellers, Short_Interest_Pct
    """
    path = Path(file_path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="Institutional")

    results = []
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        if not ticker:
            continue

        buyers = _safe_int(row.get("Buyers"))
        sellers = _safe_int(row.get("Sellers"))
        net = (buyers or 0) - (sellers or 0) if buyers is not None or sellers is not None else None

        results.append(InstitutionalData(
            ticker=ticker,
            institutional_buyers=buyers,
            institutional_sellers=sellers,
            net_institutional=net,
            short_interest_pct=_safe_float(row.get("Short_Interest_Pct")),
        ))

    return results


def export_results(scores: list[StockScore], file_path: str | Path) -> Path:
    """Export scored results to an Excel file."""
    path = Path(file_path)
    rows = []
    for s in scores:
        rows.append({
            "Ticker": s.ticker,
            "Date": s.date,
            "Signal": s.signal.value,
            "Composite": round(s.composite_score, 1),
            "Technical": round(s.technical_score, 1),
            "Fundamental": round(s.fundamental_score, 1),
            "Institutional": round(s.institutional_score, 1),
            "Entry": s.entry_price,
            "Stop Loss": s.stop_loss,
            "Target": s.target_price,
            "Notes": s.notes,
        })

    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, sheet_name="Scores")
    return path


def get_available_sheets(file_path: str | Path) -> list[str]:
    """Return list of sheet names in an Excel file."""
    path = Path(file_path)
    if path.suffix == ".csv":
        return ["csv"]
    xls = pd.ExcelFile(path)
    return xls.sheet_names


def _safe_float(val) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> int | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None
