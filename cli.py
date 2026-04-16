import typer
from typing import List, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from data.excel_io import (
    read_revenue_data,
    read_institutional_data,
    get_available_sheets,
    export_results,
)
from data.yahoo import get_price_data, get_basic_fundamentals
from data.fintel import FintelClient
from analysis.technicals import compute_technicals
from analysis.scoring import score_stock

from schemas import InstitutionalData

app = typer.Typer(help="Swing Trader Analysis Tool")
console = Console()


def _merge_institutional(excel: InstitutionalData | None, fintel: InstitutionalData) -> InstitutionalData:
    """Merge Fintel data over Excel data. Fintel values take precedence when present."""
    if excel is None:
        return fintel
    return InstitutionalData(
        ticker=fintel.ticker,
        institutional_buyers=fintel.institutional_buyers if fintel.institutional_buyers is not None else excel.institutional_buyers,
        institutional_sellers=fintel.institutional_sellers if fintel.institutional_sellers is not None else excel.institutional_sellers,
        net_institutional=fintel.net_institutional if fintel.net_institutional is not None else excel.net_institutional,
        short_interest_pct=fintel.short_interest_pct if fintel.short_interest_pct is not None else excel.short_interest_pct,
        short_interest_change=fintel.short_interest_change if fintel.short_interest_change is not None else excel.short_interest_change,
    )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Swing Trader — analyze stocks for swing trade opportunities."""
    if ctx.invoked_subcommand is None:
        console.print("[bold]Swing Trader[/bold] — use --help to see commands")


@app.command()
def analyze(
    file: Path = typer.Option(..., "--file", "-f", help="Excel file with Revenue and/or Institutional sheets"),
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="Export scored results to Excel file"),
):
    """Analyze stocks from an Excel file — scores them and recommends buy/pass."""
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    sheets = get_available_sheets(file)
    console.print(f"\n[bold]File:[/bold] {file}")
    console.print(f"[bold]Sheets found:[/bold] {', '.join(sheets)}\n")

    # Load data from Excel
    fund_map: dict = {}
    inst_map: dict = {}

    if "Revenue" in sheets:
        for f in read_revenue_data(file):
            fund_map[f.ticker] = f
    else:
        console.print("[yellow]No 'Revenue' sheet — skipping revenue data[/yellow]")

    if "Institutional" in sheets:
        for i in read_institutional_data(file):
            inst_map[i.ticker] = i
    else:
        console.print("[yellow]No 'Institutional' sheet — skipping institutional data[/yellow]")

    # Fintel integration — merge with Excel data
    fintel = FintelClient()
    if fintel.enabled:
        console.print("[green]Fintel.io API connected[/green]")
    else:
        console.print("[dim]Fintel.io — no API key (using Excel data only)[/dim]")

    # Get all unique tickers
    all_tickers = sorted(set(list(fund_map.keys()) + list(inst_map.keys())))

    if not all_tickers:
        console.print("[red]No tickers found in file[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Scoring {len(all_tickers)} tickers:[/bold] {', '.join(all_tickers)}\n")

    # Fetch technicals and score each ticker
    scores = []
    data_sources: dict[str, list[str]] = {}
    for ticker in all_tickers:
        sources = []
        tech = None
        try:
            df = get_price_data(ticker)
            tech = compute_technicals(ticker, df)
            sources.append("yfinance")
        except Exception as e:
            console.print(f"[yellow]{ticker}: Could not fetch price data — {e}[/yellow]")

        # Merge Fintel data over Excel data (Fintel takes precedence)
        inst = inst_map.get(ticker)
        if fintel.enabled:
            fintel_data = fintel.get_institutional_data(ticker)
            if fintel_data is not None:
                inst = _merge_institutional(inst, fintel_data)
                sources.append("Fintel")

        if inst:
            sources.append("Excel" if ticker in inst_map and "Fintel" not in sources else "")
        if ticker in fund_map:
            sources.append("Excel")

        data_sources[ticker] = [s for s in sources if s]

        result = score_stock(
            tech=tech,
            fund=fund_map.get(ticker),
            inst=inst,
        )
        scores.append(result)

    # Display scores table
    table = Table(title="Swing Trade Scores")
    table.add_column("Ticker", style="cyan")
    table.add_column("Signal", justify="center")
    table.add_column("Composite", justify="right")
    table.add_column("Technical", justify="right")
    table.add_column("Fundamental", justify="right")
    table.add_column("Institutional", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Stop", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Sources", style="dim")

    signal_styles = {
        "strong_buy": "bold green",
        "buy": "green",
        "neutral": "yellow",
        "pass": "red",
    }

    for s in sorted(scores, key=lambda x: x.composite_score, reverse=True):
        style = signal_styles.get(s.signal.value, "dim")
        entry = f"${s.entry_price:.2f}" if s.entry_price else "N/A"
        stop = f"${s.stop_loss:.2f}" if s.stop_loss else "N/A"
        target = f"${s.target_price:.2f}" if s.target_price else "N/A"
        sources = ", ".join(sorted(set(data_sources.get(s.ticker, []))))

        table.add_row(
            s.ticker,
            f"[{style}]{s.signal.value.upper()}[/{style}]",
            f"{s.composite_score:.1f}",
            f"{s.technical_score:.1f}",
            f"{s.fundamental_score:.1f}",
            f"{s.institutional_score:.1f}",
            entry,
            stop,
            target,
            sources,
        )

    console.print(table)

    # Export if requested
    if export:
        out_path = export_results(scores, export)
        console.print(f"\n[green]Results exported to {out_path}[/green]")


@app.command()
def scan(
    tickers: List[str] = typer.Argument(..., help="One or more ticker symbols (e.g. AAPL MSFT NVDA)"),
):
    """Fetch live price data and compute technical indicators for given tickers."""
    table = Table(title="Technical Scan")
    table.add_column("Ticker", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("RSI(14)", justify="right")
    table.add_column("MACD", justify="center")
    table.add_column(">SMA50", justify="center")
    table.add_column(">SMA200", justify="center")
    table.add_column("Vol Ratio", justify="right")
    table.add_column("ATR(14)", justify="right")

    for ticker in tickers:
        ticker = ticker.upper()
        try:
            df = get_price_data(ticker)
            snap = compute_technicals(ticker, df)

            rsi_str = f"{snap.rsi_14:.1f}" if snap.rsi_14 is not None else "N/A"
            rsi_style = "green" if snap.rsi_14 and 30 <= snap.rsi_14 <= 70 else "red"

            macd_str = snap.macd_signal or "N/A"
            macd_style = "green" if macd_str == "bullish" else ("red" if macd_str == "bearish" else "dim")

            sma50 = "[green]Yes[/green]" if snap.above_sma_50 else "[red]No[/red]" if snap.above_sma_50 is not None else "N/A"
            sma200 = "[green]Yes[/green]" if snap.above_sma_200 else "[red]No[/red]" if snap.above_sma_200 is not None else "N/A"

            vol_str = f"{snap.volume_vs_avg:.2f}x" if snap.volume_vs_avg is not None else "N/A"
            atr_str = f"${snap.atr_14:.2f}" if snap.atr_14 is not None else "N/A"

            table.add_row(
                snap.ticker,
                f"${snap.price:.2f}",
                f"[{rsi_style}]{rsi_str}[/{rsi_style}]",
                f"[{macd_style}]{macd_str}[/{macd_style}]",
                sma50,
                sma200,
                vol_str,
                atr_str,
            )
        except Exception as e:
            table.add_row(ticker, f"[red]Error: {e}[/red]", "", "", "", "", "", "")

    console.print(table)

    # Show fundamentals summary
    console.print()
    for ticker in tickers:
        ticker = ticker.upper()
        try:
            info = get_basic_fundamentals(ticker)
            console.print(f"[bold cyan]{ticker}[/bold cyan] — {info['name']} | "
                          f"Sector: {info.get('sector', 'N/A')} | "
                          f"P/E: {info.get('pe_ratio', 'N/A')} | "
                          f"52w: ${info.get('fifty_two_week_low', 0):.0f}-${info.get('fifty_two_week_high', 0):.0f}")
        except Exception:
            pass

    # Show Fintel data if available
    fintel = FintelClient()
    if fintel.enabled:
        console.print()
        inst_table = Table(title="Institutional Data (Fintel.io)")
        inst_table.add_column("Ticker", style="cyan")
        inst_table.add_column("Buyers", justify="right")
        inst_table.add_column("Sellers", justify="right")
        inst_table.add_column("Net", justify="right")
        inst_table.add_column("Short %", justify="right")
        inst_table.add_column("Short Chg", justify="right")

        for ticker in tickers:
            ticker = ticker.upper()
            data = fintel.get_institutional_data(ticker)
            if data:
                buyers = str(data.institutional_buyers) if data.institutional_buyers is not None else "N/A"
                sellers = str(data.institutional_sellers) if data.institutional_sellers is not None else "N/A"
                net = str(data.net_institutional) if data.net_institutional is not None else "N/A"
                net_style = "green" if (data.net_institutional or 0) > 0 else "red"
                short_pct = f"{data.short_interest_pct:.1f}%" if data.short_interest_pct is not None else "N/A"
                short_chg = f"{data.short_interest_change:+.2f}%" if data.short_interest_change is not None else "N/A"
                inst_table.add_row(ticker, buyers, sellers, f"[{net_style}]{net}[/{net_style}]", short_pct, short_chg)
            else:
                inst_table.add_row(ticker, "[dim]N/A[/dim]", "", "", "", "")

        console.print(inst_table)


if __name__ == "__main__":
    app()
