"""One-time script to create sample_data/example_portfolio.xlsx"""
import pandas as pd
from pathlib import Path

output = Path("sample_data/example_portfolio.xlsx")
output.parent.mkdir(exist_ok=True)

revenue_df = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "Revenue_Current": [94_930, 65_585, 39_331, 96_469, 187_792],
    "Revenue_Prior": [89_498, 56_189, 22_103, 86_311, 170_000],
})

institutional_df = pd.DataFrame({
    "Ticker": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "Buyers": [120, 95, 200, 80, 110],
    "Sellers": [45, 30, 50, 60, 40],
    "Short_Interest_Pct": [0.7, 0.5, 1.2, 0.8, 0.9],
})

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    revenue_df.to_excel(writer, sheet_name="Revenue", index=False)
    institutional_df.to_excel(writer, sheet_name="Institutional", index=False)

print(f"Created {output}")
