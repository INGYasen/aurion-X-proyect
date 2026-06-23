"""Test connectivity to yfinance for DXY, US10Y, VIX."""
import yfinance as yf

SYMBOLS = [("DX-Y.NYB", "DXY"), ("^TNX", "US10Y"), ("^VIX", "VIX")]
for sym, label in SYMBOLS:
    try:
        t = yf.Ticker(sym)
        h = t.history(period="5d", interval="1h", auto_adjust=False)
        if h.empty:
            print(f"  {label} ({sym}): EMPTY")
        else:
            last_close = float(h["Close"].iloc[-1])
            print(f"  {label} ({sym}): {len(h)} bars, last_close={last_close:.4f}")
    except Exception as exc:
        print(f"  {label} ({sym}): FAIL {exc}")
