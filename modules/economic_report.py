# modules/economic_report.py

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
import pandas_datareader.data as web

# ---------- Robust price fetcher ----------
def _fetch_prices(symbol: str, start, end) -> pd.DataFrame | None:
    """Return df with columns: Date, Close (float). Handles Adj Close/multi-index/fallbacks."""
    def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        # Flatten multi-index columns if any
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join([c for c in tup if c]) for tup in df.columns.to_list()]
        cols = list(df.columns)
        price_col = None
        for cand in ["Adj Close", "Adj Close_"+symbol, "Close", f"Close_{symbol}"]:
            if cand in cols:
                price_col = cand
                break
        if price_col is None:
            return None
        out = df.reset_index().rename(columns={price_col: "Close"})
        if "Date" not in out.columns:
            if "index" in out.columns:
                out = out.rename(columns={"index": "Date"})
            elif "Datetime" in out.columns:
                out = out.rename(columns={"Datetime": "Date"})
        if "Date" not in out.columns:
            return None
        return out[["Date", "Close"]].dropna()

    # Try Yahoo Finance (download)
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end + timedelta(days=1),
            interval="1d",
            auto_adjust=True,
            actions=False,
            progress=False,
            group_by="column"
        )
        norm = _normalize(df)
        if norm is not None and not norm.empty:
            return norm
    except Exception:
        pass

    # Fallback to yfinance Ticker().history
    try:
        tk = yf.Ticker(symbol)
        df2 = tk.history(start=start, end=end + timedelta(days=1), interval="1d", auto_adjust=True)
        norm2 = _normalize(df2)
        if norm2 is not None and not norm2.empty:
            return norm2
    except Exception:
        pass

    return None


def run_economic_report():
    st.title("📈 Economic Report Generator")
    st.write("Generate summarized reports of economic and market indicators using real data and LLMs.")

    # ---------- Sidebar ----------
    st.sidebar.header("Report Settings")
    country = st.sidebar.selectbox(
        "Select a country / market:",
        ["India", "United States", "United Kingdom", "Japan", "Germany"]
    )

    index_options = {
        "India": {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK", "NIFTY IT": "^CNXIT"},
        "United States": {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC"},
        "United Kingdom": {"FTSE 100": "^FTSE"},
        "Japan": {"Nikkei 225": "^N225"},
        "Germany": {"DAX 40": "^GDAXI"}
    }
    symbol_map = index_options[country]
    index_name = st.sidebar.selectbox("Select Stock Index:", list(symbol_map.keys()))
    index_symbol = symbol_map[index_name]

    today = datetime.today()
    default_start = today - timedelta(days=90)
    start_date = st.sidebar.date_input("Start date:", default_start)
    end_date = st.sidebar.date_input("End date:", today)

    compare = st.sidebar.checkbox("Compare with another country's index (e.g., NIFTY vs S&P 500)")
    other_index_symbol = other_index_name = None
    if compare:
        other_country = st.sidebar.selectbox(
            "Select comparison country:",
            [c for c in index_options.keys() if c != country]
        )
        other_map = index_options[other_country]
        other_index_name = st.sidebar.selectbox("Select comparison index:", list(other_map.keys()))
        other_index_symbol = other_map[other_index_name]

    if not st.sidebar.button("📊 Generate Report"):
        st.info("Select your options and click **Generate Report** to begin.")
        return

    # ---------- Fetch primary data ----------
    st.subheader("📉 Market Overview")
    data = _fetch_prices(index_symbol, start_date, end_date)
    if data is None or data.empty:
        st.error(f"Error fetching data: ['Close']  \n(Ticker: {index_symbol}). Try changing the date range or index.")
        return

    st.caption(f"📆 Data from {data['Date'].min().date()} to {data['Date'].max().date()}")
    st.line_chart(data.set_index("Date")["Close"], height=300)

    latest_price = round(float(data["Close"].iloc[-1]), 2)
    pct_change = round((data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) * 100, 2)
    col1, col2 = st.columns(2)
    col1.metric(label=f"{index_name} ({index_symbol})", value=f"{latest_price:,.2f}")
    col2.metric(label="% Change", value=f"{pct_change}%")

    # ---------- Key statistics ----------
    st.subheader("📘 Key Statistics")
    summary = {
        "Mean Price": round(data["Close"].mean(), 2),
        "Median Price": round(data["Close"].median(), 2),
        "Max Price": round(data["Close"].max(), 2),
        "Min Price": round(data["Close"].min(), 2),
        "Volatility (Std Dev)": round(data["Close"].std(), 2)
    }
    st.table(pd.DataFrame(summary, index=[0]))

    # ---------- Comparison ----------
    if compare and other_index_symbol:
        st.subheader(f"🌐 Comparison: {index_name} vs {other_index_name}")
        comp = _fetch_prices(other_index_symbol, start_date, end_date)
        if comp is None or comp.empty:
            st.warning("Comparison index data not available for the chosen period.")
        else:
            merged = pd.merge(
                data[["Date", "Close"]].rename(columns={"Close": "Close_main"}),
                comp[["Date", "Close"]].rename(columns={"Close": "Close_compare"}),
                on="Date"
            ).dropna()
            st.line_chart(merged.set_index("Date")[["Close_main", "Close_compare"]])
            corr = merged["Close_main"].corr(merged["Close_compare"])
            st.metric("Correlation (Price Movements)", f"{corr:.2f}")

    # ---------- Macroeconomic indicators ----------
    st.subheader("🌍 Key Macroeconomic Indicators")

    macro_symbols = {
        "India": {"Repo Rate (RBI)": "INREPO=ECI", "Inflation YoY": "INCPIMOM"},
        "United States": {"GDP (Quarterly)": "USGDP", "CPI (Inflation)": "CPIAUCSL"},
        "United Kingdom": {"GDP (Quarterly)": "UKRGDPNP", "CPI (Inflation)": "GBRCPIALLMINMEI"},
        "Japan": {"GDP (Quarterly)": "JPNRGDPEXP", "CPI (Inflation)": "JPNCPIALLMINMEI"},
        "Germany": {"GDP (Quarterly)": "DEUGDPNQDSMEI", "CPI (Inflation)": "DEUCPIALLMINMEI"},
    }

    macro_df = pd.DataFrame()
    try:
        # Try Yahoo first
        for name, ticker in macro_symbols.get(country, {}).items():
            m = yf.download(ticker, period="1y", interval="1mo", auto_adjust=True, progress=False)
            if not m.empty:
                col = "Adj Close" if "Adj Close" in m.columns else ("Close" if "Close" in m.columns else None)
                if col:
                    macro_df[name] = m[col].tail(6)

        # If Yahoo failed, use World Bank fallback
        if macro_df.empty:
            wb_country_codes = {
                "India": "IND", "United States": "USA", "United Kingdom": "GBR",
                "Japan": "JPN", "Germany": "DEU"
            }
            code = wb_country_codes.get(country)
            gdp = web.DataReader("NY.GDP.MKTP.CD", "wb", start="2020", end="2025")[code]
            cpi = web.DataReader("FP.CPI.TOTL.ZG", "wb", start="2020", end="2025")[code]
            macro_df = pd.DataFrame({"GDP (USD)": gdp.tail(6), "CPI (%)": cpi.tail(6)})

        if not macro_df.empty:
            st.line_chart(macro_df, height=250)
        else:
            st.warning("Macro data not available for this country and period.")

    except Exception as e:
        st.warning(f"Unable to fetch macroeconomic indicators ({e}).")

    # ---------- AI summary ----------
    st.subheader("🧠 AI-Generated Economic Summary")
    try:
        summarizer = pipeline("text2text-generation", model="google/flan-t5-small")
        prompt = (
            f"Summarize market trends for {country}'s {index_name} from {start_date} to {end_date}. "
            f"Change {pct_change}%. Volatility {summary['Volatility (Std Dev)']}. "
            f"Explain key drivers in 4–6 concise bullet points."
        )
        ai_output = summarizer(prompt, max_new_tokens=220)[0]["generated_text"]
        st.success("✅ Report Generated!")
        st.write(ai_output)
    except Exception as e:
        st.warning(f"AI summarization skipped: {e}")
        ai_output = "AI summary unavailable."

    # ---------- Download ----------
    report_text = f"""
📈 Economic Report for {country}
Index: {index_name} ({index_symbol})
Period: {start_date} → {end_date}

Market:
- Latest Price: {latest_price}
- % Change: {pct_change}%
- Volatility: {summary['Volatility (Std Dev)']}

AI Summary:
{ai_output}
"""
    st.download_button("💾 Download Report", data=report_text, file_name=f"{country}_{index_name}_report.txt")
