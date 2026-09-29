import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai

st.set_page_config(page_title="Sektor- & Marknadsanalys AI", layout="wide")

st.title("📊 Sektor- & Marknadsanalys AI")

# --- SIDEBAR & API-HANTERING ---
st.sidebar.header("Inställningar")

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("🔑 API-nyckel laddad automatiskt!")
else:
    api_key = st.sidebar.text_input("Google Gemini API-nyckel", type="password", help="Klistra in din API-nyckel från Google AI Studio")

if st.sidebar.button("Uppdatera marknadsdata"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Pris- och nyhetsdata cacheas i appen.")

# --- WILDER'S RSI & SIGNALFUNKTIONER ---
def compute_rsi_wilders(series, period=14):
    """RSI beräknad med Wilder's Smoothing (Exakt samma som TradingView)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_signal(rsi, dist_sma200, ret_1m):
    if pd.isna(rsi) or pd.isna(dist_sma200):
        return "⚪ Neutral"
    if rsi > 68:
        return "🟡 TA VINST"
    elif dist_sma200 < 0:
        return "🔴 SÄLJLÄGE"
    elif dist_sma200 > 0 and (rsi < 52 or (ret_1m is not None and ret_1m < 0)):
        return "🟢 KÖPLÄGE"
    return "⚪ Neutral"

# --- HÄMTA MAKRO- OCH RÅVARUDATA ---
@st.cache_data(ttl=3600)
def fetch_macro_data():
    macro_tickers = {
        "USD/SEK": "USDSEK=X",
        "Olja (Brent)": "BZ=F",
        "Guld": "GC=F",
        "US 10Y Ränta": "^TNX"
    }
    df_macro = yf.download(list(macro_tickers.values()), period="5d", interval="1d")["Close"]
    results = {}
    for name, ticker in macro_tickers.items():
        if ticker in df_macro.columns:
            series = df_macro[ticker].dropna()
            if len(series) >= 2:
                last_val = series.iloc[-1]
                prev_val = series.iloc[-2]
                change_pct = ((last_val - prev_val) / prev_val) * 100
                results[name] = f"{round(last_val, 2)} ({'+' if change_pct>0 else ''}{round(change_pct, 2)}%)"
            else:
                results[name] = "N/A"
    return results

# --- HÄMTA NYHETER LIVE ---
@st.cache_data(ttl=1800)
def fetch_recent_news(tickers):
    news_items = []
    for t in tickers[:12]:  # Hämta nyheter för ett urval av nyckelaktier/sektorer
        try:
            tk = yf.Ticker(t)
            news_list = tk.news
            if news_list:
                for item in news_list[:2]:
                    title = item.get("title")
                    if not title and "content" in item:
                        title = item["content"].get("title")
                    publisher = item.get("publisher", "Marknadsnyhet")
                    if title:
                        news_items.append(f"• [{t}] {title} ({publisher})")
        except:
            continue
    return news_items[:12]

# --- HÄMTA PRISDATA & SIGNALER ---
@st.cache_data(ttl=3600)
def fetch_market_data(tickers, benchmark_ticker):
    all_tickers = list(set(tickers + [benchmark_ticker]))
    df = yf.download(all_tickers, period="1y", interval="1d")["Close"]
    
    bench = df[benchmark_ticker]
    bench_ret_1m = ((bench.iloc[-1] / bench.iloc[-21]) - 1) * 100 if len(bench) >= 21 else 0
    bench_ret_3m = ((bench.iloc[-1] / bench.iloc[-63]) - 1) * 100 if len(bench) >= 63 else 0

    results = []
    for t in tickers:
        if t not in df.columns:
            continue
        series = df[t].dropna()
        if len(series) < 200:
            continue

        last_price = series.iloc[-1]
        sma200 = series.rolling(200).mean().iloc[-1]
        dist_sma200 = ((last_price - sma200) / sma200) * 100

        rsi_series = compute_rsi_wilders(series)
        last_rsi = rsi_series.iloc[-1]

        ret_1m = ((last_price / series.iloc[-21]) - 1) * 100 if len(series) >= 21 else np.nan
        ret_3m = ((last_price / series.iloc[-63]) - 1) * 100 if len(series) >= 63 else np.nan

        rs_1m = ret_1m - bench_ret_1m if not pd.isna(ret_1m) else np.nan
        rs_3m = ret_3m - bench_ret_3m if not pd.isna(ret_3m) else np.nan

        signal = get_signal(last_rsi, dist_sma200, ret_1m)

        results.append({
            "Ticker": t,
            "Signal": signal,
            "Senaste Pris": round(last_price, 2),
            "RSI (14)": round(last_rsi, 1),
            "SMA200 Avstånd (%)": round(dist_sma200, 1),
            "1M Avkastning (%)": round(ret_1m, 1),
            "3M Avkastning (%)": round(ret_3m, 1),
            "RS 1M (%)": round(rs_1m, 1),
            "RS 3M (%)": round(rs_3m, 1)
        })
    return pd.DataFrame(results)

@st.cache_data(ttl=3600)
def fetch_fundamentals(tickers):
    fund_data = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            fund_data[t] = {
                "Trailing P/E": round(info.get("trailingPE"), 1) if info.get("trailingPE") else "N/A",
                "Forward P/E": round(info.get("forwardPE"), 1) if info.get("forwardPE") else "N/A",
                "Vinstmarginal (%)": round(info.get("profitMargins", 0) * 100, 1) if info.get("profitMargins") else "N/A"
            }
        except:
            fund_data[t] = {"Trailing P/E": "N/A", "Forward P/E": "N/A", "Vinstmarginal (%)": "N/A"}
    return fund_data

# --- MAKROBAR LÄNGST UPP ---
macro_info = fetch_macro_data()
if macro_info:
    m_cols = st.columns(len(macro_info))
    for idx, (k, v) in enumerate(macro_info.items()):
        m_cols[idx].metric(label=k, value=v)

# --- FLIKAR ---
tab1, tab2 = st.tabs(["🇺🇸 US Sektorer (SPY)", "🇸🇪 Svenska Aktier (^OMX)"])

# --- FLIK 1: AMERIKANSKA SEKTORER ---
with tab1:
    st.subheader("Amerikanska Sektor-ETF:er med Signaler (Jämfört mot SPY)")
    us_tickers = ["XLK", "XLE", "XLF", "XLV", "XLP", "XLY", "XLI", "XLB", "XLU", "XLRE", "XLC"]
    us_names = {
        "XLK": "Teknik", "XLE": "Energi", "XLF": "Finans", "XLV": "Hälsovård",
        "XLP": "Dagligvaror", "XLY": "Sällanköp", "XLI": "Industri", "XLB": "Råvaror",
        "XLU": "Kraftförsörjning", "XLRE": "Fastigheter", "XLC": "Kommunikation"
    }
    
    df_us = fetch_market_data(us_tickers, "SPY")
    if not df_us.empty:
        df_us["Sektor"] = df_us["Ticker"].map(us_names)
        cols_us = ["Sektor", "Ticker", "Signal", "Senaste Pris ($)", "RSI (14)", "SMA200 Avstånd (%)", "1M Avkastning (%)", "RS 1M (%)", "RS 3M (%)"]
        df_us = df_us.rename(columns={"Senaste Pris": "Senaste Pris ($)"})
        st.dataframe(df_us[cols_us], use_container_width=True)

# --- FLIK 2: SVENSKA AKTIER PER SEKTOR ---
with tab2:
    st.subheader("Svenska Aktier per Sektor med Signaler (Jämfört mot OMXS30)")
    se_sectors = {
        "Bank & Finans": ["SEB-A.ST", "SWED-A.ST", "SHB-A.ST", "NDA-SE.ST"],
        "Industri & Verkstad": ["VOLV-B.ST", "ATCO-A.ST", "SAND.ST", "ALFA.ST", "SKF-B.ST"],
        "Investmentbolag": ["INVE-B.ST", "INDU-C.ST", "LUND-B.ST", "KINV-B.ST"],
        "Fastigheter": ["CAST.ST", "BALD-B.ST", "SAGA-B.ST", "FABG.ST"],
        "Tech, Gaming & Tillväxt": ["EVO.ST", "SINCH.ST", "SPOT", "EMBRAC-B.ST"],
        "Hälsovård & Medtech": ["AZN.ST", "GETI-B.ST", "EKTAB-B.ST"],
        "Konsument & Handel": ["HM-B.ST", "DOM.ST", "VOLCAR-B.ST"],
        "Dagligvaror": ["ESSITY-B.ST", "AAK.ST"],
        "Telekom & Kommunikation": ["TELIA.ST", "ERIC-B.ST"],
        "Råvaror & Skog": ["SCA-B.ST", "BOL.ST"]
    }
    
    all_se_tickers = [t for list_t in se_sectors.values() for t in list_t]
    df_se = fetch_market_data(all_se_tickers, "^OMX")
    
    if not df_se.empty:
        ticker_to_sector = {}
        for sec, t_list in se_sectors.items():
            for t in t_list:
                ticker_to_sector[t] = sec
        
        df_se["Sektor"] = df_se["Ticker"].map(ticker_to_sector)
        
        funds = fetch_fundamentals(all_se_tickers)
        df_se["Trailing P/E"] = df_se["Ticker"].apply(lambda t: funds.get(t, {}).get("Trailing P/E", "N/A"))
        df_se["Forward P/E"] = df_se["Ticker"].apply(lambda t: funds.get(t, {}).get("Forward P/E", "N/A"))
        df_se["Vinstmarginal (%)"] = df_se["Ticker"].apply(lambda t: funds.get(t, {}).get("Vinstmarginal (%)", "N/A"))

        df_se = df_se.rename(columns={"Senaste Pris": "Senaste Pris (SEK)"})
        cols_se = ["Sektor", "Ticker", "Signal", "Senaste Pris (SEK)", "RSI (14)", "SMA200 Avstånd (%)", "1M Avkastning (%)", "RS 3M (%)", "Trailing P/E", "Forward P/E", "Vinstmarginal (%)"]
        st.dataframe(df_se[cols_se], use_container_width=True)

# --- AI-RAPPORT ---
st.markdown("---")
st.header("🤖 AI-marknadsanalys & Nyhetsperspektiv")

if st.button("Generera AI-analys", type="primary"):
    if not api_key or not api_key.strip():
        st.error("Mata in din Gemini API-nyckel i sidomenyn till vänster.")
    else:
        with st.spinner("AI-analytikern hämtar nyheter, analyserar makro, tekniska signaler och fundamenta..."):
            try:
                genai.configure(api_key=api_key.strip())
                model = genai.GenerativeModel("gemini-3.8-flash")
                
                # Hämtar färska nyheter för analysen
                sample_tickers = ["VOLV-B.ST", "INVE-B.ST", "CAST.ST", "EVO.ST", "XLK", "XLE", "XLF"]
                live_news = fetch_recent_news(sample_tickers)
                news_text = "\n".join(live_news) if live_news else "Inga färska nyhetsrubriker hittades."

                prompt = f"""
                Du är en erfaren chefsanalytiker och makrostrateg på en ledande investeringsbank.
                Gör en djupgående marknads- och nyhetsanalys utifrån följande data:

                SENASTE NYHETSRUBRIKER & HÄNDELSER:
                {news_text}

                MAKROINDIKATORER JUST NU:
                {macro_info}

                AMERIKANSKA SEKTORER (Jämfört mot SPY):
                {df_us.to_string() if 'df_us' in locals() and not df_us.empty else 'Ingen data'}

                SVENSKA AKTIER, SIGNALER OCH FUNDAMENTA (Jämfört mot OMXS30):
                {df_se.to_string() if 'df_se' in locals() and not df_se.empty else 'Ingen data'}

                Skriv en strukturerad och professionell rapport på SVENSKA med följande rubriker:

                1. **📰 Nyhetsläget & Viktiga Katalysatorer**
                   - Sammanfatta de viktigaste nyhetsrubrikerna och marknadshändelserna just nu.
                   - Förklara hur nyhetsflödet påverkar investerarnas humör.

                2. **🌍 Makroläge & Omvärldstrender**
                   - Analysera hur räntor, valuta (USD/SEK), olja och guld påverkar klimatet.

                3. **💡 Vem gynnas & Vem drabbas? (Sektor & Fundamenta)**
                   - Vilka branscher har medvind utifrån nyheter, makro och värderingar (P/E)?
                   - Vilka sektorer har motvind?

                4. **🟢 Bästa Köplägena (Teknik, Nyheter & Fundamenta)**
                   - Lyft fram de starkaste aktierna/sektorerna med köpsignal (🟢).

                5. **⚠️ Överköpta eller Riskfyllda Aktier (🟡 / 🔴)**
                   - Vilka aktier är tekniskt överköpta eller under sin SMA200 där vinsthemtagning är klokt.

                6. **🎯 Konkret Handelsplan**
                   - Tydliga råd för hur en investerare bör positionera sig den kommande veckan.
                """
                
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Fel vid AI-analys: {e}")