import os
import time
import streamlit as st
import fitz  # PyMuPDF for reading PDF filings
import yfinance as yf
import pandas as pd
from groq import Groq

# Safe import for live web search
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

# --- 1. WEBSITE PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Autonomous Equity Research & Valuation Engine",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Autonomous M&A Due Diligence & Live Valuation Engine")
st.caption("Neuro-Symbolic Architecture: Live Market Multiples (yfinance) + Deterministic 2-Stage DCF & Sensitivity Matrix + 4-Node Multi-Agent RAG (Groq)")

# --- 2. CRASH-PREVENTION HELPER FUNCTIONS ---
def safe_float(val, default=0.0):
    """Safely converts any input to a number so missing fields never crash the app."""
    try:
        if val is None or pd.isna(val):
            return float(default)
        return float(val)
    except Exception:
        return float(default)

# Safely fetch GROQ_API_KEY from either Streamlit Cloud Secrets or system environment
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None

client = Groq(api_key=api_key) if api_key else None

def ask_ai_agent(system_role, user_task, primary_model="llama-3.3-70b-versatile"):
    """Calls the Groq AI brain with automatic model fallback and rate-limit retry."""
    if not client:
        return "⚠️ **Setup Alert:** `GROQ_API_KEY` not found! In Streamlit Cloud, click **Manage app (bottom right) -> Settings -> Secrets** and paste `GROQ_API_KEY = \"your_key_here\"`."
    
    models_to_try = [primary_model, "llama-3.1-8b-instant"]
    for model_choice in models_to_try:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_task}
                    ],
                    temperature=0.1,  # Strict factual mode to prevent hallucinations
                    max_tokens=2048
                )
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    time.sleep(2.5)
                    continue
                break
    return "⚠️ Free-tier API rate limit reached. Please wait 15 seconds and click Run again."

def extract_high_signal_pdf_pages(uploaded_file, max_pages=18):
    """Scans a 300+ page Annual Report and extracts only the top 18 financial & risk pages."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    keywords = [
        "revenue", "ebitda", "balance sheet", "cash flow", "profit",
        "risk", "contingent", "segment", "order book", "debt", "guidance", "margin", "roce", "roe"
    ]
    scored_pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        text_lower = text.lower()
        kw_score = sum(text_lower.count(k) for k in keywords)
        scored_pages.append((kw_score, i + 1, text))
    
    top_pages = sorted(scored_pages, key=lambda x: x[0], reverse=True)[:max_pages]
    top_pages_ordered = sorted(top_pages, key=lambda x: x[1])
    
    compiled_text = ""
    for score, page_no, txt in top_pages_ordered:
        compiled_text += f"\n--- [Source PDF Page {page_no}] ---\n{txt}"
    return compiled_text, len(doc)

@st.cache_data(ttl=1800)
def fetch_bulletproof_financials(ticker_symbol):
    """Pulls live stock data from Yahoo Finance with a 3-layer waterfall fallback for Free Cash Flow."""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info or {}
        
        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
        if price <= 0:
            return {"valid": False}

        currency = info.get("currency", "INR" if (".NS" in ticker_symbol or ".BO" in ticker_symbol) else "USD")
        unit_divisor = 1e7 if currency == "INR" else 1e6
        unit_label = "Cr" if currency == "INR" else "M"

        # Waterfall Layer 1: Direct Free Cash Flow field
        fcf = safe_float(info.get("freeCashflow"), 0.0)
        fcf_source = "Reported Free Cash Flow (Yahoo Finance)"

        # Waterfall Layer 2: Calculate Operating Cash Flow minus CapEx from Cash Flow Statement
        if fcf == 0.0:
            try:
                cf_df = stock.cashflow
                if cf_df is not None and not cf_df.empty:
                    if "Free Cash Flow" in cf_df.index:
                        fcf = safe_float(cf_df.loc["Free Cash Flow"].iloc[0])
                        fcf_source = "Cash Flow Statement (Direct FCF)"
                    elif "Operating Cash Flow" in cf_df.index:
                        ocf = safe_float(cf_df.loc["Operating Cash Flow"].iloc[0])
                        capex = safe_float(cf_df.loc["Capital Expenditure"].iloc[0]) if "Capital Expenditure" in cf_df.index else 0.0
                        fcf = ocf + capex
                        fcf_source = "Calculated (Operating Cash Flow - CapEx)"
            except Exception:
                pass

        # Waterfall Layer 3: Net Income Proxy (80% cash conversion)
        if fcf == 0.0:
            net_inc = safe_float(info.get("netIncomeToCommon"), 0.0)
            if net_inc > 0:
                fcf = net_inc * 0.80
                fcf_source = "Estimated Proxy (80% of Net Income)"
            else:
                fcf = 1000.0 * unit_divisor
                fcf_source = "Default Baseline (Adjust in DCF Box Below)"

        yf_news_items = []
        try:
            for item in (stock.news or [])[:5]:
                content = item.get("content", {})
                title = item.get("title") or content.get("title", "")
                summary = content.get("summary", "")
                if title:
                    yf_news_items.append(f"- **{title}**: {summary}")
        except Exception:
            pass

        raw_de = safe_float(info.get("debtToEquity"), 0.0)
        de_ratio = raw_de / 100.0 if raw_de > 10 else raw_de

        return {
            "valid": True,
            "name": info.get("longName") or info.get("shortName") or ticker_symbol,
            "sector": info.get("sector", "Corporate / Industrial"),
            "industry": info.get("industry", "General"),
            "currency": currency,
            "unit_divisor": unit_divisor,
            "unit_label": unit_label,
            "current_price": price,
            "market_cap": safe_float(info.get("marketCap")),
            "trailing_pe": safe_float(info.get("trailingPE")),
            "forward_pe": safe_float(info.get("forwardPE")),
            "ev_to_ebitda": safe_float(info.get("enterpriseToEbitda")),
            "price_to_book": safe_float(info.get("priceToBook")),
            "roe": safe_float(info.get("returnOnEquity")) * 100,
            "operating_margin": safe_float(info.get("operatingMargins")) * 100,
            "debt_to_equity": de_ratio,
            "revenue_growth": safe_float(info.get("revenueGrowth")) * 100,
            "free_cash_flow": fcf,
            "fcf_source": fcf_source,
            "total_debt": safe_float(info.get("totalDebt")),
            "total_cash": safe_float(info.get("totalCash")),
            "shares_outstanding": safe_float(info.get("sharesOutstanding"), 100e6),
            "yf_news": "\n".join(yf_news_items)
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

# --- 3. SIDEBAR USER CONTROLS ---
with st.sidebar:
    st.header("1. Target Company Setup")
    st.caption("Indian NSE: Add `.NS` (`LT.NS`, `RELIANCE.NS`, `TCS.NS`, `ITC.NS`). US: `NVDA`, `AAPL`, `MSFT`.")
    ticker_input = st.text_input("Stock Ticker Symbol", value="LT.NS").upper().strip()
    
    fin_data = fetch_bulletproof_financials(ticker_input)
    default_name = fin_data.get("name", "Larsen & Toubro") if fin_data.get("valid") else ticker_input
    company_name = st.text_input("Company Display Name", value=default_name)
    
    uploaded_pdf = st.file_uploader("Upload Annual Report / Investor Deck (PDF - Optional)", type=["pdf"])
    
    st.markdown("---")
    st.header("2. Interactive DCF Assumptions")
    growth_rate = st.slider("5-Yr Cash Flow Growth (CAGR %)", 1.0, 35.0, 14.0, 0.5) / 100.0
    wacc = st.slider("Discount Rate / WACC (%)", 7.0, 20.0, 11.5, 0.5) / 100.0
    terminal_growth = st.slider("Terminal Growth Rate (%)", 1.0, 6.0, 4.0, 0.25) / 100.0

    run_button = st.button("🚀 Run Full Valuation & AI Audit", type="primary", use_container_width=True)

# --- 4. LIVE MULTIPLES & DETERMINISTIC DCF ENGINE ---
if fin_data.get("valid"):
    curr = fin_data["currency"]
    u_div = fin_data["unit_divisor"]
    u_lbl = fin_data["unit_label"]

    st.subheader(f"📊 Live Exchange Multiples: {fin_data['name']} ({ticker_input})")
    st.caption(f"Sector: **{fin_data['sector']}** | Industry: **{fin_data['industry']}** | Cash Flow Source: *{fin_data['fcf_source']}*")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Live Share Price", f"{curr} {fin_data['current_price']:,.2f}")
    m2.metric("Trailing P/E", f"{fin_data['trailing_pe']:.1f}x" if fin_data['trailing_pe'] > 0 else "N/A")
    m3.metric("EV / EBITDA", f"{fin_data['ev_to_ebitda']:.1f}x" if fin_data['ev_to_ebitda'] > 0 else "N/A")
    m4.metric("ROE (%)", f"{fin_data['roe']:.1f}%")
    m5.metric("Operating Margin", f"{fin_data['operating_margin']:.1f}%")
    m6.metric("Debt / Equity", f"{fin_data['debt_to_equity']:.2f}x")

    with st.expander("🧮 Interactive 2-Stage DCF Valuation & WACC Sensitivity Matrix", expanded=True):
        col_left, col_right = st.columns([1, 2])
        
        default_fcf_u = round(fin_data["free_cash_flow"] / u_div, 2)
        if default_fcf_u <= 0:
            default_fcf_u = round((fin_data["market_cap"] * 0.035) / u_div, 2) or 1000.0

        default_shares_u = round(fin_data["shares_outstanding"] / u_div, 4)
        default_net_debt_u = round((fin_data["total_debt"] - fin_data["total_cash"]) / u_div, 2)

        with col_left:
            st.markdown(f"**Editable Model Inputs (in {curr} {u_lbl}):**")
            base_fcf = st.number_input(f"Base Free Cash Flow ({curr} {u_lbl})", value=float(default_fcf_u), step=50.0)
            net_debt = st.number_input(f"Net Debt: Total Debt - Cash ({curr} {u_lbl})", value=float(default_net_debt_u), step=50.0)
            shares_u = st.number_input(f"Shares Outstanding (in {u_lbl})", value=float(default_shares_u), format="%.4f")

        def compute_dcf_price(fcf_val, g_rate, wacc_rate, tg_rate, n_debt, sh_count):
            if wacc_rate <= tg_rate or sh_count <= 0:
                return 0.0, [], []
            proj, disc = [], []
            for yr in range(1, 6):
                f_yr = fcf_val * ((1 + g_rate) ** yr)
                d_yr = f_yr / ((1 + wacc_rate) ** yr)
                proj.append(round(f_yr, 2))
                disc.append(round(d_yr, 2))
            term_fcf = proj[-1] * (1 + tg_rate)
            term_val = term_fcf / (wacc_rate - tg_rate)
            pv_term = term_val / ((1 + wacc_rate) ** 5)
            ev = sum(disc) + pv_term
            eq_val = ev - n_debt
            return max(0.0, eq_val / sh_count), proj, disc

        intrinsic_val, proj_list, disc_list = compute_dcf_price(
            base_fcf, growth_rate, wacc, terminal_growth, net_debt, shares_u
        )
        upside_pct = ((intrinsic_val - fin_data["current_price"]) / fin_data["current_price"]) * 100

        with col_right:
            k1, k2, k3 = st.columns(3)
            k1.metric("Implied DCF Value / Share", f"{curr} {intrinsic_val:,.2f}")
            k2.metric("Current Market Price", f"{curr} {fin_data['current_price']:,.2f}")
            k3.metric("Margin of Safety / Upside", f"{upside_pct:+.1f}%")

            st.markdown("**🎯 2D Valuation Sensitivity Matrix (Implied Share Price across WACC vs. Growth)**")
            wacc_steps = [wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01]
            g_steps = [growth_rate - 0.02, growth_rate, growth_rate + 0.02]
            sens_rows = {}
            for g_s in g_steps:
                row_label = f"Growth {g_s*100:.1f}%"
                sens_rows[row_label] = [
                    f"{curr} {compute_dcf_price(base_fcf, g_s, w_s, terminal_growth, net_debt, shares_u)[0]:,.0f}"
                    for w_s in wacc_steps
                ]
            sens_df = pd.DataFrame(sens_rows, index=[f"WACC {w*100:.1f}%" for w in wacc_steps]).T
            st.dataframe(sens_df, use_container_width=True)
else:
    st.warning(f"Could not load live exchange data for `{ticker_input}`. Check ticker spelling (e.g., `LT.NS` for NSE India or `NVDA` for US). You can still upload a PDF and run the AI audit.")
    intrinsic_val, upside_pct = 0.0, 0.0

st.markdown("---")

# --- 5. 4-NODE MULTI-AGENT EXECUTION ---
# --- 5. 4-NODE MULTI-AGENT EXECUTION ---
if run_button:
    with st.status("Running 4-Node Autonomous Due Diligence Pipeline...", expanded=True) as status:
        if uploaded_pdf:
            st.write("📄 Scanning uploaded PDF for high-density financial & risk disclosures...")
            pdf_text, total_pages = extract_high_signal_pdf_pages(uploaded_pdf, max_pages=18)
            st.write(f"✅ Extracted top 18 financial pages out of {total_pages} total pages.")
        else:
            st.write("ℹ️ No PDF uploaded — running autonomous agents on live exchange multiples & news feed.")
            pdf_text = "No external PDF provided. Rely on live exchange financials and web news."

        quant_context = f"""
        Company: {company_name} ({ticker_input})
        Live Share Price: {fin_data.get('current_price', 'N/A')} {fin_data.get('currency', '')}
        DCF Intrinsic Value Estimate: {intrinsic_val:.2f} (Implied Upside / Margin of Safety: {upside_pct:+.1f}%)
        DCF Assumptions: 5Y FCF CAGR = {growth_rate*100:.1f}%, WACC = {wacc*100:.1f}%, Terminal Growth = {terminal_growth*100:.1f}%
        Trailing P/E: {fin_data.get('trailing_pe', 0):.1f}x | Forward P/E: {fin_data.get('forward_pe', 0):.1f}x
        EV/EBITDA: {fin_data.get('ev_to_ebitda', 0):.1f}x | Price/Book: {fin_data.get('price_to_book', 0):.1f}x
        ROE: {fin_data.get('roe', 0):.1f}% | Operating Margin: {fin_data.get('operating_margin', 0):.1f}%
        Debt/Equity: {fin_data.get('debt_to_equity', 0):.2f}x | YoY Revenue Growth: {fin_data.get('revenue_growth', 0):.1f}%
        """

        st.write("🧮 Node 1 (Quant & Filing Auditor): Auditing valuation multiples, ROE drivers, and filing numbers...")
        fin_report = ask_ai_agent(
            system_role="You are a CFA Charterholder and Quantitative Equity Analyst. Audit the company's valuation multiples, DCF implied upside, DuPont ROE drivers, and PDF financial statements. Cite specific figures.",
            user_task=f"QUANT SNAPSHOT:\n{quant_context}\n\nHIGH-SIGNAL PDF PAGES:\n{pdf_text[:14000]}",
            primary_model="llama-3.1-8b-instant"
        )
        time.sleep(3)  # Pause to respect Groq free tier rate limits

        st.write("🔍 Node 2 (Forensic Risk Detective): Auditing contingent liabilities, margin headwinds, and governance...")
        risk_report = ask_ai_agent(
            system_role="You are a Forensic Accounting & Corporate Risk Investigator. Identify the top 5 structural, margin, working-capital, regulatory, or governance risks.",
            user_task=f"QUANT SNAPSHOT:\n{quant_context}\n\nHIGH-SIGNAL PDF PAGES:\n{pdf_text[:14000]}",
            primary_model="llama-3.1-8b-instant"
        )
        time.sleep(3)  # Pause to respect Groq free tier rate limits

        st.write("🌐 Node 3 (Market Intelligence Scout): Fetching live news catalysts & competitor actions...")
        market_news = ""
        if DDGS is not None:
            try:
                with DDGS() as ddgs:
                    ddg_res = list(ddgs.text(f"{company_name} financial results order book guidance news", max_results=5))
                    if ddg_res:
                        market_news = "\n".join([f"- **{r.get('title', '')}**: {r.get('body', '')}" for r in ddg_res])
            except Exception:
                pass
        if not market_news:
            market_news = fin_data.get("yf_news") or "Live search temporarily rate-limited; relying on exchange fundamentals and PDF disclosures."

        st.write("📝 Node 4 (IC Managing Director): Synthesizing institutional Investment Committee Memorandum...")
        final_memo = ask_ai_agent(
            system_role="You are a Managing Director on an Institutional Investment Committee. Synthesize the quantitative audit, DCF sensitivity, forensic risk report, and live news into an executive-ready Investment Memorandum.",
            user_task=f"""
            Compile an Institutional Equity Research & M&A Due Diligence Memo for {company_name} ({ticker_input}).

            [1. QUANT & DCF SNAPSHOT]:
            {quant_context}

            [2. QUANTITATIVE & FILING AUDIT]:
            {fin_report}

            [3. FORENSIC RISK REPORT]:
            {risk_report}

            [4. LIVE NEWS & MARKET SIGNALS]:
            {market_news}

            Structure the report with clean Markdown headings:
            ## 1. Executive Thesis & Verdict (Buy / Hold / Sell + Conviction Score)
            ## 2. Valuation Breakdown (Multiples vs. 2-Stage DCF & Margin of Safety)
            ## 3. Financial Quality & DuPont ROE / Margin Drivers
            ## 4. Key Risk Matrix & Bear-Case Scenarios
            ## 5. 5 Forensic Due Diligence Questions for Management
            """,
            primary_model="llama-3.3-70b-versatile"
        )
        status.update(label="Multi-Agent Due Diligence Complete!", state="complete", expanded=False)

    # --- 6. RENDER OUTPUT TABS & DOWNLOAD BUTTON ---
    if not final_memo.startswith("⚠️"):
        st.download_button(
            label="📥 Download Investment Committee Memo (.md)",
            data=final_memo,
            file_name=f"{ticker_input.replace('.','_')}_IC_Due_Diligence_Memo.md",
            mime="text/markdown"
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Final IC Memorandum",
        "🧮 Node 1: Quant & Filing Audit",
        "⚠️ Node 2: Forensic Risk Audit",
        "🌐 Node 3: Live Market Signals"
    ])
    with tab1:
        st.markdown(final_memo)
    with tab2:
        st.markdown(fin_report)
    with tab3:
        st.markdown(risk_report)
    with tab4:
        st.markdown(market_news)

    # --- 6. RENDER OUTPUT TABS & DOWNLOAD BUTTON ---
    st.download_button(
        label="📥 Download Investment Committee Memo (.md)",
        data=final_memo,
        file_name=f"{ticker_input}_IC_Due_Diligence_Memo.md",
        mime="text/markdown"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Final IC Memorandum",
        "🧮 Node 1: Quant & Filing Audit",
        "⚠️ Node 2: Forensic Risk Audit",
        "🌐 Node 3: Live Market Signals"
    ])
    with tab1:
        st.markdown(final_memo)
    with tab2:
        st.markdown(fin_report)
    with tab3:
        st.markdown(risk_report)
    with tab4:
        st.markdown(market_news)
