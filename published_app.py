import streamlit as st
import json
import pandas as pd
import re
import io
from pathlib import Path

# ────────────────────────────────────────────────────────────
#   Page Config & Custom CSS (mobile-first, professional)
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Biotech Screening Results",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",  # better for mobile
)

st.markdown("""
<style>
/* ── Global typography ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
h1 { font-size: 1.6rem !important; font-weight: 700 !important; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1a365d !important; }

/* ── Compact metrics on mobile ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetricLabel"] { font-size: 0.72rem !important; color: #64748b !important; }
[data-testid="stMetricValue"] { font-size: 1.1rem !important; font-weight: 700 !important; }

/* ── Tables ── */
.stDataFrame { font-size: 0.82rem !important; }
table { width: 100% !important; border-collapse: collapse !important; }
th {
    background: #1a365d !important; color: white !important;
    padding: 8px 10px !important; font-size: 0.78rem !important;
    font-weight: 600 !important; text-align: left !important;
    position: sticky; top: 0; z-index: 1;
}
td {
    padding: 7px 10px !important; font-size: 0.8rem !important;
    border-bottom: 1px solid #e2e8f0 !important; vertical-align: top !important;
}
tr:nth-child(even) { background: #f8fafc !important; }
tr:hover { background: #edf2f7 !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 14px !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
}

/* ── Section dividers ── */
hr { border: none !important; border-top: 2px solid #e2e8f0 !important; margin: 18px 0 !important; }

/* ── Mobile adjustments ── */
@media (max-width: 768px) {
    h1 { font-size: 1.3rem !important; }
    h2 { font-size: 1.1rem !important; }
    h3 { font-size: 0.95rem !important; }
    [data-testid="stMetric"] { padding: 8px 10px !important; }
    [data-testid="stMetricLabel"] { font-size: 0.65rem !important; }
    [data-testid="stMetricValue"] { font-size: 0.95rem !important; }
    th { font-size: 0.7rem !important; padding: 5px 6px !important; }
    td { font-size: 0.72rem !important; padding: 5px 6px !important; }
    button[data-baseweb="tab"] { font-size: 0.72rem !important; padding: 6px 8px !important; }
    .block-container { padding: 0.8rem 0.8rem !important; }
}

/* ── Caption / footer ── */
.run-meta {
    font-size: 0.78rem; color: #64748b;
    background: #f1f5f9; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 16px;
    border-left: 4px solid #3182ce;
}
.run-meta b { color: #1a365d; }

/* ── Badge styling ── */
.badge-pass {
    display: inline-block; background: #38a169; color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;
}
.badge-fail {
    display: inline-block; background: #e53e3e; color: white;
    padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;
}

/* ── Section card ── */
.section-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px; margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
#   Helper: parse Markdown tables in LLM output to DataFrames
# ────────────────────────────────────────────────────────────
def _extract_md_tables(md_text):
    """Extract markdown tables from text, return list of DataFrames."""
    tables = []
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "|" in line and i + 1 < len(lines) and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i + 1].strip()):
            header = [c.strip() for c in line.strip("|").split("|")]
            i += 2  # skip separator
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(row)
                i += 1
            if header and rows:
                try:
                    df = pd.DataFrame(rows, columns=header[:len(rows[0])] if len(header) >= len(rows[0]) else header + [""] * (len(rows[0]) - len(header)))
                    tables.append(df)
                except Exception:
                    pass
        else:
            i += 1
    return tables


def _render_md_with_tables(md_text, max_table_rows=50):
    """Render markdown text, converting embedded tables to st.dataframe for better mobile UX."""
    if not md_text:
        st.info("No data available.")
        return

    lines = md_text.split("\n")
    buffer = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect start of a markdown table
        if "|" in stripped and i + 1 < len(lines) and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[i + 1].strip()):
            # Flush text buffer
            if buffer:
                st.markdown("\n".join(buffer))
                buffer = []

            # Parse header
            header = [c.strip() for c in stripped.strip("|").split("|")]
            header = [h.strip("*").strip() for h in header]  # strip bold markers
            i += 2  # skip separator line

            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                row_raw = lines[i].strip().strip("|").split("|")
                row = [c.strip() for c in row_raw]
                rows.append(row)
                i += 1

            if header and rows:
                try:
                    n_cols = len(header)
                    clean_rows = []
                    for r in rows[:max_table_rows]:
                        if len(r) >= n_cols:
                            clean_rows.append(r[:n_cols])
                        else:
                            clean_rows.append(r + [""] * (n_cols - len(r)))
                    df = pd.DataFrame(clean_rows, columns=header)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                except Exception:
                    # Fallback: render as markdown
                    st.markdown(stripped)
            continue
        else:
            buffer.append(line)
            i += 1

    # Flush remaining
    if buffer:
        st.markdown("\n".join(buffer))


def _render_screening_reasons(reasons_text):
    """Parse screening reasons string into a clean structured display."""
    if not reasons_text:
        st.info("No screening reasons available.")
        return

    criteria = reasons_text.split("\n")
    for criterion in criteria:
        criterion = criterion.strip()
        if not criterion:
            continue

        # Determine pass/fail
        is_pass = any(kw in criterion.lower() for kw in [
            "qualifying", "bpiq found", "upside", "received",
            "breakthrough", "orphan", "accelerated", "fast track",
        ])
        is_fail = any(kw in criterion.lower() for kw in [
            "no upcoming catalysts", "no original research", "no qualifying",
            "does not have", "below threshold", "no papers",
        ])

        if "Criterion 1" in criterion or "criterion 1" in criterion:
            icon = "📅"
            label = "Catalysts"
        elif "Criterion 2" in criterion or "criterion 2" in criterion or "Criteria 2" in criterion:
            icon = "📄"
            label = "Academic Papers & Evidence"
        elif "Criterion 3" in criterion or "criterion 3" in criterion:
            icon = "🐁"
            label = "Animal Studies"
        elif "Criterion 4" in criterion or "criterion 4" in criterion:
            icon = "🏥"
            label = "FDA Designation"
        elif "Criterion 5" in criterion or "criterion 5" in criterion:
            icon = "📈"
            label = "Analyst Upside"
        else:
            icon = "•"
            label = ""

        # Extract the text after the colon
        parts = criterion.split(":", 1)
        detail = parts[1].strip() if len(parts) > 1 else criterion

        if is_fail:
            st.markdown(f"{icon} **{label}:** :red[{detail}]" if label else f"{icon} :red[{detail}]")
        elif is_pass:
            st.markdown(f"{icon} **{label}:** :green[{detail}]" if label else f"{icon} :green[{detail}]")
        else:
            st.markdown(f"{icon} **{label}:** {detail}" if label else f"{icon} {detail}")


# ────────────────────────────────────────────────────────────
#   Load Data
# ────────────────────────────────────────────────────────────
DATA_FILE = Path("published_results.json")

if not DATA_FILE.exists():
    st.error("No published results found.")
    st.info("In the main app (app.py), run screening and click **Publish Current Results**.")
    st.stop()

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ────────────────────────────────────────────────────────────
#   Header
# ────────────────────────────────────────────────────────────
st.markdown("# 🧬 Biotech Screening Results")

run_date = data.get("run_date", "Unknown")
trial_window = data.get("trial_window", "N/A")
min_if = data.get("min_impact_factor", "N/A")
n_screened = len(data.get("all_screened", []))
n_qualified = len(data.get("detailed_results", []))

st.markdown(f"""
<div class="run-meta">
    <b>Published:</b> {run_date} &nbsp;|&nbsp;
    <b>Trial Window:</b> {trial_window} &nbsp;|&nbsp;
    <b>Min Impact Factor:</b> &gt; {min_if}<br/>
    <b>Stocks Screened:</b> {n_screened} &nbsp;|&nbsp;
    <b>Qualifying:</b> {n_qualified}
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
#   Main Tabs
# ────────────────────────────────────────────────────────────
tab_summary, tab_details = st.tabs(["📊 Screening Summary", "🔬 Detailed Results"])

# ── Tab 1: Screening Summary ──
with tab_summary:
    st.subheader("All Screened Stocks")

    if data.get("all_screened"):
        summary_rows = []
        for s in data["all_screened"]:
            summary_rows.append({
                "Ticker": s.get("ticker", ""),
                "Company": s.get("company", ""),
                "Exchange": s.get("exchange", ""),
                "Market Cap": s.get("market_cap", ""),
                "Lead Asset": s.get("asset_name", "N/A"),
                "Passed": "Yes" if s.get("meets_criteria") else "No",
            })
        df_summary = pd.DataFrame(summary_rows)

        # Highlight qualifying rows
        st.dataframe(
            df_summary,
            column_config={
                "Passed": st.column_config.TextColumn("Passed?", width="small"),
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Exchange": st.column_config.TextColumn("Exch", width="small"),
                "Market Cap": st.column_config.TextColumn("Mkt Cap", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Quick stats
        pass_count = sum(1 for s in data["all_screened"] if s.get("meets_criteria"))
        fail_count = len(data["all_screened"]) - pass_count
        cols = st.columns(3)
        cols[0].metric("Total Screened", n_screened)
        cols[1].metric("Passed All Criteria", pass_count)
        cols[2].metric("Did Not Pass", fail_count)
    else:
        st.info("No screened stocks available.")

# ── Tab 2: Detailed Results ──
with tab_details:
    if not data.get("detailed_results"):
        st.info("No qualifying stocks in this publish.")
    else:
        st.subheader(f"Qualifying Stocks ({len(data['detailed_results'])})")

        for res_idx, res in enumerate(data["detailed_results"]):
            ticker = res.get("ticker", "N/A")
            company = res.get("company", "N/A")
            asset_name = res.get("asset_name", "N/A")
            passed = res.get("meets_criteria", False)
            badge = '<span class="badge-pass">PASSED</span>' if passed else '<span class="badge-fail">FAILED</span>'

            with st.expander(f"{company} ({ticker}) — {asset_name}", expanded=(res_idx == 0)):
                st.markdown(f"{badge}", unsafe_allow_html=True)

                # ── Inner tabs matching app.py structure ──
                inner_tabs = st.tabs([
                    "📋 Asset",
                    "🏢 Company & Stock",
                    "👥 Shareholders",
                    "📝 Screening Criteria",
                ])

                # ─────────────────────────────────────
                #  Inner Tab 0: Asset
                # ─────────────────────────────────────
                with inner_tabs[0]:
                    st.markdown(f"### {asset_name}")
                    asset_md = res.get("asset_details_md", "")
                    if asset_md:
                        _render_md_with_tables(asset_md)
                    else:
                        st.info("No asset details saved.")

                # ─────────────────────────────────────
                #  Inner Tab 1: Company & Stock
                # ─────────────────────────────────────
                with inner_tabs[1]:
                    st.markdown(f"### {company} ({ticker})")

                    # ── BPIQ Pipeline Table ──
                    bpiq_pipeline = res.get("bpiq_pipeline", [])
                    if bpiq_pipeline:
                        st.markdown(f"**Pipeline Assets** ({len(bpiq_pipeline)} programs from BPIQ)")

                        pipeline_rows = []
                        for prog in bpiq_pipeline:
                            pipeline_rows.append({
                                "Drug & Indication": prog.get("drug_indication", ""),
                                "Stage": prog.get("stage_event", ""),
                                "Catalyst Date": prog.get("catalyst_date", "TBD"),
                            })
                        pipeline_df = pd.DataFrame(pipeline_rows)
                        st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

                        # Expandable notes per drug
                        for prog_idx, prog in enumerate(bpiq_pipeline):
                            label = prog.get("drug_indication", "Unknown Drug")
                            note = prog.get("note", "")
                            source = prog.get("source", "")
                            if note or source:
                                with st.expander(f"Details: {label}", expanded=False):
                                    if note:
                                        st.markdown(f"**Notes:** {note}")
                                    if source:
                                        st.markdown(f"**Source:** [{source}]({source})")
                    else:
                        st.info("No BPIQ pipeline data available.")

                    # ── Grok stock/pipeline analysis ──
                    st.markdown("---")
                    stock_md = res.get("stock_relative_md", "")
                    if stock_md:
                        _render_md_with_tables(stock_md)
                    else:
                        st.info("No pipeline analysis saved.")

                    # ── Analyst Consensus ──
                    ac = res.get("analyst_consensus", {})
                    if ac and ac.get("avg_price_target"):
                        st.markdown("---")
                        st.markdown("### Sell-Side Analyst Consensus Price Targets")

                        cur_p = ac.get("current_price")
                        avg_t = ac.get("avg_price_target")
                        up_pct = ac.get("upside_pct")
                        rating = ac.get("analyst_consensus_rating", "")
                        p_src = ac.get("price_source", "Unknown")
                        p_ref = ac.get("price_ref_date", "N/A")
                        n_a = ac.get("num_analysts")
                        tgt_hi = ac.get("price_target_high")
                        tgt_lo = ac.get("price_target_low")

                        cols_an = st.columns(4)
                        cols_an[0].metric(
                            f"Price ({p_ref})",
                            f"${cur_p:.2f}" if isinstance(cur_p, (int, float)) else str(cur_p or "N/A"),
                            help=f"Source: {p_src}",
                        )
                        cols_an[1].metric(
                            "Avg 12-Mo Target",
                            f"${avg_t:.2f}" if isinstance(avg_t, (int, float)) else str(avg_t or "N/A"),
                        )
                        cols_an[2].metric(
                            "Implied Upside",
                            f"{up_pct:.1f}%" if isinstance(up_pct, (int, float)) else str(up_pct or "N/A"),
                        )
                        cols_an[3].metric("Consensus", rating.upper() if rating else "N/A")

                        # Detail row
                        detail_parts = []
                        if n_a:
                            detail_parts.append(f"Based on **{n_a} analyst(s)**")
                        if tgt_hi and tgt_lo:
                            detail_parts.append(f"Range: **${tgt_lo}** (low) – **${tgt_hi}** (high)")
                        if detail_parts:
                            st.markdown(" | ".join(detail_parts))

                        st.caption(
                            f"Stock price as of {p_ref} ({p_src}). "
                            "Analyst targets from Yahoo Finance consensus."
                        )

                # ─────────────────────────────────────
                #  Inner Tab 2: Shareholders
                # ─────────────────────────────────────
                with inner_tabs[2]:
                    st.markdown(f"### Shareholders & Institutional Investors")
                    shareholders_md = res.get("shareholders_md", "")
                    if shareholders_md:
                        _render_md_with_tables(shareholders_md)
                    else:
                        st.info("No shareholder data saved.")

                # ─────────────────────────────────────
                #  Inner Tab 3: Screening Criteria
                # ─────────────────────────────────────
                with inner_tabs[3]:
                    st.markdown("### Screening Criteria Results")
                    reasons = res.get("reasons", "")
                    if reasons:
                        _render_screening_reasons(reasons)
                    else:
                        st.info("No screening reasons available.")

# ────────────────────────────────────────────────────────────
#   Footer
# ────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Generated by Biotech Stock Screener | Published {run_date} | "
    f"Data sources: BPIQ, Yahoo Finance, xAI Grok, Google Gemini, DeepSeek"
)
