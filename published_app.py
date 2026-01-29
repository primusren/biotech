import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Biotech Screening Results", layout="wide")

st.title("Published Biotech Screening Results")

DATA_FILE = Path("published_results.json")

if not DATA_FILE.exists():
    st.error("No published results found.")
    st.info("In the main app (app.py), run screening and click 'Publish Current Results'.")
    st.stop()

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

st.caption(f"Published on: {data.get('run_date', 'Unknown')}")
st.caption(f"Trial window: {data.get('trial_window', 'N/A')}")
st.caption(f"Min Impact Factor filter: > {data.get('min_impact_factor', 'N/A')}")

tab_all, tab_details = st.tabs(["All Screened Stocks", "Detailed Results"])

with tab_all:
    st.subheader("All Screened Stocks")
    if data.get("all_screened"):
        st.dataframe(
            data["all_screened"],
            column_config={
                "company": "Company",
                "ticker": "Ticker",
                "exchange": "Exchange",
                "market_cap": "Market Cap",
                "meets_criteria": st.column_config.CheckboxColumn("Passed?"),
                "reasons_summary": st.column_config.TextColumn("Reasons (Summary)", width="large"),
                "asset_name": "Lead Asset"
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No screened stocks.")

with tab_details:
    st.subheader("Qualifying Stocks – Full Details")
    if data.get("detailed_results"):
        for res in data["detailed_results"]:
            ticker = res["ticker"]
            company = res["company"]
            asset_name = res.get("asset_name", "N/A")

            with st.expander(f"{company} ({ticker}) – {asset_name}"):
                st.markdown(f"**Passed criteria?** {'Yes' if res['meets_criteria'] else 'No'}")
                st.markdown("**Screening Reasons:**")
                st.markdown(res["reasons"])

                st.markdown("---")
                st.subheader("Asset Characteristics")
                st.markdown(res.get("asset_details_md", "No details saved."))

                st.markdown("---")
                st.subheader("Shareholders & Investors")
                st.markdown(res.get("shareholders_md", "No shareholder data saved."))

                st.markdown("---")
                st.subheader("Stock Relative Characteristics")
                st.markdown(res.get("stock_relative_md", "No relative data saved."))
    else:
        st.info("No qualifying stocks in this publish.")
