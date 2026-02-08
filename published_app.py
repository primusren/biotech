import streamlit as st
import requests
import json
import time
import uuid
import hashlib
import io
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import re
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Initialize session state for BPIQ cache
if 'bpiq_drugs' not in st.session_state:
    st.session_state.bpiq_drugs = None
    st.session_state.bpiq_last_fetch = None

# Load environment variables
load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
BPIQ_API_KEY = os.getenv("BPIQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Only show sidebar inputs for keys NOT found in .env
if not XAI_API_KEY:
    XAI_API_KEY = st.sidebar.text_input("Enter your xAI API Key", type="password")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.sidebar.text_input("Enter your Google Gemini API Key (free — get at aistudio.google.com)", type="password")

# File-based persistence
BPIQ_CACHE_FILE = Path("bpiq_drugs.json")
CACHE_FILE = Path("grok_cache.json")
VERIFY_CACHE_FILE = Path("verification_cache.json")

# Load Grok cache on startup
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            grok_cache = json.load(f)
    except Exception:
        grok_cache = {}
else:
    grok_cache = {}

# Load Gemini/DeepSeek verification cache on startup
if VERIFY_CACHE_FILE.exists():
    try:
        with open(VERIFY_CACHE_FILE, "r", encoding="utf-8") as f:
            verify_cache = json.load(f)
    except Exception:
        verify_cache = {}
else:
    verify_cache = {}


def _save_verify_cache():
    """Persist verification cache to disk."""
    try:
        with open(VERIFY_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(verify_cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def load_bpiq_from_disk():
    if BPIQ_CACHE_FILE.exists():
        try:
            with open(BPIQ_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                st.session_state.bpiq_drugs = data.get("drugs", [])
                st.session_state.bpiq_last_fetch = data.get("last_fetch", None)
                st.sidebar.info(f"Loaded {len(st.session_state.bpiq_drugs)} drug records from disk cache (last fetch: {st.session_state.bpiq_last_fetch})")
                return True
        except Exception as e:
            st.sidebar.warning(f"Failed to load BPIQ cache from disk: {e}")
    return False

def save_bpiq_to_disk():
    if st.session_state.bpiq_drugs is not None:
        cache_data = {
            "drugs": st.session_state.bpiq_drugs,
            "last_fetch": st.session_state.bpiq_last_fetch
        }
        try:
            with open(BPIQ_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            st.sidebar.success(f"Saved {len(st.session_state.bpiq_drugs)} records to disk cache.")
        except Exception as e:
            st.sidebar.error(f"Failed to save BPIQ cache to disk: {e}")

GROK_API_URL = "https://api.x.ai/v1/chat/completions"

def call_grok(prompt, _max_retries=3):
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 16384
    }
    for attempt in range(_max_retries):
        try:
            response = requests.post(GROK_API_URL, headers=headers, json=data, timeout=120)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                wait = 10 * (attempt + 1)
                st.warning(f"Grok rate-limited, waiting {wait}s… (attempt {attempt+1}/{_max_retries})")
                time.sleep(wait)
            else:
                return f"Error: {response.status_code} - {response.text}"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = 5 * (attempt + 1)
            if attempt < _max_retries - 1:
                st.warning(f"Grok connection error, retrying in {wait}s… (attempt {attempt+1}/{_max_retries})")
                time.sleep(wait)
            else:
                return f"Error: Connection failed after {_max_retries} attempts — {e}"
        except Exception as e:
            return f"Error: {e}"
    return "Error: Grok API failed after all retries"

def call_gemini_with_search(prompt):
    """Call Google Gemini API with Google Search Grounding for fact-checking.
    
    Uses the Gemini 2.0 Flash model with dynamic search retrieval,
    which lets Gemini search Google to verify facts and return citations.
    Returns the text response or an error string.
    """
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not set"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{
            "google_search_retrieval": {
                "dynamic_retrieval_config": {
                    "mode": "MODE_DYNAMIC",
                    "dynamic_threshold": 0.3
                }
            }
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192
        }
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return "Error: No candidates in Gemini response"
                parts = candidates[0].get("content", {}).get("parts", [])
                text = " ".join(p.get("text", "") for p in parts)
                return text
            elif resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                return f"Error: Gemini {resp.status_code} - {resp.text[:300]}"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                return "Error: Gemini connection failed after retries"
        except Exception as e:
            return f"Error: {e}"
    return "Error: Gemini API failed after all retries"


def call_deepseek(prompt):
    """Call DeepSeek API (OpenAI-compatible). Very cheap, strong on academic content."""
    if not DEEPSEEK_API_KEY:
        return "Error: DeepSeek API key not set"
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 8192
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=90)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
            else:
                return f"Error: DeepSeek {resp.status_code} - {resp.text[:300]}"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                return "Error: DeepSeek connection failed after retries"
        except Exception as e:
            return f"Error: {e}"
    return "Error: DeepSeek API failed after all retries"


# ────────────────────────────────────────────────
#   Paper Verification & Rescue Functions
# ────────────────────────────────────────────────

_PAPER_CHECK_JSON_TEMPLATE = """
Output ONLY a JSON object (no other text):
{{
  "criterion2_pass": true/false,
  "criterion3_pass": true/false,
  "papers": [
    {{
      "title": "Full paper title",
      "journal": "Journal name",
      "year": number,
      "impact_factor": number_or_null,
      "is_original_research": true/false,
      "is_animal_study": true/false,
      "pubmed_url": "url or null",
      "notes": "brief note"
    }}
  ],
  "summary": "Brief explanation"
}}

"criterion2_pass" = true if at least one original research paper (not review) with IF > {min_if} supports the POC.
"criterion3_pass" = true if at least one original animal study paper with IF >= 10 exists for this drug.
"""


def verify_papers_with_gemini(ticker, company, asset_name, grok_reasons, min_if):
    """Use Gemini + Google Search to VERIFY papers Grok cited (catch hallucinations).
    Results are cached to avoid redundant API calls."""
    if not GEMINI_API_KEY:
        return {"action": "skip", "details": "Gemini key not set"}

    # ── Check verification cache ──
    reasons_hash = hashlib.sha256(grok_reasons.encode()).hexdigest()[:12]
    cache_key = f"verify__{ticker}__{asset_name}__{min_if}__{reasons_hash}"
    if cache_key in verify_cache:
        st.info(f"Using cached Gemini verification for {ticker}")
        return verify_cache[cache_key]

    prompt = f"""You are a fact-checker for biotech investment research. VERIFY the academic papers cited below using Google Search (PubMed, Google Scholar, journal websites).

COMPANY: {company} ({ticker})
LEAD ASSET: {asset_name}

--- CLAIMS TO VERIFY ---
{grok_reasons}
--- END ---

For EACH paper: Does it exist? Correct journal? Original research (not review)? Actual Impact Factor? Animal study?
{_PAPER_CHECK_JSON_TEMPLATE.format(min_if=min_if)}
"""
    raw = call_gemini_with_search(prompt)
    result = parse_json_safely(raw, dict)
    if not result or isinstance(result, list):
        return {"action": "skip", "details": f"Unparseable Gemini response: {raw[:300]}"}
    out = {
        "action": "verify",
        "source": "Gemini (Google Search)",
        "criterion2_pass": result.get("criterion2_pass", True),
        "criterion3_pass": result.get("criterion3_pass", True),
        "papers": result.get("papers", []),
        "details": result.get("summary", ""),
    }
    # ── Persist to cache ──
    verify_cache[cache_key] = out
    _save_verify_cache()
    return out


def rescue_papers_search(ticker, company, asset_name, min_if):
    """When Grok says NO qualifying papers, use Gemini + DeepSeek to search independently.
    
    Runs up to 2 LLMs in sequence. If ANY finds qualifying papers, the stock is rescued.
    Results are cached to avoid redundant API calls.
    Returns a dict with rescue results.
    """
    # ── Check verification cache ──
    cache_key = f"rescue__{ticker}__{asset_name}__{min_if}"
    if cache_key in verify_cache:
        st.info(f"Using cached rescue search for {ticker}")
        return verify_cache[cache_key]

    rescue_prompt = f"""You are a biotech research analyst. Another AI was unable to find qualifying academic papers for this company's lead drug candidate. Your job is to INDEPENDENTLY SEARCH for papers that the other AI may have missed.

COMPANY: {company} ({ticker})
LEAD ASSET: {asset_name}

Search thoroughly for:
1. **POC papers**: Original research papers (NOT reviews) supporting the proof-of-concept behind {asset_name}, published in journals with Clarivate Impact Factor > {min_if}. Search PubMed, Google Scholar, and journal websites for the drug name, mechanism of action, and target pathway.
2. **Animal studies**: Original papers (NOT reviews) describing animal/preclinical studies of {asset_name} or closely related compounds, published in journals with Clarivate IF >= 10.

Search using: the drug name "{asset_name}", the company name "{company}", the drug's mechanism of action, the target protein/pathway, and synonyms or development codes.

{_PAPER_CHECK_JSON_TEMPLATE.format(min_if=min_if)}
"""

    results = []

    # ── Gemini with Google Search (primary — can actually search the web) ──
    if GEMINI_API_KEY:
        raw_gemini = call_gemini_with_search(rescue_prompt)
        parsed = parse_json_safely(raw_gemini, dict)
        if parsed and isinstance(parsed, dict):
            results.append({
                "source": "Gemini (Google Search)",
                "criterion2_pass": parsed.get("criterion2_pass", False),
                "criterion3_pass": parsed.get("criterion3_pass", False),
                "papers": parsed.get("papers", []),
                "details": parsed.get("summary", ""),
            })

    # ── DeepSeek (secondary — strong academic training data, very cheap) ──
    if DEEPSEEK_API_KEY:
        raw_ds = call_deepseek(rescue_prompt)
        parsed = parse_json_safely(raw_ds, dict)
        if parsed and isinstance(parsed, dict):
            results.append({
                "source": "DeepSeek",
                "criterion2_pass": parsed.get("criterion2_pass", False),
                "criterion3_pass": parsed.get("criterion3_pass", False),
                "papers": parsed.get("papers", []),
                "details": parsed.get("summary", ""),
            })

    if not results:
        return {
            "rescued": False,
            "details": "No verification LLMs available (set Gemini and/or DeepSeek API keys)",
            "results": [],
        }

    # A stock is rescued if ANY LLM found qualifying papers for BOTH criteria
    any_c2 = any(r["criterion2_pass"] for r in results)
    any_c3 = any(r["criterion3_pass"] for r in results)
    rescued = any_c2 and any_c3

    # Collect all unique papers found across LLMs
    all_papers = []
    seen_titles = set()
    for r in results:
        for p in r.get("papers", []):
            title = p.get("title", "").strip().lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                p["found_by"] = r["source"]
                all_papers.append(p)

    details_parts = []
    for r in results:
        status = "FOUND papers" if (r["criterion2_pass"] or r["criterion3_pass"]) else "No qualifying papers found"
        details_parts.append(f"{r['source']}: {status}. {r.get('details', '')}")

    out = {
        "rescued": rescued,
        "criterion2_rescued": any_c2,
        "criterion3_rescued": any_c3,
        "details": " | ".join(details_parts),
        "papers": all_papers,
        "results": results,
    }
    # ── Persist to cache ──
    verify_cache[cache_key] = out
    _save_verify_cache()
    return out


def verify_catalysts_with_llms(company, ticker, asset_name, grok_catalysts_md, start_date_str, bpiq_catalysts_json):
    """Use Gemini (with web search) and DeepSeek to verify and supplement catalyst dates.

    - Flags any catalyst dates that have already passed relative to start_date_str.
    - Searches for upcoming presentations, conferences, or company announcements
      that Grok may have missed or hallucinated.
    - Returns verified Markdown text to append/replace the catalysts section.
    - Results are cached in verify_cache.
    """
    cache_key = f"catalysts__{ticker}__{asset_name}__{start_date_str}"
    if cache_key in verify_cache:
        return verify_cache[cache_key]

    verify_prompt = f"""You are a biotech analyst fact-checker. Your job is to verify and supplement the near-term catalysts listed below for {company} ({ticker}), lead asset: {asset_name}.

Today's reference date (start of trial window): **{start_date_str}**

BPIQ catalyst data (trusted source):
{bpiq_catalysts_json}

Grok's catalyst output to verify:
---
{grok_catalysts_md[:3000]}
---

Tasks:
1. **Flag past dates**: Identify any catalyst dates that are BEFORE {start_date_str}. Mark them as "(PAST — already occurred)" in your output.
2. **Verify upcoming events**: For each future catalyst, verify if the date and event are accurate by searching company press releases, SEC filings, ClinicalTrials.gov, and biotech news sites.
3. **Search for missing catalysts**: Search for any upcoming presentations at medical conferences (e.g., ASCO, AHA, ASH, AACR, ESMO, JPM Healthcare Conference, etc.), earnings calls, PDUFA dates, advisory committee meetings, or data readouts in the next 6 months that Grok missed.
4. **Search for recent company announcements**: Check the company's investor relations page and recent press releases (last 30 days) for any newly announced events.

Output as structured Markdown:
### Verified Near-Term Catalysts
A table with columns: **Date**, **Event**, **Type**, **Status** (Verified/Unverified/Past), **Source**

### Additional Catalysts Found
Any events you found that were NOT in Grok's output. Same table format.

If you cannot verify a date, mark Status as "Unverified".
Include source links where possible.
"""

    result = {"verified_md": "", "source": ""}

    # ── Try Gemini first (has web search grounding) ──
    if GEMINI_API_KEY:
        gemini_out = call_gemini_with_search(verify_prompt)
        if gemini_out and not gemini_out.startswith("Error:"):
            result["verified_md"] = gemini_out
            result["source"] = "Gemini (Google Search)"
            verify_cache[cache_key] = result
            _save_verify_cache()
            return result

    # ── Fallback to DeepSeek ──
    if DEEPSEEK_API_KEY:
        ds_out = call_deepseek(verify_prompt)
        if ds_out and not ds_out.startswith("Error:"):
            result["verified_md"] = ds_out
            result["source"] = "DeepSeek"
            verify_cache[cache_key] = result
            _save_verify_cache()
            return result

    result["verified_md"] = ""
    result["source"] = "No verification LLMs available"
    verify_cache[cache_key] = result
    _save_verify_cache()
    return result


def parse_json_safely(raw_response, expected_type=list):
    if not raw_response:
        return [] if expected_type == list else {}
    cleaned = re.sub(r'^```json\s*|\s*```$', '', raw_response.strip())
    cleaned = re.sub(r'^.*?(\[|\{)', r'\1', cleaned)
    cleaned = re.sub(r'(\]|\}).*?$', r'\1', cleaned)
    cleaned = cleaned.replace('\n', '')
    attempts = 0
    while attempts < 10:
        try:
            parsed = json.loads(cleaned)
            if expected_type == list and not isinstance(parsed, list):
                parsed = [parsed] if isinstance(parsed, dict) else []
            elif expected_type == dict and isinstance(parsed, list) and len(parsed) == 1:
                parsed = parsed[0]
            return parsed
        except json.JSONDecodeError:
            cleaned = cleaned[:cleaned.rfind(',')] + cleaned[cleaned.rfind(',')+1:]
            attempts += 1
    st.error(f"Invalid JSON from Grok: {raw_response}")
    return [] if expected_type == list else {}

# ────────────────────────────────────────────────
#   BPIQ API – Fetch & Filter Upcoming Catalysts
# ────────────────────────────────────────────────

BPIQ_BASE_URL = "https://api.bpiq.com/api/v1/"
def fetch_all_bpiq_drugs():
    """Fetch all drugs from BPIQ with retries and longer timeout"""
    headers = {"Authorization": f"Token {BPIQ_API_KEY}"} if BPIQ_API_KEY else {}
    if not headers:
        st.error("BPIQ API Key not set. Enter it in the sidebar.")
        return []
    session = requests.Session()
    
    # Automatic retries on failure (Heroku cold starts, network blips)
    retries = Retry(
        total=5,
        backoff_factor=1,           # wait 1s, 2s, 4s, etc.
        status_forcelist=[502, 503, 504],  # retry on these status codes
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    
    drugs = []
    url = f"{BPIQ_BASE_URL}drugs/"
    
    while url:
        try:
            r = session.get(
                url,
                headers=headers,
                timeout=90          # ← increased from 20 to 90 seconds
            )
            r.raise_for_status()
            
            data = r.json()
            page_drugs = data.get("results", [])
            drugs.extend(page_drugs)
            
            url = data.get("next")
            st.info(f"Fetched page with {len(page_drugs)} drugs (total so far: {len(drugs)})")
            
            time.sleep(1)  # polite 1-second delay between pages to avoid rate-limiting
            
        except requests.exceptions.Timeout:
            st.error("BPIQ request timed out after 90 seconds – server is very slow or overloaded.")
            break
        except requests.exceptions.RequestException as e:
            st.error(f"BPIQ fetch failed: {str(e)}")
            break
    
    st.success(f"Successfully fetched {len(drugs)} drug records from BPIQ")
    return drugs

def get_upcoming_catalysts_for_tickers(drugs_data, tickers, start_dt, end_dt):
    """Filter drugs for given tickers with catalysts in date window. Exclude initiation-only; include conference presentations."""
    results = []
    tickers_upper = {t.upper() for t in tickers}

    for drug in drugs_data:
        ticker = drug.get("ticker", "").upper()
        if ticker not in tickers_upper:
            continue

        cat_date_str = drug.get("catalyst_date") or drug.get("catalyst_date_text", "")
        note = (drug.get("note") or "").lower()
        stage_label = drug.get("stage_event", {}).get("label", "") if isinstance(drug.get("stage_event"), dict) else ""
        stage = (stage_label or "").lower()

        # Screen out initiation of trial only
        if "initiation" in note or "start" in note or "enroll start" in stage or "trial initiation" in stage:
            continue

        if not cat_date_str or cat_date_str.strip() in ["", "TBA", "TBD"]:
            continue

        # Parse date flexibly
        parsed_date = None
        for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%b %d %Y", "%B %d %Y"]:
            try:
                parsed_date = datetime.strptime(cat_date_str.split("T")[0].strip(), fmt)
                break
            except Exception:
                pass

        if parsed_date and start_dt <= parsed_date <= end_dt:
            # Flag conference presentations for inclusion in near-term catalysts description
            is_presentation = (
                "presentation" in note or "conference" in note or "symposium" in note
                or "presentation" in stage or "conference" in stage or "symposium" in stage
            )
            out = {
                "ticker": ticker,
                "company": drug.get("company", {}).get("name", "") if isinstance(drug.get("company"), dict) else drug.get("company", ""),
                "drug": drug.get("drug_name", ""),
                "indication": drug.get("indications_text", ""),
                "catalyst_date": cat_date_str,
                "stage": stage_label,
                "note": drug.get("note", ""),
                "source": drug.get("catalyst_source", ""),
                "has_catalyst": drug.get("has_catalyst", False),
                "type": "conference_presentation" if is_presentation else "other",
            }
            results.append(out)

    return results


def get_all_bpiq_programs_for_ticker(drugs_data, ticker):
    """Return all programs for a ticker from BPIQ with rich detail (no date filter).
    
    Extracts: drug_name, indications_text, stage_event label/stage_label/score,
    catalyst_date, catalyst_date_text, note, catalyst_source, mechanism_of_action,
    and boolean flags.  Sorted by stage score descending (most mature first).
    """
    results = []
    t = ticker.upper()
    for drug in drugs_data:
        if (drug.get("ticker") or "").upper() != t:
            continue
        se = drug.get("stage_event") or {}
        if not isinstance(se, dict):
            se = {}
        results.append({
            "drug_name": drug.get("drug_name", ""),
            "indications_text": drug.get("indications_text", ""),
            "stage_event_label": se.get("label", ""),
            "stage_label": se.get("stage_label", ""),
            "event_label": se.get("event_label", ""),
            "stage_score": se.get("score", 0),
            "catalyst_date": drug.get("catalyst_date") or "",
            "catalyst_date_text": drug.get("catalyst_date_text") or "",
            "note": drug.get("note") or "",
            "catalyst_source": drug.get("catalyst_source") or "",
            "mechanism_of_action": re.sub(r"<[^>]+>", "", drug.get("mechanism_of_action") or ""),  # strip HTML
            "has_catalyst": drug.get("has_catalyst", False),
            "is_big_mover": drug.get("is_big_mover", False),
            "is_suspected_mover": drug.get("is_suspected_mover", False),
            "is_hedge_fund_pick": drug.get("is_hedge_fund_pick", False),
        })
    # Sort by stage score descending (most mature first; higher score = more advanced)
    results.sort(key=lambda x: x["stage_score"], reverse=True)
    return results


def get_cached_or_compute(ticker, params_dict, compute_func, key_suffix=""):
    """
    Return cached Grok result or compute and cache.
    params_dict: screening params that affect the result (trial_window, min_impact_factor, run_date).
    compute_func: callable that calls Grok and returns the Markdown string.
    key_suffix: "asset", "shareholders", or "stock_relative" so we store three entries per (ticker, params).
    """
    param_str = json.dumps(params_dict, sort_keys=True)
    hash_obj = hashlib.sha256(param_str.encode()).hexdigest()
    cache_key = f"{ticker}__{hash_obj}" + (f"__{key_suffix}" if key_suffix else "")

    if cache_key in grok_cache:
        st.info(f"Using cached result for {ticker}" + (f" ({key_suffix})" if key_suffix else ""))
        return grok_cache[cache_key]

    with st.spinner(f"Generating fresh description for {ticker}" + (f" ({key_suffix})..." if key_suffix else "...")):
        result = compute_func()
        grok_cache[cache_key] = result
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(grok_cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return result


def _fetch_price_yahoo_direct(ticker, ref_date):
    """Fetch closing price via Yahoo Finance chart API directly (bypasses yfinance rate limits).
    
    Uses the raw Yahoo v8 chart endpoint with a custom User-Agent header.
    Returns float price or None.
    """
    try:
        end_ts = int(datetime.combine(ref_date + timedelta(days=1), datetime.min.time()).timestamp())
        start_ts = int(datetime.combine(ref_date - timedelta(days=10), datetime.min.time()).timestamp())

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"period1": start_ts, "period2": end_ts, "interval": "1d", "includePrePost": "false"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}

        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None

        result = resp.json().get("chart", {}).get("result")
        if not result:
            return None

        timestamps = result[0].get("timestamp", [])
        closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
        if not timestamps or not closes:
            return None

        # Find last valid close on or before ref_date
        ref_epoch = datetime.combine(ref_date, datetime.max.time()).timestamp()
        best_price = None
        for ts, close in zip(timestamps, closes):
            if ts <= ref_epoch and close is not None:
                best_price = close
        return round(float(best_price), 2) if best_price else None
    except Exception:
        return None


def _fetch_price_stooq(ticker, ref_date):
    """Fetch closing price from Stooq.com (free, no API key, reliable backup).
    
    Uses the direct CSV download endpoint. Returns float price or None.
    """
    try:
        start = ref_date - timedelta(days=10)
        d1 = start.strftime("%Y%m%d")
        d2 = ref_date.strftime("%Y%m%d")
        url = f"https://stooq.com/q/d/l/?s={ticker}.us&d1={d1}&d2={d2}&i=d"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200 or "Date" not in resp.text:
            return None
        reader = csv.DictReader(io.StringIO(resp.text))
        rows = list(reader)
        if rows:
            return round(float(rows[-1]["Close"]), 2)
        return None
    except Exception:
        return None


def build_stock_universe_from_bpiq(drugs_data, max_market_cap_b=10.0):
    """Build a deduplicated list of biotech stocks from BPIQ data.

    Extracts unique (ticker, company) pairs from the BPIQ drugs database.
    Returns list of dicts: [{"company": ..., "ticker": ..., "exchange": "US", "market_cap": ""}]
    Market cap will be populated later via Yahoo Finance.
    """
    seen = {}
    for drug in drugs_data:
        ticker = (drug.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            continue
        company = drug.get("company", {})
        if isinstance(company, dict):
            name = company.get("name", "")
        else:
            name = str(company) if company else ""
        seen[ticker] = {
            "company": name,
            "ticker": ticker,
            "exchange": "US",
            "market_cap": "",
        }
    result = sorted(seen.values(), key=lambda x: x["ticker"])
    return result


def fetch_market_caps_batch(tickers, max_cap_b=10.0):
    """Fetch market caps for a list of tickers via Yahoo Finance direct API.

    Returns dict {TICKER: {"market_cap": float, "market_cap_str": str, "exchange": str}}.
    Uses the v8 chart API meta field (fast, single call per ticker).
    Filters out tickers with market cap above max_cap_b (billions).
    """
    results = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    total = len(tickers)
    progress = st.progress(0, text="Fetching market caps...")

    for idx, t in enumerate(tickers):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
            params = {"interval": "1d", "range": "1d", "includePrePost": "false"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                meta = resp.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
                # Market cap isn't in chart meta — use regularMarketPrice * sharesOutstanding
                price = meta.get("regularMarketPrice")
                exch = meta.get("exchangeName", "")
                full_exch = meta.get("fullExchangeName", "")

                # Skip non-US exchanges
                if exch and exch.upper() not in ("NMS", "NYQ", "NGM", "NCM", "PCX", "ASE", "BTS",
                                                  "NASDAQ", "NYSE", "AMEX", "NYSE ARCA"):
                    continue

                results[t] = {
                    "price": price,
                    "exchange": full_exch or exch or "US",
                }
        except Exception:
            pass

        # Throttle: 0.15s per call
        if idx < total - 1:
            time.sleep(0.15)
        if (idx + 1) % 20 == 0 or idx == total - 1:
            progress.progress((idx + 1) / total, text=f"Market data: {idx+1}/{total} tickers...")

    progress.empty()

    # Now batch-fetch market caps via quoteSummary (more reliable for market cap)
    yahoo = _get_yahoo_crumb_session()
    session = yahoo["session"]
    crumb = yahoo["crumb"]

    if crumb:
        tickers_to_check = list(results.keys())
        cap_progress = st.progress(0, text="Fetching market caps...")
        for idx, t in enumerate(tickers_to_check):
            try:
                url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{t}"
                params = {"modules": "price", "crumb": crumb}
                r = session.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    price_mod = r.json().get("quoteSummary", {}).get("result", [{}])[0].get("price", {})
                    mc_raw = price_mod.get("marketCap", {}).get("raw")
                    if mc_raw:
                        mc_b = mc_raw / 1e9
                        if mc_b <= max_cap_b:
                            if mc_b >= 1:
                                results[t]["market_cap"] = f"{mc_b:.1f}B"
                            else:
                                results[t]["market_cap"] = f"{mc_raw / 1e6:.0f}M"
                            results[t]["market_cap_num"] = mc_raw
                        else:
                            # Too large — remove
                            results.pop(t, None)
                    else:
                        # No market cap data — keep but mark unknown
                        results[t]["market_cap"] = "N/A"
                        results[t]["market_cap_num"] = 0
                else:
                    results[t]["market_cap"] = "N/A"
                    results[t]["market_cap_num"] = 0
            except Exception:
                results[t]["market_cap"] = "N/A"
                results[t]["market_cap_num"] = 0

            time.sleep(0.15)
            if (idx + 1) % 20 == 0 or idx == len(tickers_to_check) - 1:
                cap_progress.progress((idx + 1) / len(tickers_to_check),
                                      text=f"Market caps: {idx+1}/{len(tickers_to_check)}...")
        cap_progress.empty()

    return results


def _fetch_chart_yahoo_direct(ticker, days=1825):
    """Fetch OHLCV chart data via Yahoo Finance v8 chart API directly (bypasses yfinance).
    
    Returns a pandas DataFrame with Date index and OHLCV+Volume columns, or None.
    """
    try:
        end_ts = int(datetime.now().timestamp())
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"period1": start_ts, "period2": end_ts, "interval": "1d", "includePrePost": "false"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"}

        resp = requests.get(url, params=params, headers=headers, timeout=20)
        if resp.status_code != 200:
            return None

        result = resp.json().get("chart", {}).get("result")
        if not result:
            return None

        timestamps = result[0].get("timestamp", [])
        quote = result[0].get("indicators", {}).get("quote", [{}])[0]
        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        if not timestamps or not closes:
            return None

        dates = pd.to_datetime(timestamps, unit='s').normalize()
        df = pd.DataFrame({
            "Open": opens, "High": highs, "Low": lows,
            "Close": closes, "Volume": volumes,
        }, index=dates)
        df.index.name = "Date"
        df = df.dropna(subset=["Close"])
        return df if not df.empty else None
    except Exception:
        return None


def _fetch_chart_stooq(ticker, days=1825):
    """Fetch OHLCV chart data from Stooq.com (free backup for chart data).
    
    Returns a pandas DataFrame with Date index and OHLCV+Volume columns, or None.
    """
    try:
        end = datetime.now().date()
        start = end - timedelta(days=days)
        d1 = start.strftime("%Y%m%d")
        d2 = end.strftime("%Y%m%d")
        url = f"https://stooq.com/q/d/l/?s={ticker}.us&d1={d1}&d2={d2}&i=d"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200 or "Date" not in resp.text:
            return None
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")
        df = df.sort_index()
        if "Close" in df.columns and not df.empty:
            return df
        return None
    except Exception:
        return None


def _fetch_market_cap_direct(ticker):
    """Fetch market cap via Yahoo Finance quoteSummary API (bypasses yfinance).
    
    Returns dict with market_cap (int) and date (str), or None.
    """
    try:
        yahoo = _get_yahoo_crumb_session()
        session = yahoo["session"]
        crumb = yahoo["crumb"]
        url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        params = {"modules": "summaryDetail,price", "crumb": crumb}
        resp = session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json().get("quoteSummary", {}).get("result", [])
        if not data:
            return None
        # Try 'price' module first, then 'summaryDetail'
        price_mod = data[0].get("price", {})
        mkt_cap = price_mod.get("marketCap", {}).get("raw")
        if not mkt_cap:
            summary = data[0].get("summaryDetail", {})
            mkt_cap = summary.get("marketCap", {}).get("raw")
        if mkt_cap:
            return {"market_cap": int(mkt_cap), "date": datetime.now().strftime("%Y-%m-%d")}
        return None
    except Exception:
        return None


def fetch_reference_prices(tickers, ref_date):
    """Fetch verified closing prices on/near the reference date.
    
    Strategy (in order):
      1. Direct Yahoo v8 chart API per ticker (fastest, bypasses yfinance rate limits)
      2. Stooq.com per ticker (free, no API key, reliable backup)
      3. Batch yf.download() as last resort (single attempt only)
    Results are cached in st.session_state so Streamlit reruns don't re-fetch.
    """
    if not tickers:
        return {}

    ticker_list = sorted(set(tickers))
    cache_key = f"yf_prices_{ref_date}_{'_'.join(ticker_list)}"

    # Return session-state cache if available
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        st.info(f"Using cached Yahoo Finance prices ({len(cached)}/{len(ticker_list)} tickers, ref {ref_date})")
        return cached

    prices = {}
    start_dt = ref_date - timedelta(days=10)
    end_dt = ref_date + timedelta(days=1)
    ref_ts = pd.Timestamp(ref_date)

    # ── Method 1 (primary): Direct Yahoo chart API — fast, no rate-limit issues ──
    for t in ticker_list:
        price = _fetch_price_yahoo_direct(t, ref_date)
        if price:
            prices[t] = price
        time.sleep(0.3)  # throttle between requests

    # ── Method 2 (backup): Stooq.com — free, no API key, no rate limits ──
    missing = [t for t in ticker_list if t not in prices]
    if missing:
        st.info(f"Trying Stooq for {len(missing)} remaining ticker(s)…")
        for t in missing:
            price = _fetch_price_stooq(t, ref_date)
            if price:
                prices[t] = price
            time.sleep(0.3)

    # ── Method 3 (last resort): single yf.download() attempt ──
    # Only triggers if both direct Yahoo API and Stooq failed
    still_missing = [t for t in ticker_list if t not in prices]
    if still_missing:
        st.info(f"Trying yfinance as last resort for {len(still_missing)} ticker(s)…")
        try:
            time.sleep(2)  # extra pause to avoid rate limits
            arg = still_missing[0] if len(still_missing) == 1 else still_missing
            df = yf.download(arg, start=start_dt, end=end_dt, progress=False)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    for t in still_missing:
                        try:
                            close_col = df['Close'][t].dropna()
                            if not close_col.empty:
                                valid = close_col[close_col.index <= ref_ts]
                                if not valid.empty:
                                    prices[t] = round(float(valid.iloc[-1]), 2)
                        except (KeyError, TypeError):
                            pass
                else:
                    if 'Close' in df.columns:
                        close_col = df['Close'].dropna()
                        if not close_col.empty:
                            valid = close_col[close_col.index <= ref_ts]
                            if not valid.empty:
                                prices[still_missing[0]] = round(float(valid.iloc[-1]), 2)
        except Exception:
            pass  # silently skip — direct APIs already tried

    # Report
    final_missing = [t for t in ticker_list if t not in prices]
    if final_missing:
        st.warning(f"⚠️ No price data for: {', '.join(final_missing)} — likely delisted or invalid ticker")
    if prices:
        st.success(f"✅ Fetched verified prices for {len(prices)}/{len(ticker_list)} tickers (ref date {ref_date})")
    else:
        st.error("❌ Could not fetch any prices. Try again in a few minutes.")

    # Cache in session state
    st.session_state[cache_key] = prices
    return prices


def _get_yahoo_crumb_session():
    """Create a requests session with Yahoo Finance cookie + crumb for authenticated endpoints.
    
    Cached in st.session_state so the crumb/cookie persist across Streamlit reruns.
    """
    cache_key = "_yahoo_crumb_session"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    session = requests.Session()
    session.headers['User-Agent'] = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
    )
    try:
        session.get('https://fc.yahoo.com', timeout=10, allow_redirects=True)
        crumb = session.get(
            'https://query2.finance.yahoo.com/v1/test/getcrumb', timeout=10
        ).text.strip()
    except Exception:
        crumb = None

    result = {"session": session, "crumb": crumb}
    st.session_state[cache_key] = result
    return result


def fetch_analyst_targets(tickers):
    """Fetch analyst consensus price targets from Yahoo Finance (free, no paid API needed).
    
    Uses Yahoo's quoteSummary endpoint with cookie + crumb auth.
    Returns dict of {TICKER: {mean, median, high, low, num_analysts, rec_key, current_price}}.
    Cached in st.session_state.
    """
    if not tickers:
        return {}

    ticker_list = sorted(set(tickers))
    cache_key = f"analyst_targets_{'_'.join(ticker_list)}"

    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        st.info(f"Using cached analyst targets ({len(cached)}/{len(ticker_list)} tickers)")
        return cached

    yahoo = _get_yahoo_crumb_session()
    session = yahoo["session"]
    crumb = yahoo["crumb"]

    if not crumb:
        st.warning("Could not obtain Yahoo Finance auth crumb — analyst targets unavailable")
        return {}

    targets = {}
    for t in ticker_list:
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{t}"
            params = {"modules": "financialData", "crumb": crumb}
            r = session.get(url, params=params, timeout=15)
            if r.status_code != 200:
                continue
            fd = r.json().get("quoteSummary", {}).get("result", [{}])[0].get("financialData", {})
            mean_target = fd.get("targetMeanPrice", {}).get("raw")
            if mean_target is not None:
                targets[t] = {
                    "mean": mean_target,
                    "median": fd.get("targetMedianPrice", {}).get("raw"),
                    "high": fd.get("targetHighPrice", {}).get("raw"),
                    "low": fd.get("targetLowPrice", {}).get("raw"),
                    "num_analysts": fd.get("numberOfAnalystOpinions", {}).get("raw"),
                    "rec_key": fd.get("recommendationKey", ""),
                }
            time.sleep(0.4)  # throttle to avoid Yahoo rate limits
        except Exception:
            pass

    missing = [t for t in ticker_list if t not in targets]
    if missing:
        st.warning(f"No analyst targets for: {', '.join(missing)}")
    if targets:
        st.success(f"✅ Fetched analyst targets for {len(targets)}/{len(ticker_list)} tickers (Yahoo Finance)")

    st.session_state[cache_key] = targets
    return targets


st.title("Biotech Stock Screener App")

st.sidebar.header("Screening Parameters")
start_date = datetime(2026, 1, 1)
end_date = datetime(2026, 7, 1)
trial_window = st.sidebar.text_input("Trial Window", "2026-01-01 to 2026-07-01")

# Stock price reference date — used to fetch verified closing prices from Yahoo Finance
price_ref_date = st.sidebar.date_input(
    "Stock Price Reference Date",
    value=datetime.now().date() - timedelta(days=1),
    help="Closing price on this date (or nearest prior trading day) from Yahoo Finance. "
         "Used for the analyst-upside filter (Criterion 5). Adjust if market data is delayed."
)
st.sidebar.caption(f"Verified prices will be pulled from Yahoo Finance as of **{price_ref_date}**.")

# ────────────────────────────────────────────────
#   Max Stocks Control (with Unlimited option)
# ────────────────────────────────────────────────

st.sidebar.header("Screening Limits")

max_stocks_slider = st.sidebar.slider(
    "Maximum Stocks to Screen",
    min_value=10,
    max_value=1000,
    value=200,
    step=10,
    help="Screens top N stocks by market cap from BPIQ universe. Higher = slower but more coverage."
)

unlimited_mode = st.sidebar.checkbox(
    "Unlimited (screen all BPIQ stocks)",
    value=False,
    help="Overrides the slider — screens every US-listed biotech in BPIQ with market cap < $10B"
)

# Determine final limit
if unlimited_mode:
    max_stocks = None  # No limit
    st.sidebar.info("Unlimited mode — will screen all qualifying BPIQ stocks (may take a while)")
else:
    max_stocks = max_stocks_slider

batch_size = st.sidebar.slider("Batch Size for Filtering", 1, 20, 6)
min_impact_factor = st.sidebar.slider("Min Impact Factor for POC", 5.0, 50.0, 20.0, 5.0)
min_upside_pct = st.sidebar.slider("Min Upside vs Analyst Consensus Target (%)", 10, 100, 30, 5,
                                    help="Stock must have at least this % upside vs average sell-side 12-month price target")

st.sidebar.header("BPIQ Data Management")

# Try to load from disk on startup
if st.session_state.bpiq_drugs is None:
    load_bpiq_from_disk()

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("Load / Refresh BPIQ Data"):
        with st.spinner("Fetching full BPIQ drugs database (may take 1–3 minutes)..."):
            fresh_drugs = fetch_all_bpiq_drugs()
            if fresh_drugs:
                st.session_state.bpiq_drugs = fresh_drugs
                st.session_state.bpiq_last_fetch = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.pop("stock_universe", None)  # invalidate universe cache
                st.session_state.pop("stock_universe_bpiq_count", None)
                save_bpiq_to_disk()  # ← saves to disk automatically
                st.success(f"Refreshed and saved {len(fresh_drugs)} records!")
            else:
                st.error("Failed to fetch from API — check key or network.")

with col2:
    if st.button("Clear Cache"):
        st.session_state.bpiq_drugs = None
        st.session_state.bpiq_last_fetch = None
        st.session_state.pop("stock_universe", None)
        st.session_state.pop("stock_universe_bpiq_count", None)
        if BPIQ_CACHE_FILE.exists():
            BPIQ_CACHE_FILE.unlink()
        st.success("Cache cleared. Click 'Load / Refresh' to fetch again.")

# Status display
if st.session_state.bpiq_drugs is None:
    st.sidebar.warning("No BPIQ data loaded. Click 'Load / Refresh BPIQ Data' to start screening.")
else:
    st.sidebar.info(f"BPIQ cache ready: {len(st.session_state.bpiq_drugs)} records\nLast fetch: {st.session_state.bpiq_last_fetch}")

st.sidebar.markdown("---")
if st.sidebar.button("🔬 Run Screening", use_container_width=True, type="primary"):
    # Check if BPIQ data is loaded
    if st.session_state.bpiq_drugs is None:
        st.error("Cannot run screening: BPIQ data not loaded. Please load it first using the sidebar button.")
        st.stop()  # Stops execution here
    
        # Step 1: Build stock universe from BPIQ data (cached in session state)
        start_date_str = start_date.strftime("%Y-%m-%d")

        # Use cached universe if BPIQ data hasn't changed
        bpiq_count = len(st.session_state.bpiq_drugs)
        universe_cache_valid = (
            st.session_state.get("stock_universe") is not None
            and st.session_state.get("stock_universe_bpiq_count") == bpiq_count
        )

        if universe_cache_valid:
            stock_list_full = st.session_state["stock_universe"]
            st.success(f"Using cached stock universe: **{len(stock_list_full)}** US-listed biotech < $10B")
        else:
            with st.spinner("Building stock universe from BPIQ and fetching market data (one-time)..."):
                st.info("Extracting tickers from BPIQ database...")
                bpiq_universe = build_stock_universe_from_bpiq(st.session_state.bpiq_drugs)
                st.info(f"BPIQ contains **{len(bpiq_universe)}** unique tickers. Fetching market caps to filter US-listed biotech < $10B...")

                # Fetch market caps and filter by cap + US exchange
                all_tickers = [s["ticker"] for s in bpiq_universe]
                mkt_data = fetch_market_caps_batch(all_tickers, max_cap_b=10.0)

                # Build final stock list with market cap data
                stock_list_full = []
                for s in bpiq_universe:
                    t = s["ticker"]
                    if t not in mkt_data:
                        continue  # filtered out (non-US or cap > $10B or no data)
                    md = mkt_data[t]
                    stock_list_full.append({
                        "company": s["company"],
                        "ticker": t,
                        "exchange": md.get("exchange", "US"),
                        "market_cap": md.get("market_cap", "N/A"),
                        "_market_cap_num": md.get("market_cap_num", 0),
                    })

                # Sort by market cap descending
                stock_list_full.sort(key=lambda x: x.get("_market_cap_num", 0), reverse=True)

                # Cache in session state
                st.session_state["stock_universe"] = stock_list_full
                st.session_state["stock_universe_bpiq_count"] = bpiq_count

                st.success(f"Stock universe: **{len(stock_list_full)}** US-listed biotech stocks with market cap < $10B (from {len(bpiq_universe)} BPIQ tickers)")

        # Apply slider limit if not in unlimited mode
        stock_list = list(stock_list_full)  # copy so we don't mutate the cache
        if max_stocks is not None and len(stock_list) > max_stocks:
            stock_list = stock_list[:max_stocks]
            st.info(f"Screening top {max_stocks} by market cap (adjust slider or enable Unlimited)")

        # Step 2: Batched filtering (both criteria)
        filtered_stocks = []
        all_results = []
        progress = st.progress(0)

        for i in range(0, len(stock_list), batch_size):
            batch = stock_list[i:i+batch_size]
            batch_desc = "\n".join([f"- {s['company']} ({s['ticker']})" for s in batch])
            
            # Get BPIQ catalysts for this batch (use cached data)
            tickers_in_batch = [s['ticker'].upper() for s in batch]
            near_term_events = get_upcoming_catalysts_for_tickers(
                st.session_state.bpiq_drugs, 
                tickers_in_batch, 
                start_date, 
                end_date
            )
            
            # Group by ticker for easy lookup
            catalysts_by_ticker = {}
            for ev in near_term_events:
                t = ev['ticker']
                if t not in catalysts_by_ticker:
                    catalysts_by_ticker[t] = []
                catalysts_by_ticker[t].append(ev)
            
            # Format BPIQ data for Grok prompt
            bpiq_context = ""
            for ticker, cats in catalysts_by_ticker.items():
                bpiq_context += f"\n{ticker} catalysts from BPIQ:\n"
                for cat in cats:
                    bpiq_context += f"  - {cat['drug']}: {cat['catalyst_date']} ({cat['stage']}) - {cat['indication']}\n"

            # Criterion 1: Check BPIQ catalysts
            for stock in batch:
                ticker = stock['ticker'].upper()
                catalysts = catalysts_by_ticker.get(ticker, [])
                stock['criterion1_pass'] = len(catalysts) > 0
                stock['criterion1_reasons'] = f"BPIQ found {len(catalysts)} upcoming catalyst(s) in window" if stock['criterion1_pass'] else "No upcoming catalysts in Jan–Jul 2026 per BPIQ"
                stock['near_term_catalysts'] = catalysts  # Save for later asset description

            # ── Fetch verified stock prices + analyst targets from Yahoo Finance ──
            verified_prices = fetch_reference_prices(tickers_in_batch, price_ref_date)
            analyst_targets = fetch_analyst_targets(tickers_in_batch)

            # Criterion 5: Analyst upside (computed in Python, no Grok needed)
            for stock in batch:
                t = stock['ticker'].upper()
                vp = verified_prices.get(t)
                at = analyst_targets.get(t, {})
                stock['verified_price'] = vp
                stock['analyst_data'] = at

                avg_target = at.get('mean')
                if vp is not None and avg_target is not None and vp > 0:
                    upside = round((avg_target - vp) / vp * 100, 1)
                    stock['criterion5_pass'] = upside >= min_upside_pct
                    stock['criterion5_reasons'] = (
                        f"Price ${vp:.2f} (Yahoo Finance, {price_ref_date}) vs "
                        f"avg analyst target ${avg_target:.2f} ({at.get('num_analysts', '?')} analysts) "
                        f"= {upside:.1f}% upside (threshold: {min_upside_pct}%)"
                    )
                    stock['upside_pct'] = upside
                else:
                    stock['criterion5_pass'] = False
                    stock['criterion5_reasons'] = (
                        f"Price: {'$'+f'{vp:.2f}' if vp else 'N/A'}, "
                        f"Analyst target: {'$'+f'{avg_target:.2f}' if avg_target else 'N/A'} — insufficient data"
                    )
                    stock['upside_pct'] = None

            # Criteria 2, 3 & 4: Use Grok ONLY for POC papers + animal study + orphan/accelerated
            filter_prompt = f"""
For each of the following stocks, check ALL THREE criteria as of {start_date_str}:
2. The theory / proof-of-concept (POC) behind the lead drug candidate is cited and supported by **original research papers** (not review articles) published in journals with Clarivate Impact Factor > {min_impact_factor} (e.g., Nature, Cell, Science, Nature Medicine; search PubMed, Google Scholar, Clarivate for max IF among key POC citations). Only count original research, not reviews.
   **In your reasons for criterion 2, you MUST list each qualifying POC paper with: full title, journal name, year, and Clarivate Impact Factor.**
3. There must be at least one **original paper** (not a review) on the lead drug candidate that is an **animal study** and was published in a journal with Clarivate impact factor ≥ 10.
   **In your reasons for criterion 3, you MUST specify the qualifying animal study paper(s) with: full title, journal name, year, and Clarivate Impact Factor.**
4. The lead drug candidate must have **orphan drug designation** OR **accelerated approval pathway** by the FDA (e.g., Breakthrough Therapy, Fast Track, Priority Review, Accelerated Approval; check FDA Orange Book, drug labels, company press releases).
   **In your reasons for criterion 4, state the specific FDA designation type and source.**

Stocks:
{batch_desc}

{bpiq_context}

A stock meets_criteria only if it passes ALL of criteria 2, 3, and 4. Output ONLY JSON array (same order):
[{{"company":"Name","ticker":"SYM","meets_criteria":true/false,"reasons":"For criterion 2: list each qualifying POC original research paper with full title, journal, year, IF. For criterion 3: specify qualifying animal study paper(s) with full title, journal, year, IF. For criterion 4: state FDA designation type and source.","asset_name":"Name"}}]
"""
            response = call_grok(filter_prompt)
            filter_results = parse_json_safely(response, list)

            for j, filter_res in enumerate(filter_results):
                if j >= len(batch):
                    break
                stock = batch[j]
                grok_meets_criteria = filter_res.get("meets_criteria", False)
                filter_reasons = filter_res.get("reasons", "No data")
                asset_name = filter_res.get("asset_name", stock.get("asset_name", "Unknown"))

                # ────────────────────────────────────────────
                #   Bidirectional paper verification / rescue
                # ────────────────────────────────────────────
                cross_check = None
                c234 = grok_meets_criteria
                has_verification_llm = bool(GEMINI_API_KEY or DEEPSEEK_API_KEY)

                if grok_meets_criteria and has_verification_llm:
                    # ── PATH A: Grok says PASS → verify papers aren't hallucinated ──
                    with st.spinner(f"🔍 {stock['ticker']}: Verifying Grok's papers with Gemini…"):
                        cross_check = verify_papers_with_gemini(
                            stock['ticker'], stock['company'], asset_name, filter_reasons, min_impact_factor
                        )
                    if cross_check.get("action") == "skip":
                        st.info(f"ℹ️ {stock['ticker']}: Verification skipped — {cross_check.get('details', '')}")
                    elif not (cross_check.get("criterion2_pass", True) and cross_check.get("criterion3_pass", True)):
                        c234 = False
                        st.warning(
                            f"⚠️ {stock['ticker']}: Gemini could NOT verify Grok's papers — "
                            f"{cross_check.get('details', 'papers failed verification')}"
                        )
                    else:
                        st.success(f"✅ {stock['ticker']}: Papers verified by Gemini")

                elif not grok_meets_criteria and has_verification_llm:
                    # ── PATH B: Grok says FAIL → rescue search for missing papers ──
                    with st.spinner(f"🔎 {stock['ticker']}: Grok found no papers — searching with Gemini + DeepSeek…"):
                        cross_check = rescue_papers_search(
                            stock['ticker'], stock['company'], asset_name, min_impact_factor
                        )
                    if cross_check.get("rescued"):
                        c234 = True  # Override Grok's fail — other LLMs found papers
                        # Build rescued reasons from the papers found
                        rescued_papers_text = []
                        for p in cross_check.get("papers", []):
                            rescued_papers_text.append(
                                f"  - {p.get('title', '?')} ({p.get('journal', '?')}, {p.get('year', '?')}, "
                                f"IF={p.get('impact_factor', '?')}) [found by {p.get('found_by', '?')}]"
                            )
                        filter_reasons = (
                            f"RESCUED by secondary LLMs — Grok missed these papers:\n"
                            + "\n".join(rescued_papers_text)
                            + f"\nOriginal Grok result: {filter_reasons}"
                        )
                        st.success(
                            f"🆘 {stock['ticker']}: RESCUED — Gemini/DeepSeek found qualifying papers "
                            f"that Grok missed! ({len(cross_check.get('papers', []))} papers found)"
                        )
                    else:
                        st.info(
                            f"ℹ️ {stock['ticker']}: Confirmed — no qualifying papers found by any LLM. "
                            f"{cross_check.get('details', '')}"
                        )

                # Combine all criteria: BPIQ (1) + papers (2,3 via Grok ± rescue) + FDA (4) + Yahoo Finance (5)
                c1 = stock.get('criterion1_pass', False)
                c5 = stock.get('criterion5_pass', False)
                meets_criteria = c1 and c234 and c5

                # Build comprehensive reasons string
                reasons = (
                    f"Criterion 1 (Catalysts - BPIQ): {stock.get('criterion1_reasons', 'No data')}\n"
                    f"Criteria 2–4 (POC + Animal study + Orphan/Accelerated - Grok): {filter_reasons}\n"
                    f"Criterion 5 (Analyst upside - Yahoo Finance): {stock.get('criterion5_reasons', 'No data')}"
                )
                if cross_check:
                    cc = cross_check
                    reasons += f"\n--- Cross-Check ({cc.get('source', 'Gemini+DeepSeek')}) ---\n"
                    if cc.get("rescued"):
                        reasons += f"RESCUED: Yes — papers found by secondary LLMs\n"
                    elif cc.get("action") == "verify":
                        reasons += f"Verification: C2={cc.get('criterion2_pass')}, C3={cc.get('criterion3_pass')}\n"
                    reasons += f"Details: {cc.get('details', 'N/A')}\n"
                    for p in cc.get("papers", []):
                        reasons += f"  Paper: {p.get('title', '?')} | {p.get('journal', '?')} | IF={p.get('impact_factor', '?')} | Original={p.get('is_original_research', '?')} | Animal={p.get('is_animal_study', '?')}\n"

                stock_result = stock.copy()
                stock_result['meets_criteria'] = meets_criteria
                stock_result['reasons'] = reasons
                stock_result['asset_name'] = asset_name
                stock_result['poc_reasons'] = filter_reasons
                stock_result['cross_check'] = cross_check

                # Verified price + analyst data (all from Yahoo Finance)
                at = stock.get('analyst_data', {})
                stock_result['current_price'] = stock.get('verified_price')
                stock_result['price_source'] = "Yahoo Finance"
                stock_result['price_ref_date'] = str(price_ref_date)
                stock_result['avg_price_target'] = at.get('mean')
                stock_result['upside_pct'] = stock.get('upside_pct')
                stock_result['analyst_consensus_rating'] = at.get('rec_key', '')
                stock_result['num_analysts'] = at.get('num_analysts')
                stock_result['price_target_high'] = at.get('high')
                stock_result['price_target_low'] = at.get('low')

                all_results.append(stock_result)
                if meets_criteria:
                    filtered_stocks.append(stock_result)

            progress.progress((i + len(batch)) / len(stock_list))

        # Store latest results for Publish button (survives rerun)
        st.session_state.publish_all_results = all_results
        st.session_state.publish_filtered_stocks = filtered_stocks

        # Display all (for debugging)
        if all_results:
            st.subheader("All Screened Stocks")
            st.dataframe(all_results)

        if filtered_stocks:
            st.success(f"Found {len(filtered_stocks)} stocks meeting criteria!")
            st.table(filtered_stocks)

            for idx, stock in enumerate(filtered_stocks):
                # Persistent UUID per stock (survives reruns; UUID + index for globally unique widget keys)
                uuid_key = f"uuid_{stock['ticker']}_{idx}"
                if uuid_key not in st.session_state:
                    st.session_state[uuid_key] = str(uuid.uuid4())
                unique_id = st.session_state[uuid_key]

                with st.expander(f"Details for {stock['company']} ({stock['ticker']})", expanded=False):
                    # Cache key for LLM content: only params that change the prompts
                    # (trial_window and min_impact_factor affect what Grok writes)
                    # price_ref_date and min_upside_pct are used in Python math, not LLM prompts
                    # run_date excluded so cache survives across days
                    params_for_cache = {
                        "trial_window": trial_window,
                        "min_impact_factor": min_impact_factor,
                    }
                    tabs = st.tabs([
                        "Asset",
                        "Company and Stock",
                        "Shareholders & Investors",
                        "Stock Price & Volume Chart",
                    ])

                    # ─── Tab 0: Asset ───
                    with tabs[0]:
                        st.subheader(f"Asset — {stock.get('asset_name', 'Lead Asset')}")
                        catalysts_text = json.dumps(stock.get('near_term_catalysts', []), indent=2)
                        # Inject screening filter results as ground truth for papers
                        screening_evidence = stock.get('poc_reasons', 'No screening data available.')
                        asset_prompt = f"""
For {stock['company']}'s {stock.get('asset_name', 'lead asset')}:

Known upcoming catalysts from BPIQ API:
{catalysts_text}

Use this BPIQ data as the primary source for catalyst dates. Include presentations at upcoming conferences in the near-term catalysts section.

**IMPORTANT — Ground truth from screening filter (you MUST use this):**
During the screening/filtering step, the following papers and evidence were identified for this stock.
You MUST include ALL of these papers in your "Key Academic Papers" and "Animal Studies" sections below.
Do NOT omit, contradict, or replace any of them. You may add additional papers you find, but these are the mandatory baseline:
---
{screening_evidence}
---

Research and output as structured Markdown. Use the EXACT section headers below (### header) so the output is cleanly organized:

### Near-Term Catalysts
Date of near-term clinical trial results, data readouts, presentations at upcoming conferences, regulatory milestones (e.g., PDUFA dates, BLA decisions), or key events within 0-6 months AFTER {start_date_str}. **IMPORTANT: Only include events dated ON OR AFTER {start_date_str}. Do NOT include past events.** Primarily focus on company press releases, earnings transcripts from the last four quarters leading up to {start_date_str}, and stock commentary websites (e.g., Seeking Alpha, Yahoo Finance). Present as a Markdown table with columns: **Date**, **Event**, **Type**, **Potential Impact**.

### Drug Overview
Present the following as a Markdown table with columns **Attribute** and **Detail**:
| Attribute | Detail |
|---|---|
| Modality | ... |
| First in Class | Yes/No — reason |
| Best in Class | Yes/No — reason |
| Route of Administration & Dosing | ... |
| Proposed Benefit | ... (assess if significant) |
| Worldwide Incidence | ... |
| Target Population Size | ... |
| Potential Market Size ($) | ... |

### Current Standard of Care
Description, cost, frequency, insurance coverage, administration, dosage, side effects, outcomes.

### Key Academic Papers
Include ALL papers from the screening ground truth above FIRST, then add any additional papers that are (a) specific to this drug candidate, or (b) relate to the theory / proof-of-concept (POC) behind the lead drug candidate, published in journals with Clarivate Impact Factor > {min_impact_factor}. No limit on number of papers. For each paper present as a table: **Title**, **Authors**, **Journal (Year)**, **Impact Factor**, **Citations**, **Abstract Summary**, **Link**.

### Animal Studies
Include the animal study paper(s) from the screening ground truth above. Cite at least one original paper (not a review) on the lead drug that is an animal study, published in a journal with Clarivate impact factor ≥ 10. **Assess how good the animal studies were and the quality of the animal studies** (study design, endpoints, reproducibility, relevance to human disease).

### Clinical Data Consistency
- Preclinical vs. Phase 1/2 data (assess consistency)
- Consistency with preclinical nonhuman studies
- Outcome measures accepted? (assess)

### Safety Profile
**Clearly highlight any side effects, adverse events, or deaths** reported in trials or preclinical studies. Present any known AEs in a table if data is available.

### Alternative Approaches
Descriptions, stages, and **upcoming catalysts for competing alternative approaches** — check the BPIQ database and public sources/filings of the company sponsoring each alternative approach asset; highlight near-term events (0-6 months) that could impact this stock. Present as a table: **Competing Asset**, **Company**, **Stage**, **Near-Term Catalyst**, **Date**.

### Manufacturing & Scalability
Assess manufacturing readiness and scalability.

### Insurance Coverage Potential
Potential for insurance coverage in US and other markets.

### Other Indications
List any other indications being explored.

### Investigator Background
Key investigators, their CV highlights, and track record.

Include links to sources in [link](url) format.
Use tools like web search, X search, PubMed, Google Scholar, Clarivate for IFs, browse pages for accuracy. Prioritize company press releases, earnings transcripts from the last four quarters before {start_date_str}, and stock commentary websites for catalyst supplements.
"""
                        asset_md = get_cached_or_compute(
                            stock["ticker"],
                            params_for_cache,
                            lambda ap=asset_prompt: call_grok(ap),
                            "asset",
                        )
                        st.markdown(asset_md)

                        # ── Verify catalysts with Gemini/DeepSeek ──
                        # Extract the catalysts section from Grok's output for verification
                        catalyst_section = ""
                        if "### Near-Term Catalysts" in asset_md:
                            parts = asset_md.split("### Near-Term Catalysts", 1)
                            if len(parts) > 1:
                                rest = parts[1]
                                # Take content up to the next ### section
                                next_section = rest.find("\n### ")
                                catalyst_section = rest[:next_section] if next_section > 0 else rest[:2000]

                        if catalyst_section.strip():
                            bpiq_cats_json = json.dumps(stock.get('near_term_catalysts', []), indent=2)
                            with st.spinner(f"Verifying catalysts for {stock['ticker']} with Gemini/DeepSeek..."):
                                cat_verify = verify_catalysts_with_llms(
                                    stock['company'], stock['ticker'],
                                    stock.get('asset_name', 'lead asset'),
                                    catalyst_section, start_date_str, bpiq_cats_json
                                )
                            if cat_verify.get("verified_md"):
                                st.markdown("---")
                                st.markdown(f"#### Catalyst Verification ({cat_verify.get('source', 'LLM')})")
                                st.markdown(cat_verify["verified_md"])

                    # ─── Tab 1: Company and Stock ───
                    with tabs[1]:
                        st.subheader("Company and Stock")

                        # ── Pipeline table: directly from BPIQ data ──
                        bpiq_pipeline = get_all_bpiq_programs_for_ticker(st.session_state.bpiq_drugs or [], stock["ticker"])

                        if bpiq_pipeline:
                            st.markdown(f"**Pipeline Assets ({len(bpiq_pipeline)} programs from BPIQ)**")

                            pipeline_summary_rows = []
                            for prog in bpiq_pipeline:
                                pipeline_summary_rows.append({
                                    "Drug & Indication": f"{prog['drug_name']} — {prog['indications_text']}",
                                    "Stage & Event": prog["stage_event_label"],
                                    "Catalyst Date": prog["catalyst_date_text"] or prog["catalyst_date"] or "TBD",
                                })
                            pipeline_df = pd.DataFrame(pipeline_summary_rows)
                            st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

                            for prog in bpiq_pipeline:
                                with st.expander(f"{prog['drug_name']} — {prog['indications_text']}", expanded=False):
                                    st.markdown(f"**Stage & Event:** {prog['stage_event_label']}")
                                    st.markdown(f"**Catalyst Date:** {prog['catalyst_date_text'] or prog['catalyst_date'] or 'TBD'}")
                                    if prog["note"]:
                                        st.markdown(f"**Notes:** {prog['note']}")
                                    if prog["mechanism_of_action"]:
                                        st.markdown(f"**Mechanism of Action:** {prog['mechanism_of_action']}")
                                    if prog["catalyst_source"]:
                                        st.markdown(f"**Source:** [{prog['catalyst_source']}]({prog['catalyst_source']})")
                                    flags = []
                                    if prog.get("is_big_mover"):
                                        flags.append("Big Mover")
                                    if prog.get("is_suspected_mover"):
                                        flags.append("Suspected Mover")
                                    if prog.get("is_hedge_fund_pick"):
                                        flags.append("Hedge Fund Pick")
                                    if flags:
                                        st.markdown(f"**BPIQ Flags:** {', '.join(flags)}")

                            drug_list_for_grok = "\n".join([
                                f"- {p['drug_name']} ({p['indications_text']}) — {p['stage_event_label']}"
                                for p in bpiq_pipeline
                            ])
                            stock_prompt = f"""
For {stock['company']} ({stock['ticker']}), estimate the following for each pipeline asset listed below.
Output a **Markdown table** with columns: **Drug & Indication**, **Estimated Peak Annual Revenue ($)**.
Rank from most mature to least mature (order given).

Pipeline assets:
{drug_list_for_grok}

After the table, also provide:
- Analyst probability for positive trial results (for the lead asset)
- Market-implied probability for positive trial results
- Stock price impact if lead asset fails (considering full pipeline)
Include links to sources (analyst reports, consensus estimates, etc.).
Use web search for analyst reports, consensus estimates, etc.
"""
                        else:
                            st.warning("No pipeline data found in BPIQ for this ticker.")
                            stock_prompt = f"""
For {stock['company']} ({stock['ticker']}):
Research and output as structured Markdown:
- A **Markdown table** of ALL known clinical-stage pipeline assets with columns: **Drug & Indication**, **Stage**, **Upcoming Catalyst**, **Estimated Peak Annual Revenue ($)**. Rank most mature first.
- Analyst probability for positive trial results
- Market-implied probability for positive trial results
- Stock price impact if asset fails (considering pipeline)
Include links to sources.
Use web search for analyst reports, etc.
"""

                        stock_md = get_cached_or_compute(
                            stock["ticker"],
                            params_for_cache,
                            lambda sp=stock_prompt: call_grok(sp),
                            "stock_relative",
                        )
                        st.markdown(stock_md)

                        # ── Sell-Side Analyst Consensus Section ──
                        st.markdown("---")
                        st.subheader("Sell-Side Analyst Consensus Price Targets")

                        cur_price = stock.get("current_price")
                        avg_tgt = stock.get("avg_price_target")
                        upside = stock.get("upside_pct")
                        rating = stock.get("analyst_consensus_rating", "")
                        n_analysts = stock.get("num_analysts")
                        tgt_high = stock.get("price_target_high")
                        tgt_low = stock.get("price_target_low")

                        price_src = stock.get("price_source", "Unknown")
                        ref_dt = stock.get("price_ref_date", "N/A")

                        if avg_tgt and cur_price:
                            cols_an = st.columns(4)
                            cols_an[0].metric(
                                f"Stock Price ({ref_dt})",
                                f"${cur_price:.2f}" if isinstance(cur_price, (int, float)) else str(cur_price),
                                help=f"Source: {price_src}"
                            )
                            cols_an[1].metric("Avg 12-Mo Target", f"${avg_tgt:.2f}" if isinstance(avg_tgt, (int, float)) else str(avg_tgt))
                            cols_an[2].metric("Implied Upside", f"{upside:.1f}%" if isinstance(upside, (int, float)) else str(upside or "N/A"))
                            cols_an[3].metric("Consensus Rating", rating or "N/A")

                            detail_parts = []
                            if n_analysts:
                                detail_parts.append(f"Based on **{n_analysts} analyst(s)**")
                            if tgt_high and tgt_low:
                                detail_parts.append(f"Range: **${tgt_low}** (low) – **${tgt_high}** (high)")
                            if detail_parts:
                                st.markdown(" | ".join(detail_parts))

                            st.caption(
                                f"All data from Yahoo Finance. "
                                f"Stock price as of {ref_dt}. "
                                "Analyst targets sourced from Yahoo Finance consensus (aggregated from sell-side analysts)."
                            )
                        else:
                            st.info("Analyst consensus data not available for this stock from Yahoo Finance.")

                    # ─── Tab 2: Shareholders & Investors ───
                    with tabs[2]:
                        st.subheader("Disclosed Major Shareholders & Institutional Investors")
                        shareholders_prompt = f"""
For {stock['company']} ({stock['ticker']}):
Research the latest disclosed major shareholders and institutional investors (from 13F filings, SEC EDGAR, WhaleWisdom, company investor presentations, or recent cap table data).

Include:
- Top 8–12 holders with approximate % ownership and any recent stake changes.
- **Highlight in bold** and explicitly note any matches from this list of key biotech/specialty investors: Orbimed, Arch Ventures, Lily Asia Ventures, Decheng, NEA (New Enterprise Associates), Samsara, Vivo Capital, RA Capital, Third Rock Ventures, Atlas Venture, Novo Holdings, Flagship Pioneering, Sofinnova Partners, Baker Brothers.
- For the highlighted VCs/specialty investors (and any other top holders), include recent purchase/sale activity over the last 1–4 quarters (e.g., Q4 2025, Q3 2025, etc.). Specify shares bought/sold, % change, approximate date/quarter, and source (13F or Form 4).
- If data is limited (early-stage, small float, pre-IPO), state that clearly and mention known venture rounds or lead investors from press releases.

Output as structured Markdown with:
- Bullet list of top holders
- Bold highlights for key VCs/specialty names
- Sub-bullets for recent trade activity when available
Include source links in [link](url) format (SEC filings, WhaleWisdom, etc.).
Use web search, browse SEC EDGAR/WhaleWisdom/company IR pages, recent 13F/Form 4 filings for accuracy.
"""
                        shareholders_md = get_cached_or_compute(
                            stock["ticker"],
                            params_for_cache,
                            lambda sp=shareholders_prompt: call_grok(sp),
                            "shareholders",
                        )
                        st.markdown(shareholders_md)

                    # ─── Tab 3: Stock Price & Volume Chart ───
                    with tabs[3]:
                        st.subheader(f"Stock Price & Trading Volume — {stock['ticker']} ({stock.get('exchange', '')})")

                        ticker_symbol = stock['ticker']
                        if stock.get('exchange') == "HKEX":
                            ticker_symbol += ".HK"

                        chart_data_key = f"chart_data_{unique_id}"
                        period_key = f"period_select_{unique_id}"

                        if chart_data_key not in st.session_state:
                            with st.spinner(f"Loading data for {stock['ticker']}..."):
                                # Try direct Yahoo API first (bypasses yfinance rate limits)
                                df = _fetch_chart_yahoo_direct(ticker_symbol, days=1825)
                                if df is None:
                                    # Backup: Stooq (free, no API key)
                                    df = _fetch_chart_stooq(ticker_symbol, days=1825)
                                if df is None:
                                    # Last resort: yfinance
                                    try:
                                        end = datetime.now()
                                        start = end - timedelta(days=1825)
                                        df = yf.download(ticker_symbol, start=start, end=end, progress=False)
                                        if not df.empty and isinstance(df.columns, pd.MultiIndex):
                                            df.columns = df.columns.get_level_values(0)
                                        if df.empty:
                                            df = None
                                    except Exception:
                                        df = None
                                st.session_state[chart_data_key] = df

                        df = st.session_state.get(chart_data_key)

                        if df is None or df.empty:
                            st.warning(f"No data for {stock['ticker']}.")
                        else:
                            period_options = {
                                "1 Month": 30,
                                "3 Months": 90,
                                "6 Months": 180,
                                "1 Year": 365,
                                "2 Years": 730,
                                "5 Years": 1825,
                                "Max": len(df)
                            }

                            selected_label = st.selectbox(
                                "Select time period",
                                options=list(period_options.keys()),
                                index=3,
                                key=period_key
                            )

                            days = period_options[selected_label]
                            chart_df = df.tail(days)

                            if chart_df.empty:
                                st.warning("No data in selected period.")
                            else:
                                fig = go.Figure()
                                fig.add_trace(go.Candlestick(
                                    x=chart_df.index,
                                    open=chart_df['Open'], high=chart_df['High'],
                                    low=chart_df['Low'], close=chart_df['Close'],
                                    name='Price',
                                    increasing_line_color='green', decreasing_line_color='red'
                                ))
                                fig.add_trace(go.Bar(
                                    x=chart_df.index, y=chart_df['Volume'],
                                    name='Volume', yaxis='y2',
                                    marker_color='rgba(100, 149, 237, 0.6)', opacity=0.6
                                ))
                                fig.update_layout(
                                    title=f"{stock['ticker']} — {selected_label}",
                                    yaxis_title="Price (USD)",
                                    xaxis_rangeslider_visible=True,
                                    hovermode="x unified",
                                    template="plotly_white",
                                    height=600, showlegend=True,
                                    yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False)
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                current_price = chart_df['Close'].iloc[-1]
                                change_period = ((current_price - chart_df['Close'].iloc[0]) / chart_df['Close'].iloc[0] * 100)
                                vol_avg = chart_df['Volume'].mean()
                                cols = st.columns(3)
                                cols[0].metric("Latest Close", f"${current_price:.2f}")
                                cols[1].metric(f"{selected_label} Change", f"{change_period:.1f}%",
                                               delta_color="normal" if change_period >= 0 else "inverse")
                                cols[2].metric("Avg Daily Volume", f"{int(vol_avg):,}")

                        # Today's Market Cap (direct API — bypasses yfinance rate limits)
                        mkt_data = _fetch_market_cap_direct(ticker_symbol)
                        if mkt_data and mkt_data.get("market_cap"):
                            mc = mkt_data["market_cap"]
                            if mc >= 1e9:
                                market_cap_display = f"${mc / 1e9:.2f}B"
                            elif mc >= 1e6:
                                market_cap_display = f"${mc / 1e6:.2f}M"
                            else:
                                market_cap_display = f"${mc:,.0f}"
                            st.metric(label="Market Cap (as of today)", value=market_cap_display,
                                      delta_color="normal", help=f"Data as of {mkt_data['date']} from Yahoo Finance")
                        else:
                            st.metric("Market Cap (as of today)", "N/A", help="Could not fetch from Yahoo Finance")

                        # Recent News & Catalysts
                        st.markdown("---")
                        st.subheader("Recent News & Catalysts (Last 12 Months)")
                        news_prompt = f"""
For {stock['company']} ({stock['ticker']} on {stock.get('exchange', 'NASDAQ')}):
Summarize key news, catalysts, and events from the last 12 months (since {(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')}).

Include:
- Date (YYYY-MM-DD or month/year)
- Short description (e.g., "Positive Phase 2 data readout", "Partnership announcement", "FDA fast track designation", "Stock jumped 40% on news")
- Type (e.g., clinical, regulatory, financial, partnership)
- Link if available

Prioritize events that impacted stock price.
Sources: company press releases, BioSpace, FierceBiotech, ClinicalTrials.gov, Seeking Alpha, Yahoo Finance, SEC filings.

Output as Markdown bullet list, newest to oldest. If none found, say so.
"""
                        news_response = call_grok(news_prompt)
                        st.markdown(news_response)
        else:
            st.warning("No stocks passed both filters.")

# ────────────────────────────────────────────────
#   Publish Results Button (with full details)
# ────────────────────────────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("📤 Publish Current Results", use_container_width=True):
    filtered_stocks = st.session_state.get("publish_filtered_stocks", [])
    if not filtered_stocks:
        st.sidebar.error("No results to publish — run screening first.")
    else:
        all_results = st.session_state.get("publish_all_results", [])
        start_date_str = start_date.strftime("%Y-%m-%d")

        publish_data = {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trial_window": trial_window,
            "min_impact_factor": min_impact_factor,
            "all_screened": [
                {
                    "company": s.get("company", ""),
                    "ticker": s.get("ticker", ""),
                    "exchange": s.get("exchange", ""),
                    "market_cap": s.get("market_cap", ""),
                    "meets_criteria": s.get("meets_criteria", False),
                    "reasons_summary": (s.get("reasons", "N/A")[:200] + "...") if len(s.get("reasons", "")) > 200 else s.get("reasons", "N/A"),
                    "asset_name": s.get("asset_name", "N/A")
                }
                for s in all_results
            ],
            "detailed_results": []
        }

        # Generate and save full details for each qualifier (use cache when available)
        params_for_cache = {
            "trial_window": trial_window,
            "min_impact_factor": min_impact_factor,
        }
        for stock in filtered_stocks:
            catalysts_text = json.dumps(stock.get("near_term_catalysts", []), indent=2)
            screening_evidence = stock.get('poc_reasons', 'No screening data available.')
            asset_prompt = f"""
For {stock['company']}'s {stock.get('asset_name', 'lead asset')}:

Known upcoming catalysts from BPIQ API:
{catalysts_text}

Use this BPIQ data as the primary source for catalyst dates. Include presentations at upcoming conferences in the near-term catalysts section.

**IMPORTANT — Ground truth from screening filter (you MUST use this):**
During the screening/filtering step, the following papers and evidence were identified for this stock.
You MUST include ALL of these papers in your "Key Academic Papers" and "Animal Studies" sections below.
Do NOT omit, contradict, or replace any of them. You may add additional papers you find, but these are the mandatory baseline:
---
{screening_evidence}
---

Then research and output as structured Markdown:
- Date of near-term clinical trial results, data readouts, presentations at upcoming conferences, regulatory milestones (e.g., PDUFA dates, BLA decisions), or key events (e.g., interim analyses, conference presentations) within 0-6 months that could significantly impact stock price (e.g., events analysts highlight as catalysts). Primarily focus on company press releases, earnings transcripts from the last four quarters leading up to {start_date_str}, and stock commentary websites (e.g., Seeking Alpha, Yahoo Finance).
- Modality
- First in class candidate (assess yes/no with reason)
- Best in class candidate (assess yes/no with reason)
- Key academic papers: Include ALL papers from the screening ground truth above FIRST, then add any additional papers that are (a) specific to this drug candidate, or (b) relate to the theory / proof-of-concept (POC) behind the lead drug candidate, published in journals with Clarivate Impact Factor > {min_impact_factor}. No limit on number of papers. For each paper: title, authors, journal, year, Impact Factor, citation count (if available), links (DOI/PubMed), and a **short summary of the abstract**.
- Animal studies: Include the animal study paper(s) from the screening ground truth above. Cite at least one original paper (not a review) on the lead drug that is an animal study, published in a journal with Clarivate impact factor ≥ 10. **Assess how good the animal studies were and the quality of the animal studies** (study design, endpoints, reproducibility, relevance to human disease).
- Worldwide incidence of disease
- Current standard of care: description, cost, frequency, insurance coverage, administration, dosage, side effects, outcomes
- Route of Administration and Dosing
- Proposed benefit (assess if significant)
- Target population size
- Potential market size (in dollars)
- Preclinical vs. Phase 1/2 data (assess consistency)
- Consistency with preclinical nonhuman studies
- Outcome measures accepted? (assess)
- Safety profile. **Clearly highlight any side effects, adverse events, or deaths** reported in trials or preclinical studies.
- Alternative approaches: descriptions, stages, and **upcoming catalysts for competing alternative approaches** — check the BPIQ database and public sources/filings of the company sponsoring each alternative approach asset; highlight near-term events (0-6 months) that could impact this stock.
- Manufacturing and Scalability (assess)
- Potential for insurance coverage in US/other
- Other indications
- Investigator background/CV/track record
Include links to sources in [link](url) format.
Use tools like web search, X search, PubMed, Google Scholar, Clarivate for IFs, browse pages for accuracy. Prioritize company press releases, earnings transcripts from the last four quarters before {start_date_str}, and stock commentary websites for catalyst supplements.
"""
            asset_md = get_cached_or_compute(
                stock["ticker"],
                params_for_cache,
                lambda ap=asset_prompt: call_grok(ap),
                "asset",
            )

            # Verify catalysts with Gemini/DeepSeek for published output
            catalyst_section_pub = ""
            if "### Near-Term Catalysts" in asset_md:
                parts = asset_md.split("### Near-Term Catalysts", 1)
                if len(parts) > 1:
                    rest = parts[1]
                    next_section = rest.find("\n### ")
                    catalyst_section_pub = rest[:next_section] if next_section > 0 else rest[:2000]
            if catalyst_section_pub.strip():
                bpiq_cats_pub = json.dumps(stock.get('near_term_catalysts', []), indent=2)
                cat_verify_pub = verify_catalysts_with_llms(
                    stock['company'], stock['ticker'],
                    stock.get('asset_name', 'lead asset'),
                    catalyst_section_pub, start_date_str, bpiq_cats_pub
                )
                if cat_verify_pub.get("verified_md"):
                    asset_md += f"\n\n---\n#### Catalyst Verification ({cat_verify_pub.get('source', 'LLM')})\n{cat_verify_pub['verified_md']}"

            shareholders_prompt = f"""
For {stock['company']} ({stock['ticker']}):
Research the latest disclosed major shareholders and institutional investors (from 13F filings, SEC EDGAR, WhaleWisdom, company investor presentations, or recent cap table data).

Include:
- Top 8–12 holders with approximate % ownership and any recent stake changes.
- **Highlight in bold** and explicitly note any matches from this list of key biotech/specialty investors: Orbimed, Arch Ventures, Lily Asia Ventures, Decheng, NEA (New Enterprise Associates), Samsara, Vivo Capital, RA Capital, Third Rock Ventures, Atlas Venture, Novo Holdings, Flagship Pioneering, Sofinnova Partners, Baker Brothers.
- For the highlighted VCs/specialty investors (and any other top holders), include recent purchase/sale activity over the last 1–4 quarters (e.g., Q4 2025, Q3 2025, etc.). Specify shares bought/sold, % change, approximate date/quarter, and source (13F or Form 4).
- If data is limited (early-stage, small float, pre-IPO), state that clearly and mention known venture rounds or lead investors from press releases.

Output as structured Markdown with:
- Bullet list of top holders
- Bold highlights for key VCs/specialty names
- Sub-bullets for recent trade activity when available
Include source links in [link](url) format (SEC filings, WhaleWisdom, etc.).
Use web search, browse SEC EDGAR/WhaleWisdom/company IR pages, recent 13F/Form 4 filings for accuracy.
"""
            shareholders_md = get_cached_or_compute(
                stock["ticker"],
                params_for_cache,
                lambda sp=shareholders_prompt: call_grok(sp),
                "shareholders",
            )

            bpiq_pipeline_pub = get_all_bpiq_programs_for_ticker(st.session_state.bpiq_drugs or [], stock["ticker"])

            # Build BPIQ pipeline summary for published JSON
            pipeline_records = []
            for prog in bpiq_pipeline_pub:
                pipeline_records.append({
                    "drug_indication": f"{prog['drug_name']} — {prog['indications_text']}",
                    "stage_event": prog["stage_event_label"],
                    "catalyst_date": prog["catalyst_date_text"] or prog["catalyst_date"] or "TBD",
                    "note": prog["note"],
                    "source": prog["catalyst_source"],
                })

            # Ask Grok only for peak sales + probabilities
            if bpiq_pipeline_pub:
                drug_list_for_grok = "\n".join([
                    f"- {p['drug_name']} ({p['indications_text']}) — {p['stage_event_label']}"
                    for p in bpiq_pipeline_pub
                ])
                stock_prompt = f"""
For {stock['company']} ({stock['ticker']}), estimate the following for each pipeline asset listed below.
Output a **Markdown table** with columns: **Drug & Indication**, **Estimated Peak Annual Revenue ($)**.
Rank from most mature to least mature (order given).

Pipeline assets:
{drug_list_for_grok}

After the table, also provide:
- Analyst probability for positive trial results (for the lead asset)
- Market-implied probability for positive trial results
- Stock price impact if lead asset fails (considering full pipeline)
Include links to sources (analyst reports, consensus estimates, etc.).
Use web search for analyst reports, consensus estimates, etc.
"""
            else:
                stock_prompt = f"""
For {stock['company']} ({stock['ticker']}):
Research and output as structured Markdown:
- A **Markdown table** of ALL known clinical-stage pipeline assets with columns: **Drug & Indication**, **Stage**, **Upcoming Catalyst**, **Estimated Peak Annual Revenue ($)**. Rank most mature first.
- Analyst probability for positive trial results
- Market-implied probability for positive trial results
- Stock price impact if asset fails (considering pipeline)
Include links to sources.
Use web search for analyst reports, etc.
"""

            stock_md = get_cached_or_compute(
                stock["ticker"],
                params_for_cache,
                lambda sp=stock_prompt: call_grok(sp),
                "stock_relative",
            )

            publish_data["detailed_results"].append({
                "ticker": stock.get("ticker", ""),
                "company": stock.get("company", ""),
                "asset_name": stock.get("asset_name", ""),
                "meets_criteria": stock.get("meets_criteria", False),
                "reasons": stock.get("reasons", ""),
                "asset_details_md": asset_md,
                "shareholders_md": shareholders_md,
                "stock_relative_md": stock_md,
                "bpiq_pipeline": pipeline_records,
                "analyst_consensus": {
                    "current_price": stock.get("current_price"),
                    "price_source": stock.get("price_source", "Unknown"),
                    "price_ref_date": stock.get("price_ref_date", "N/A"),
                    "avg_price_target": stock.get("avg_price_target"),
                    "upside_pct": stock.get("upside_pct"),
                    "analyst_consensus_rating": stock.get("analyst_consensus_rating", ""),
                    "num_analysts": stock.get("num_analysts"),
                    "price_target_high": stock.get("price_target_high"),
                    "price_target_low": stock.get("price_target_low"),
                },
            })

        with open("published_results.json", "w", encoding="utf-8") as f:
            json.dump(publish_data, f, indent=2, ensure_ascii=False)

        st.sidebar.success(f"Published {len(filtered_stocks)} detailed results!")
        st.sidebar.info("File saved: published_results.json\nRun published_app.py to view.")

# ────────────────────────────────────────────────
#   Cache Controls (Grok descriptions)
# ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Cache Controls")

if st.sidebar.button("Clear All LLM Cache"):
    grok_cache.clear()
    verify_cache.clear()
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    if VERIFY_CACHE_FILE.exists():
        VERIFY_CACHE_FILE.unlink()
    st.sidebar.success("All LLM caches cleared (Grok, Gemini, DeepSeek)! Next run will regenerate everything.")

if st.sidebar.button("Refresh All Descriptions (Force Recompute)"):
    grok_cache.clear()
    verify_cache.clear()
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    if VERIFY_CACHE_FILE.exists():
        VERIFY_CACHE_FILE.unlink()
    st.sidebar.success("All caches cleared — descriptions and verifications will refresh on next screening.")
