"""
Spam Email AI — Streamlit UI

Modes:
  - Direct mode (default): imports predictor directly, no API needed.
  - API mode (USE_API=true): calls the FastAPI backend.

Set USE_API=true and API_URL=http://localhost:8000 to use API mode.
"""

import os
import sys
import requests
import streamlit as st
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

API_URL = os.environ.get("API_URL", "http://localhost:8000")
USE_API = os.environ.get("USE_API", "false").lower() == "true"

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spam Email AI",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .spam-badge   { background:#ff4b4b; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1.1rem; }
    .ham-badge    { background:#21c55d; color:white; padding:6px 16px; border-radius:20px; font-weight:700; font-size:1.1rem; }
    .word-spam    { background:#ffd6d6; color:#b91c1c; padding:3px 10px; border-radius:12px; margin:3px; display:inline-block; font-size:0.85rem; }
    .word-ham     { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:12px; margin:3px; display:inline-block; font-size:0.85rem; }
    .metric-card  { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Predictor Setup (Direct Mode) ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_predictor():
    """Load predictor directly (no API). Cached across sessions."""
    import yaml
    from src.model.predictor import SpamPredictor
    from src.data.preprocessor import TextPreprocessor

    config_path = ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    preprocessor = TextPreprocessor(
        short_forms=config["preprocessing"].get("short_forms"),
        max_input_length=config["preprocessing"].get("max_input_length", 5000),
    )
    predictor = SpamPredictor(
        model_dir=str(ROOT / config["model"]["save_dir"]),
        preprocessor=preprocessor,
    )
    predictor.load()
    return predictor


def predict_direct(message: str) -> dict:
    predictor = load_predictor()
    result = predictor.predict(message)
    return result.to_dict()


def predict_batch_direct(messages: list[str]) -> list[dict]:
    predictor = load_predictor()
    results = predictor.predict_batch(messages)
    return [r.to_dict() for r in results]


# ─── API Mode Calls ───────────────────────────────────────────────────────────

def predict_api(message: str) -> dict:
    resp = requests.post(f"{API_URL}/api/v1/predict", json={"message": message}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def predict_batch_api(messages: list[str]) -> list[dict]:
    resp = requests.post(f"{API_URL}/api/v1/predict/batch", json={"messages": messages}, timeout=30)
    resp.raise_for_status()
    return resp.json()["results"]


def predict(message: str) -> dict:
    return predict_api(message) if USE_API else predict_direct(message)


def predict_batch(messages: list[str]) -> list[dict]:
    return predict_batch_api(messages) if USE_API else predict_batch_direct(messages)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/spam.png", width=64)
    st.title("Spam Email AI")
    st.caption("Powered by Machine Learning")
    st.divider()

    mode = st.radio("Mode", ["Single Message", "Batch Check"], index=0)
    st.divider()

    if USE_API:
        try:
            health = requests.get(f"{API_URL}/api/v1/health", timeout=3).json()
            st.success(f"API Online | Model: {health.get('model_name', '?')}")
        except Exception:
            st.error("API Offline")
    else:
        st.info("Running in Direct Mode")

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "This tool detects spam using a trained ML model. "
        "It highlights the words that most contributed to the result."
    )


# ─── Session State ────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []


# ─── Single Message Mode ──────────────────────────────────────────────────────

def render_result(result: dict, message: str = ""):
    is_spam = result["is_spam"]
    spam_prob = result["spam_probability"]
    ham_prob = result["ham_probability"]

    col1, col2, col3 = st.columns(3)
    with col1:
        badge = "spam-badge" if is_spam else "ham-badge"
        label = "SPAM" if is_spam else "NOT SPAM"
        st.markdown(f'<div class="metric-card"><span class="{badge}">{label}</span></div>', unsafe_allow_html=True)
    with col2:
        st.metric("Spam Probability", f"{spam_prob:.1%}")
    with col3:
        st.metric("Ham Probability", f"{ham_prob:.1%}")

    # Probability bar
    st.progress(spam_prob, text=f"Spam confidence: {spam_prob:.1%}")

    # Word contributions
    col_spam, col_ham = st.columns(2)
    with col_spam:
        st.markdown("**Top spam-contributing words**")
        if result["top_spam_words"]:
            words_html = " ".join(f'<span class="word-spam">{w}</span>' for w in result["top_spam_words"])
            st.markdown(words_html, unsafe_allow_html=True)
        else:
            st.caption("None detected")
    with col_ham:
        st.markdown("**Top ham-contributing words**")
        if result["top_ham_words"]:
            words_html = " ".join(f'<span class="word-ham">{w}</span>' for w in result["top_ham_words"])
            st.markdown(words_html, unsafe_allow_html=True)
        else:
            st.caption("None detected")

    st.caption(f"Model: {result['model_name']}")


if mode == "Single Message":
    st.header("📩 Single Message Check")

    user_input = st.text_area(
        "Enter your message:",
        height=150,
        placeholder="Paste an email or SMS message here...",
        max_chars=5000,
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        check = st.button("Check", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    if check:
        if not user_input.strip():
            st.warning("Please enter a message.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    result = predict(user_input)
                    st.session_state.history.insert(
                        0, {"message": user_input[:80] + "..." if len(user_input) > 80 else user_input, "result": result}
                    )
                    render_result(result, user_input)
                except Exception as e:
                    st.error(f"Error: {e}")

    # History
    if st.session_state.history:
        st.divider()
        st.subheader("Recent Checks")
        for i, item in enumerate(st.session_state.history[:10]):
            r = item["result"]
            icon = "🔴" if r["is_spam"] else "🟢"
            with st.expander(f"{icon} {item['message']}"):
                render_result(r)


# ─── Batch Mode ───────────────────────────────────────────────────────────────

elif mode == "Batch Check":
    st.header("📋 Batch Message Check")
    st.caption("Enter one message per line (max 100 messages).")

    batch_input = st.text_area(
        "Messages (one per line):",
        height=250,
        placeholder="Message 1\nMessage 2\nMessage 3",
    )

    uploaded = st.file_uploader("Or upload a .txt file (one message per line)", type=["txt"])

    if uploaded:
        batch_input = uploaded.read().decode("utf-8")

    if st.button("Check All", type="primary"):
        messages = [m.strip() for m in batch_input.strip().splitlines() if m.strip()]
        if not messages:
            st.warning("No messages found.")
        elif len(messages) > 100:
            st.error("Max 100 messages per batch.")
        else:
            with st.spinner(f"Analyzing {len(messages)} messages..."):
                try:
                    results = predict_batch(messages)

                    spam_count = sum(1 for r in results if r["is_spam"])
                    ham_count = len(results) - spam_count

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total", len(results))
                    col2.metric("Spam", spam_count)
                    col3.metric("Ham", ham_count)

                    st.divider()

                    import pandas as pd
                    rows = []
                    for msg, r in zip(messages, results):
                        rows.append({
                            "Message": msg[:100] + ("..." if len(msg) > 100 else ""),
                            "Label": r["label"].upper(),
                            "Spam Prob": f"{r['spam_probability']:.1%}",
                            "Top Spam Words": ", ".join(r["top_spam_words"][:5]),
                        })

                    df = pd.DataFrame(rows)

                    def highlight_spam(row):
                        if row["Label"] == "SPAM":
                            return ["background-color: #fef2f2"] * len(row)
                        return ["background-color: #f0fdf4"] * len(row)

                    st.dataframe(df.style.apply(highlight_spam, axis=1), use_container_width=True)

                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results (CSV)",
                        data=csv,
                        file_name="spam_results.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
