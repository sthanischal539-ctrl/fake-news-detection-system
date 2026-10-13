"""Streamlit web interface + evaluation dashboard.

Run from the project root:
    streamlit run src/app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data_loader import get_dataset_info
from src.explain import explain_prediction, top_global_features
from src.predict import get_available_models, predict, validate_input_text

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "⚠️ **This prediction is a probabilistic signal, not proof.** The system analyses "
    "writing style, vocabulary, and linguistic patterns — not factual correctness. "
    "It can be confidently wrong. **Always verify claims against trusted, authoritative "
    "news organizations and primary sources before acting on them.**"
)

SAMPLE_REAL = (
    "BRUSSELS — European Union antitrust regulators on Tuesday approved the proposed "
    "merger between two major telecommunications companies, subject to full compliance "
    "with agreed commitments. Officials said the remedies adequately address competition "
    "concerns in the wholesale mobile market. The decision follows an in-depth phase II "
    "investigation launched earlier this year."
)

SAMPLE_FAKE = (
    "SHOCKING BOMBSHELL: Hidden medical documents exposed by whistleblower reveal that "
    "global elites have been suppressing a 100% natural miracle cure for decades! "
    "Mainstream media is desperately trying to DELETE this report before the public "
    "wakes up to the truth. Share this now before the establishment bans it forever!!!"
)


# --------------------------------------------------------------------------- #
# Sidebar: System Status & Model Availability
# --------------------------------------------------------------------------- #
def render_sidebar():
    st.sidebar.title("📰 System Status")

    available_models = get_available_models()
    info = get_dataset_info()

    st.sidebar.subheader("Active Dataset")
    if info.get("is_synthetic", True):
        st.sidebar.warning("⚠️ Synthetic Demo Data")
        st.sidebar.caption("High accuracy is expected due to deliberate stylistic generation.")
    else:
        st.sidebar.success(f"✓ {info.get('source', 'Real Dataset')}")

    st.sidebar.subheader("Model Status")
    all_models = ["Logistic Regression", "Naive Bayes", "Linear SVM", "DistilBERT"]
    for m in all_models:
        if m in available_models:
            st.sidebar.markdown(f"🟢 **{m}** `Ready`")
        else:
            st.sidebar.markdown(f"⚪ **{m}** `Not trained`")

    if "DistilBERT" not in available_models:
        st.sidebar.caption("To train DistilBERT:\n`python -m src.train --with-transformer`")

    st.sidebar.markdown("---")
    st.sidebar.caption("Fake News Detection System (NLP)\nAcademic Submission & Demo")


# --------------------------------------------------------------------------- #
# Tab 1: Predict
# --------------------------------------------------------------------------- #
def predict_tab():
    st.header("📰 Analyse a News Article")
    st.caption("Paste the full text of an article (or multiple paragraphs) for best linguistic analysis.")

    available = get_available_models()
    if not available:
        st.error("No trained models found! Please train the models first using `python -m src.train`.")
        return

    col_select, col_samples = st.columns([2, 3])
    with col_select:
        model_name = st.selectbox("Select Model for Analysis", list(available.keys()))

    with col_samples:
        st.write("Need sample text?")
        b1, b2, b3 = st.columns(3)
        if b1.button("📋 Sample Real News"):
            st.session_state["article_input"] = SAMPLE_REAL
        if b2.button("⚠️ Sample Fake News"):
            st.session_state["article_input"] = SAMPLE_FAKE
        if b3.button("🔄 Clear"):
            st.session_state["article_input"] = ""

    from pypdf import PdfReader

    uploaded_pdf = st.file_uploader(
        "📄 Upload a news article (PDF)",
        type=["pdf"],
        help="Upload a text-based PDF to extract its article text.",
    )

    if uploaded_pdf is not None:
        try:
            reader = PdfReader(uploaded_pdf)
            extracted_text = "\\n\\n".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()

            if extracted_text:
                st.session_state["article_input"] = extracted_text
                st.success("PDF text extracted successfully.")
                st.text_area(
                    "Extracted PDF Text",
                    value=extracted_text,
                    height=200,
                    disabled=True,
                )
            else:
                st.warning(
                    "No selectable text was found. This PDF may be scanned "
                    "and require OCR."
                )
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")

    default_text = st.session_state.get("article_input", "")
    text = st.text_area(
        "Article Content",
        value=default_text,
        height=240,
        placeholder="Paste full article text here (minimum 20 characters)...",
    )

    if st.button("🔎 Analyse Article", type="primary"):
        is_valid, validation_msg, sanitized_text = validate_input_text(text)
        if not is_valid:
            st.warning(validation_msg)
            return

        with st.spinner(f"Analysing with {model_name}..."):
            try:
                res = predict(sanitized_text, model_name=model_name)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        label = res["label"]
        confidence = res["confidence"]
        prob_real = res["probabilities"]["REAL"]
        prob_fake = res["probabilities"]["FAKE"]

        st.markdown("---")
        st.subheader("Analysis Results")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if label == "FAKE":
                st.error(f"### Predicted Class: **{label}**")
                st.caption(f"Model identified sensational or unverified stylistic patterns ({model_name}).")
            else:
                st.success(f"### Predicted Class: **{label}**")
                st.caption(f"Model identified measured or conventional reporting style ({model_name}).")

        with res_col2:
            st.metric("Primary Confidence", f"{confidence * 100:.1f}%")
            st.progress(confidence)

        st.markdown("**Estimated Class Probabilities:**")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.info(f"🟢 **REAL:** {prob_real * 100:.1f}%")
        with p_col2:
            st.warning(f"🔴 **FAKE:** {prob_fake * 100:.1f}%")

        if "warning" in res and res["warning"]:
            st.info(f"ℹ️ {res['warning']}")

        # ---- Explainability (for classical models) ----------------------- #
        if model_name in config.CLASSICAL_MODELS and available.get(model_name):
            st.subheader("🔍 Linguistic Attribution (LIME)")
            st.caption("Words pushing the model towards FAKE (positive) or REAL (negative):")

            with st.spinner("Generating LIME word-level explanation..."):
                try:
                    exp = explain_prediction(sanitized_text, model_path=available[model_name], num_features=10)
                    ww = pd.DataFrame(exp["word_weights"], columns=["Word", "Weight"])
                    ww["Direction"] = np.where(ww["Weight"] > 0, "Pushes toward FAKE", "Pushes toward REAL")
                    ww["Magnitude"] = ww["Weight"].abs()
                    ww = ww.sort_values(by="Magnitude", ascending=False)

                    e_col1, e_col2 = st.columns([1, 1])
                    with e_col1:
                        st.dataframe(
                            ww[["Word", "Direction", "Weight"]],
                            hide_index=True,
                            use_container_width=True,
                        )
                    with e_col2:
                        chart_data = ww.set_index("Word")["Weight"].sort_values()
                        st.bar_chart(chart_data)

                except Exception as exc:
                    st.info(f"LIME explanation unavailable: {exc}")
        elif model_name == "DistilBERT":
            st.info(
                "💡 **DistilBERT Transformer Analysis**: Prediction was computed using deep subword "
                "contextual embeddings from fine-tuned DistilBERT sequence classification layers."
            )

        st.warning(DISCLAIMER)


# --------------------------------------------------------------------------- #
# Tab 2: Model Comparison Dashboard
# --------------------------------------------------------------------------- #
def dashboard_tab():
    st.header("📊 Model Comparison Dashboard")

    if not config.METRICS_JSON.exists():
        st.warning("No evaluation metrics found on disk. Run `python -m src.evaluate` to generate results.")
        return

    with open(config.METRICS_JSON) as f:
        data = json.load(f)

    metadata = data.get("_metadata", {})
    model_results = {k: v for k, v in data.items() if not k.startswith("_")}

    if not model_results:
        st.info("No evaluated models found in metrics.json. Run `python -m src.evaluate`.")
        return

    # Dataset indicator banner
    is_synthetic = metadata.get("is_synthetic", True)
    dataset_name = metadata.get("dataset_source", "Unknown")

    if is_synthetic:
        st.warning(
            f"⚠️ **Results evaluated on:** `{dataset_name}`\n\n"
            "**Notice on 100% metrics:** The synthetic demo dataset was algorithmically generated with "
            "deliberately contrasting vocabulary (clickbait sensationalism vs formal attribution). "
            "These near-perfect scores demonstrate model convergence on synthetic stylistic cues, "
            "**not real-world fake news detection capability**. For an academic benchmark, drop in the "
            "real ISOT dataset (`True.csv` and `Fake.csv`) into `data/raw/` and rerun training."
        )
    else:
        st.success(f"✓ **Results evaluated on genuine dataset:** `{dataset_name}`")

    # Metrics table
    rows = []
    for name, m in model_results.items():
        row = {
            "Model": name,
            "Accuracy": m.get("accuracy", 0.0),
            "Precision": m.get("precision", 0.0),
            "Recall": m.get("recall", 0.0),
            "F1-Score": m.get("f1", 0.0),
        }
        if m.get("roc_auc") is not None:
            row["ROC-AUC"] = m["roc_auc"]
        rows.append(row)

    df_metrics = pd.DataFrame(rows).set_index("Model")

    st.subheader("Held-Out Test Set Performance")
    test_count = metadata.get("test_samples", "held-out")
    st.caption(f"Evaluated on {test_count} independent test records across all models.")

    st.dataframe(
        df_metrics.style.format("{:.4f}").highlight_max(axis=0, color="#2e7d32"),
        use_container_width=True,
    )

    st.subheader("Metric Comparison Chart")
    st.bar_chart(df_metrics[["Accuracy", "Precision", "Recall", "F1-Score"]])

    # Confusion matrices
    st.subheader("Confusion Matrices (Actual vs Predicted)")
    cm_cols = st.columns(min(len(model_results), 4))
    for col, (name, m) in zip(cm_cols, model_results.items()):
        with col:
            slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            fig_path = config.RESULTS_DIR / f"confusion_{slug}.png"
            if fig_path.exists():
                st.image(str(fig_path), caption=f"{name}", use_container_width=True)
            else:
                st.write(f"**{name}**")
                cm = np.array(m.get("confusion_matrix", [[0, 0], [0, 0]]))
                st.table(pd.DataFrame(cm, index=["Actual REAL", "Actual FAKE"], columns=["Pred REAL", "Pred FAKE"]))

    # Detailed Classification Reports
    with st.expander("📋 Detailed Classification Reports (Per-Class Breakdown)"):
        for name, m in model_results.items():
            rep = m.get("classification_report")
            if rep:
                st.write(f"**{name}**")
                df_rep = pd.DataFrame(rep).transpose()
                st.dataframe(df_rep.style.format(precision=4), use_container_width=True)

    # Global Indicator Words (from Logistic Regression)
    if config.LOGREG_PATH.exists() and config.TFIDF_PATH.exists():
        st.subheader("Global Vocabulary Indicators (Logistic Regression Coefficients)")
        st.caption("Top words associated with FAKE vs REAL across the training corpus.")
        try:
            indicators = top_global_features(n=12)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("🔴 **Strongest FAKE Indicators**")
                st.dataframe(
                    pd.DataFrame(indicators["fake_indicators"], columns=["Word", "Coefficient"]),
                    hide_index=True,
                    use_container_width=True,
                )
            with c2:
                st.markdown("🟢 **Strongest REAL Indicators**")
                st.dataframe(
                    pd.DataFrame(indicators["real_indicators"], columns=["Word", "Coefficient"]),
                    hide_index=True,
                    use_container_width=True,
                )
        except Exception as exc:
            st.info(f"Global indicator words unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Tab 3: Dataset & Architecture Information
# --------------------------------------------------------------------------- #
def dataset_tab():
    st.header("📚 Dataset & System Documentation")

    info = get_dataset_info()

    st.subheader("Current Dataset Profile")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{info.get('total_records', 0):,}")
    with col2:
        st.metric("Training Split", f"{info.get('train_records', 0):,}")
    with col3:
        st.metric("Test Split", f"{info.get('test_records', 0):,}")
    with col4:
        st.metric("Label Balance", f"{info.get('real_count', 0):,} Real / {info.get('fake_count', 0):,} Fake")

    st.markdown(f"**Dataset Source:** `{info.get('source', 'Unknown')}`")

    st.markdown("---")
    st.subheader("Academic Note: Synthetic Dataset vs. Real-World Benchmarks")

    st.markdown(
        """
        ### Why the Demonstration Dataset Scores 100%
        The project bundles an auto-generated synthetic dataset (`demo_news.csv`) so that
        the pipeline runs completely out-of-the-box without requiring large manual downloads.

        However, because the generator uses distinguishable vocabulary templates
        (*e.g., clickbait sensationalism with capitalizations for Fake vs. measured passive-voice reporting for Real*),
        classical TF-IDF models effortlessly achieve near-perfect classification.

        > **Crucial Takeaway for Academic Submission:**
        > Near-perfect accuracy on synthetic data demonstrates pipeline correctness and model convergence,
        > **not real-world fake news detection capability**. Real-world misinformation often mimics
        > authoritative journalistic style.

        ### Using the ISOT Fake News Dataset (Recommended)
        For rigorous academic evaluation:
        1. Download the **ISOT Fake News Dataset** (University of Victoria).
        2. Place `True.csv` and `Fake.csv` into the `data/raw/` directory.
        3. Run `python -m src.train --force-reprocess` followed by `python -m src.evaluate`.
        4. The pipeline will automatically detect the ISOT files, combine title and body,
           deduplicate records, and regenerate realistic performance metrics.
        """
    )

    st.markdown("---")
    st.subheader("Architecture & Pipeline")
    st.markdown(
        """
        ```
        Raw News Data (ISOT / Demo)
                    ↓
        Data Cleaning & Deduplication (leakage prevention)
                    ↓
        Stratified Train / Val / Test Split (reproducible seed=42)
                    ↓
        Text Preprocessing (lowercasing, URL/dateline stripping, lemmatization)
                    ↓
        Feature Extraction (TF-IDF 1-2 grams)  /  Tokenization (DistilBERT subwords)
                    ↓
        Models: Logistic Regression | Naive Bayes | Linear SVM | DistilBERT
                    ↓
        Evaluation on Held-Out Test Set (Accuracy, Precision, Recall, F1, ROC-AUC, CM)
                    ↓
        Prediction Pipeline & Interactive Web Interface (Streamlit + LIME)
        ```
        """
    )


# --------------------------------------------------------------------------- #
# Main Layout
# --------------------------------------------------------------------------- #
render_sidebar()

st.title("📰 Fake News Detection System")
st.markdown(
    "An end-to-end NLP system for classifying news articles as **Fake** or **Real**, "
    "comparing classical machine learning and transformer models, and providing "
    "transparent word-level explanations."
)

tab1, tab2, tab3 = st.tabs(["🔎 Predict", "📊 Model Comparison", "📚 Dataset & Overview"])

with tab1:
    predict_tab()

with tab2:
    dashboard_tab()

with tab3:
    dataset_tab()
