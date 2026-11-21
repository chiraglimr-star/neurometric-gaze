from pathlib import Path

code = r'''"""
BEYOND SILENCE 1.0
A research and public-health media communication platform.

Run:
    pip install -r requirements.txt
    streamlit run beyond_silence.py

Research safeguards:
- This is not a diagnostic, screening, or clinical decision-support tool.
- It analyzes research datasets at participant level.
- If an official split column is supplied, the held-out test split is not used
  for cross-validation or model selection.
- If no official split is supplied, stratified cross-validation is reported
  as internal validation only.
- Do not upload identifiable or sensitive personal data unless you have the
  legal authority, consent, and ethics approval to do so.
"""

from __future__ import annotations

import io
import json
import re
import hashlib
import textwrap
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.utils import resample

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy.stats import spearmanr, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------
# PAGE CONFIG + DESIGN
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="BEYOND SILENCE | Research & Media Lab",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --ink: #17243A;
  --muted: #718096;
  --line: #E7ECF2;
  --blue: #315C91;
  --teal: #2F827B;
}
.stApp { background: #F5F7FA; }
.block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #101C2D; }
[data-testid="stSidebar"] * { color: #EAF0F7 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 0.18rem 0; }
.hero {
  background: linear-gradient(120deg, #13243A 0%, #24476A 60%, #2D7776 100%);
  border-radius: 22px; padding: 2.2rem 2.4rem; color: white;
  margin-bottom: 1.2rem;
}
.hero-kicker { color: #BBD7E1; letter-spacing: .16em; font-size: .72rem; font-weight: 700; }
.hero-title { color: white; font-size: 2.55rem; line-height: 1.1; font-weight: 750; margin: .55rem 0; }
.hero-sub { color: #E1EBF2; max-width: 780px; font-size: 1.02rem; }
.section-kicker { color: #397B82; font-size: .72rem; font-weight: 800; letter-spacing: .14em; }
.soft-card {
  background: white; border: 1px solid #E7ECF2; border-radius: 16px;
  padding: 1.1rem 1.2rem; height: 100%;
}
.soft-card h3 { margin-top: .1rem; color: #17243A; }
.small-muted { color: #718096; font-size: .88rem; }
.metric-card {
  background: white; border: 1px solid #E7ECF2; border-radius: 14px;
  padding: .9rem 1rem;
}
.metric-label { color: #718096; font-size: .78rem; }
.metric-value { color: #17243A; font-size: 1.7rem; font-weight: 750; }
div[data-testid="stMetric"] {
  background: white; border: 1px solid #E7ECF2; padding: 14px 16px;
  border-radius: 14px;
}
div.stButton > button, div.stDownloadButton > button {
  border-radius: 10px; font-weight: 650; min-height: 2.7rem;
}
hr { border-color: #E7ECF2; }
footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# CONSTANTS + SESSION STATE
# ---------------------------------------------------------------------

APP_VERSION = "1.0.0"
PROJECT_NAME = "BEYOND SILENCE"
DISCLAIMER = (
    "Research and educational use only. This platform does not diagnose, "
    "screen, predict an individual's mental-health status, or provide "
    "clinical advice."
)

if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "evidence_items" not in st.session_state:
    st.session_state.evidence_items = []
if "impact_data" not in st.session_state:
    st.session_state.impact_data = None
if "story_draft" not in st.session_state:
    st.session_state.story_draft = ""


# ---------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(columns, candidates):
    mapping = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in mapping:
            return mapping[key]
    return None


def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower(), flags=re.UNICODE)


def participant_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transcript rows to one row per participant."""
    rows = []
    for participant_id, group in df.groupby("participant_id", dropna=False, sort=False):
        texts = [clean_text(x) for x in group["text"] if clean_text(x)]
        labels = pd.to_numeric(group["label"], errors="coerce").dropna().unique()
        scores = (
            pd.to_numeric(group["score"], errors="coerce").dropna().unique()
            if "score" in group else []
        )
        splits = (
            group["split"].dropna().astype(str).str.strip().str.lower().unique()
            if "split" in group else []
        )
        rows.append({
            "participant_id": str(participant_id),
            "text": " ".join(texts),
            "label": int(labels[0]) if len(labels) == 1 else np.nan,
            "score": float(scores[0]) if len(scores) == 1 else np.nan,
            "split": str(splits[0]) if len(splits) == 1 else (
                "conflict" if len(splits) > 1 else ""
            ),
            "source_rows": len(group),
        })
    result = pd.DataFrame(rows)
    result["text"] = result["text"].map(clean_text)
    result["token_count"] = result["text"].map(lambda x: len(tokenize(x)))
    result["character_count"] = result["text"].str.len()
    return result


def normalize_research_csv(df: pd.DataFrame) -> pd.DataFrame:
    id_col = find_column(df.columns, [
        "participant_id", "participant", "subject_id", "subject", "id"
    ])
    text_col = find_column(df.columns, [
        "text", "transcript", "utterance", "narrative", "content"
    ])
    label_col = find_column(df.columns, [
        "phq8_binary", "phq_8_binary", "label", "target", "class"
    ])
    score_col = find_column(df.columns, [
        "phq8_score", "phq_8_score", "score"
    ])
    split_col = find_column(df.columns, [
        "split", "set", "partition", "subset"
    ])

    if text_col is None:
        raise ValueError("Could not identify a text/transcript column.")
    if label_col is None:
        raise ValueError(
            "Could not identify a binary research label column. "
            "Expected phq8_binary or label."
        )

    out = pd.DataFrame()
    out["participant_id"] = (
        df[id_col].astype(str) if id_col else
        [f"participant_{i+1}" for i in range(len(df))]
    )
    out["text"] = df[text_col].map(clean_text)
    out["label"] = pd.to_numeric(df[label_col], errors="coerce")
    if score_col:
        out["score"] = pd.to_numeric(df[score_col], errors="coerce")
    if split_col:
        out["split"] = df[split_col].astype(str).str.strip().str.lower()
    return out


def load_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("Unable to read CSV: " + " | ".join(errors[-2:]))


def make_model():
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=10000,
        sublinear_tf=True,
        lowercase=True,
    )
    features = FeatureUnion([("word", word), ("character", char)])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )),
    ])


def bootstrap_auc_ci(y_true, probabilities, repetitions=1000, seed=42):
    """Participant-level bootstrap CI over fixed out-of-fold predictions.

    This interval quantifies sampling variability conditional on the generated
    out-of-fold predictions; it does not fully capture model-training
    uncertainty. It is not a substitute for external validation.
    """
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(repetitions):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], p[idx]))
    if len(values) < 50:
        return None, None
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def evaluate_cv(data: pd.DataFrame, folds_requested: int = 5):
    y = data["label"].astype(int).to_numpy()
    X = data["text"].astype(str).to_numpy()
    counts = pd.Series(y).value_counts()
    if len(counts) != 2:
        raise ValueError("Both binary classes (0 and 1) are required.")
    n_splits = min(folds_requested, int(counts.min()))
    if n_splits < 2:
        raise ValueError("At least two participants in each class are required.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model = make_model()
    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)

    dummy = DummyClassifier(strategy="prior")
    baseline_preds = cross_val_predict(dummy, X.reshape(-1, 1), y, cv=cv, method="predict")

    auc = roc_auc_score(y, probs)
    pr_auc = average_precision_score(y, probs)
    ci_low, ci_high = bootstrap_auc_ci(y, probs)

    metrics = {
        "Participants": int(len(y)),
        "CV folds": int(n_splits),
        "Accuracy": float(accuracy_score(y, preds)),
        "Balanced accuracy": float(balanced_accuracy_score(y, preds)),
        "Macro F1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "ROC AUC": float(auc),
        "PR AUC": float(pr_auc),
        "Brier score": float(brier_score_loss(y, probs)),
        "Baseline accuracy": float(accuracy_score(y, baseline_preds)),
        "Baseline macro F1": float(
            f1_score(y, baseline_preds, average="macro", zero_division=0)
        ),
        "ROC AUC bootstrap CI lower": ci_low,
        "ROC AUC bootstrap CI upper": ci_high,
    }
    matrix = confusion_matrix(y, preds, labels=[0, 1])
    report = classification_report(
        y, preds, labels=[0, 1], output_dict=True, zero_division=0
    )
    predictions = pd.DataFrame({
        "participant_id": data["participant_id"].astype(str).values,
        "observed_research_label": y,
        "out_of_fold_probability": probs,
        "out_of_fold_class": preds,
    })
    return {
        "metrics": metrics,
        "matrix": matrix,
        "report": report,
        "predictions": predictions,
        "y": y,
        "probabilities": probs,
        "predictions_array": preds,
        "validation": "Internal stratified cross-validation",
    }


def evaluate_official_split(data: pd.DataFrame, folds_requested: int = 5):
    """Train on recognized train/dev rows; evaluate once on official test rows."""
    split = data["split"].astype(str).str.lower().str.strip()
    train_mask = split.isin(["train", "training", "dev", "development", "validation", "valid"])
    test_mask = split.isin(["test", "testing"])
    train = data.loc[train_mask].copy()
    test = data.loc[test_mask].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Official split detected, but both train/dev and test records "
            "could not be identified. Use split values such as train, dev, test."
        )
    if train["label"].nunique() != 2 or test["label"].nunique() != 2:
        raise ValueError("Train/dev and test must each contain both classes.")

    counts = train["label"].value_counts()
    n_splits = min(folds_requested, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Insufficient train/dev class counts for CV.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_y = train["label"].astype(int).to_numpy()
    train_x = train["text"].astype(str).to_numpy()

    # CV metrics on train/dev are used for internal model assessment.
    cv_probs = cross_val_predict(
        make_model(), train_x, train_y, cv=cv, method="predict_proba"
    )[:, 1]
    cv_preds = (cv_probs >= 0.5).astype(int)

    # Final model is fitted only on train/dev, then evaluated once on test.
    final_model = make_model()
    final_model.fit(train_x, train_y)
    test_x = test["text"].astype(str).to_numpy()
    test_y = test["label"].astype(int).to_numpy()
    test_probs = final_model.predict_proba(test_x)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    test_auc = roc_auc_score(test_y, test_probs)
    ci_low, ci_high = bootstrap_auc_ci(test_y, test_probs)
    metrics = {
        "Train/dev participants": int(len(train)),
        "Held-out test participants": int(len(test)),
        "CV folds (train/dev)": int(n_splits),
        "Test accuracy": float(accuracy_score(test_y, test_preds)),
        "Test balanced accuracy": float(balanced_accuracy_score(test_y, test_preds)),
        "Test macro F1": float(f1_score(test_y, test_preds, average="macro", zero_division=0)),
        "Test ROC AUC": float(test_auc),
        "Test PR AUC": float(average_precision_score(test_y, test_probs)),
        "Test Brier score": float(brier_score_loss(test_y, test_probs)),
        "Test ROC AUC bootstrap CI lower": ci_low,
        "Test ROC AUC bootstrap CI upper": ci_high,
        "Train/dev CV ROC AUC": float(roc_auc_score(train_y, cv_probs)),
    }
    return {
        "metrics": metrics,
        "matrix": confusion_matrix(test_y, test_preds, labels=[0, 1]),
        "report": classification_report(
            test_y, test_preds, labels=[0, 1], output_dict=True, zero_division=0
        ),
        "predictions": pd.DataFrame({
            "participant_id": test["participant_id"].astype(str).values,
            "observed_research_label": test_y,
            "held_out_test_probability": test_probs,
            "held_out_test_class": test_preds,
        }),
        "y": test_y,
        "probabilities": test_probs,
        "predictions_array": test_preds,
        "validation": "Official held-out test evaluation; model fitted on train/dev",
        "final_model": final_model,
    }


def top_model_terms(data: pd.DataFrame, top_n=15):
    """Exploratory model coefficients; not causal or clinical explanations."""
    y = data["label"].astype(int).to_numpy()
    if len(np.unique(y)) != 2 or min(np.bincount(y)) < 2:
        return pd.DataFrame()
    model = make_model()
    model.fit(data["text"].astype(str).to_numpy(), y)
    feature_union = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    names = feature_union.get_feature_names_out()
    coefs = classifier.coef_[0]
    out = pd.DataFrame({"feature": names, "coefficient": coefs})
    out["direction_in_model"] = np.where(out["coefficient"] > 0, "Class 1", "Class 0")
    out["absolute_coefficient"] = out["coefficient"].abs()
    return out.sort_values("absolute_coefficient", ascending=False).head(top_n)


def render_plotly_or_table(fig, fallback_df=None):
    if PLOTLY_AVAILABLE:
        st.plotly_chart(fig, use_container_width=True)
    elif fallback_df is not None:
        st.dataframe(fallback_df, use_container_width=True, hide_index=True)
    else:
        st.info("Install plotly to enable interactive charts.")


def make_pdf_report(title: str, sections: list[tuple[str, str]]) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("Install reportlab to export PDF.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
        topMargin=18*mm, bottomMargin=18*mm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SubheadCustom", parent=styles["Heading2"],
        textColor=colors.HexColor("#245A70"), spaceBefore=12, spaceAfter=6
    ))
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]),
        Spacer(1, 10),
        Paragraph(DISCLAIMER, styles["Italic"]),
        Spacer(1, 12),
    ]
    for heading, body in sections:
        story.append(Paragraph(heading, styles["SubheadCustom"]))
        safe_body = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe_body.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 8))
    doc.build(story)
    return buffer.getvalue()


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------

st.sidebar.markdown("## ◉ BEYOND SILENCE")
st.sidebar.caption("RESEARCH · STORYTELLING · PUBLIC HEALTH")
page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Research Lab",
        "Narrative Explorer",
        "Evidence Observatory",
        "Story Studio",
        "Impact Lab",
        "Ethics & Bias",
        "Research Reports",
    ],
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.markdown("**Research safeguards**")
st.sidebar.caption("No diagnosis. No individual risk scores. No automated clinical decisions.")
st.sidebar.caption(f"Version {APP_VERSION}")


# ---------------------------------------------------------------------
# PAGE: OVERVIEW
# ---------------------------------------------------------------------

if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">MEDIA · MEDICINE · HUMAN STORIES</div>
          <div class="hero-title">BEYOND SILENCE</div>
          <div class="hero-sub">
            Understanding mental health through the stories we tell.
            An interdisciplinary research and media platform connecting
            language, evidence, storytelling and audience impact.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(DISCLAIMER)
    st.markdown("### A research-to-impact workflow")
    cols = st.columns(4)
    cards = [
        ("01", "Research", "Explore language, narratives and communication patterns."),
        ("02", "Evidence", "Organize claims and verify them against credible sources."),
        ("03", "Storytelling", "Translate evidence into accessible media experiences."),
        ("04", "Impact", "Evaluate comprehension, knowledge and audience response."),
    ]
    for col, (num, title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="soft-card"><div class="section-kicker">{num}</div>'
                f'<h3>{title}</h3><div class="small-muted">{body}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Workspace status")
    research_loaded = st.session_state.research_data is not None
    impact_loaded = st.session_state.impact_data is not None
    evidence_count = len(st.session_state.evidence_items)
    a, b, c = st.columns(3)
    a.metric("Research dataset", "Loaded" if research_loaded else "Not loaded")
    b.metric("Evidence records", evidence_count)
    c.metric("Impact dataset", "Loaded" if impact_loaded else "Not loaded")

    st.markdown("### Start here")
    st.markdown(
        """
        1. Upload a de-identified research CSV in **Research Lab**.
        2. Inspect participant-level data quality before analysis.
        3. Explore aggregate language patterns in **Narrative Explorer**.
        4. Add source-backed claims in **Evidence Observatory**.
        5. Create a media concept in **Story Studio**.
        6. Evaluate a consented audience study in **Impact Lab**.
        """
    )
    st.info(
        "Suggested research focus: investigate how mental-health stories are "
        "framed and understood by young adults in Azerbaijan. Treat this as a "
        "research question to test, not as an established finding."
    )


# ---------------------------------------------------------------------
# PAGE: RESEARCH LAB
# ---------------------------------------------------------------------

elif page == "Research Lab":
    st.markdown("## Research Lab")
    st.caption("Participant-level computational text research. Not a clinical tool.")
    st.warning(
        "Upload only de-identified data that you are authorized to use. "
        "Do not upload names, contact details, raw clinical records, or "
        "unconsented personal narratives."
    )

    uploaded = st.file_uploader(
        "Upload normalized research CSV",
        type=["csv"],
        help="Required: text and a binary research label. Recommended: participant_id and split."
    )
    with st.expander("Expected CSV structure"):
        st.code(
            "participant_id,text,phq8_binary,phq8_score,split\n"
            'P001,"De-identified interview text...",0,4,train\n'
            'P002,"Another de-identified narrative...",1,14,train\n'
            'P003,"Held-out example...",0,3,test',
            language="csv",
        )
        st.caption(
            "Accepted text columns include text/transcript/narrative; labels include "
            "phq8_binary/label/target; split values should include train, dev and test."
        )

    if uploaded is not None:
        try:
            raw_df = load_csv(uploaded)
            normalized = normalize_research_csv(raw_df)
            aggregated = participant_aggregate(normalized)
            aggregated["label"] = pd.to_numeric(aggregated["label"], errors="coerce")
            aggregated = aggregated[
                aggregated["label"].isin([0, 1]) &
                aggregated["text"].str.len().gt(0)
            ].copy()
            aggregated["label"] = aggregated["label"].astype(int)
            st.session_state.research_data = aggregated.reset_index(drop=True)
            st.session_state.research_results = None
            st.success(
                f"Loaded {len(aggregated):,} eligible participants. "
                "Rows were aggregated by participant ID."
            )
        except Exception as exc:
            st.error(f"Could not process the file: {exc}")

    data = st.session_state.research_data
    if data is not None:
        st.markdown("### Dataset overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Participants", f"{len(data):,}")
        c2.metric("Class 0", f"{int((data.label == 0).sum()):,}")
        c3.metric("Class 1", f"{int((data.label == 1).sum()):,}")
        c4.metric("Median tokens", f"{data.token_count.median():.0f}")

        st.markdown("### Data quality audit")
        split_counts = (
            data["split"].replace("", "not supplied").value_counts().to_dict()
            if "split" in data else {"not supplied": len(data)}
        )
        audit = pd.DataFrame([
            ("Unique participants", len(data)),
            ("Empty narratives after aggregation", int(data.text.str.strip().eq("").sum())),
            ("Median tokens", float(data.token_count.median())),
            ("Participants with conflicting split labels", int((data.split == "conflict").sum())),
            ("Participants with missing score", int(data.score.isna().sum())),
        ], columns=["Audit item", "Value"])
        st.dataframe(audit, use_container_width=True, hide_index=True)
        st.write("Split distribution:", split_counts)

        min_tokens = st.slider("Minimum participant token count", 0, 500, 20, 10)
        filtered = data.loc[data.token_count >= min_tokens].copy()
        st.caption(f"{len(filtered):,} participants remain after the token threshold.")

        if st.button("Run research evaluation", type="primary", disabled=filtered.empty):
            try:
                has_official_split = (
                    "split" in filtered.columns
                    and filtered["split"].isin(["train", "training", "dev", "development",
                                                "validation", "valid", "test", "testing"]).any()
                )
                if has_official_split:
                    result = evaluate_official_split(filtered)
                else:
                    result = evaluate_cv(filtered)
                st.session_state.research_results = result
                st.session_state.research_analysis_data = filtered
                st.success("Evaluation completed.")
            except Exception as exc:
                st.error(f"Evaluation could not be completed: {exc}")

    result = st.session_state.research_results
    if result is not None:
        st.markdown("### Model evaluation")
        st.info(result["validation"])
        metrics = result["metrics"]
        display_metrics = pd.DataFrame(
            [{"Metric": k, "Value": ("Not available" if v is None else
                                     f"{v:.4f}" if isinstance(v, float) else v)}
             for k, v in metrics.items()]
        )
        st.dataframe(display_metrics, use_container_width=True, hide_index=True)

        cm = result["matrix"]
        cm_df = pd.DataFrame(cm, index=["Observed 0", "Observed 1"],
                             columns=["Predicted 0", "Predicted 1"])
        left, right = st.columns(2)
        with left:
            st.markdown("#### Confusion matrix")
            if PLOTLY_AVAILABLE:
                fig = px.imshow(cm_df, text_auto=True, aspect="auto",
                                color_continuous_scale="Blues")
                render_plotly_or_table(fig)
            else:
                st.dataframe(cm_df)
        with right:
            st.markdown("#### Classification report")
            report_df = pd.DataFrame(result["report"]).T.reset_index()
            report_df = report_df.rename(columns={"index": "Class / metric"})
            st.dataframe(report_df, use_container_width=True)

        st.markdown("#### Prediction-level research output")
        st.caption(
            "Predictions are for research evaluation only. Do not use them to "
            "label, rank, contact, diagnose, or make decisions about individuals."
        )
        st.dataframe(result["predictions"], use_container_width=True, hide_index=True)
        st.download_button(
            "Download evaluation predictions CSV",
            result["predictions"].to_csv(index=False).encode("utf-8"),
            file_name="beyond_silence_research_predictions.csv",
            mime="text/csv",
        )

        st.markdown("#### Exploratory model terms")
        st.caption(
            "Model coefficients describe associations learned by this fitted model. "
            "They are not causal explanations, validated psychological markers, "
            "or evidence that a person experiences a particular emotion."
        )
        if st.button("Calculate exploratory model terms"):
            analysis_data = st.session_state.get("research_analysis_data")
            if analysis_data is not None and "final_model" not in result:
                try:
                    terms = top_model_terms(analysis_data)
                    st.session_state.exploratory_terms = terms
                except Exception as exc:
                    st.error(str(exc))
            elif "final_model" in result:
                st.info("For official split evaluation, terms should be calculated from train/dev only.")
                terms = top_model_terms(
                    st.session_state.research_analysis_data[
                        st.session_state.research_analysis_data.split.isin(
                            ["train", "training", "dev", "development", "validation", "valid"]
                        )
                    ]
                )
                st.session_state.exploratory_terms = terms
        if st.session_state.get("exploratory_terms") is not None:
            st.dataframe(st.session_state.exploratory_terms,
                         use_container_width=True, hide_index=True)

        record = {
            "project": PROJECT_NAME,
            "version": APP_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "validation_design": result["validation"],
            "metrics": metrics,
            "limitations": [
                "Not a diagnostic or screening instrument.",
                "PHQ-8 is a self-reported symptom measure, not a clinical diagnosis.",
                "Internal validation does not establish generalizability.",
                "Text features do not directly measure emotion, silence, gaze, or vocal behavior.",
                "Model coefficients are associative, not causal explanations.",
                "External validation and independent replication are required."
            ],
        }
        st.download_button(
            "Download research record JSON",
            json.dumps(record, indent=2, allow_nan=False, default=str).encode("utf-8"),
            file_name="beyond_silence_research_record.json",
            mime="application/json",
        )
    else:
        st.info("Upload a dataset to activate participant-level research analysis.")


# ---------------------------------------------------------------------
# PAGE: NARRATIVE EXPLORER
# ---------------------------------------------------------------------

elif page == "Narrative Explorer":
    st.markdown("## Narrative Explorer")
    st.caption("Descriptive analysis of language and narrative structure.")
    data = st.session_state.research_data

    if data is None:
        st.info("Load a de-identified dataset in Research Lab first.")
    else:
        min_tokens = st.slider("Minimum tokens for narrative exploration", 0, 500, 20, 10)
        subset = data[data.token_count >= min_tokens].copy()
        a, b, c = st.columns(3)
        a.metric("Participants", len(subset))
        b.metric("Median narrative length", f"{subset.token_count.median():.0f} tokens")
        c.metric("Mean narrative length", f"{subset.token_count.mean():.1f} tokens")

        st.markdown("### Narrative length by research label")
        summary = subset.groupby("label").agg(
            Participants=("participant_id", "count"),
            Mean_tokens=("token_count", "mean"),
            Median_tokens=("token_count", "median"),
            Mean_characters=("character_count", "mean"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if PLOTLY_AVAILABLE:
            fig = px.box(subset, x="label", y="token_count",
                         points="outliers", labels={"label": "Research label",
                                                    "token_count": "Tokens"})
            render_plotly_or_table(fig)

        st.markdown("### Exploratory vocabulary")
        label_choice = st.selectbox("Select research group", ["All", "Class 0", "Class 1"])
        word_limit = st.slider("Number of terms", 10, 50, 25)
        selected = subset
        if label_choice == "Class 0":
            selected = subset[subset.label == 0]
        elif label_choice == "Class 1":
            selected = subset[subset.label == 1]
        stopwords = {
            "the", "and", "that", "this", "with", "have", "from", "they", "them",
            "there", "their", "then", "when", "what", "would", "could", "should",
            "about", "because", "been", "were", "your", "you", "are", "was",
            "for", "but", "not", "can", "all", "our", "out", "just", "into",
            "like", "one", "also", "very", "its", "it's", "had", "has", "his",
            "her", "she", "him", "who", "how", "why", "where", "which",
        }
        from collections import Counter
        words = Counter()
        for txt in selected.text:
            words.update(w for w in tokenize(txt) if len(w) > 2 and w not in stopwords)
        vocab = pd.DataFrame(words.most_common(word_limit), columns=["Term", "Frequency"])
        st.dataframe(vocab, use_container_width=True, hide_index=True)
        if PLOTLY_AVAILABLE and not vocab.empty:
            fig = px.bar(vocab.sort_values("Frequency"), x="Frequency", y="Term",
                         orientation="h")
            render_plotly_or_table(fig)
        st.caption(
            "Word frequency is descriptive, sensitive to topic and interview prompts, "
            "and must not be interpreted as a direct measure of mental state."
        )

        st.markdown("### Optional score association")
        if "score" in subset and subset.score.notna().sum() >= 5 and SCIPY_AVAILABLE:
            corr = subset[["token_count", "score"]].dropna()
            if len(corr) >= 5 and corr.score.nunique() > 1 and corr.token_count.nunique() > 1:
                rho, pval = spearmanr(corr.token_count, corr.score)
                x, y = st.columns(2)
                x.metric("Spearman correlation", f"{rho:.3f}")
                y.metric("P-value (exploratory)", f"{pval:.4f}")
                st.caption("Exploratory association only; not causal evidence.")
        else:
            st.caption("A score column and scipy are required for this optional analysis.")


# ---------------------------------------------------------------------
# PAGE: EVIDENCE OBSERVATORY
# ---------------------------------------------------------------------

elif page == "Evidence Observatory":
    st.markdown("## Evidence Observatory")
    st.caption("A structured workspace for source-backed health communication claims.")
    st.warning(
        "This module does not automatically verify facts. A claim should be "
        "marked as reviewed only after a human checks the cited source."
    )

    with st.form("evidence_form", clear_on_submit=True):
        claim = st.text_area("Claim or message", placeholder="Write one specific health-related claim...")
        source_title = st.text_input("Source title")
        organization = st.text_input("Publisher / organization")
        url = st.text_input("Source URL")
        evidence_type = st.selectbox(
            "Evidence type",
            ["Peer-reviewed research", "Government / public-health agency",
             "Clinical guideline", "Systematic review", "Other"]
        )
        review_status = st.selectbox(
            "Human review status",
            ["Not reviewed", "Reviewed — supports claim",
             "Reviewed — partially supports claim",
             "Reviewed — does not support claim"]
        )
        notes = st.text_area("Reviewer notes / limitations")
        submitted = st.form_submit_button("Add evidence record", type="primary")
        if submitted:
            if not claim.strip() or not source_title.strip():
                st.error("Please provide at least a claim and source title.")
            else:
                st.session_state.evidence_items.append({
                    "claim": claim.strip(),
                    "source_title": source_title.strip(),
                    "organization": organization.strip(),
                    "url": url.strip(),
                    "evidence_type": evidence_type,
                    "review_status": review_status,
                    "notes": notes.strip(),
                    "added_utc": datetime.now(timezone.utc).isoformat(),
                })
                st.success("Evidence record added.")

    if st.session_state.evidence_items:
        evidence_df = pd.DataFrame(st.session_state.evidence_items)
        st.markdown("### Evidence register")
        st.dataframe(evidence_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download evidence register CSV",
            evidence_df.to_csv(index=False).encode("utf-8"),
            file_name="beyond_silence_evidence_register.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download evidence register JSON",
            json.dumps(st.session_state.evidence_items, indent=2).encode("utf-8"),
            file_name="beyond_silence_evidence_register.json",
            mime="application/json",
        )
        if st.button("Clear evidence register"):
            st.session_state.evidence_items = []
            st.rerun()
    else:
        st.info("No evidence records yet. Add claims and their sources above.")


# ---------------------------------------------------------------------
# PAGE: STORY STUDIO
# ---------------------------------------------------------------------

elif page == "Story Studio":
    st.markdown("## Story Studio")
    st.caption("Turn reviewed evidence into accessible, audience-aware media concepts.")
    st.warning(
        "Generated drafts require editorial review and source verification. "
        "Do not fabricate personal testimonials, clinical advice, or statistics."
    )

    audience = st.selectbox(
        "Intended audience",
        ["Young adults in Azerbaijan", "Parents and caregivers",
         "University students", "Educators", "General public"]
    )
    format_choice = st.selectbox(
        "Media format",
        ["Short video (60–90 seconds)", "Podcast episode (5–8 minutes)",
         "Instagram carousel", "Interactive story", "Public information article"]
    )
    objective = st.selectbox(
        "Communication objective",
        ["Improve mental-health literacy",
         "Reduce stigma and encourage respectful conversation",
         "Explain how to find trustworthy information",
         "Encourage help-seeking from qualified services",
         "Correct a specific misconception"]
    )
    tone = st.selectbox(
        "Tone",
        ["Empathetic and non-judgmental", "Clear and educational",
         "Personal and reflective", "Youth-friendly and conversational"]
    )
    evidence_records = st.session_state.evidence_items
    evidence_options = ["No evidence record selected"] + [
        f"{i+1}. {item['source_title']}" for i, item in enumerate(evidence_records)
    ]
    selected_evidence = st.selectbox("Evidence record to build around", evidence_options)

    if st.button("Create story brief", type="primary"):
        selected_item = None
        if selected_evidence != "No evidence record selected":
            idx = evidence_options.index(selected_evidence) - 1
            selected_item = evidence_records[idx]

        source_line = (
            f"Evidence source: {selected_item['source_title']} "
            f"({selected_item['organization']}). URL: {selected_item['url']}. "
            f"Human review status: {selected_item['review_status']}."
            if selected_item else
            "Evidence source: Not yet selected. Do not publish factual claims until sources are added and reviewed."
        )
        claim_line = (
            f"Evidence claim to communicate: {selected_item['claim']}"
            if selected_item else
            "Evidence claim: To be defined and verified before publication."
        )
        brief = f"""# Story brief: {objective}

**Audience:** {audience}
**Format:** {format_choice}
**Tone:** {tone}

## Core message
Write one clear, non-stigmatizing message that supports the objective without diagnosing or labeling individuals.

## Suggested structure
1. **Opening hook:** Begin with a relatable question or everyday communication situation.
2. **Human context:** Explain why the topic matters without inventing a personal story.
3. **Evidence:** Present one carefully qualified, source-backed point in plain language.
4. **Practical takeaway:** Offer a safe, realistic action such as checking a reputable source or speaking with a qualified professional.
5. **Closing:** End with an empathetic, non-judgmental invitation to continue learning.

## Evidence to verify
{claim_line}
{source_line}

## Editorial safeguards
- Do not imply that a language pattern reveals an individual's diagnosis.
- Do not use fear, shame, or sensationalism.
- Do not invent statistics, quotations, or testimonials.
- Distinguish established evidence from interpretation.
- Add local support-service information only after verifying the service, eligibility, and contact details.
- Obtain consent for any real person's story, voice, image, or identifiable details.
"""
        st.session_state.story_draft = brief

    if st.session_state.story_draft:
        st.markdown("### Your story brief")
        st.markdown(st.session_state.story_draft)
        st.download_button(
            "Download story brief (Markdown)",
            st.session_state.story_draft.encode("utf-8"),
            file_name="beyond_silence_story_brief.md",
            mime="text/markdown",
        )


# ---------------------------------------------------------------------
# PAGE: IMPACT LAB
# ---------------------------------------------------------------------

elif page == "Impact Lab":
    st.markdown("## Impact Lab")
    st.caption("Evaluate audience learning and communication outcomes.")
    st.warning(
        "Use only consented, appropriately de-identified research data. "
        "This module is not a clinical outcome assessment."
    )

    st.markdown("### Upload a pre/post study CSV")
    st.code(
        "participant_id,group,pre_score,post_score\n"
        "A001,story,4,7\n"
        "A002,story,5,8\n"
        "B001,comparison,4,5",
        language="csv",
    )
    impact_file = st.file_uploader(
        "Upload impact evaluation CSV",
        type=["csv"],
        key="impact_csv",
    )
    if impact_file is not None:
        try:
            impact_raw = load_csv(impact_file)
            id_col = find_column(impact_raw.columns, ["participant_id", "id", "subject_id"])
            group_col = find_column(impact_raw.columns, ["group", "condition", "arm"])
            pre_col = find_column(impact_raw.columns, ["pre_score", "baseline", "pre"])
            post_col = find_column(impact_raw.columns, ["post_score", "followup", "post"])
            if not all([id_col, pre_col, post_col]):
                raise ValueError("Required columns: participant_id, pre_score, post_score. group is optional.")
            clean = pd.DataFrame({
                "participant_id": impact_raw[id_col].astype(str),
                "pre_score": pd.to_numeric(impact_raw[pre_col], errors="coerce"),
                "post_score": pd.to_numeric(impact_raw[post_col], errors="coerce"),
            })
            clean["group"] = impact_raw[group_col].astype(str) if group_col else "all"
            clean = clean.dropna(subset=["pre_score", "post_score"])
            clean["change"] = clean["post_score"] - clean["pre_score"]
            st.session_state.impact_data = clean
            st.success(f"Loaded {len(clean)} complete paired records.")
        except Exception as exc:
            st.error(f"Could not process impact data: {exc}")

    impact = st.session_state.impact_data
    if impact is not None and not impact.empty:
        st.markdown("### Descriptive outcomes")
        summary = impact.groupby("group").agg(
            Participants=("participant_id", "nunique"),
            Mean_pre=("pre_score", "mean"),
            Mean_post=("post_score", "mean"),
            Mean_change=("change", "mean"),
            Median_change=("change", "median"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if PLOTLY_AVAILABLE:
            plot_df = impact.melt(
                id_vars=["participant_id", "group"],
                value_vars=["pre_score", "post_score"],
                var_name="timepoint", value_name="score"
            )
            fig = px.box(plot_df, x="timepoint", y="score", color="group",
                         points="all", title="Pre/post score distributions")
            render_plotly_or_table(fig)

        if SCIPY_AVAILABLE:
            st.markdown("### Paired comparisons (exploratory)")
            from scipy.stats import wilcoxon
            rows = []
            for group_name, group_df in impact.groupby("group"):
                differences = group_df["post_score"] - group_df["pre_score"]
                if len(differences) >= 5 and differences.abs().sum() > 0:
                    stat, p = wilcoxon(group_df["post_score"], group_df["pre_score"])
                    rows.append({
                        "Group": group_name,
                        "N paired": len(group_df),
                        "Wilcoxon statistic": float(stat),
                        "P-value": float(p),
                        "Mean change": float(differences.mean()),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("At least five non-identical paired observations per group are needed for this exploratory test.")
        st.caption(
            "Pre/post change alone does not establish that the media intervention caused "
            "the change. Consider a comparison group, randomization where feasible, "
            "validated measures, preregistration, and follow-up measurement."
        )
        st.download_button(
            "Download cleaned impact data",
            impact.to_csv(index=False).encode("utf-8"),
            file_name="beyond_silence_impact_data_cleaned.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a paired pre/post dataset to explore audience outcomes.")


# ---------------------------------------------------------------------
# PAGE: ETHICS & BIAS
# ---------------------------------------------------------------------

elif page == "Ethics & Bias":
    st.markdown("## Ethics & Bias")
    st.caption("A practical governance checklist for responsible health communication research.")
    ethics_items = [
        ("Purpose limitation", "The project has a clearly defined research or educational purpose."),
        ("Legal basis and consent", "Appropriate consent, permissions, and ethics review have been considered."),
        ("Data minimization", "Only variables necessary for the stated research question are collected."),
        ("De-identification", "Direct identifiers and unnecessary quasi-identifiers are removed."),
        ("Dataset provenance", "Dataset source, license, permitted uses, and documentation are recorded."),
        ("Split integrity", "Official train/dev/test partitions are preserved where supplied."),
        ("No clinical use", "No diagnosis, screening, triage, or individual decision-making is performed."),
        ("Human review", "Health claims and media outputs are reviewed by qualified humans."),
        ("Source transparency", "Public-facing factual claims are traceable to credible sources."),
        ("Fairness review", "Language, cultural, demographic, and accessibility limitations are documented."),
        ("Security", "Access, storage, retention, and deletion procedures are defined."),
        ("Participant dignity", "Stories and quotations are used with permission and without sensationalism."),
        ("Limitations", "Uncertainty, sampling bias, and generalizability limits are reported."),
        ("Incident plan", "A process exists for reporting data exposure or harmful outputs."),
    ]
    checked = []
    for title, description in ethics_items:
        value = st.checkbox(f"**{title}** — {description}", key=f"ethics_{title}")
        checked.append(value)
    completed = sum(checked)
    st.progress(completed / len(ethics_items))
    st.metric("Checklist completion", f"{completed}/{len(ethics_items)}")
    if completed == len(ethics_items):
        st.success("All checklist items are marked complete. This is not a substitute for formal ethics approval.")
    else:
        st.info("Unmarked items should be reviewed before collecting or publishing research data.")

    st.markdown("### Model and communication limitations")
    st.markdown(
        """
        - Text is shaped by interview prompts, language, culture, transcription quality,
          education, and context.
        - A model can learn dataset-specific artifacts rather than meaningful constructs.
        - Performance on one dataset does not establish performance in Azerbaijan or
          another population.
        - Fairness cannot be established from an overall accuracy score.
        - A word, phrase, pause, or writing style is not a diagnosis.
        - Health information should be understandable, accessible, non-stigmatizing,
          and linked to credible evidence.
        """
    )


# ---------------------------------------------------------------------
# PAGE: RESEARCH REPORTS
# ---------------------------------------------------------------------

elif page == "Research Reports":
    st.markdown("## Research Reports")
    st.caption("Export a reproducible summary of the current research workspace.")

    sections = [
        ("Project", "BEYOND SILENCE — Research, Storytelling and Public Health Communication."),
        ("Purpose", "An interdisciplinary research prototype exploring language, evidence, storytelling and audience impact."),
        ("Safeguards", DISCLAIMER),
        ("Methodological note",
         "Analyses are exploratory. Predictive associations are not causal findings. "
         "Internal validation is not external validation. Human review is required."),
    ]

    data = st.session_state.research_data
    if data is not None:
        sections.append((
            "Research dataset summary",
            f"Participants: {len(data)}\n"
            f"Class 0: {int((data.label == 0).sum())}\n"
            f"Class 1: {int((data.label == 1).sum())}\n"
            f"Median tokens: {data.token_count.median():.1f}"
        ))
    result = st.session_state.research_results
    if result is not None:
        metrics_text = "\n".join(
            f"{key}: {value}" for key, value in result["metrics"].items()
        )
        sections.append(("Model evaluation", result["validation"] + "\n" + metrics_text))

    sections.append((
        "Evidence register",
        f"Evidence records currently stored in this session: {len(st.session_state.evidence_items)}"
    ))
    impact = st.session_state.impact_data
    if impact is not None:
        sections.append((
            "Impact study",
            f"Paired records: {len(impact)}\n"
            f"Mean change: {impact.change.mean():.3f}"
        ))

    report_text = "\n\n".join(
        f"## {heading}\n{body}" for heading, body in sections
    )
    st.markdown(report_text)
    st.download_button(
        "Download research report (Markdown)",
        report_text.encode("utf-8"),
        file_name="beyond_silence_research_report.md",
        mime="text/markdown",
    )

    if REPORTLAB_AVAILABLE:
        try:
            pdf_bytes = make_pdf_report("BEYOND SILENCE — Research Report", sections)
            st.download_button(
                "Download research report (PDF)",
                pdf_bytes,
                file_name="beyond_silence_research_report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.warning(f"PDF generation unavailable: {exc}")
    else:
        st.caption("Install reportlab to enable PDF export.")

    evidence = st.session_state.evidence_items
    if evidence:
        st.download_button(
            "Download evidence register (JSON)",
            json.dumps(evidence, indent=2).encode("utf-8"),
            file_name="beyond_silence_evidence.json",
            mime="application/json",
        )


# ---------------------------------------------------------------------
# GLOBAL FOOTER
# ---------------------------------------------------------------------

st.divider()
st.caption(
    "BEYOND SILENCE · Research and educational prototype · "
    "No diagnosis, individual risk scoring, or clinical decision support."
)
'''
path = Path("/mnt/data/beyond_silence.py")
path.write_text(code, encoding="utf-8")

requirements = """streamlit>=1.35
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
plotly>=5.18
scipy>=1.10
reportlab>=4.0
"""
req_path = Path("/mnt/data/requirements.txt")
req_path.write_text(requirements, encoding="utf-8")

print(f"Created {path} ({path.stat().st_size:,} bytes)")
print(f"Created {req_path} ({req_path.stat().st_size:,} bytes)")
