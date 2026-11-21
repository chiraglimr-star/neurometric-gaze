from pathlib import Path

app_code = r'''"""
THE NEUROMETRIC GAZE
Clinical Silence Analyzer

Research/educational prototype for exploring associations between
participant interview text and self-reported PHQ-8 measures.

IMPORTANT:
- Not a diagnostic or clinical screening instrument.
- Do not use outputs for decisions about individuals.
- Text features do not directly measure emotion, silence, gaze, or intent.
- Use only data you are authorized to process.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import warnings
import zipfile
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
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="The Neurometric Gaze",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1020; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }
    h1, h2, h3 { letter-spacing: -0.45px; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        padding: 16px;
        border-radius: 13px;
    }
    div[data-testid="stAlert"] { border-radius: 10px; }
    .small-note { color: #aebbd0; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("◉ The Neurometric Gaze")
st.subheader("Clinical Silence Analyzer")
st.caption(
    "An exploratory computational humanities research prototype "
    "examining language patterns in interview narratives alongside "
    "self-reported psychological distress measures."
)

st.warning(
    "Research use only — not a diagnostic or screening instrument. "
    "PHQ-8 scores are self-reported symptom measures, not clinical diagnoses. "
    "Language patterns cannot establish a person's actual emotional state. "
    "Never use these results to make decisions about an individual."
)

with st.expander("Research scope, ethics and methodological limitations"):
    st.markdown(
        """
        **Research question:** Are linguistic characteristics in interview
        narratives associated with self-reported PHQ-8 measures?

        **Interpretation boundaries**
        - PHQ-8 is a self-report symptom measure, not a clinical diagnosis.
        - Lexical patterns are not direct measurements of emotions or mental states.
        - Transcript text alone cannot measure acoustic silence, pauses, prosody,
          facial expression or gaze.
        - DAIC-WOZ interviews use a virtual interviewer and are not equivalent
          to ordinary doctor–patient consultations.
        - Internal cross-validation is not external validation.
        - Predictive association does not establish causality.
        - Performance on one dataset does not establish generalizability to
          another population, language, culture or setting.

        **Ethical position:** Use only data for which you have appropriate
        authorization. Follow the dataset's licence, consent conditions,
        institutional ethics requirements and data-protection rules. Avoid
        exposing participant-level predictions or identifiable text.
        """
    )

# ---------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------

def clean_text(value: Any) -> str:
    """Convert a cell to normalized text, safely handling missing values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_column(name: Any) -> str:
    """Normalize a column label for tolerant matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def locate_column(columns, alternatives):
    """Return the original column name matching one of the aliases."""
    mapping = {normalize_column(col): col for col in columns}
    for alternative in alternatives:
        key = normalize_column(alternative)
        if key in mapping:
            return mapping[key]
    return None


def read_csv_bytes(raw: bytes, filename: str = "data.csv") -> pd.DataFrame:
    """Read CSV bytes using common encodings and provide a useful error."""
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(
        f"Could not read {filename} as CSV. "
        + " | ".join(errors[-2:])
    )


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert common CSV column names to the app's internal schema.

    Required: text/transcript/narrative and PHQ-8 binary label.
    Optional: participant ID, PHQ-8 total score, split/partition.
    Optional fields are always created so downstream code cannot fail
    merely because a column is absent.
    """
    id_col = locate_column(
        df.columns,
        ["participant_id", "participant", "subject_id", "subject", "id"],
    )
    text_col = locate_column(
        df.columns,
        ["text", "transcript", "utterance", "narrative", "speech"],
    )
    label_col = locate_column(
        df.columns,
        ["phq8_binary", "phq_8_binary", "phq8binary", "label", "target"],
    )
    score_col = locate_column(
        df.columns,
        ["phq8_score", "phq_8_score", "phq8", "phq_8", "total_score"],
    )
    split_col = locate_column(
        df.columns,
        ["split", "set", "partition", "subset"],
    )

    if text_col is None:
        raise ValueError(
            "Text column not found. Expected a column named text, transcript "
            "or narrative."
        )
    if label_col is None:
        raise ValueError(
            "PHQ-8 binary label not found. Expected phq8_binary, "
            "phq_8_binary, label or target."
        )

    result = pd.DataFrame(index=df.index)

    if id_col is not None:
        ids = df[id_col].map(clean_text)
        result["participant_id"] = [
            value if value else f"participant_{i}"
            for i, value in enumerate(ids)
        ]
    else:
        result["participant_id"] = [
            f"participant_{i}" for i in range(len(df))
        ]

    result["text"] = df[text_col].map(clean_text)
    result["phq8_binary"] = pd.to_numeric(
        df[label_col], errors="coerce"
    )

    if score_col is not None:
        result["phq8_score"] = pd.to_numeric(
            df[score_col], errors="coerce"
        )
    else:
        result["phq8_score"] = np.nan

    if split_col is not None:
        result["split"] = df[split_col].map(clean_text).str.lower()
    else:
        result["split"] = ""

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------
# DAIC-STYLE ZIP PARSING
# ---------------------------------------------------------------------

def parse_transcript_csv(raw: bytes, filename: str) -> dict:
    """
    Parse one DAIC-style transcript CSV and retain participant speech.

    Expected transcript columns commonly include speaker and value.
    Speaker names vary, so the function searches for participant/patient/
    subject labels. If there is no speaker column, all rows in the text
    column are joined.
    """
    frame = read_csv_bytes(raw, filename)

    speaker_col = locate_column(frame.columns, ["speaker", "role", "speaker_id"])
    text_col = locate_column(
        frame.columns, ["value", "text", "utterance", "transcript"]
    )

    if text_col is None:
        raise ValueError(f"No transcript text column in {filename}")

    if speaker_col is not None:
        speakers = frame[speaker_col].map(clean_text).str.lower()
        participant_mask = speakers.str.contains(
            r"participant|patient|subject|participant_id",
            regex=True,
            na=False,
        )
        participant_rows = frame.loc[participant_mask, text_col]

        # Some transcript exports use numeric speaker codes. If no known
        # participant label is found, do not silently assume the interviewer
        # is the participant; return a clear parsing error.
        if participant_rows.empty:
            raise ValueError(
                f"Could not identify participant speech in {filename}. "
                "Inspect the speaker labels before using this archive."
            )
        text_values = participant_rows
    else:
        text_values = frame[text_col]

    transcript = " ".join(
        cleaned for item in text_values
        if (cleaned := clean_text(item))
    )

    # Typical filenames include a participant number, e.g. 300_TRANSCRIPT.csv.
    match = re.search(r"(\d+)", filename)
    participant_id = match.group(1) if match else filename

    if not transcript:
        raise ValueError(f"Transcript is empty: {filename}")

    return {"participant_id": str(participant_id), "text": transcript}


def parse_daic_zip(raw: bytes) -> pd.DataFrame:
    """
    Read transcript CSVs and a metadata CSV from a DAIC-style ZIP.

    This is a best-effort parser, not a guarantee of compatibility with every
    DAIC-WOZ archive release or folder structure. Always verify IDs and labels
    against the official metadata before analysis.
    """
    transcripts = []
    metadata_frames = []

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded ZIP file is invalid or corrupted.") from exc

    with archive:
        for member in archive.namelist():
            if member.endswith("/") or "__MACOSX" in member:
                continue
            base = member.rsplit("/", 1)[-1]
            if not base.lower().endswith(".csv"):
                continue

            content = archive.read(member)

            if "transcript" in base.lower():
                try:
                    transcripts.append(parse_transcript_csv(content, base))
                except Exception:
                    # Invalid transcript files are skipped, but a count is
                    # displayed later so users can check archive compatibility.
                    continue
            else:
                try:
                    metadata_frames.append(read_csv_bytes(content, base))
                except Exception:
                    continue

    if not transcripts:
        raise ValueError(
            "No readable participant transcript CSV files were found. "
            "The archive should contain CSV filenames including 'TRANSCRIPT'. "
            "If speaker labels differ from participant/patient/subject, "
            "use a normalized CSV instead."
        )

    result = pd.DataFrame(transcripts)
    metadata_found = False

    for metadata in metadata_frames:
        id_col = locate_column(
            metadata.columns,
            ["participant_id", "participant", "subject_id", "id"],
        )
        label_col = locate_column(
            metadata.columns,
            ["phq8_binary", "phq_8_binary", "phq8binary"],
        )
        if id_col is None or label_col is None:
            continue

        meta = pd.DataFrame()
        raw_ids = metadata[id_col].map(clean_text)
        extracted = raw_ids.str.extract(r"(\d+)", expand=False)
        meta["participant_id"] = extracted.where(extracted.notna(), raw_ids)
        meta["phq8_binary"] = pd.to_numeric(
            metadata[label_col], errors="coerce"
        )

        score_col = locate_column(
            metadata.columns, ["phq8_score", "phq_8_score", "phq8", "phq_8"]
        )
        split_col = locate_column(
            metadata.columns, ["split", "partition", "set", "subset"]
        )

        meta["phq8_score"] = (
            pd.to_numeric(metadata[score_col], errors="coerce")
            if score_col is not None else np.nan
        )
        meta["split"] = (
            metadata[split_col].map(clean_text).str.lower()
            if split_col is not None else ""
        )

        # Remove duplicated metadata rows only after checking for conflicts.
        meta = meta.drop_duplicates(subset=["participant_id"], keep="first")
        result = result.merge(
            meta[["participant_id", "phq8_binary", "phq8_score", "split"]],
            on="participant_id",
            how="left",
            validate="many_to_one",
        )
        metadata_found = True
        break

    if not metadata_found:
        result["phq8_binary"] = np.nan
        result["phq8_score"] = np.nan
        result["split"] = ""

    return result


# ---------------------------------------------------------------------
# PARTICIPANT-LEVEL DATA PREPARATION
# ---------------------------------------------------------------------

def aggregate_participants(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple transcript rows per participant.

    A participant receives a label only when exactly one unique non-missing
    binary label is present. Conflicting labels are left missing and excluded.
    """
    required = {"participant_id", "text", "phq8_binary"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "Internal dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    work = frame.copy()
    if "phq8_score" not in work.columns:
        work["phq8_score"] = np.nan
    if "split" not in work.columns:
        work["split"] = ""

    rows = []
    for participant_id, group in work.groupby(
        "participant_id", dropna=False, sort=False
    ):
        texts = [clean_text(value) for value in group["text"]]
        joined_text = " ".join(value for value in texts if value)

        labels = (
            pd.to_numeric(group["phq8_binary"], errors="coerce")
            .dropna()
            .unique()
        )
        label = labels[0] if len(labels) == 1 else np.nan

        scores = (
            pd.to_numeric(group["phq8_score"], errors="coerce")
            .dropna()
            .unique()
        )
        score = scores[0] if len(scores) == 1 else np.nan

        split_values = (
            group["split"].map(clean_text).str.lower()
            .replace("", np.nan).dropna().unique()
        )
        split = split_values[0] if len(split_values) == 1 else ""

        rows.append(
            {
                "participant_id": str(participant_id),
                "text": joined_text,
                "phq8_binary": label,
                "phq8_score": score,
                "split": split,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No participant records were produced.")

    result["token_count"] = result["text"].map(
        lambda value: len(re.findall(r"\b[\w'-]+\b", value))
    )
    return result


def prepare_eligible_data(raw_frame: pd.DataFrame, minimum_tokens: int):
    participant_data = aggregate_participants(raw_frame)

    participant_data["phq8_binary"] = pd.to_numeric(
        participant_data["phq8_binary"], errors="coerce"
    )

    eligible = participant_data[
        participant_data["phq8_binary"].isin([0, 1])
        & participant_data["text"].str.len().gt(0)
        & participant_data["token_count"].ge(minimum_tokens)
    ].copy()

    if not eligible.empty:
        eligible["phq8_binary"] = eligible["phq8_binary"].astype(int)

    eligible = eligible.reset_index(drop=True)
    return participant_data, eligible


# ---------------------------------------------------------------------
# TEXT MODEL AND EVALUATION
# ---------------------------------------------------------------------

def build_model() -> Pipeline:
    """Word + character TF-IDF with regularized logistic regression."""
    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=15000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    character_features = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=15000,
        sublinear_tf=True,
    )
    combined_features = FeatureUnion(
        [
            ("word", word_features),
            ("character", character_features),
        ]
    )
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    return Pipeline(
        [
            ("features", combined_features),
            ("classifier", classifier),
        ]
    )


def safe_auc(y_true, probabilities) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, probabilities))
    except Exception:
        return float("nan")


def bootstrap_auc_ci(
    y_true,
    probabilities,
    repetitions: int = 1000,
    seed: int = 42,
):
    """Participant-level bootstrap CI for out-of-fold ROC AUC."""
    y = np.asarray(y_true)
    probs = np.asarray(probabilities)
    if len(y) < 4:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(repetitions):
        indices = rng.integers(0, len(y), size=len(y))
        sampled_y = y[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        values.append(roc_auc_score(sampled_y, probs[indices]))

    if len(values) < 30:
        return float("nan"), float("nan")

    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def evaluate_model(data: pd.DataFrame):
    """Generate out-of-fold predictions; preprocessing stays inside each fold."""
    X = data["text"].astype(str).to_numpy()
    y = data["phq8_binary"].astype(int).to_numpy()

    counts = pd.Series(y).value_counts()
    if len(counts) != 2:
        raise ValueError(
            "Both PHQ-8 binary classes (0 and 1) are required."
        )
    if int(counts.min()) < 2:
        raise ValueError(
            "At least two participants in each class are required "
            "for stratified cross-validation."
        )

    folds = min(5, int(counts.min()))
    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    model = build_model()
    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    baseline = DummyClassifier(strategy="prior")
    baseline_predictions = cross_val_predict(
        baseline,
        X.reshape(-1, 1),
        y,
        cv=cv,
        method="predict",
    )

    auc_low, auc_high = bootstrap_auc_ci(y, probabilities)

    metrics = {
        "Participants": int(len(y)),
        "Cross-validation folds": int(folds),
        "Accuracy": float(accuracy_score(y, predictions)),
        "Balanced accuracy": float(balanced_accuracy_score(y, predictions)),
        "Macro F1": float(
            f1_score(y, predictions, average="macro", zero_division=0)
        ),
        "ROC AUC": safe_auc(y, probabilities),
        "PR AUC": float(average_precision_score(y, probabilities)),
        "Brier score": float(brier_score_loss(y, probabilities)),
        "Baseline accuracy": float(
            accuracy_score(y, baseline_predictions)
        ),
        "Baseline macro F1": float(
            f1_score(
                y, baseline_predictions, average="macro", zero_division=0
            )
        ),
        "ROC AUC 95% CI lower": auc_low,
        "ROC AUC 95% CI upper": auc_high,
    }

    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )
    prediction_table = pd.DataFrame(
        {
            "participant_id": data["participant_id"].values,
            "actual_label": y,
            "predicted_probability": probabilities,
            "predicted_label": predictions,
        }
    )

    return metrics, matrix, report, prediction_table


# ---------------------------------------------------------------------
# REPORT HELPERS
# ---------------------------------------------------------------------

def json_safe(value):
    """Convert NumPy/Pandas values to JSON-safe Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def make_research_record(data, metrics):
    return {
        "project": "The Neurometric Gaze",
        "analysis": "Participant-level stratified cross-validation",
        "participants": int(len(data)),
        "class_counts": {
            str(key): int(value)
            for key, value in data["phq8_binary"].value_counts().items()
        },
        "metrics": {key: json_safe(value) for key, value in metrics.items()},
        "limitations": [
            "Research prototype, not a diagnostic or screening instrument.",
            "PHQ-8 is a self-reported symptom measure, not a diagnosis.",
            "Cross-validation is internal evaluation.",
            "No external validation or clinical utility has been established.",
            "Transcript text cannot measure acoustic silence, gaze or emotion.",
            "Predictive associations do not establish causality.",
            "Dataset provenance, licence and consent must be independently verified.",
        ],
    }


# ---------------------------------------------------------------------
# SIDEBAR / UPLOAD
# ---------------------------------------------------------------------

st.sidebar.title("Research workspace")
st.sidebar.caption("Upload a normalized CSV or a DAIC-style ZIP archive.")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset",
    type=["csv", "zip"],
    help=(
        "CSV requires a text/transcript column and a PHQ-8 binary label. "
        "Participant ID and PHQ-8 total score are optional."
    ),
)

minimum_tokens = st.sidebar.slider(
    "Minimum participant token count",
    min_value=0,
    max_value=500,
    value=20,
    step=10,
)

with st.sidebar.expander("CSV format"):
    st.code(
        "participant_id,text,phq8_binary,phq8_score\n"
        'P001,"Example participant narrative",0,4\n'
        'P002,"Another participant narrative",1,14',
        language="csv",
    )
    st.caption(
        "Accepted label aliases include phq8_binary, phq_8_binary, label "
        "and target. Accepted text aliases include text, transcript "
        "and narrative."
    )

# ---------------------------------------------------------------------
# MAIN APP FLOW
# ---------------------------------------------------------------------

if uploaded_file is None:
    st.info("Upload a CSV or ZIP file from the left sidebar to begin.")
    st.markdown("## Research workflow")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 01 · Prepare\nImport participant narratives and verified labels.")
    c2.markdown("### 02 · Explore\nReview data quality and descriptive patterns.")
    c3.markdown("### 03 · Evaluate\nRun participant-level out-of-fold model evaluation.")
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue()
    upload_fingerprint = hashlib.sha256(raw_bytes).hexdigest()

    if uploaded_file.name.lower().endswith(".zip"):
        raw_data = parse_daic_zip(raw_bytes)
        if raw_data["phq8_binary"].notna().sum() == 0:
            st.error(
                "Transcript files were found, but matching PHQ-8 metadata "
                "was not identified. Please upload a normalized CSV with "
                "verified phq8_binary labels."
            )
            st.stop()
    else:
        source_frame = read_csv_bytes(raw_bytes, uploaded_file.name)
        raw_data = normalize_dataset(source_frame)

    participant_data, data = prepare_eligible_data(
        raw_data, minimum_tokens
    )

    # Avoid showing results from a previous file or token threshold.
    analysis_key = f"{upload_fingerprint}:{minimum_tokens}"
    if st.session_state.get("analysis_key") != analysis_key:
        st.session_state["analysis_key"] = analysis_key
        st.session_state.pop("model_results", None)

    if data.empty:
        st.error(
            "No eligible participant records remain after filtering. "
            "Check the PHQ-8 labels, transcript text and minimum token setting."
        )
        st.stop()

    # -----------------------------------------------------------------
    # 01 DATASET OVERVIEW
    # -----------------------------------------------------------------

    st.header("01 — Dataset overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eligible participants", f"{len(data):,}")
    m2.metric("Class 0", f"{int((data.phq8_binary == 0).sum()):,}")
    m3.metric("Class 1", f"{int((data.phq8_binary == 1).sum()):,}")
    m4.metric("Median tokens", f"{data.token_count.median():.0f}")

    st.caption(
        "The binary categories are displayed as supplied in the dataset. "
        "Confirm their official meaning and threshold in the source documentation."
    )

    # -----------------------------------------------------------------
    # 02 DATA QUALITY AUDIT
    # -----------------------------------------------------------------

    st.header("02 — Data quality audit")
    raw_participant_count = raw_data["participant_id"].nunique()
    empty_text_count = int(
        participant_data["text"].map(lambda value: len(clean_text(value)) == 0).sum()
    )
    missing_label_count = int(participant_data["phq8_binary"].isna().sum())
    conflicting_label_count = int(
        (
            raw_data.groupby("participant_id")["phq8_binary"]
            .nunique(dropna=True)
            .gt(1)
        ).sum()
    )
    below_threshold_count = int(
        (participant_data["token_count"] < minimum_tokens).sum()
    )

    audit = pd.DataFrame(
        {
            "Measure": [
                "Uploaded transcript rows",
                "Unique participant IDs",
                "Eligible labelled participants",
                "Participants with missing/conflicting labels",
                "Participants with empty text",
                "Participants below token threshold",
            ],
            "Value": [
                len(raw_data),
                raw_participant_count,
                len(data),
                missing_label_count + conflicting_label_count,
                empty_text_count,
                below_threshold_count,
            ],
        }
    )
    st.dataframe(audit, use_container_width=True, hide_index=True)

    if conflicting_label_count:
        st.warning(
            f"{conflicting_label_count} participant(s) have conflicting "
            "binary labels across rows. Their label is treated as missing."
        )

    # -----------------------------------------------------------------
    # 03 DESCRIPTIVE ANALYSIS
    # -----------------------------------------------------------------

    st.header("03 — Descriptive analysis")
    descriptive = (
        data.groupby("phq8_binary")
        .agg(
            Participants=("participant_id", "count"),
            Mean_tokens=("token_count", "mean"),
            Median_tokens=("token_count", "median"),
            SD_tokens=("token_count", "std"),
        )
        .reset_index()
    )
    st.dataframe(descriptive, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Participant distribution")
        distribution = (
            data["phq8_binary"]
            .value_counts()
            .sort_index()
            .rename_axis("PHQ-8 binary category")
            .reset_index(name="Participants")
        )
        st.bar_chart(
            distribution.set_index("PHQ-8 binary category"),
            use_container_width=True,
        )

    with right:
        st.markdown("#### Participant token-count distribution")
        st.bar_chart(
            data.set_index("participant_id")[["token_count"]],
            use_container_width=True,
        )

    score_data = data.dropna(subset=["phq8_score"])
    if len(score_data) >= 3:
        st.markdown("#### Exploratory PHQ-8 total-score analysis")
        score_distribution = (
            score_data["phq8_score"].value_counts().sort_index()
        )
        st.bar_chart(score_distribution, use_container_width=True)

        correlation_data = score_data[["token_count", "phq8_score"]]
        if (
            len(correlation_data) >= 3
            and correlation_data["token_count"].nunique() > 1
            and correlation_data["phq8_score"].nunique() > 1
        ):
            correlation = correlation_data.corr(method="spearman").iloc[0, 1]
            st.metric(
                "Exploratory Spearman correlation",
                f"{correlation:.3f}",
            )
            st.caption(
                "Descriptive association only; it does not establish causality "
                "or a psychological interpretation."
            )

    # -----------------------------------------------------------------
    # 04 COMPUTATIONAL TEXT ANALYSIS
    # -----------------------------------------------------------------

    st.header("04 — Computational text analysis")
    st.markdown(
        """
        The model combines word-level and character-level TF-IDF features
        with regularized logistic regression. TF-IDF fitting occurs inside
        each cross-validation fold through a scikit-learn Pipeline, reducing
        preprocessing leakage between training and held-out folds.

        Every prediction shown below is out-of-fold: the participant is
        excluded from the model that generates that participant's prediction.
        """
    )

    st.info(
        "This is internal cross-validation, not an untouched external test set. "
        "Do not describe these results as clinical validation or evidence that "
        "the model can identify mental health conditions."
    )

    run_button = st.button(
        "Run full evaluation",
        type="primary",
        use_container_width=True,
    )

    if run_button:
        with st.spinner("Evaluating the model. This may take a few moments..."):
            try:
                st.session_state["model_results"] = evaluate_model(data)
                st.success("Evaluation completed.")
            except Exception as model_error:
                st.session_state.pop("model_results", None)
                st.error(f"Model evaluation could not be completed: {model_error}")

    if "model_results" in st.session_state:
        metrics, matrix, report, prediction_table = st.session_state["model_results"]

        st.subheader("Model performance")
        metric_frame = pd.DataFrame(
            {
                "Metric": list(metrics.keys()),
                "Value": [
                    "—" if pd.isna(value) else f"{value:.4f}"
                    if isinstance(value, (float, np.floating))
                    else str(value)
                    for value in metrics.values()
                ],
            }
        )
        st.dataframe(metric_frame, use_container_width=True, hide_index=True)

        st.subheader("Confusion matrix")
        confusion_frame = pd.DataFrame(
            matrix,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(confusion_frame, use_container_width=True)

        st.subheader("Classification report")
        report_frame = (
            pd.DataFrame(report)
            .T.reset_index()
            .rename(columns={"index": "Class / metric"})
        )
        st.dataframe(report_frame, use_container_width=True, hide_index=True)

        st.subheader("Out-of-fold predictions")
        st.caption(
            "Participant-level predictions are sensitive research outputs. "
            "Do not publish identifiable records or use them to label individuals."
        )
        st.dataframe(
            prediction_table,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download out-of-fold predictions CSV",
            data=prediction_table.to_csv(index=False).encode("utf-8"),
            file_name="neurometric_gaze_oof_predictions.csv",
            mime="text/csv",
        )

        record = make_research_record(data, metrics)
        st.download_button(
            "Download research record JSON",
            data=json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="neurometric_gaze_research_record.json",
            mime="application/json",
        )

    # -----------------------------------------------------------------
    # 05 DATA EXPORT
    # -----------------------------------------------------------------

    st.header("05 — Cleaned participant-level dataset")
    st.caption(
        "Review the cleaned data before export. Ensure that sharing or exporting "
        "participant-level text is permitted by the dataset licence and consent."
    )
    display_columns = [
        "participant_id",
        "token_count",
        "phq8_binary",
        "phq8_score",
        "split",
    ]
    st.dataframe(
        data[display_columns],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download cleaned participant summary CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="neurometric_gaze_cleaned_data.csv",
        mime="text/csv",
    )

    # -----------------------------------------------------------------
    # 06 RESEARCH INTEGRITY
    # -----------------------------------------------------------------

    st.header("06 — Research integrity checklist")
    checklist = [
        "Verify participant IDs against the official dataset documentation.",
        "Verify PHQ-8 labels and binary thresholds against the original source.",
        "Document missing-data and participant-exclusion decisions.",
        "Check the dataset licence, consent conditions and ethics requirements.",
        "Keep any official test set untouched if planning external evaluation.",
        "Compare the model against simple baselines.",
        "Report class-specific metrics and uncertainty, not accuracy alone.",
        "Use a written codebook for any human annotation.",
        "Calculate inter-rater reliability only from genuinely independent ratings.",
        "Do not interpret lexical patterns as actual emotions or diagnoses.",
        "Do not publish participant-level predictions or identifiable narratives.",
    ]
    for index, item in enumerate(checklist):
        st.checkbox(item, key=f"integrity_check_{index}")

    st.caption(
        "For reproducibility, record dataset version, source, preprocessing "
        "decisions, software versions, random seeds and analysis parameters."
    )

except Exception as error:
    st.error(f"Application processing error: {error}")
    st.info(
        "Check that the uploaded file is a valid CSV or supported ZIP. "
        "A normalized CSV must contain a text/transcript/narrative column "
        "and a PHQ-8 binary label column. Labels must be numeric 0 or 1."
    )
    with st.expander("Technical details for debugging"):
        st.exception(error)

st.divider()
st.caption(
    "The Neurometric Gaze · Research prototype · "
    "Not intended for diagnosis, screening, treatment or individual-level decisions."
)
'''

requirements = """streamlit>=1.36,<2.0
pandas>=2.1,<4.0
numpy>=1.26,<3.0
scikit-learn>=1.4,<2.0
"""

out_dir = Path("/mnt/data/neurometric_gaze_fixed")
out_dir.mkdir(parents=True, exist_ok=True)
app_path = out_dir / "app.py"
req_path = out_dir / "requirements.txt"
app_path.write_text(app_code, encoding="utf-8")
req_path.write_text(requirements, encoding="utf-8")

# Syntax validation
compile(app_code, str(app_path), "exec")

print(f"Created: {app_path} ({app_path.stat().st_size:,} bytes)")
print(f"Created: {req_path} ({req_path.stat().st_size:,} bytes)")
print(f"Python lines: {len(app_code.splitlines())}")
print("Syntax check: PASSED")
