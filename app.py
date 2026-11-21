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
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

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
# SAMPLE / HELPER FUNCTIONS
# ---------------------------------------------------------------------

def get_sample_dataframe() -> pd.DataFrame:
    """Synthetic placeholder rows for instant interface testing."""
    rows = [
        ("S01", "I usually keep things to myself during appointments and just nod along.", 1, 15),
        ("S02", "Sometimes it is hard to explain what I am feeling so I stay quiet.", 1, 13),
        ("S03", "I do not want to sound dramatic so I downplay my symptoms.", 1, 12),
        ("S04", "I pause a lot before answering personal questions about my mood.", 1, 14),
        ("S05", "I tell the doctor directly when something bothers me or feels off.", 0, 3),
        ("S06", "Sleep has been steady and I feel comfortable talking openly.", 0, 2),
        ("S07", "I keep a daily routine and share clear updates during check-ins.", 0, 4),
        ("S08", "Overall my energy is normal and I rarely hold back concerns.", 0, 5),
        ("S09", "I often feel tired but brush it off instead of bringing it up.", 1, 16),
        ("S10", "I feel hesitant to speak openly even when asked directly.", 1, 11),
        ("S11", "My mood feels balanced and I have no trouble describing my week.", 0, 3),
        ("S12", "I communicate clearly and feel heard in conversations.", 0, 2),
    ]
    return pd.DataFrame(
        rows, columns=["participant_id", "text", "phq8_binary", "phq8_score"]
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_column(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def locate_column(columns, alternatives):
    mapping = {normalize_column(col): col for col in columns}
    for alternative in alternatives:
        key = normalize_column(alternative)
        if key in mapping:
            return mapping[key]
    return None


def read_csv_bytes(raw: bytes, filename: str = "data.csv") -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Could not read {filename} as CSV. " + " | ".join(errors[-2:]))


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
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
            "Text column not found. Expected text, transcript or narrative."
        )
    if label_col is None:
        raise ValueError(
            "PHQ-8 binary label not found. Expected phq8_binary, phq_8_binary, label or target."
        )

    result = pd.DataFrame(index=df.index)
    if id_col is not None:
        ids = df[id_col].map(clean_text)
        result["participant_id"] = [
            val if val else f"participant_{i}" for i, val in enumerate(ids)
        ]
    else:
        result["participant_id"] = [f"participant_{i}" for i in range(len(df))]

    result["text"] = df[text_col].map(clean_text)
    result["phq8_binary"] = pd.to_numeric(df[label_col], errors="coerce")
    result["phq8_score"] = (
        pd.to_numeric(df[score_col], errors="coerce")
        if score_col is not None
        else np.nan
    )
    result["split"] = (
        df[split_col].map(clean_text).str.lower() if split_col is not None else ""
    )
    return result.reset_index(drop=True)


def parse_transcript_csv(raw: bytes, filename: str) -> dict:
    frame = read_csv_bytes(raw, filename)
    speaker_col = locate_column(frame.columns, ["speaker", "role", "speaker_id"])
    text_col = locate_column(frame.columns, ["value", "text", "utterance", "transcript"])

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
        if participant_rows.empty:
            raise ValueError(f"Could not identify participant speech in {filename}.")
        text_values = participant_rows
    else:
        text_values = frame[text_col]

    transcript = " ".join(cleaned for item in text_values if (cleaned := clean_text(item)))
    match = re.search(r"(\d+)", filename)
    participant_id = match.group(1) if match else filename

    if not transcript:
        raise ValueError(f"Transcript is empty: {filename}")

    return {"participant_id": str(participant_id), "text": transcript}


def parse_daic_zip(raw: bytes) -> pd.DataFrame:
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
                    continue
            else:
                try:
                    metadata_frames.append(read_csv_bytes(content, base))
                except Exception:
                    continue

    if not transcripts:
        raise ValueError(
            "No readable participant transcript CSV files were found in the ZIP archive."
        )

    result = pd.DataFrame(transcripts)
    metadata_found = False

    for metadata in metadata_frames:
        id_col = locate_column(
            metadata.columns, ["participant_id", "participant", "subject_id", "id"]
        )
        label_col = locate_column(
            metadata.columns, ["phq8_binary", "phq_8_binary", "phq8binary"]
        )
        if id_col is None or label_col is None:
            continue

        meta = pd.DataFrame()
        raw_ids = metadata[id_col].map(clean_text)
        extracted = raw_ids.str.extract(r"(\d+)", expand=False)
        meta["participant_id"] = extracted.where(extracted.notna(), raw_ids)
        meta["phq8_binary"] = pd.to_numeric(metadata[label_col], errors="coerce")

        score_col = locate_column(
            metadata.columns, ["phq8_score", "phq_8_score", "phq8", "phq_8"]
        )
        split_col = locate_column(
            metadata.columns, ["split", "partition", "set", "subset"]
        )

        meta["phq8_score"] = (
            pd.to_numeric(metadata[score_col], errors="coerce")
            if score_col is not None
            else np.nan
        )
        meta["split"] = (
            metadata[split_col].map(clean_text).str.lower()
            if split_col is not None
            else ""
        )

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


def aggregate_participants(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"participant_id", "text", "phq8_binary"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    work = frame.copy()
    if "phq8_score" not in work.columns:
        work["phq8_score"] = np.nan
    if "split" not in work.columns:
        work["split"] = ""

    rows = []
    for participant_id, group in work.groupby("participant_id", dropna=False, sort=False):
        texts = [clean_text(value) for value in group["text"]]
        joined_text = " ".join(value for value in texts if value)

        labels = pd.to_numeric(group["phq8_binary"], errors="coerce").dropna().unique()
        label = labels[0] if len(labels) == 1 else np.nan

        scores = pd.to_numeric(group["phq8_score"], errors="coerce").dropna().unique()
        score = scores[0] if len(scores) == 1 else np.nan

        split_values = (
            group["split"].map(clean_text).str.lower().replace("", np.nan).dropna().unique()
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

    return participant_data, eligible.reset_index(drop=True)


# ---------------------------------------------------------------------
# MODELING & EVALUATION
# ---------------------------------------------------------------------

def build_model() -> Pipeline:
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
        [("word", word_features), ("character", character_features)]
    )
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    return Pipeline([("features", combined_features), ("classifier", classifier)])


def safe_auc(y_true, probabilities) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, probabilities))
    except Exception:
        return float("nan")


def bootstrap_auc_ci(y_true, probabilities, repetitions: int = 500, seed: int = 42):
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
    X = data["text"].astype(str).to_numpy()
    y = data["phq8_binary"].astype(int).to_numpy()

    counts = pd.Series(y).value_counts()
    if len(counts) != 2:
        raise ValueError("Both PHQ-8 binary classes (0 and 1) are required.")
    if int(counts.min()) < 2:
        raise ValueError(
            "At least two participants in each class are required for cross-validation."
        )

    folds = min(5, int(counts.min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    model = build_model()
    probabilities = cross_val_predict(
        model, X, y, cv=cv, method="predict_proba", n_jobs=None
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    baseline = DummyClassifier(strategy="prior")
    baseline_predictions = cross_val_predict(
        baseline, X.reshape(-1, 1), y, cv=cv, method="predict"
    )

    auc_low, auc_high = bootstrap_auc_ci(y, probabilities)

    metrics = {
        "Participants": int(len(y)),
        "Cross-validation folds": int(folds),
        "Accuracy": float(accuracy_score(y, predictions)),
        "Balanced accuracy": float(balanced_accuracy_score(y, predictions)),
        "Macro F1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "ROC AUC": safe_auc(y, probabilities),
        "PR AUC": float(average_precision_score(y, probabilities)),
        "Brier score": float(brier_score_loss(y, probabilities)),
        "Baseline accuracy": float(accuracy_score(y, baseline_predictions)),
        "Baseline macro F1": float(
            f1_score(y, baseline_predictions, average="macro", zero_division=0)
        ),
        "ROC AUC 95% CI lower": auc_low,
        "ROC AUC 95% CI upper": auc_high,
    }

    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    report = classification_report(
        y, predictions, labels=[0, 1], output_dict=True, zero_division=0
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


def json_safe(value):
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
            str(k): int(v) for k, v in data["phq8_binary"].value_counts().items()
        },
        "metrics": {k: json_safe(v) for k, v in metrics.items()},
        "limitations": [
            "Research prototype, not a diagnostic or screening instrument.",
            "PHQ-8 is a self-reported symptom measure, not a diagnosis.",
            "Cross-validation is internal evaluation.",
            "No external validation or clinical utility has been established.",
            "Transcript text cannot measure acoustic silence, gaze or emotion.",
            "Predictive associations do not establish causality.",
        ],
    }


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

st.sidebar.title("Research workspace")
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset",
    type=["csv", "zip"],
    help="CSV requires text/transcript and phq8_binary.",
)

use_sample = st.sidebar.checkbox(
    "Use built-in synthetic sample dataset",
    value=(uploaded_file is None),
    help="Loads synthetic placeholder examples when no file is uploaded.",
)

minimum_tokens = st.sidebar.slider(
    "Minimum participant token count",
    min_value=0,
    max_value=500,
    value=5,
    step=5,
)

with st.sidebar.expander("CSV format"):
    st.code(
        "participant_id,text,phq8_binary,phq8_score\n"
        'P001,"Example participant narrative",0,4\n'
        'P002,"Another participant narrative",1,14',
        language="csv",
    )

# ---------------------------------------------------------------------
# MAIN FLOW
# ---------------------------------------------------------------------

if uploaded_file is None and not use_sample:
    st.info("Upload a CSV/ZIP file or enable the synthetic sample dataset in the sidebar.")
    st.stop()

try:
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        source_tag = hashlib.sha256(raw_bytes).hexdigest()
        if uploaded_file.name.lower().endswith(".zip"):
            raw_data = parse_daic_zip(raw_bytes)
            if raw_data["phq8_binary"].notna().sum() == 0:
                st.error(
                    "Transcript files were found, but matching PHQ-8 metadata was not found."
                )
                st.stop()
        else:
            source_frame = read_csv_bytes(raw_bytes, uploaded_file.name)
            raw_data = normalize_dataset(source_frame)
    else:
        raw_data = get_sample_dataframe()
        source_tag = "synthetic_sample_v1"
        st.caption("Showing built-in synthetic sample dataset for demonstration.")

    participant_data, data = prepare_eligible_data(raw_data, minimum_tokens)

    analysis_key = f"{source_tag}:{minimum_tokens}"
    if st.session_state.get("analysis_key") != analysis_key:
        st.session_state["analysis_key"] = analysis_key
        st.session_state.pop("model_results", None)

    if data.empty:
        st.error("No eligible participant records remain after filtering.")
        st.stop()

    st.header("01 — Dataset overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eligible participants", f"{len(data):,}")
    m2.metric("Class 0", f"{int((data.phq8_binary == 0).sum()):,}")
    m3.metric("Class 1", f"{int((data.phq8_binary == 1).sum()):,}")
    m4.metric("Median tokens", f"{data.token_count.median():.0f}")

    st.header("02 — Data quality audit")
    audit = pd.DataFrame(
        {
            "Measure": [
                "Uploaded / sample rows",
                "Unique participant IDs",
                "Eligible labelled participants",
                "Participants with missing/conflicting labels",
                "Participants with empty text",
                "Participants below token threshold",
            ],
            "Value": [
                len(raw_data),
                raw_data["participant_id"].nunique(),
                len(data),
                int(participant_data["phq8_binary"].isna().sum())
                + int(
                    (
                        raw_data.groupby("participant_id")["phq8_binary"]
                        .nunique(dropna=True)
                        .gt(1)
                    ).sum()
                ),
                int(participant_data["text"].map(lambda v: len(clean_text(v)) == 0).sum()),
                int((participant_data["token_count"] < minimum_tokens).sum()),
            ],
        }
    )
    st.dataframe(audit, use_container_width=True, hide_index=True)

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
        dist = (
            data["phq8_binary"]
            .value_counts()
            .sort_index()
            .rename_axis("PHQ-8 binary category")
            .reset_index(name="Participants")
        )
        st.bar_chart(dist.set_index("PHQ-8 binary category"), use_container_width=True)

    with right:
        st.markdown("#### Participant token-count distribution")
        st.bar_chart(data.set_index("participant_id")[["token_count"]], use_container_width=True)

    st.header("04 — Computational text analysis")
    if st.button("Run full evaluation", type="primary", use_container_width=True):
        with st.spinner("Evaluating model..."):
            try:
                st.session_state["model_results"] = evaluate_model(data)
                st.success("Evaluation completed.")
            except Exception as model_error:
                st.session_state.pop("model_results", None)
                st.error(f"Evaluation error: {model_error}")

    if "model_results" in st.session_state:
        metrics, matrix, report, prediction_table = st.session_state["model_results"]

        st.subheader("Model performance")
        metric_frame = pd.DataFrame(
            {
                "Metric": list(metrics.keys()),
                "Value": [
                    "—"
                    if pd.isna(v)
                    else f"{v:.4f}"
                    if isinstance(v, float)
                    else str(v)
                    for v in metrics.values()
                ],
            }
        )
        st.dataframe(metric_frame, use_container_width=True, hide_index=True)

        st.subheader("Confusion matrix")
        st.dataframe(
            pd.DataFrame(
                matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
            ),
            use_container_width=True,
        )

        st.subheader("Classification report")
        st.dataframe(
            pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class / metric"}),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Out-of-fold predictions")
        st.dataframe(prediction_table, use_container_width=True, hide_index=True)

        st.download_button(
            "Download predictions CSV",
            prediction_table.to_csv(index=False).encode("utf-8"),
            file_name="neurometric_predictions.csv",
            mime="text/csv",
        )

        record = make_research_record(data, metrics)
        st.download_button(
            "Download research record JSON",
            json.dumps(record, indent=2).encode("utf-8"),
            file_name="neurometric_research_record.json",
            mime="application/json",
        )

    st.header("05 — Research integrity checklist")
    checklist = [
        "Verify participant IDs against official metadata.",
        "Verify PHQ-8 labels against the original source.",
        "Document missing-data and exclusion decisions.",
        "Keep the official test set untouched when applicable.",
        "Compare the model against simple baselines.",
        "Report uncertainty and class-specific performance.",
        "Conduct independent human annotation using a written codebook.",
        "Calculate inter-rater reliability only from actual independent annotations.",
        "Do not interpret lexical associations as actual emotions.",
        "Respect dataset licensing, consent and ethics requirements.",
    ]
    for idx, item in enumerate(checklist):
        st.checkbox(item, key=f"check_{idx}")

except Exception as error:
    st.error(f"Application processing error: {error}")
