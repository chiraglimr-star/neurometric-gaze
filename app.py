
import io
import re
import json
import zipfile
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="The Neurometric Gaze",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background-color: #0b1020;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}
h1, h2, h3 {
    letter-spacing: -0.5px;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("◉ The Neurometric Gaze")
st.subheader("Clinical Silence Analyzer")
st.caption(
    "An exploratory computational humanities research prototype "
    "examining language patterns and self-reported psychological distress."
)

st.warning(
    "This application is not a diagnostic instrument. "
    "PHQ-8 scores are self-reported symptom measures, not clinical diagnoses. "
    "Language patterns cannot establish a person's actual emotional state."
)

# ============================================================
# METHODOLOGICAL TRANSPARENCY
# ============================================================

with st.expander("Research scope and methodological limitations"):
    st.markdown("""
    ### Research purpose

    This prototype explores whether linguistic characteristics in interview
    narratives are associated with self-reported PHQ-8 scores.

    ### Important distinctions

    - PHQ-8 is a self-report measure, not a clinical diagnosis.
    - Lexical associations do not directly measure emotions.
    - Text-only transcripts cannot establish acoustic silence or pause duration.
    - DAIC-WOZ interviews involve a virtual interviewer and are not equivalent
      to ordinary doctor-patient consultations.
    - Internal cross-validation does not constitute external validation.
    - Predictive associations do not establish causality.

    ### Ethical position

    The application is intended for research and educational exploration.
    It must not be used to diagnose, screen, rank, or make decisions about
    individual patients.
    """)

# ============================================================
# DATA PROCESSING FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_column(name):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(name).strip().lower()
    ).strip("_")


def locate_column(columns, alternatives):
    mapping = {
        normalise_column(c): c
        for c in columns
    }

    for alternative in alternatives:
        key = normalise_column(alternative)
        if key in mapping:
            return mapping[key]

    return None


def read_csv_file(raw, filename="data.csv"):
    errors = []

    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=encoding
            )
        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        f"Unable to read {filename}: {'; '.join(errors)}"
    )


def normalise_dataset(df):

    id_col = locate_column(
        df.columns,
        ["participant_id", "participant", "subject_id", "id"]
    )

    text_col = locate_column(
        df.columns,
        ["text", "transcript", "utterance", "narrative"]
    )

    label_col = locate_column(
        df.columns,
        ["phq8_binary", "phq_8_binary", "label", "target"]
    )

    score_col = locate_column(
        df.columns,
        ["phq8_score", "phq_8_score", "phq8", "phq_8"]
    )

    split_col = locate_column(
        df.columns,
        ["split", "set", "partition"]
    )

    if text_col is None or label_col is None:
        raise ValueError(
            "Required columns are missing. "
            "The dataset must contain text and phq8_binary."
        )

    result = pd.DataFrame()

    if id_col:
        result["participant_id"] = df[id_col].astype(str)
    else:
        result["participant_id"] = [
            f"participant_{i}" for i in range(len(df))
        ]

    result["text"] = df[text_col].map(clean_text)

    result["phq8_binary"] = pd.to_numeric(
        df[label_col],
        errors="coerce"
    )

    if score_col:
        result["phq8_score"] = pd.to_numeric(
            df[score_col],
            errors="coerce"
        )

    if split_col:
        result["split"] = df[split_col].astype(str)

    return result


def parse_transcript(raw, filename):

    df = read_csv_file(raw, filename)

    speaker_col = locate_column(
        df.columns,
        ["speaker", "role"]
    )

    text_col = locate_column(
        df.columns,
        ["value", "text", "utterance", "transcript"]
    )

    if text_col is None:
        raise ValueError(
            f"No transcript text column found in {filename}"
        )

    if speaker_col is not None:

        speakers = (
            df[speaker_col]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        participant_mask = speakers.str.contains(
            r"participant|patient|subject",
            regex=True,
            na=False
        )

        participant_rows = df.loc[
            participant_mask,
            text_col
        ]

        if participant_rows.empty:
            raise ValueError(
                f"No participant speech identified in {filename}. "
                "Check speaker labels."
            )

        text = " ".join(
            clean_text(x)
            for x in participant_rows
            if clean_text(x)
        )

    else:

        text = " ".join(
            clean_text(x)
            for x in df[text_col]
            if clean_text(x)
        )

    match = re.search(r"(\d+)", filename)

    participant_id = (
        match.group(1)
        if match
        else filename
    )

    return {
        "participant_id": str(participant_id),
        "text": text
    }


def parse_daic_zip(raw):

    transcripts = []
    metadata_frames = []

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:

        for filename in archive.namelist():

            base = filename.split("/")[-1]

            if not base.lower().endswith(".csv"):
                continue

            if base.startswith("__MACOSX"):
                continue

            content = archive.read(filename)

            if "transcript" in base.lower():

                try:
                    transcripts.append(
                        parse_transcript(content, base)
                    )
                except Exception:
                    continue

            else:

                try:
                    metadata_frames.append(
                        read_csv_file(content, base)
                    )
                except Exception:
                    continue

    if not transcripts:
        raise ValueError(
            "No readable transcript CSV files were found. "
            "Expected filenames containing TRANSCRIPT."
        )

    result = pd.DataFrame(transcripts)

    # Attempt to locate metadata with PHQ-8 labels.
    for metadata in metadata_frames:

        id_col = locate_column(
            metadata.columns,
            ["participant_id", "participant", "subject_id", "id"]
        )

        label_col = locate_column(
            metadata.columns,
            ["phq8_binary", "phq_8_binary"]
        )

        if id_col is None or label_col is None:
            continue

        meta = metadata.copy()

        meta["participant_id"] = (
            meta[id_col]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna(meta[id_col].astype(str))
        )

        meta["phq8_binary"] = pd.to_numeric(
            meta[label_col],
            errors="coerce"
        )

        score_col = locate_column(
            meta.columns,
            ["phq8_score", "phq_8_score"]
        )

        split_col = locate_column(
            meta.columns,
            ["split", "partition", "set"]
        )

        columns = [
            "participant_id",
            "phq8_binary"
        ]

        if score_col:
            meta["phq8_score"] = pd.to_numeric(
                meta[score_col],
                errors="coerce"
            )
            columns.append("phq8_score")

        if split_col:
            meta["split"] = meta[split_col].astype(str)
            columns.append("split")

        meta = meta[columns].drop_duplicates(
            "participant_id"
        )

        result = result.merge(
            meta,
            on="participant_id",
            how="left"
        )

        break

    return result


def aggregate_participants(df):

    rows = []

    for participant_id, group in df.groupby(
        "participant_id",
        dropna=False
    ):

        texts = [
            clean_text(x)
            for x in group["text"]
            if clean_text(x)
        ]

        labels = (
            pd.to_numeric(
                group["phq8_binary"],
                errors="coerce"
            )
            .dropna()
            .unique()
        )

        label = (
            labels[0]
            if len(labels) == 1
            else np.nan
        )

        score = np.nan

        if "phq8_score" in group:

            scores = (
                pd.to_numeric(
                    group["phq8_score"],
                    errors="coerce"
                )
                .dropna()
                .unique()
            )

            if len(scores) == 1:
                score = scores[0]

        rows.append({
            "participant_id": str(participant_id),
            "text": " ".join(texts),
            "phq8_binary": label,
            "phq8_score": score
        })

    result = pd.DataFrame(rows)

    result["token_count"] = result["text"].map(
        lambda x: len(
            re.findall(r"\b[\w'-]+\b", x)
        )
    )

    return result


# ============================================================
# MODELING
# ============================================================

def build_model():

    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=15000,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    character_features = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=15000,
        sublinear_tf=True
    )

    features = FeatureUnion([
        ("word", word_features),
        ("character", character_features)
    ])

    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42
        ))
    ])

    return model


def calculate_auc(y_true, probabilities):

    try:
        return float(
            roc_auc_score(
                y_true,
                probabilities
            )
        )
    except Exception:
        return np.nan


def bootstrap_auc(
    y_true,
    probabilities,
    repetitions=1000,
    seed=42
):

    rng = np.random.default_rng(seed)

    y = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    results = []

    for _ in range(repetitions):

        indices = rng.integers(
            0,
            len(y),
            len(y)
        )

        sampled_y = y[indices]

        if len(np.unique(sampled_y)) < 2:
            continue

        score = roc_auc_score(
            sampled_y,
            probabilities[indices]
        )

        results.append(score)

    if len(results) < 30:
        return np.nan, np.nan

    return tuple(
        np.quantile(
            results,
            [0.025, 0.975]
        )
    )


def evaluate_model(data):

    X = data["text"].astype(str).to_numpy()

    y = data["phq8_binary"].astype(int).to_numpy()

    class_counts = pd.Series(y).value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            "Both PHQ-8 binary classes are required."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "At least two participants per class are required "
            "for cross-validation."
        )

    folds = min(
        5,
        int(class_counts.min())
    )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    model = build_model()

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba"
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    baseline = DummyClassifier(
        strategy="prior"
    )

    baseline_predictions = cross_val_predict(
        baseline,
        X,
        y,
        cv=cv,
        method="predict"
    )

    metrics = {
        "Participants": len(y),
        "Cross-validation folds": folds,
        "Accuracy": accuracy_score(
            y,
            predictions
        ),
        "Balanced accuracy": balanced_accuracy_score(
            y,
            predictions
        ),
        "Macro F1": f1_score(
            y,
            predictions,
            average="macro",
            zero_division=0
        ),
        "ROC AUC": calculate_auc(
            y,
            probabilities
        ),
        "PR AUC": average_precision_score(
            y,
            probabilities
        ),
        "Brier score": brier_score_loss(
            y,
            probabilities
        ),
        "Baseline accuracy": accuracy_score(
            y,
            baseline_predictions
        ),
        "Baseline macro F1": f1_score(
            y,
            baseline_predictions,
            average="macro",
            zero_division=0
        )
    }

    ci_low, ci_high = bootstrap_auc(
        y,
        probabilities
    )

    metrics["ROC AUC CI lower"] = ci_low
    metrics["ROC AUC CI upper"] = ci_high

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        output_dict=True,
        zero_division=0
    )

    prediction_table = pd.DataFrame({
        "participant_id": data["participant_id"].values,
        "actual_label": y,
        "predicted_probability": probabilities,
        "predicted_label": predictions
    })

    return (
        metrics,
        matrix,
        report,
        prediction_table
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Research workspace")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset",
    type=["csv", "zip"],
    help=(
        "Upload a normalized CSV or a DAIC-style ZIP. "
        "CSV requires text and phq8_binary."
    )
)

minimum_tokens = st.sidebar.slider(
    "Minimum participant token count",
    min_value=0,
    max_value=500,
    value=20,
    step=10
)

# ============================================================
# DATA UPLOAD
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload your dataset to start the analysis."
    )

    st.markdown("### Expected CSV structure")

    st.code(
        "participant_id,text,phq8_binary,phq8_score\n"
        "P001,\"Example participant narrative\",0,4\n"
        "P002,\"Another participant narrative\",1,14",
        language="csv"
    )

    st.stop()


try:

    raw = uploaded_file.getvalue()

    if uploaded_file.name.lower().endswith(".zip"):

        raw_data = parse_daic_zip(raw)

        if "phq8_binary" not in raw_data.columns:
            st.error(
                "Transcripts were found, but matching PHQ-8 metadata "
                "was not found. Upload a verified normalized CSV."
            )
            st.stop()

    else:

        csv_data = read_csv_file(
            raw,
            uploaded_file.name
        )

        raw_data = normalise_dataset(csv_data)

    data = aggregate_participants(raw_data)

    data["phq8_binary"] = pd.to_numeric(
        data["phq8_binary"],
        errors="coerce"
    )

    data = data[
        data["phq8_binary"].isin([0, 1])
    ].copy()

    data["phq8_binary"] = (
        data["phq8_binary"].astype(int)
    )

    data = data[
        data["text"].str.len() > 0
    ]

    data = data[
        data["token_count"] >= minimum_tokens
    ].reset_index(drop=True)

    if data.empty:
        st.error(
            "No eligible participant records remain."
        )
        st.stop()

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.header("01 — Dataset overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Participants",
        f"{len(data):,}"
    )

    col2.metric(
        "Class 0",
        f"{int((data.phq8_binary == 0).sum()):,}"
    )

    col3.metric(
        "Class 1",
        f"{int((data.phq8_binary == 1).sum()):,}"
    )

    col4.metric(
        "Median tokens",
        f"{data.token_count.median():.0f}"
    )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    st.header("02 — Data quality audit")

    audit = pd.DataFrame({
        "Measure": [
            "Uploaded rows",
            "Unique participants",
            "Eligible labelled participants",
            "Participants with empty text",
            "Participants below token threshold"
        ],
        "Value": [
            len(raw_data),
            raw_data["participant_id"].nunique(),
            len(data),
            int(
                raw_data["text"].map(
                    lambda x: len(clean_text(x)) == 0
                ).sum()
            ),
            int(
                (
                    aggregate_participants(raw_data)["token_count"]
                    < minimum_tokens
                ).sum()
            )
        ]
    })

    st.dataframe(
        audit,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # DESCRIPTIVE STATISTICS
    # ========================================================

    st.header("03 — Descriptive analysis")

    descriptive = (
        data.groupby("phq8_binary")
        .agg(
            Participants=("participant_id", "count"),
            Mean_tokens=("token_count", "mean"),
            Median_tokens=("token_count", "median"),
            Standard_deviation=("token_count", "std")
        )
        .reset_index()
    )

    st.dataframe(
        descriptive,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Participant distribution")

    distribution = (
        data["phq8_binary"]
        .value_counts()
        .sort_index()
        .rename_axis("PHQ-8 binary category")
        .reset_index(name="Participants")
    )

    st.bar_chart(
        distribution.set_index(
            "PHQ-8 binary category"
        )
    )

    if data["phq8_score"].notna().sum() >= 3:

        st.markdown("#### PHQ-8 score distribution")

        score_distribution = (
            data["phq8_score"]
            .dropna()
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            score_distribution
        )

        score_corr = data[
            ["token_count", "phq8_score"]
        ].dropna()

        if (
            len(score_corr) >= 3
            and score_corr["phq8_score"].nunique() > 1
            and score_corr["token_count"].nunique() > 1
        ):

            correlation = score_corr.corr(
                method="spearman"
            ).iloc[0, 1]

            st.metric(
                "Exploratory Spearman correlation",
                f"{correlation:.3f}"
            )

            st.caption(
                "This is a descriptive association, not evidence of causation."
            )

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    st.header("04 — Computational text analysis")

    st.markdown("""
    The model combines word-level and character-level TF-IDF features
    with logistic regression.

    Predictions are generated out-of-fold using stratified
    cross-validation. Each participant is held out from the model
    that generates their prediction.
    """)

    if st.button(
        "Run full evaluation",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Evaluating the model..."
        ):

            results = evaluate_model(data)

            st.session_state["results"] = results

    if "results" in st.session_state:

        metrics, matrix, report, prediction_table = (
            st.session_state["results"]
        )

        st.subheader("Model performance")

        metric_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values())
        })

        st.dataframe(
            metric_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Confusion matrix")

        confusion_df = pd.DataFrame(
            matrix,
            index=[
                "Actual 0",
                "Actual 1"
            ],
            columns=[
                "Predicted 0",
                "Predicted 1"
            ]
        )

        st.dataframe(
            confusion_df,
            use_container_width=True
        )

        st.subheader("Classification report")

        report_df = (
            pd.DataFrame(report)
            .T
            .reset_index()
            .rename(
                columns={"index": "Class / metric"}
            )
        )

        st.dataframe(
            report_df,
            use_container_width=True
        )

        st.subheader("Out-of-fold predictions")

        st.dataframe(
            prediction_table,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download predictions CSV",
            prediction_table.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="neurometric_predictions.csv",
            mime="text/csv"
        )

        # ====================================================
        # RESEARCH RECORD
        # ====================================================

        record = {
            "project": "The Neurometric Gaze",
            "analysis": "Stratified cross-validation",
            "participants": int(len(data)),
            "class_counts": {
                str(k): int(v)
                for k, v in data[
                    "phq8_binary"
                ].value_counts().to_dict().items()
            },
            "metrics": {
                k: (
                    None if pd.isna(v)
                    else float(v)
                )
                for k, v in metrics.items()
            },
            "limitations": [
                "Research prototype, not a diagnostic instrument.",
                "PHQ-8 is a self-reported symptom measure.",
                "Cross-validation is internal evaluation.",
                "No external validation has been established.",
                "Transcript text cannot measure acoustic silence or gaze.",
                "DAIC-WOZ is not an ordinary doctor-patient consultation dataset."
            ]
        }

        st.download_button(
            "Download research record JSON",
            json.dumps(
                record,
                indent=2
            ).encode("utf-8"),
            file_name="neurometric_research_record.json",
            mime="application/json"
        )

    # ========================================================
    # RESEARCH INTEGRITY
    # ========================================================

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
        "Respect dataset licensing, consent and ethics requirements."
    ]

    for index, item in enumerate(checklist):
        st.checkbox(
            item,
            key=f"research_check_{index}"
        )

    st.caption(
        "Reproducibility requires documenting dataset provenance, "
        "preprocessing, software versions, random seeds and analysis decisions."
    )

except Exception as error:

    st.error(
        f"Application processing error: {error}"
    )

    st.info(
        "Check the uploaded dataset. The normalized CSV requires "
        "participant_id, text and phq8_binary columns. "
        "The binary label must contain 0 or 1."
    )
