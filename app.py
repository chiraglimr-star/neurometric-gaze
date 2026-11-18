
from __future__ import annotations

import io
import re
import zipfile
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# APPLICATION CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="The Neurometric Gaze",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("The Neurometric Gaze")
st.subheader(
    "A Computational and Humanistic Study of Psychological Distress in Digital Narratives"
)

st.caption(
    "Research prototype | Reproducible analysis | "
    "Not a diagnostic or clinical decision-support system"
)

st.warning(
    "This application does not diagnose mental illness or determine an individual's "
    "emotional state. PHQ-8 is a self-reported symptom measure, not a clinical diagnosis. "
    "Only upload data you are authorized to process."
)

# ---------------------------------------------------------
# DATA IMPORT
# ---------------------------------------------------------

COLUMN_ALIASES = {
    "participant_id": [
        "participant_id", "participant", "id", "subject_id"
    ],
    "text": [
        "text", "transcript", "narrative", "value"
    ],
    "phq8_binary": [
        "phq8_binary", "phq_8_binary", "label", "target"
    ],
    "phq8_score": [
        "phq8_score", "phq_8_score", "phq8"
    ],
    "split": [
        "split", "set", "partition"
    ],
    "speaker": [
        "speaker", "role"
    ],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    rename = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = standard_name
                break

    return df.rename(columns=rename)


def read_csv_file(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


def read_daic_zip(raw: bytes) -> pd.DataFrame:
    """
    Extract participant speech from DAIC-style transcript files.

    Expected transcript columns:
    speaker, value

    This importer does not invent or infer PHQ-8 labels.
    """

    records = []

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:

        transcript_files = [
            name for name in archive.namelist()
            if name.lower().endswith("_transcript.csv")
        ]

        if not transcript_files:
            raise ValueError(
                "No *_TRANSCRIPT.csv files were found in this ZIP."
            )

        for filename in transcript_files:

            with archive.open(filename) as file:
                transcript = pd.read_csv(file)

            if not {"speaker", "value"}.issubset(transcript.columns):
                continue

            speaker = (
                transcript["speaker"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            participant_rows = transcript.loc[
                speaker.eq("participant"), "value"
            ]

            participant_text = " ".join(
                participant_rows.dropna().astype(str)
            )

            participant_id = (
                filename.split("/")[-1].split("_")[0]
            )

            records.append({
                "participant_id": str(participant_id),
                "text": participant_text,
            })

        if not records:
            raise ValueError(
                "Transcript files were found, but no participant speech "
                "could be extracted."
            )

        data = pd.DataFrame(records)

        # Merge metadata only if an identifiable participant key and
        # a documented binary outcome are both present.
        metadata_candidates = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv")
            and "transcript" not in name.lower()
        ]

        for filename in metadata_candidates:

            try:
                with archive.open(filename) as file:
                    metadata = normalize_columns(pd.read_csv(file))

                if (
                    "participant_id" in metadata.columns
                    and "phq8_binary" in metadata.columns
                ):
                    data["participant_id"] = (
                        data["participant_id"].astype(str)
                    )

                    metadata["participant_id"] = (
                        metadata["participant_id"].astype(str)
                    )

                    data = data.merge(
                        metadata,
                        on="participant_id",
                        how="left",
                    )

                    break

            except Exception:
                continue

    return data


def load_data(uploaded_file) -> pd.DataFrame:

    raw = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return read_csv_file(raw)

    if filename.endswith(".zip"):
        return read_daic_zip(raw)

    raise ValueError("Please upload a CSV or ZIP file.")


# ---------------------------------------------------------
# DATA PREPARATION AND AUDIT
# ---------------------------------------------------------

def prepare_data(
    raw_df: pd.DataFrame,
    minimum_tokens: int = 20,
):
    df = normalize_columns(raw_df)

    required = {
        "participant_id",
        "text",
        "phq8_binary",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    original_rows = len(df)

    if "speaker" in df.columns:
        speaker = (
            df["speaker"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[speaker.eq("participant")].copy()

    df["participant_id"] = (
        df["participant_id"].astype(str).str.strip()
    )

    df["text"] = (
        df["text"].fillna("").astype(str)
    )

    df["phq8_binary"] = pd.to_numeric(
        df["phq8_binary"],
        errors="coerce",
    )

    invalid_labels = int(
        (~df["phq8_binary"].isin([0, 1])).sum()
    )

    df = df[
        df["phq8_binary"].isin([0, 1])
    ].copy()

    if "phq8_score" in df.columns:
        df["phq8_score"] = pd.to_numeric(
            df["phq8_score"],
            errors="coerce",
        )

    df["token_count_row"] = df["text"].map(
        lambda text: len(
            re.findall(r"\b[\w']+\b", text)
        )
    )

    # Exclude participants with contradictory binary labels.
    label_counts = (
        df.groupby("participant_id")["phq8_binary"]
        .nunique()
    )

    conflicting_ids = set(
        label_counts[label_counts > 1].index
    )

    df = df[
        ~df["participant_id"].isin(conflicting_ids)
    ].copy()

    aggregation = {
        "text": lambda values: " ".join(
            value for value in values.astype(str)
            if value.strip()
        ),
        "phq8_binary": "first",
        "token_count_row": "sum",
    }

    if "phq8_score" in df.columns:
        aggregation["phq8_score"] = "first"

    if "split" in df.columns:
        aggregation["split"] = "first"

    participant_data = (
        df.groupby("participant_id", as_index=False)
        .agg(aggregation)
    )

    participant_data = participant_data.rename(
        columns={"token_count_row": "token_count"}
    )

    participant_data["phq8_binary"] = (
        participant_data["phq8_binary"].astype(int)
    )

    before_filter = len(participant_data)

    participant_data = participant_data[
        participant_data["token_count"] >= minimum_tokens
    ].copy()

    participant_data = participant_data[
        participant_data["text"].str.strip().ne("")
    ].copy()

    audit = {
        "original_rows": int(original_rows),
        "participant_records_after_aggregation": int(before_filter),
        "final_participant_count": int(len(participant_data)),
        "invalid_label_rows": invalid_labels,
        "conflicting_label_participants_excluded": int(
            len(conflicting_ids)
        ),
        "participants_excluded_by_minimum_length": int(
            before_filter - len(participant_data)
        ),
        "analysis_unit": "participant",
    }

    return participant_data.reset_index(drop=True), audit


# ---------------------------------------------------------
# STATISTICAL ANALYSIS
# ---------------------------------------------------------

def survey_summary(data):

    result = {
        "participant_count": int(len(data)),
        "class_counts": (
            data["phq8_binary"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }

    if "phq8_score" in data.columns:

        scores = pd.to_numeric(
            data["phq8_score"],
            errors="coerce",
        ).dropna()

        if not scores.empty:
            result["phq8_score_summary"] = (
                scores.describe().to_dict()
            )

    return result


def correlation_analysis(data):

    if "phq8_score" not in data.columns:
        return pd.DataFrame()

    numeric_columns = [
        column for column in data.columns
        if column in ["token_count"]
    ]

    results = []

    for column in numeric_columns:

        temp = data[
            [column, "phq8_score"]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()

        if (
            len(temp) < 5
            or temp[column].nunique() < 2
            or temp["phq8_score"].nunique() < 2
        ):
            continue

        rho, p_value = spearmanr(
            temp[column],
            temp["phq8_score"],
        )

        results.append({
            "variable": column,
            "N": len(temp),
            "spearman_rho": rho,
            "p_raw": p_value,
        })

    result = pd.DataFrame(results)

    if not result.empty:
        result["p_holm"] = multipletests(
            result["p_raw"],
            method="holm",
        )[1]

    return result


def nominal_krippendorff_alpha(annotation_df):

    required = {
        "participant_id",
        "segment_id",
        "annotator_id",
        "label",
    }

    if not required.issubset(annotation_df.columns):
        raise ValueError(
            "Annotation CSV must contain: "
            "participant_id, segment_id, annotator_id, label."
        )

    df = annotation_df.copy()

    df["unit"] = (
        df["participant_id"].astype(str)
        + "::"
        + df["segment_id"].astype(str)
    )

    wide = df.pivot_table(
        index="unit",
        columns="annotator_id",
        values="label",
        aggfunc="first",
    )

    matrix = wide.to_numpy(dtype=object)

    coincidence = {}

    for row in matrix:

        ratings = [
            value for value in row
            if pd.notna(value)
        ]

        count = len(ratings)

        if count < 2:
            continue

        for i, j in combinations(range(count), 2):

            a = ratings[i]
            b = ratings[j]

            coincidence[(a, b)] = (
                coincidence.get((a, b), 0)
                + 1 / (count - 1)
            )

            coincidence[(b, a)] = (
                coincidence.get((b, a), 0)
                + 1 / (count - 1)
            )

    total = sum(coincidence.values())

    if total == 0:
        return float("nan")

    categories = set()

    for a, b in coincidence:
        categories.add(a)
        categories.add(b)

    marginals = {
        category: sum(
            value
            for (a, _), value in coincidence.items()
            if a == category
        )
        for category in categories
    }

    observed_disagreement = sum(
        value
        for (a, b), value in coincidence.items()
        if a != b
    ) / total

    marginal_total = sum(marginals.values())

    denominator = total * (total - 1)

    if denominator <= 0:
        return float("nan")

    expected_disagreement = sum(
        marginals[category]
        * (marginal_total - marginals[category])
        for category in categories
    ) / denominator

    if expected_disagreement == 0:
        return float("nan")

    return float(
        1 - observed_disagreement / expected_disagreement
    )


# ---------------------------------------------------------
# MODEL DEVELOPMENT
# ---------------------------------------------------------

def make_pipeline(max_features):

    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=max_features,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "character_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=max(500, max_features // 3),
                sublinear_tf=True,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=2026,
            ),
        ),
    ])


def calculate_splits(y, requested_splits):

    counts = pd.Series(y).value_counts()

    if len(counts) != 2:
        raise ValueError(
            "Both outcome classes must be present."
        )

    return min(
        requested_splits,
        int(counts.min()),
    )


def evaluate_model(
    data,
    max_features=3000,
    outer_splits=5,
    inner_splits=3,
):

    X = data["text"].astype(str).to_numpy()
    y = data["phq8_binary"].astype(int).to_numpy()
    groups = data["participant_id"].astype(str).to_numpy()

    n_outer = calculate_splits(
        y,
        outer_splits,
    )

    if n_outer < 2:
        raise ValueError(
            "There are not enough participants in each class "
            "for grouped cross-validation."
        )

    outer_cv = StratifiedGroupKFold(
        n_splits=n_outer,
        shuffle=True,
        random_state=2026,
    )

    oof_probabilities = np.full(
        len(data),
        np.nan,
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups),
        start=1,
    ):

        X_train = X[train_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx]

        n_inner = calculate_splits(
            y_train,
            inner_splits,
        )

        if n_inner < 2:
            raise ValueError(
                f"Fold {fold}: insufficient class support "
                "for inner cross-validation."
            )

        inner_cv = StratifiedGroupKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=2026 + fold,
        )

        inner_splits_list = list(
            inner_cv.split(
                X_train,
                y_train,
                groups_train,
            )
        )

        search = GridSearchCV(
            estimator=make_pipeline(max_features),
            param_grid={
                "classifier__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ]
            },
            scoring="f1_macro",
            cv=inner_splits_list,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        search.fit(
            X_train,
            y_train,
        )

        probabilities = (
            search.best_estimator_
            .predict_proba(X[test_idx])[:, 1]
        )

        oof_probabilities[test_idx] = probabilities

        fold_results.append({
            "fold": fold,
            "test_participants": len(test_idx),
            "best_C": search.best_params_[
                "classifier__C"
            ],
        })

    valid = np.isfinite(oof_probabilities)

    y_true = y[valid]
    y_prob = oof_probabilities[valid]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "N": int(len(y_true)),
        "Accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "Balanced accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "Macro F1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "Brier score": float(
            brier_score_loss(y_true, y_prob)
        ),
        "ROC AUC": float(
            roc_auc_score(y_true, y_prob)
        ),
        "PR AUC": float(
            average_precision_score(y_true, y_prob)
        ),
    }

    # Baseline using the same outer fold allocation.
    baseline_probabilities = np.full(
        len(data),
        np.nan,
    )

    for train_idx, test_idx in outer_cv.split(
        X, y, groups
    ):

        baseline = DummyClassifier(
            strategy="prior"
        )

        baseline.fit(
            np.zeros((len(train_idx), 1)),
            y[train_idx],
        )

        baseline_probabilities[test_idx] = (
            baseline.predict_proba(
                np.zeros((len(test_idx), 1))
            )[:, 1]
        )

    baseline_valid = np.isfinite(
        baseline_probabilities
    )

    baseline_y = y[baseline_valid]
    baseline_prob = baseline_probabilities[baseline_valid]
    baseline_pred = (
        baseline_prob >= 0.5
    ).astype(int)

    baseline_metrics = {
        "Accuracy": float(
            accuracy_score(baseline_y, baseline_pred)
        ),
        "Balanced accuracy": float(
            balanced_accuracy_score(
                baseline_y,
                baseline_pred,
            )
        ),
        "Macro F1": float(
            f1_score(
                baseline_y,
                baseline_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "ROC AUC": float(
            roc_auc_score(
                baseline_y,
                baseline_prob,
            )
        ),
    }

    predictions = pd.DataFrame({
        "participant_id": groups[valid],
        "observed": y_true,
        "predicted_probability": y_prob,
        "predicted_class": y_pred,
    })

    return {
        "metrics": metrics,
        "baseline": baseline_metrics,
        "folds": pd.DataFrame(fold_results),
        "predictions": predictions,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


def bootstrap_auc(
    predictions,
    iterations=1000,
    seed=2026,
):

    ids = predictions[
        "participant_id"
    ].astype(str).unique()

    if len(ids) < 5:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    grouped = {
        pid: predictions[
            predictions["participant_id"].astype(str) == pid
        ]
        for pid in ids
    }

    values = []

    for _ in range(iterations):

        sampled_ids = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        sample = pd.concat(
            [grouped[pid] for pid in sampled_ids],
            ignore_index=True,
        )

        if sample["observed"].nunique() < 2:
            continue

        values.append(
            roc_auc_score(
                sample["observed"],
                sample["predicted_probability"],
            )
        )

    if not values:
        return np.nan, np.nan

    return tuple(
        np.quantile(values, [0.025, 0.975])
    )


def permutation_auc_test(
    predictions,
    iterations=1000,
    seed=2026,
):

    y = predictions["observed"].to_numpy()
    probabilities = predictions[
        "predicted_probability"
    ].to_numpy()

    observed_auc = roc_auc_score(
        y,
        probabilities,
    )

    rng = np.random.default_rng(seed)

    null_aucs = []

    for _ in range(iterations):

        permuted_y = rng.permutation(y)

        null_aucs.append(
            roc_auc_score(
                permuted_y,
                probabilities,
            )
        )

    p_value = (
        1 + np.sum(
            np.asarray(null_aucs) >= observed_auc
        )
    ) / (iterations + 1)

    return {
        "observed_auc": float(observed_auc),
        "permutation_p": float(p_value),
        "iterations": int(iterations),
        "null_auc_mean": float(
            np.mean(null_aucs)
        ),
    }


# ---------------------------------------------------------
# USER INTERFACE
# ---------------------------------------------------------

with st.sidebar:

    st.header("Research configuration")

    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "zip"],
        help=(
            "CSV or DAIC-style ZIP. "
            "The preferred CSV includes participant_id, text and PHQ8_Binary."
        ),
    )

    minimum_tokens = st.number_input(
        "Minimum participant tokens",
        min_value=1,
        max_value=500,
        value=20,
    )

    max_features = st.select_slider(
        "Maximum TF-IDF features",
        options=[
            1000,
            2000,
            3000,
            5000,
            8000,
        ],
        value=3000,
    )

    run_model = st.button(
        "Run nested grouped evaluation",
        type="primary",
    )


tabs = st.tabs([
    "Research overview",
    "Dataset audit",
    "Survey analysis",
    "Model laboratory",
    "Human annotation",
    "Research report",
])

data = None
audit = None
model_results = None

if uploaded_file is not None:

    try:

        raw_data = load_data(uploaded_file)

        data, audit = prepare_data(
            raw_data,
            minimum_tokens=minimum_tokens,
        )

        if len(data) == 0:
            raise ValueError(
                "No participant records remain after preprocessing."
            )

        st.success(
            f"Prepared {len(data):,} participant-level records."
        )

    except Exception as error:

        st.error(
            f"Data preparation failed: {error}"
        )


# ---------------------------------------------------------
# TAB 1: RESEARCH OVERVIEW
# ---------------------------------------------------------

with tabs[0]:

    st.header("Research framework")

    st.markdown("""
### Primary research question

To what extent are linguistic patterns in interview narratives
associated with self-reported psychological distress, and what
aspects of lived experience remain inaccessible to computational
interpretation?

### Research dimensions

1. Computational linguistics
2. Self-report survey analysis
3. Statistical modelling
4. Human annotation reliability
5. Bias and methodological limitations
6. Narrative Medicine and Medical Humanities

### Important interpretation boundaries

- PHQ-8 is a self-report symptom measure, not a diagnosis.
- DAIC-WOZ uses a virtual interviewer; it is not an ordinary doctor–patient consultation.
- Word-level emotion associations do not measure a participant's actual emotional state.
- Predictive association does not establish causation or clinical usefulness.
- Internal cross-validation is not external validation.
""")

    st.info(
        "The purpose is to study language patterns and the limits "
        "of computational interpretation—not to classify or diagnose people."
    )


# ---------------------------------------------------------
# TAB 2: DATASET AUDIT
# ---------------------------------------------------------

with tabs[1]:

    st.header("Dataset integrity")

    if data is None:

        st.info(
            "Upload a dataset to begin the data audit."
        )

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Participants",
            f"{len(data):,}",
        )

        col2.metric(
            "Group 0",
            int((data["phq8_binary"] == 0).sum()),
        )

        col3.metric(
            "Group 1",
            int((data["phq8_binary"] == 1).sum()),
        )

        st.subheader("Preprocessing audit")

        st.json(audit)

        st.subheader("Outcome distribution")

        st.bar_chart(
            data["phq8_binary"].value_counts().sort_index()
        )

        st.subheader("Participant-level data")

        st.dataframe(
            data.head(30),
            use_container_width=True,
        )

        if "split" in data.columns:

            st.subheader("Supplied dataset split metadata")

            st.dataframe(
                pd.crosstab(
                    data["split"],
                    data["phq8_binary"],
                ),
                use_container_width=True,
            )

            st.caption(
                "This displays the supplied split metadata. "
                "It does not independently certify that the split is official."
            )


# ---------------------------------------------------------
# TAB 3: SURVEY ANALYSIS
# ---------------------------------------------------------

with tabs[2]:

    st.header("Self-report survey analysis")

    if data is None:

        st.info(
            "Upload a dataset to enable survey analysis."
        )

    else:

        st.subheader("Descriptive statistics")

        st.json(
            survey_summary(data)
        )

        if "phq8_score" in data.columns:

            scores = pd.to_numeric(
                data["phq8_score"],
                errors="coerce",
            ).dropna()

            if not scores.empty:

                fig, ax = plt.subplots()

                ax.hist(
                    scores,
                    bins=min(
                        12,
                        max(4, scores.nunique()),
                    ),
                )

                ax.set_title(
                    "Observed PHQ-8 score distribution"
                )

                ax.set_xlabel("PHQ-8 score")
                ax.set_ylabel("Participants")

                st.pyplot(fig)

                st.caption(
                    "This is a description of the uploaded sample, "
                    "not a population prevalence estimate."
                )

                correlations = correlation_analysis(data)

                if not correlations.empty:

                    st.subheader(
                        "Exploratory association analysis"
                    )

                    st.dataframe(
                        correlations,
                        use_container_width=True,
                    )

                    st.caption(
                        "Spearman associations are exploratory and unadjusted "
                        "for potential confounding. Holm correction is shown "
                        "for the reported family of tests."
                    )

        else:

            st.info(
                "No PHQ8_Score column was found. "
                "The application will not fabricate total-score results."
            )


# ---------------------------------------------------------
# TAB 4: MODEL LABORATORY
# ---------------------------------------------------------

with tabs[3]:

    st.header("Leakage-aware model evaluation")

    st.markdown("""
The model uses word- and character-level TF-IDF features with
logistic regression.

The evaluation uses:

- Participant-level grouping
- Stratified grouped outer cross-validation
- Inner-fold hyperparameter selection
- Out-of-fold predictions
- Majority/prior baseline comparison
- Classification metrics
- Participant-level bootstrap uncertainty interval
""")

    if data is None:

        st.info(
            "Upload a dataset before running the model."
        )

    elif run_model:

        try:

            with st.spinner(
                "Running nested grouped cross-validation..."
            ):

                model_results = evaluate_model(
                    data,
                    max_features=max_features,
                )

            metrics = model_results["metrics"]
            baseline = model_results["baseline"]

            st.subheader("Out-of-fold model metrics")

            columns = st.columns(4)

            metric_names = [
                "Macro F1",
                "Balanced accuracy",
                "ROC AUC",
                "PR AUC",
            ]

            for column, metric_name in zip(
                columns,
                metric_names,
            ):

                column.metric(
                    metric_name,
                    f"{metrics[metric_name]:.3f}",
                )

            st.subheader("Baseline comparison")

            comparison = pd.DataFrame([
                {
                    "Model": "TF-IDF + Logistic Regression",
                    **metrics,
                },
                {
                    "Model": "Prior baseline",
                    **baseline,
                },
            ])

            st.dataframe(
                comparison,
                use_container_width=True,
            )

            st.subheader("Participant-bootstrap 95% ROC-AUC interval")

            lower, upper = bootstrap_auc(
                model_results["predictions"]
            )

            if np.isfinite(lower):

                st.write(
                    f"{lower:.3f} – {upper:.3f}"
                )

            else:

                st.write(
                    "Not estimable from the available sample."
                )

            st.subheader("Fold information")

            st.dataframe(
                model_results["folds"],
                use_container_width=True,
            )

            st.subheader("Classification report")

            st.code(
                model_results["classification_report"]
            )

            st.subheader("Confusion matrix")

            cm = model_results["confusion_matrix"]

            fig, ax = plt.subplots()

            ax.imshow(cm)

            ax.set_xticks(
                [0, 1],
                ["Predicted 0", "Predicted 1"],
            )

            ax.set_yticks(
                [0, 1],
                ["Observed 0", "Observed 1"],
            )

            for (i, j), value in np.ndenumerate(cm):

                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                )

            ax.set_title(
                "Out-of-fold confusion matrix"
            )

            st.pyplot(fig)

            predictions = model_results["predictions"]

            st.download_button(
                "Download out-of-fold predictions",
                predictions.to_csv(index=False),
                file_name="oof_predictions.csv",
                mime="text/csv",
            )

            if st.button(
                "Run exploratory permutation test"
            ):

                with st.spinner(
                    "Running permutation test..."
                ):

                    permutation_result = permutation_auc_test(
                        predictions,
                        iterations=1000,
                    )

                st.json(permutation_result)

                st.caption(
                    "This permutation test is conditional on the fixed "
                    "out-of-fold predictions. It does not replace independent "
                    "external validation."
                )

            st.warning(
                "These results describe this dataset and evaluation design. "
                "They do not establish clinical validity, causality, "
                "or generalizability to other populations."
            )

        except Exception as error:

            st.error(
                f"Model evaluation failed: {error}"
            )

            st.caption(
                "Check the number of participants in each class, "
                "the participant identifiers, and the availability "
                "of sufficient independent groups."
            )

    else:

        st.info(
            "Use the sidebar button to run the evaluation after checking the dataset."
        )


# ---------------------------------------------------------
# TAB 5: HUMAN ANNOTATION
# ---------------------------------------------------------

with tabs[4]:

    st.header("Human annotation reliability")

    st.markdown("""
Upload a long-format CSV containing:

- participant_id
- segment_id
- annotator_id
- label

At least two human annotators should independently label the same
text segments using a written annotation codebook.
""")

    annotation_file = st.file_uploader(
        "Upload independent annotation CSV",
        type=["csv"],
        key="annotation_upload",
    )

    if annotation_file is not None:

        try:

            annotation_data = pd.read_csv(
                annotation_file
            )

            alpha = nominal_krippendorff_alpha(
                annotation_data
            )

            if np.isfinite(alpha):

                st.metric(
                    "Krippendorff's alpha (nominal)",
                    f"{alpha:.3f}",
                )

            else:

                st.info(
                    "Alpha could not be estimated from the available annotations."
                )

            st.dataframe(
                annotation_data.head(30),
                use_container_width=True,
            )

            st.caption(
                "Interpret reliability in context. Report the codebook, "
                "annotator training, category prevalence, missing ratings "
                "and disagreements. A reliability coefficient does not "
                "prove that the annotation construct is valid."
            )

        except Exception as error:

            st.error(
                f"Annotation analysis failed: {error}"
            )


# ---------------------------------------------------------
# TAB 6: RESEARCH REPORT
# ---------------------------------------------------------

with tabs[5]:

    st.header("Research documentation")

    researcher_notes = st.text_area(
        "Researcher notes",
        height=160,
        placeholder=(
            "Describe the dataset, inclusion criteria, protocol, "
            "deviations, limitations and interpretation."
        ),
    )

    report = {
        "project": "The Neurometric Gaze",
        "dataset": (
            uploaded_file.name
            if uploaded_file is not None
            else None
        ),
        "data_audit": audit,
        "researcher_notes": researcher_notes,
        "model_metrics": (
            model_results["metrics"]
            if model_results is not None
            else None
        ),
        "baseline_metrics": (
            model_results["baseline"]
            if model_results is not None
            else None
        ),
        "interpretation_boundary": (
            "Associations with PHQ-8 self-report labels do not establish "
            "diagnosis, causation, clinical utility or external validity."
        ),
    }

    st.download_button(
        "Download research record (JSON)",
        data=__import__("json").dumps(
            report,
            indent=2,
            default=str,
        ),
        file_name="neurometric_gaze_research_record.json",
        mime="application/json",
    )

    st.markdown("""
### Academic reporting checklist

- Cite the exact dataset release and documentation.
- Define the analysis unit and label definition.
- Report exclusions and missing data.
- Report baseline comparisons.
- Describe the cross-validation and tuning procedure.
- Report uncertainty and error analysis.
- Avoid causal or clinical claims.
- Obtain ethics approval or an institutional determination where required.
- Validate on an independent dataset before making generalizability claims.
""")

st.divider()

st.caption(
    "The Neurometric Gaze | Reproducibility-first research prototype | "
    "No synthetic survey responses, annotations, metrics or validation claims."
)
