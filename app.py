from pathlib import Path
import zipfile, textwrap

out = Path("/mnt/data/neurometric_gaze_fixed")
out.mkdir(exist_ok=True)

app_code = r'''"""
The Neurometric Gaze | Clinical Silence Analyzer
Research prototype for exploratory analysis of interview narratives and self-reported PHQ-8 scores.

Important:
- This is a research/educational prototype, not a diagnostic tool.
- PHQ-8 scores are self-reported symptom measures, not clinical diagnoses.
- DAIC-WOZ interviews use a virtual interviewer and are not ordinary doctor-patient consultations.
- Transcript-only analysis cannot measure acoustic silence, pause duration, prosody, or gaze.
"""

import io
import re
import zipfile
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="The Neurometric Gaze | Clinical Silence Analyzer",
    page_icon="◉",
    layout="wide",
)

APP_TITLE = "The Neurometric Gaze"
APP_SUBTITLE = "Clinical Silence Analyzer · Research prototype"

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.warning(
    "Research-use only. This application does not diagnose depression, infer a person's "
    "true emotional state, or establish clinical validity. Language patterns are not "
    "proof of fear, concealment, distress, or any other internal state."
)

with st.expander("Scope, interpretation and ethical limitations", expanded=False):
    st.markdown(
        """
        **What this application can do**
        - Explore participant-level interview text alongside supplied PHQ-8 labels.
        - Describe lexical patterns and evaluate a text classifier using stratified cross-validation.
        - Summarise performance and uncertainty on the uploaded sample.

        **What it cannot establish**
        - PHQ-8 is a self-report symptom measure, not a clinician diagnosis.
        - A word-emotion lexicon records word-level associations; it does not measure a speaker's emotion.
        - Text transcripts do not contain reliable acoustic pause duration, prosody, overlap, or gaze data.
        - DAIC-WOZ interviews involve a virtual interviewer. Results should not be generalised to real doctor-patient consultations without separate evidence.
        - Cross-validation on one dataset is internal evaluation, not external validation.
        """
    )

# ------------------------- Data ingestion -------------------------

def normalise_colname(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(columns, candidates):
    lookup = {normalise_colname(c): c for c in columns}
    for candidate in candidates:
        key = normalise_colname(candidate)
        if key in lookup:
            return lookup[key]
    return None


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_csv_bytes(raw, filename="uploaded.csv"):
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read {filename} as CSV: {last_error}")


def parse_transcript_csv(raw, filename):
    df = read_csv_bytes(raw, filename)
    speaker_col = find_column(df.columns, ["speaker", "role"])
    text_col = find_column(df.columns, ["value", "text", "utterance", "transcript"])
    if text_col is None:
        raise ValueError(f"{filename}: no transcript text column found.")

    if speaker_col is not None:
        speaker = df[speaker_col].astype(str).str.strip().str.lower()
        participant_mask = speaker.str.contains(r"participant|patient|subject", regex=True, na=False)
        # DAIC-WOZ transcript files commonly label the human speaker "Participant".
        participant_rows = df.loc[participant_mask, text_col]
        if participant_rows.empty:
            # Do not silently mix interviewer and participant speech.
            raise ValueError(
                f"{filename}: speaker column exists, but no participant rows were recognised. "
                "Check the speaker labels before proceeding."
            )
        text = " ".join(clean_text(x) for x in participant_rows if clean_text(x))
    else:
        text = " ".join(clean_text(x) for x in df[text_col] if clean_text(x))

    match = re.search(r"(\d+)", filename)
    participant_id = match.group(1) if match else re.sub(r"_TRANSCRIPT.*$", "", filename, flags=re.I)
    return {"participant_id": str(participant_id), "text": text, "source_file": filename}


def parse_zip_upload(raw):
    records = []
    metadata_frames = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        for name in names:
            base = name.rsplit("/", 1)[-1]
            if not base or base.startswith("__MACOSX"):
                continue
            if base.lower().endswith(".csv"):
                try:
                    data = archive.read(name)
                    if "transcript" in base.lower():
                        records.append(parse_transcript_csv(data, base))
                    else:
                        frame = read_csv_bytes(data, base)
                        if len(frame.columns) > 1:
                            metadata_frames.append(frame)
                except Exception:
                    # Keep scanning other CSVs; surface a useful error if no transcripts are found.
                    continue

    if not records:
        raise ValueError(
            "No readable transcript CSV files were found in the ZIP. "
            "Expected files with 'TRANSCRIPT' in their filename."
        )

    transcripts = pd.DataFrame(records)
    metadata = None
    for frame in metadata_frames:
        id_col = find_column(frame.columns, ["participant_id", "participant", "subject_id", "id"])
        label_col = find_column(frame.columns, ["phq8_binary", "phq_8_binary", "label", "target"])
        if id_col is not None and label_col is not None:
            metadata = frame.copy()
            metadata["participant_id"] = metadata[id_col].astype(str).str.extract(r"(\d+)", expand=False).fillna(metadata[id_col].astype(str))
            break

    if metadata is not None:
        metadata = metadata.rename(columns={
            find_column(metadata.columns, ["phq8_binary", "phq_8_binary", "label", "target"]): "phq8_binary"
        })
        score_col = find_column(metadata.columns, ["phq8_score", "phq_8_score", "phq8", "phq_8"])
        if score_col:
            metadata = metadata.rename(columns={score_col: "phq8_score"})
        split_col = find_column(metadata.columns, ["split", "set", "partition"])
        if split_col:
            metadata = metadata.rename(columns={split_col: "split"})
        keep = ["participant_id", "phq8_binary"]
        keep += [c for c in ["phq8_score", "split"] if c in metadata.columns]
        metadata = metadata[keep].drop_duplicates("participant_id")
        transcripts = transcripts.merge(metadata, on="participant_id", how="left")

    return transcripts


def normalise_dataset(df):
    id_col = find_column(df.columns, ["participant_id", "participant", "subject_id", "id"])
    text_col = find_column(df.columns, ["text", "transcript", "utterance", "narrative"])
    label_col = find_column(df.columns, ["phq8_binary", "phq_8_binary", "label", "target"])
    score_col = find_column(df.columns, ["phq8_score", "phq_8_score", "phq8", "phq_8"])
    split_col = find_column(df.columns, ["split", "set", "partition"])

    if text_col is None or label_col is None:
        raise ValueError(
            "CSV must contain text/transcript and phq8_binary columns. "
            "Recommended columns: participant_id, text, phq8_binary, phq8_score, split."
        )

    result = pd.DataFrame()
    result["participant_id"] = (
        df[id_col].astype(str) if id_col else [f"row_{i+1}" for i in range(len(df))]
    )
    result["text"] = df[text_col].map(clean_text)
    result["phq8_binary"] = pd.to_numeric(df[label_col], errors="coerce")
    if score_col:
        result["phq8_score"] = pd.to_numeric(df[score_col], errors="coerce")
    if split_col:
        result["split"] = df[split_col].astype(str)
    return result


def aggregate_participants(df):
    rows = []
    for pid, group in df.groupby("participant_id", dropna=False):
        texts = [clean_text(x) for x in group["text"] if clean_text(x)]
        labels = pd.to_numeric(group["phq8_binary"], errors="coerce").dropna().unique()
        scores = pd.to_numeric(group["phq8_score"], errors="coerce").dropna().unique() if "phq8_score" in group else []
        splits = group["split"].dropna().astype(str).unique() if "split" in group else []
        # Conflicting labels are excluded rather than resolved arbitrarily.
        label = labels[0] if len(labels) == 1 else np.nan
        score = scores[0] if len(scores) == 1 else np.nan
        rows.append({
            "participant_id": str(pid),
            "text": " ".join(texts),
            "phq8_binary": label,
            "phq8_score": score,
            "split": splits[0] if len(splits) == 1 else ("conflict" if len(splits) > 1 else ""),
        })
    out = pd.DataFrame(rows)
    out["token_count"] = out["text"].map(lambda x: len(re.findall(r"\b[\w'-]+\b", x)))
    return out


# ------------------------- Optional lexicon -------------------------

def load_nrc_lexicon(uploaded):
    if uploaded is None:
        return None
    raw = uploaded.getvalue()
    # NRC files are often tab-separated with word, emotion, association columns.
    for sep in ("\t", ","):
        try:
            lex = pd.read_csv(io.BytesIO(raw), sep=sep, header=None, comment="#", encoding="utf-8")
            if lex.shape[1] >= 3:
                lex = lex.iloc[:, :3]
                lex.columns = ["word", "emotion", "association"]
                lex["word"] = lex["word"].astype(str).str.lower().str.strip()
                lex["emotion"] = lex["emotion"].astype(str).str.lower().str.strip()
                lex["association"] = pd.to_numeric(lex["association"], errors="coerce")
                lex = lex.dropna(subset=["association"])
                return lex[lex["association"] == 1]
        except Exception:
            continue
    raise ValueError("Could not parse the lexicon. Expected a three-column NRC-style file.")


def lexicon_summary(text, lexicon):
    if lexicon is None:
        return {}
    word_emotions = {}
    for row in lexicon.itertuples(index=False):
        word_emotions.setdefault(row.word, set()).add(row.emotion)
    tokens = re.findall(r"\b[a-z']+\b", str(text).lower())
    counts = Counter()
    for token in tokens:
        for emotion in word_emotions.get(token, ()):
            counts[emotion] += 1
    total = max(len(tokens), 1)
    return {f"lex_{emotion}_per_1000": count / total * 1000 for emotion, count in counts.items()}


# ------------------------- Analysis helpers -------------------------

def safe_auc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return np.nan


def bootstrap_auc_ci(y_true, scores, iterations=1000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(scores)
    values = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], p[idx]))
    if len(values) < 30:
        return (np.nan, np.nan)
    return tuple(np.quantile(values, [0.025, 0.975]))


def build_text_model():
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=20000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("char", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
            max_features=20000,
            sublinear_tf=True,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )),
    ])


def run_cross_validation(data):
    X = data["text"].astype(str).to_numpy()
    y = data["phq8_binary"].astype(int).to_numpy()
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError(
            "At least two participants in each PHQ-8 binary class are required for "
            "stratified cross-validation. More participants are strongly recommended."
        )
    n_splits = min(5, int(class_counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    model = build_text_model()
    dummy = DummyClassifier(strategy="prior")

    # Every participant appears in exactly one held-out fold.
    oof_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    oof_pred = (oof_prob >= 0.5).astype(int)
    baseline_pred = cross_val_predict(dummy, X, y, cv=cv, method="predict")

    metrics = {
        "Participants": len(y),
        "CV folds": n_splits,
        "Accuracy": accuracy_score(y, oof_pred),
        "Balanced accuracy": balanced_accuracy_score(y, oof_pred),
        "Macro F1": f1_score(y, oof_pred, average="macro", zero_division=0),
        "ROC AUC": safe_auc(y, oof_prob),
        "PR AUC": average_precision_score(y, oof_prob),
        "Brier score": brier_score_loss(y, oof_prob),
        "Baseline accuracy": accuracy_score(y, baseline_pred),
        "Baseline macro F1": f1_score(y, baseline_pred, average="macro", zero_division=0),
    }
    ci_low, ci_high = bootstrap_auc_ci(y, oof_prob)
    metrics["ROC AUC 95% CI lower"] = ci_low
    metrics["ROC AUC 95% CI upper"] = ci_high

    cm = confusion_matrix(y, oof_pred, labels=[0, 1])
    report = classification_report(y, oof_pred, labels=[0, 1], output_dict=True, zero_division=0)
    predictions = pd.DataFrame({
        "participant_id": data["participant_id"].to_numpy(),
        "actual_phq8_binary": y,
        "oof_predicted_probability": oof_prob,
        "oof_predicted_class": oof_pred,
    })
    return metrics, cm, report, predictions


# ------------------------- User interface -------------------------

st.sidebar.header("Data input")
upload = st.sidebar.file_uploader(
    "Upload normalized CSV or DAIC-style ZIP",
    type=["csv", "zip"],
    help="CSV: participant_id, text, phq8_binary; optional phq8_score and split. "
         "ZIP: transcript CSVs and, if available, a metadata CSV.",
)
lex_upload = st.sidebar.file_uploader(
    "Optional NRC emotion lexicon",
    type=["txt", "tsv", "csv"],
    help="Upload the NRC Word-Emotion Association Lexicon in a three-column format.",
)

if upload is None:
    st.info(
        "Upload a normalized CSV or DAIC-style ZIP to begin. "
        "For a normalized CSV, use one row per participant where possible."
    )
    with st.expander("Required CSV format"):
        st.code(
            "participant_id,text,phq8_binary,phq8_score,split\n"
            "participant_001,\"Example interview narrative\",0,4,train\n"
            "participant_002,\"Another example narrative\",1,14,train",
            language="csv",
        )
    st.stop()

try:
    raw = upload.getvalue()
    if upload.name.lower().endswith(".zip"):
        source_df = parse_zip_upload(raw)
        if "phq8_binary" not in source_df.columns:
            st.error(
                "Transcript files were found, but no matching metadata with PHQ-8 binary "
                "labels was identified. Upload a normalized CSV with verified labels."
            )
            st.stop()
    else:
        source_df = normalise_dataset(read_csv_bytes(raw, upload.name))

    if "phq8_binary" not in source_df.columns:
        raise ValueError("The dataset does not contain phq8_binary labels.")

    data = aggregate_participants(source_df)
    data["phq8_binary"] = pd.to_numeric(data["phq8_binary"], errors="coerce")
    data = data[data["phq8_binary"].isin([0, 1])].copy()
    data["phq8_binary"] = data["phq8_binary"].astype(int)
    data = data[data["text"].str.len() > 0].copy()

    min_tokens = st.sidebar.slider("Minimum participant tokens", 0, 500, 20, 10)
    data = data[data["token_count"] >= min_tokens].reset_index(drop=True)

    if data.empty:
        st.error("No eligible participant records remain after cleaning and filtering.")
        st.stop()

    st.subheader("Dataset audit")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", f"{len(data):,}")
    c2.metric("Class 0", f"{int((data.phq8_binary == 0).sum()):,}")
    c3.metric("Class 1", f"{int((data.phq8_binary == 1).sum()):,}")
    c4.metric("Median tokens", f"{data.token_count.median():.0f}")

    audit = pd.DataFrame({
        "Audit item": [
            "Uploaded rows",
            "Unique participants after aggregation",
            "Eligible labelled participants",
            "Excluded/unlabelled rows after aggregation",
            "Participants below token threshold",
        ],
        "Count": [
            len(source_df),
            source_df["participant_id"].nunique(),
            len(data),
            int(source_df["phq8_binary"].isna().sum()) if "phq8_binary" in source_df else 0,
            int((aggregate_participants(source_df)["token_count"] < min_tokens).sum()),
        ],
    })
    st.dataframe(audit, use_container_width=True, hide_index=True)

    if "split" in data.columns:
        st.caption("Split values are displayed as supplied; this application does not certify official split provenance.")
        st.dataframe(data["split"].value_counts(dropna=False).rename_axis("split").reset_index(name="participants"),
                     use_container_width=True, hide_index=True)

    st.subheader("Descriptive survey and text overview")
    class_table = data.groupby("phq8_binary").agg(
        participants=("participant_id", "count"),
        mean_tokens=("token_count", "mean"),
        median_tokens=("token_count", "median"),
    ).reset_index()
    st.dataframe(class_table, use_container_width=True, hide_index=True)

    if "phq8_score" in data.columns:
        score_data = data.dropna(subset=["phq8_score"])
        if len(score_data) >= 3 and score_data["phq8_score"].nunique() > 1:
            corr = score_data[["token_count", "phq8_score"]].corr(method="spearman").iloc[0, 1]
            st.metric("Exploratory Spearman correlation: token count vs PHQ-8 score", f"{corr:.3f}")
            st.caption("Descriptive association only; it is not evidence of causality or clinical meaning.")
        else:
            st.caption("Not enough variation or observations to calculate the exploratory score correlation.")

    if lex_upload is not None:
        try:
            lexicon = load_nrc_lexicon(lex_upload)
            summaries = data["text"].map(lambda t: lexicon_summary(t, lexicon))
            lex_features = pd.DataFrame(list(summaries)).fillna(0)
            if not lex_features.empty:
                st.subheader("Lexicon associations (exploratory)")
                st.dataframe(lex_features.describe().T, use_container_width=True)
                st.caption(
                    "Counts indicate matches to lexicon-associated words only. "
                    "They must not be interpreted as measured emotions or psychological states."
                )
        except Exception as exc:
            st.warning(f"Lexicon was not analysed: {exc}")

    st.subheader("Text classification: internal cross-validation")
    st.caption(
        "A TF-IDF word/character logistic-regression model is evaluated using stratified "
        "cross-validation. Predictions are out-of-fold. This is internal validation only."
    )

    if st.button("Run evaluation", type="primary"):
        with st.spinner("Running cross-validation..."):
            try:
                metrics, cm, report, predictions = run_cross_validation(data)
                st.session_state["evaluation"] = {
                    "metrics": metrics,
                    "cm": cm,
                    "report": report,
                    "predictions": predictions,
                }
            except Exception as exc:
                st.error(f"Evaluation could not be completed: {exc}")

    if "evaluation" in st.session_state:
        evaluation = st.session_state["evaluation"]
        metrics = evaluation["metrics"]
        st.markdown("**Out-of-fold performance**")
        metric_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        st.markdown("**Confusion matrix**")
        cm_df = pd.DataFrame(
            evaluation["cm"],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown("**Classification report**")
        report_df = pd.DataFrame(evaluation["report"]).T.reset_index(names="Class / metric")
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        preds = evaluation["predictions"]
        st.markdown("**Out-of-fold participant predictions**")
        st.dataframe(preds, use_container_width=True, hide_index=True)
        st.download_button(
            "Download out-of-fold predictions (CSV)",
            preds.to_csv(index=False).encode("utf-8"),
            file_name="neurometric_gaze_oof_predictions.csv",
            mime="text/csv",
        )

        record = {
            "project": APP_TITLE,
            "analysis_type": "stratified_cross_validation",
            "n_participants": int(len(data)),
            "class_counts": {str(k): int(v) for k, v in data["phq8_binary"].value_counts().to_dict().items()},
            "metrics": {k: (None if pd.isna(v) else float(v)) for k, v in metrics.items()},
            "limitations": [
                "Research prototype; not a diagnostic tool.",
                "PHQ-8 is self-report, not clinician diagnosis.",
                "Cross-validation is internal evaluation, not external validation.",
                "Text-only transcripts cannot measure acoustic silence, pauses, prosody, overlap, or gaze.",
                "DAIC-WOZ virtual-interviewer interviews are not ordinary doctor-patient consultations.",
            ],
        }
        st.download_button(
            "Download analysis record (JSON)",
            data=__import__("json").dumps(record, indent=2).encode("utf-8"),
            file_name="neurometric_gaze_analysis_record.json",
            mime="application/json",
        )

    st.subheader("Research quality checklist")
    checklist = [
        "Verify every participant ID and PHQ-8 label against the official source metadata.",
        "Document inclusion/exclusion criteria and missing-data handling.",
        "Keep the official test set untouched if reporting benchmark performance.",
        "Use a pre-specified analysis plan before testing multiple modelling choices.",
        "Compare against simple baselines and report uncertainty, not only the best score.",
        "Conduct subgroup/error analysis only where sample sizes and ethics permit.",
        "Use two or more independent human annotators and a written codebook before reporting inter-rater reliability.",
        "Do not infer emotions, deception, fear, or concealment from lexical counts.",
        "Obtain appropriate permissions and follow dataset terms, consent, and institutional ethics requirements.",
    ]
    for item in checklist:
        st.checkbox(item, key="check_" + str(abs(hash(item))))

    st.caption(
        "For reproducibility, record the dataset version, metadata source, preprocessing decisions, "
        "software versions, random seed, and all deviations from the analysis plan."
    )

except Exception as exc:
    st.error(f"Data processing error: {exc}")
    st.info(
        "Check the uploaded file structure. A normalized CSV should contain participant_id, "
        "text, and phq8_binary. Labels must be numeric 0/1."
    )
'''
req = """streamlit>=1.36,<2
pandas>=2.1,<3
numpy>=1.26,<3
scikit-learn>=1.4,<2
"""
(out / "app.py").write_text(app_code, encoding="utf-8")
(out / "requirements.txt").write_text(req, encoding="utf-8")

zip_path = Path("/mnt/data/neurometric_gaze_fixed_project.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out / "app.py", "app.py")
    z.write(out / "requirements.txt", "requirements.txt")

print("Created:")
print(out / "app.py")
print(out / "requirements.txt")
print(zip_path)
