from pathlib import Path
import py_compile, zipfile, textwrap, os, json

base = Path("/mnt/data/neurometric_gaze_daic")
base.mkdir(parents=True, exist_ok=True)

app = r'''"""
THE NEUROMETRIC GAZE — Clinical Communication Research Edition

Dataset target:
  DAIC-WOZ (Distress Analysis Interview Corpus — Wizard-of-Oz subset)

Scientific scope:
  - DAIC-WOZ is access-controlled. Apply to USC ICT; do not redistribute it.
  - This app does not contain or fabricate DAIC-WOZ participant records.
  - Upload the officially obtained DAIC-WOZ archive as a ZIP.
  - The official train split is used for model development; official development
    split is kept separate for a held-out evaluation.
  - The official test split does not include PHQ-8 labels in the standard package,
    so it cannot be scored unless a legitimately obtained label file is available.
  - PHQ-8 is a self-report symptom questionnaire, not a clinical diagnosis.
  - Text-based emotion lexicon counts describe word associations, not a person's
    actual emotional state.

Expected generic CSV alternative:
  participant_id, text, PHQ8_Binary, PHQ8_Score
Optional: speaker, session_id, split

NRC Emotion Lexicon:
  Upload the official NRC Word-Emotion Association Lexicon text file.
  Expected tab-separated columns: word, emotion, association (0/1).
  NRC lexicon is free for research use subject to NRC terms; check current terms.
"""

from __future__ import annotations

import io
import re
import json
import math
import zipfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.utils import check_random_state

APP_NAME = "The Neurometric Gaze"
SEED = 42
EMOTIONS = ["anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"]
NEGATORS = {"not", "no", "never", "neither", "nor", "without", "hardly", "n't"}
INTENSIFIERS = {"very", "extremely", "really", "so", "deeply", "incredibly"}
DAMPENERS = {"slightly", "somewhat", "a little", "barely", "kind of", "sort of"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    # DAIC participant identifiers may be read as floats by pandas.
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def normalize_text(value: Any) -> str:
    return "" if pd.isna(value) else re.sub(r"\s+", " ", str(value)).strip()


def first_existing(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_nrc_lexicon(uploaded_file) -> pd.DataFrame:
    """
    Parse NRC Word-Emotion Association Lexicon.
    Common official format: word<TAB>emotion<TAB>association.
    """
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        word, emotion, assoc = parts[0].strip().lower(), parts[1].strip().lower(), parts[2].strip()
        if emotion not in EMOTIONS or not word:
            continue
        try:
            value = int(float(assoc))
        except ValueError:
            continue
        if value in (0, 1):
            rows.append((word, emotion, value))
    lex = pd.DataFrame(rows, columns=["word", "emotion", "association"])
    if lex.empty:
        raise ValueError("No valid NRC rows found. Expected tab-separated word, emotion, association columns.")
    return lex.drop_duplicates(["word", "emotion"], keep="last")


def fallback_lexicon() -> pd.DataFrame:
    """
    Small transparent demonstration lexicon only.
    It is NOT the NRC Emotion Lexicon and must never be reported as such.
    """
    seed = {
        "anger": "angry annoyed furious irritated mad rage hate",
        "anticipation": "expect hope anticipate eager await prepare",
        "disgust": "disgusted revolted gross repulsive nauseated",
        "fear": "afraid scared frightened anxious nervous worry worried panic",
        "joy": "happy joyful delighted pleased glad cheerful love",
        "sadness": "sad unhappy depressed lonely grief sorrow miserable",
        "surprise": "surprised astonished amazed unexpected shocking",
        "trust": "trust confident secure safe reliable believe",
    }
    return pd.DataFrame(
        [(w, emotion, 1) for emotion, words in seed.items() for w in words.split()],
        columns=["word", "emotion", "association"],
    )


def lexicon_lookup(lexicon: pd.DataFrame) -> Dict[str, set]:
    result = {emotion: set() for emotion in EMOTIONS}
    for row in lexicon.itertuples(index=False):
        if int(row.association) == 1 and row.emotion in result:
            result[row.emotion].add(str(row.word).lower())
    return result


def emotion_lexical_analysis(text: str, lookup: Dict[str, set]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Token-level NRC-style word association analysis.
    Negation and intensifier handling is intentionally conservative and heuristic.
    Counts are not emotion recognition or psychological assessment.
    """
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", normalize_text(text).lower())
    rows = []
    for i, token in enumerate(tokens):
        matched = [e for e in EMOTIONS if token in lookup.get(e, set())]
        if not matched:
            continue
        context = tokens[max(0, i - 3):i]
        negated = any(t in NEGATORS or t.endswith("n't") for t in context)
        intensified = any(t in INTENSIFIERS for t in context)
        dampened = any(t in DAMPENERS for t in context)
        for emotion in matched:
            rows.append({
                "token_index": i,
                "token": token,
                "lexicon_category": emotion,
                "preceding_negation_heuristic": negated,
                "preceding_intensifier_heuristic": intensified,
                "preceding_dampener_heuristic": dampened,
            })
    matches = pd.DataFrame(rows, columns=[
        "token_index", "token", "lexicon_category",
        "preceding_negation_heuristic",
        "preceding_intensifier_heuristic",
        "preceding_dampener_heuristic",
    ])
    counts = []
    total_tokens = max(len(tokens), 1)
    for emotion in EMOTIONS:
        subset = matches[matches["lexicon_category"] == emotion] if not matches.empty else matches
        raw = int(len(subset))
        negated = int(subset["preceding_negation_heuristic"].sum()) if raw else 0
        counts.append({
            "emotion lexicon category": emotion,
            "word-association matches": raw,
            "negation-window matches (heuristic)": negated,
            "matches per 1,000 tokens": raw / total_tokens * 1000,
            "share of all emotion matches": raw / max(len(matches), 1),
        })
    return pd.DataFrame(counts), matches


def parse_daic_zip(zip_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Parse official DAIC-WOZ ZIP structure:
      <session>_P/<session>_TRANSCRIPT.csv
      train_split_Depression_AVEC2017.csv
      dev_split_Depression_AVEC2017.csv
      test_split_Depression_AVEC2017.csv

    Returns participant-level transcript rows joined only to official train/dev
    labels. Does not infer labels for the unlabeled official test split.
    """
    notes = []
    transcripts = []
    labels_by_split = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for name in names:
            lower = name.lower()
            if lower.endswith(".csv") and any(k in lower for k in [
                "train_split_depression_avec2017",
                "dev_split_depression_avec2017",
                "test_split_depression_avec2017",
            ]):
                try:
                    label_df = pd.read_csv(zf.open(name))
                    split = "train" if "train_split" in lower else "dev" if "dev_split" in lower else "test"
                    labels_by_split[split] = label_df
                except Exception as exc:
                    notes.append(f"Could not read label file {name}: {exc}")

        transcript_files = [n for n in names if n.lower().endswith("_transcript.csv")]
        for name in transcript_files:
            # Session ID is the leading numeric folder/file component.
            match = re.search(r"(\d+)_TRANSCRIPT\.csv$", name, re.IGNORECASE)
            if not match:
                match = re.search(r"(\d+)", Path(name).name)
            if not match:
                continue
            participant_id = normalize_id(match.group(1))
            try:
                tr = pd.read_csv(zf.open(name))
            except Exception as exc:
                notes.append(f"Could not read transcript {name}: {exc}")
                continue
            speaker_col = first_existing(tr.columns, ["speaker"])
            text_col = first_existing(tr.columns, ["value", "text", "utterance"])
            if text_col is None:
                notes.append(f"Transcript {name} has no text/value column.")
                continue
            if speaker_col:
                # The DAIC-WOZ speaker field normally distinguishes Participant and Ellie.
                participant_rows = tr[
                    tr[speaker_col].astype(str).str.lower().str.contains("participant", na=False)
                ]
                if participant_rows.empty:
                    notes.append(
                        f"No speaker value containing 'Participant' found in {name}; "
                        "file skipped rather than mixing interviewer and participant speech."
                    )
                    continue
                tr = participant_rows
            text = " ".join(tr[text_col].map(normalize_text).tolist()).strip()
            if not text:
                continue
            transcripts.append({
                "participant_id": participant_id,
                "text": text,
                "source_file": name,
                "group_id": participant_id,
            })

    transcript_df = pd.DataFrame(transcripts)
    if transcript_df.empty:
        raise ValueError(
            "No DAIC-WOZ transcript files were found. Expected files ending "
            "with _TRANSCRIPT.csv inside session folders."
        )

    def normalize_labels(label_df, split):
        pid = first_existing(label_df.columns, ["Participant_ID", "participant_id", "participant"])
        binary = first_existing(label_df.columns, ["PHQ8_Binary", "PHQ8_Binary_Label", "phq8_binary"])
        score = first_existing(label_df.columns, ["PHQ8_Score", "PHQ8", "phq8_score"])
        if pid is None:
            notes.append(f"{split} label file has no Participant_ID column.")
            return pd.DataFrame()
        out = pd.DataFrame({"participant_id": label_df[pid].map(normalize_id)})
        if binary:
            out["PHQ8_Binary"] = pd.to_numeric(label_df[binary], errors="coerce")
        if score:
            out["PHQ8_Score"] = pd.to_numeric(label_df[score], errors="coerce")
        out["official_split"] = split
        return out

    label_frames = []
    for split in ("train", "dev", "test"):
        if split in labels_by_split:
            lab = normalize_labels(labels_by_split[split], split)
            if not lab.empty:
                label_frames.append(lab)
    labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()

    if not labels.empty:
        merged = transcript_df.merge(labels, on="participant_id", how="left", validate="one_to_one")
    else:
        merged = transcript_df.copy()
        merged["PHQ8_Binary"] = np.nan
        merged["PHQ8_Score"] = np.nan
        merged["official_split"] = "unknown"

    notes.append(
        "DAIC-WOZ files are access-controlled. Confirm the archive's provenance and "
        "permitted use. Participant transcript text is sensitive even when names are absent."
    )
    return merged, transcript_df, labels, notes


def load_generic_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    pid = first_existing(df.columns, ["participant_id", "Participant_ID", "session_id", "group_id"])
    txt = first_existing(df.columns, ["text", "transcript", "value", "utterance"])
    binary = first_existing(df.columns, ["PHQ8_Binary", "phq8_binary", "label", "target"])
    score = first_existing(df.columns, ["PHQ8_Score", "phq8_score"])
    if pid is None or txt is None:
        raise ValueError("CSV requires participant_id/Participant_ID and text/transcript columns.")
    out = pd.DataFrame({
        "participant_id": df[pid].map(normalize_id),
        "group_id": df[pid].map(normalize_id),
        "text": df[txt].map(normalize_text),
    })
    out["PHQ8_Binary"] = pd.to_numeric(df[binary], errors="coerce") if binary else np.nan
    out["PHQ8_Score"] = pd.to_numeric(df[score], errors="coerce") if score else np.nan
    out["official_split"] = df["split"].astype(str).str.lower() if "split" in df.columns else "user_supplied"
    out = out[(out["participant_id"] != "") & (out["text"] != "")]
    return out.reset_index(drop=True)


def make_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=30000,
        )),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=SEED,
        )),
    ])


def classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_class_precision": float(p[1]),
        "positive_class_recall": float(r[1]),
        "positive_class_f1": float(f[1]),
        "negative_class_precision": float(p[0]),
        "negative_class_recall": float(r[0]),
        "negative_class_f1": float(f[0]),
        "support_negative": int(s[0]),
        "support_positive": int(s[1]),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        result["brier_score"] = float(brier_score_loss(y_true, y_prob))
    return result


def nested_group_cv(train_df: pd.DataFrame, n_outer: int = 5) -> Dict[str, Any]:
    """
    Nested stratified group CV. Each participant is a group.
    Hyperparameters are tuned only inside each outer training fold.
    """
    data = train_df.dropna(subset=["PHQ8_Binary"]).copy()
    data["PHQ8_Binary"] = data["PHQ8_Binary"].astype(int)
    if data["participant_id"].duplicated().any():
        # The analysis unit is one participant-level transcript. Multiple rows
        # per participant are aggregated before CV to avoid pseudo-replication.
        data = data.groupby("participant_id", as_index=False).agg({
            "text": lambda x: " ".join(x.astype(str)),
            "PHQ8_Binary": "first",
            "PHQ8_Score": "first",
            "group_id": "first",
            "official_split": "first",
        })
    if data["PHQ8_Binary"].nunique() != 2:
        raise ValueError("PHQ8_Binary must contain both 0 and 1.")
    min_class = int(data["PHQ8_Binary"].value_counts().min())
    n_groups = data["group_id"].nunique()
    folds = min(n_outer, min_class, n_groups)
    if folds < 3:
        raise ValueError("At least three observations in each class and three groups are needed for nested CV.")
    X = data["text"].astype(str).to_numpy()
    y = data["PHQ8_Binary"].to_numpy()
    groups = data["group_id"].astype(str).to_numpy()
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)

    oof_pred = np.zeros(len(data), dtype=int)
    oof_prob = np.zeros(len(data), dtype=float)
    fold_ids = np.zeros(len(data), dtype=int)
    fold_rows = []

    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups), start=1):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        gtr = groups[tr_idx]
        inner_min = int(pd.Series(ytr).value_counts().min())
        inner_groups = len(np.unique(gtr))
        inner_n = min(3, inner_min, inner_groups)
        if inner_n < 2:
            raise ValueError(
                f"Outer fold {fold} does not contain enough class/group diversity "
                "for inner tuning. Use fewer folds only if justified, or obtain more data."
            )
        inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=SEED + fold)
        search = GridSearchCV(
            make_model(), grid, scoring="f1_macro", cv=inner,
            n_jobs=-1, refit=True, error_score="raise"
        )
        search.fit(Xtr, ytr, groups=gtr)
        pred = search.predict(Xte)
        prob = search.predict_proba(Xte)[:, list(search.classes_).index(1)]
        oof_pred[te_idx] = pred
        oof_prob[te_idx] = prob
        fold_ids[te_idx] = fold
        fold_rows.append({
            "outer_fold": fold,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        })

    return {
        "data": data,
        "truth": y,
        "pred": oof_pred,
        "prob": oof_prob,
        "fold_ids": fold_ids,
        "metrics": classification_metrics(y, oof_pred, oof_prob),
        "fold_table": pd.DataFrame(fold_rows),
    }


def bootstrap_ci(y, pred, n_boot=2000, seed=SEED):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(f1_score(y[idx], pred[idx], average="macro", zero_division=0))
    if len(values) < 100:
        raise ValueError("Too few valid bootstrap replicates; sample may be too small or highly imbalanced.")
    return {
        "estimate": float(f1_score(y, pred, average="macro", zero_division=0)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "valid_replicates": len(values),
    }


def train_on_official_train_evaluate_dev(df: pd.DataFrame) -> Dict[str, Any]:
    train = df[(df["official_split"].astype(str).str.lower() == "train")].dropna(subset=["PHQ8_Binary"]).copy()
    dev = df[(df["official_split"].astype(str).str.lower().isin(["dev", "development", "validation"]))].dropna(subset=["PHQ8_Binary"]).copy()
    if train.empty or dev.empty:
        raise ValueError(
            "Official train and dev labelled rows are required. The official DAIC-WOZ "
            "test split normally has no PHQ-8 labels in the distributed metadata."
        )
    train["PHQ8_Binary"] = train["PHQ8_Binary"].astype(int)
    dev["PHQ8_Binary"] = dev["PHQ8_Binary"].astype(int)
    if train["PHQ8_Binary"].nunique() != 2:
        raise ValueError("Training split needs both PHQ8_Binary classes.")
    model = make_model()
    model.fit(train["text"].astype(str), train["PHQ8_Binary"])
    pred = model.predict(dev["text"].astype(str))
    prob = model.predict_proba(dev["text"].astype(str))[:, list(model.classes_).index(1)]
    return {
        "train": train, "dev": dev, "model": model,
        "truth": dev["PHQ8_Binary"].to_numpy(),
        "pred": pred, "prob": prob,
        "metrics": classification_metrics(dev["PHQ8_Binary"], pred, prob),
    }


def safe_download_df(df, name):
    return st.download_button(
        name,
        df.to_csv(index=False).encode("utf-8"),
        file_name=name.lower().replace(" ", "_") + ".csv",
        mime="text/csv",
    )


# ------------------------------ Streamlit UI ------------------------------

st.set_page_config(page_title=APP_NAME, page_icon="◉", layout="wide")
st.title(APP_NAME)
st.caption("Clinical communication, participant narratives, and explainable language analysis")
st.warning(
    "Research prototype only. No diagnosis, triage, treatment recommendation, "
    "or inference of a person's actual emotional state. Do not upload identifiable "
    "patient data unless you have explicit institutional approval and lawful authority."
)

if "daic_df" not in st.session_state:
    st.session_state.daic_df = None
if "cv_result" not in st.session_state:
    st.session_state.cv_result = None
if "dev_result" not in st.session_state:
    st.session_state.dev_result = None
if "lexicon" not in st.session_state:
    st.session_state.lexicon = fallback_lexicon()
if "lexicon_is_nrc" not in st.session_state:
    st.session_state.lexicon_is_nrc = False

with st.sidebar:
    st.header("Research modules")
    page = st.radio("Navigate", [
        "Research overview",
        "Dataset ingestion",
        "Lexical emotion analysis",
        "Internal validation",
        "Official development evaluation",
        "Error analysis",
        "Methodology and ethics",
    ])
    st.divider()
    st.caption("English-only interface | Reproducibility seed 42")

df = st.session_state.daic_df

if page == "Research overview":
    st.header("Research design")
    st.markdown("""
    This application is designed around the **DAIC-WOZ** corpus, a collection of
    clinical-style interviews with transcripts and questionnaire metadata. The
    official corpus contains 189 sessions, and access is restricted to academic
    and non-profit researchers following application and consent conditions.

    **Primary research question (recommended):**  
    *To what extent do lexical and linguistic features in participant speech
    associate with PHQ-8 self-reported symptom scores in the DAIC-WOZ sample?*

    This is an association/prediction research question, not a diagnostic claim.
    """)
    st.subheader("Why this dataset?")
    st.write(
        "DAIC-WOZ contains transcribed participant-interviewer sessions and PHQ-8 "
        "questionnaire metadata. The official documentation describes 189 sessions "
        "and separate train, development, and test split files. The test metadata "
        "does not include PHQ-8 labels in the standard package."
    )
    st.subheader("Separate constructs")
    st.markdown("""
    - **Lexical emotion association:** counts words associated with eight emotion categories.
    - **PHQ-8 score:** participant-reported depressive symptom severity on a questionnaire.
    - **Model prediction:** a statistical estimate learned from the supplied development labels.
    
    These are not interchangeable. A word associated with fear does not prove fear;
    a PHQ-8 score is not a clinical diagnosis; predictive performance does not establish causation.
    """)
    st.info("Obtain the corpus from USC ICT before running any real-data analysis.")

elif page == "Dataset ingestion":
    st.header("Dataset ingestion")
    st.markdown("""
    **Option A — DAIC-WOZ archive:** upload the official ZIP containing session folders,
    `*_TRANSCRIPT.csv`, and the official train/dev label CSVs.

    **Option B — normalized CSV:** provide `participant_id`, `text`,
    `PHQ8_Binary`, `PHQ8_Score`, and optionally `split`.
    """)
    daic_zip = st.file_uploader("Official DAIC-WOZ ZIP archive", type=["zip"], key="daic_zip")
    generic = st.file_uploader("Or upload a normalized participant-level CSV", type=["csv"], key="generic_csv")
    if st.button("Load dataset", type="primary", disabled=(daic_zip is None and generic is None)):
        try:
            if daic_zip is not None:
                merged, raw_transcripts, labels, notes = parse_daic_zip(daic_zip.getvalue())
                st.session_state.daic_df = merged
                st.session_state.ingestion_notes = notes
                st.session_state.data_source = "DAIC-WOZ official archive"
            else:
                normalized = load_generic_csv(generic)
                st.session_state.daic_df = normalized
                st.session_state.ingestion_notes = [
                    "Generic CSV format supplied by user; verify provenance, permissions, and official split integrity."
                ]
                st.session_state.data_source = "User-supplied normalized CSV"
            st.session_state.cv_result = None
            st.session_state.dev_result = None
            st.success(f"Loaded {len(st.session_state.daic_df)} participant-level record(s).")
        except Exception as exc:
            st.error(f"Dataset could not be loaded: {exc}")

    df = st.session_state.daic_df
    if df is not None and not df.empty:
        st.subheader("Dataset audit")
        cols = st.columns(4)
        cols[0].metric("Records", len(df))
        cols[1].metric("Unique participants", df["participant_id"].nunique())
        cols[2].metric("PHQ-8 binary labels", int(df["PHQ8_Binary"].notna().sum()))
        cols[3].metric("PHQ-8 score labels", int(df["PHQ8_Score"].notna().sum()))
        st.write("Source:", st.session_state.get("data_source", "Unknown"))
        for note in st.session_state.get("ingestion_notes", []):
            st.caption(note)
        st.dataframe(
            df[["participant_id", "official_split", "PHQ8_Binary", "PHQ8_Score"]].head(30),
            use_container_width=True,
        )
        st.subheader("Official split distribution")
        st.dataframe(
            df.groupby("official_split", dropna=False).agg(
                participants=("participant_id", "nunique"),
                labelled=("PHQ8_Binary", "count"),
            ).reset_index(),
            use_container_width=True,
            hide_index=True,
        )
        st.warning("Avoid exporting or sharing raw transcript text. It may contain sensitive disclosures.")

elif page == "Lexical emotion analysis":
    st.header("Lexical emotion analysis")
    st.markdown("""
    This module uses the **NRC Word-Emotion Association Lexicon** when you upload
    the official lexicon file. NRC associates English words with eight emotion
    categories: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.

    These are lexical associations, not validated measurements of a speaker's internal
    emotional state. Context, negation, metaphor, quotation, sarcasm, and speaker intent
    can change meaning.
    """)
    lex_upload = st.file_uploader("Upload NRC Emotion Lexicon text file (optional)", type=["txt"], key="nrc")
    if lex_upload is not None:
        try:
            st.session_state.lexicon = load_nrc_lexicon(lex_upload)
            st.session_state.lexicon_is_nrc = True
            st.success(f"Loaded NRC-compatible lexicon: {len(st.session_state.lexicon):,} word-emotion rows.")
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.lexicon_is_nrc:
        st.success("NRC lexicon loaded.")
    else:
        st.warning(
            "Using a small built-in demonstration word list, NOT the full NRC lexicon. "
            "For a research analysis, upload the official NRC lexicon obtained under its terms."
        )
    lookup = lexicon_lookup(st.session_state.lexicon)
    text_choice = st.radio("Analysis source", ["Enter de-identified text", "Analyze loaded participant transcript"])
    text = ""
    if text_choice == "Enter de-identified text":
        text = st.text_area("English text", height=180)
    elif df is not None and not df.empty:
        choices = df["participant_id"].astype(str).tolist()
        selected_pid = st.selectbox("Participant ID", choices)
        text = df.loc[df["participant_id"].astype(str) == selected_pid, "text"].iloc[0]
        st.text_area("Selected participant text (preview)", text[:3000], height=150, disabled=True)
    else:
        st.info("Load a dataset first, or switch to manual text input.")
    if st.button("Run lexical analysis", type="primary", disabled=not bool(text.strip())):
        summary, matches = emotion_lexical_analysis(text, lookup)
        tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        st.metric("Word tokens", len(tokens))
        st.metric("Lexicon matches", len(matches))
        st.subheader("Emotion-word association profile")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(summary.set_index("emotion lexicon category")["matches per 1,000 tokens"])
        if not matches.empty:
            st.subheader("Matched lexical items")
            st.dataframe(matches, use_container_width=True, hide_index=True)
        st.download_button(
            "Download lexical analysis CSV",
            summary.to_csv(index=False).encode("utf-8"),
            file_name="nrc_lexical_emotion_summary.csv",
            mime="text/csv",
        )
        st.caption(
            "Negation, intensifier, and dampener columns are local-window heuristics. "
            "They are not syntactic scope resolution and should not be interpreted as corrected emotion scores."
        )

elif page == "Internal validation":
    st.header("Nested internal validation")
    st.write(
        "Runs nested StratifiedGroupKFold on the official training split (or all labelled "
        "rows if a custom dataset has no official train split). The participant is the "
        "grouping unit. Hyperparameter selection occurs only within outer training folds."
    )
    if df is None or df.empty:
        st.info("Load a dataset first.")
    else:
        if "official_split" in df and (df["official_split"].astype(str).str.lower() == "train").any():
            model_data = df[df["official_split"].astype(str).str.lower() == "train"].copy()
        else:
            model_data = df.copy()
        st.write(f"Records available for internal CV: {len(model_data.dropna(subset=['PHQ8_Binary']))}")
        if st.button("Run nested group cross-validation", type="primary"):
            try:
                with st.spinner("Running nested group-aware model validation..."):
                    result = nested_group_cv(model_data)
                st.session_state.cv_result = result
                st.success("Nested validation completed.")
            except Exception as exc:
                st.session_state.cv_result = None
                st.error(str(exc))
        result = st.session_state.cv_result
        if result is not None:
            m = result["metrics"]
            c = st.columns(4)
            c[0].metric("Macro F1", f"{m['macro_f1']:.3f}")
            c[1].metric("Balanced accuracy", f"{m['balanced_accuracy']:.3f}")
            c[2].metric("ROC AUC", f"{m.get('roc_auc', float('nan')):.3f}")
            c[3].metric("Participants", m["n"])
            st.subheader("Class-specific metrics")
            st.json(m)
            st.subheader("Confusion matrix")
            st.dataframe(
                pd.DataFrame(
                    confusion_matrix(result["truth"], result["pred"], labels=[0, 1]),
                    index=["Observed 0", "Observed 1"],
                    columns=["Predicted 0", "Predicted 1"],
                )
            )
            st.subheader("Outer fold details")
            st.dataframe(result["fold_table"], use_container_width=True, hide_index=True)
            if st.button("Calculate participant-level bootstrap interval"):
                try:
                    st.session_state.cv_ci = bootstrap_ci(result["truth"], result["pred"])
                except Exception as exc:
                    st.error(str(exc))
            if "cv_ci" in st.session_state:
                st.json(st.session_state.cv_ci)
            oof = pd.DataFrame({
                "participant_id": result["data"]["participant_id"].to_numpy(),
                "PHQ8_Binary_reference": result["truth"],
                "prediction": result["pred"],
                "predicted_probability_class_1": result["prob"],
                "outer_fold": result["fold_ids"],
            })
            st.download_button(
                "Download out-of-fold predictions",
                oof.to_csv(index=False).encode("utf-8"),
                file_name="daic_nested_cv_oof_predictions.csv",
                mime="text/csv",
            )
            st.caption(
                "The sample is small. Confidence intervals may be wide, and nested CV "
                "does not replace an independent external cohort."
            )

elif page == "Official development evaluation":
    st.header("Official development split evaluation")
    st.write(
        "This module trains on the official DAIC-WOZ training split and evaluates on "
        "the separately supplied official development split. Do not use development "
        "results to repeatedly tune the model and then report them as untouched test results."
    )
    if df is None or df.empty:
        st.info("Load the official DAIC-WOZ archive first.")
    else:
        if st.button("Train on official train; evaluate on official dev", type="primary"):
            try:
                with st.spinner("Training and evaluating..."):
                    result = train_on_official_train_evaluate_dev(df)
                st.session_state.dev_result = result
                st.success("Development evaluation completed.")
            except Exception as exc:
                st.session_state.dev_result = None
                st.error(str(exc))
        result = st.session_state.dev_result
        if result is not None:
            m = result["metrics"]
            st.json(m)
            st.dataframe(
                pd.DataFrame(
                    confusion_matrix(result["truth"], result["pred"], labels=[0, 1]),
                    index=["Observed 0", "Observed 1"],
                    columns=["Predicted 0", "Predicted 1"],
                )
            )
            dev_predictions = pd.DataFrame({
                "participant_id": result["dev"]["participant_id"].to_numpy(),
                "PHQ8_Binary_reference": result["truth"],
                "prediction": result["pred"],
                "predicted_probability_class_1": result["prob"],
            })
            st.download_button(
                "Download official dev predictions",
                dev_predictions.to_csv(index=False).encode("utf-8"),
                file_name="daic_official_dev_predictions.csv",
                mime="text/csv",
            )
            st.warning(
                "This is a held-out development evaluation, not the official test-set result. "
                "If you use these results for model selection, they are no longer an untouched final test."
            )

elif page == "Error analysis":
    st.header("Qualitative error analysis")
    result = st.session_state.cv_result
    if result is None:
        st.info("Run nested internal validation first.")
    else:
        data = result["data"].reset_index(drop=True)
        errors = data.loc[result["truth"] != result["pred"], ["participant_id", "text", "PHQ8_Binary"]].copy()
        errors["predicted_label"] = result["pred"][result["truth"] != result["pred"]]
        st.write(f"Misclassified participants: {len(errors)} / {len(data)}")
        if not errors.empty:
            st.dataframe(errors[["participant_id", "PHQ8_Binary", "predicted_label"]], use_container_width=True)
            st.download_button(
                "Download error records (IDs and labels only)",
                errors.drop(columns=["text"]).to_csv(index=False).encode("utf-8"),
                file_name="error_analysis_metadata.csv",
                mime="text/csv",
            )
        st.markdown("""
        Review error patterns without exposing transcript text:
        - lexical ambiguity and negation;
        - interviewer prompts and context dependence;
        - transcript omissions or ASR errors;
        - participant-level differences and potential demographic confounding;
        - label noise and PHQ-8 threshold effects;
        - whether the model learns topic or interview structure rather than the intended construct.
        
        Avoid sharing identifiable or sensitive transcript excerpts in reports.
        """)

elif page == "Methodology and ethics":
    st.header("Methodology and ethics")
    st.subheader("Dataset provenance")
    st.markdown("""
    DAIC-WOZ is an access-controlled research corpus hosted by USC ICT. Its official
    documentation describes 189 sessions, transcripts, participant audio/video,
    and PHQ-8 metadata. Access is limited to academic and non-profit researchers
    under the corpus's application and consent conditions.
    
    The official train and development label files contain participant IDs and PHQ-8
    scores/binary labels. The standard official test label file contains participant
    IDs and gender, but not PHQ-8 labels. Therefore, this app does not invent test labels.
    """)
    st.subheader("Construct validity")
    st.markdown("""
    - PHQ-8 Binary is a thresholded self-report questionnaire variable (the corpus documentation describes score >= 10).
    - It is not a clinician's diagnosis.
    - NRC word-emotion associations are lexical annotations, not a direct measurement of felt emotion.
    - A relationship between word counts and PHQ-8 does not establish causal or diagnostic meaning.
    """)
    st.subheader("Recommended analysis plan")
    st.markdown("""
    1. Pre-register the research question, unit of analysis, outcomes, exclusions, and metrics.
    2. Use participant as the independent grouping unit.
    3. Keep the official development split separate from training.
    4. Use nested, group-aware cross-validation only within the training split for internal model selection.
    5. Report macro F1, balanced accuracy, class-specific precision/recall, ROC AUC, confusion matrix, and uncertainty.
    6. Compare a majority-class baseline and a simple TF-IDF baseline.
    7. Treat lexical emotion analysis as exploratory; report raw counts and normalized rates.
    8. Seek an independent external cohort before making generalisability claims.
    9. Conduct subgroup/error analysis only where sample sizes support responsible interpretation.
    10. Follow institutional ethics, data-use agreements, and disclosure rules.
    """)
    st.subheader("Reporting guidance")
    st.markdown("""
    - TRIPOD+AI: reporting prediction model studies.
    - PROBAST+AI: risk of bias and applicability.
    - DAIC-WOZ corpus paper and official documentation: dataset provenance and field definitions.
    - NRC Emotion Lexicon paper and documentation: lexical association resource and limitations.
    """)
    st.subheader("Checklist")
    for item in [
        "I have formal permission to use this dataset.",
        "The data were obtained from the official source.",
        "I have not redistributed the dataset.",
        "The unit of analysis is participant-level.",
        "No participant appears in both training and evaluation groups.",
        "The official development split was not used for repeated tuning.",
        "PHQ-8 is described as self-reported symptom severity, not diagnosis.",
        "Emotion lexicon outputs are described as lexical associations, not felt emotions.",
        "No identifiable transcript text is included in public reports.",
    ]:
        st.checkbox(item, key="eth_" + str(abs(hash(item))))

st.divider()
st.caption(
    "The Neurometric Gaze | Research prototype. Outputs are exploratory and must not "
    "be used for diagnosis, triage, treatment, or individual clinical decisions."
)
'''

req = """streamlit>=1.36,<2.0
pandas>=2.1,<3.0
numpy>=1.26,<3.0
scikit-learn>=1.4,<2.0
"""

(base / "app.py").write_text(app, encoding="utf-8")
(base / "requirements.txt").write_text(req, encoding="utf-8")
py_compile.compile(str(base / "app.py"), doraise=True)

zip_path = Path("/mnt/data/neurometric_gaze_daic_research_project.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(base / "app.py", "app.py")
    z.write(base / "requirements.txt", "requirements.txt")

print("Created:", base / "app.py")
print("Lines:", len(app.splitlines()))
print("Python syntax compilation: PASSED")
print("Created ZIP:", zip_path)
