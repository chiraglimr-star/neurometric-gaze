from pathlib import Path
import textwrap, zipfile, py_compile, json, os, traceback

root = Path("/mnt/data/neurometric_gaze_complete")
(root / "src").mkdir(parents=True, exist_ok=True)
(root / "pages").mkdir(parents=True, exist_ok=True)
(root / "tests").mkdir(parents=True, exist_ok=True)
(root / "config").mkdir(parents=True, exist_ok=True)
(root / "data").mkdir(parents=True, exist_ok=True)

files = {}

files["requirements.txt"] = """streamlit>=1.36,<2
pandas>=2.1,<3
numpy>=1.26,<3
scipy>=1.11,<2
scikit-learn>=1.4,<2
matplotlib>=3.8,<4
PyYAML>=6,<7
statsmodels>=0.14,<1
"""

files["src/__init__.py"] = ""

files["src/ingestion.py"] = r'''
from __future__ import annotations
from pathlib import Path
import io, zipfile
import pandas as pd

ALIASES = {
    "participant_id": ["participant_id", "participant", "id", "subject_id"],
    "text": ["text", "transcript", "narrative", "value"],
    "phq8_binary": ["phq8_binary", "phq_8_binary", "label", "target"],
    "phq8_score": ["phq8_score", "phq_8_score", "phq8"],
    "split": ["split", "set", "partition"],
    "speaker": ["speaker", "role"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for target, candidates in ALIASES.items():
        for candidate in candidates:
            if candidate in lower:
                rename[lower[candidate]] = target
                break
    return df.rename(columns=rename).copy()

def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))

def read_daic_zip(raw: bytes) -> pd.DataFrame:
    """Extract participant utterances from *_TRANSCRIPT.csv files only."""
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        transcript_files = [n for n in archive.namelist()
                            if n.lower().endswith("_transcript.csv")]
        if not transcript_files:
            raise ValueError("No *_TRANSCRIPT.csv files were found in the ZIP.")
        for name in transcript_files:
            with archive.open(name) as handle:
                transcript = pd.read_csv(handle)
            if not {"speaker", "value"}.issubset(transcript.columns):
                continue
            mask = transcript["speaker"].astype(str).str.strip().str.lower().eq("participant")
            utterances = transcript.loc[mask, "value"].dropna().astype(str)
            participant_id = Path(name).name.split("_")[0]
            rows.append({
                "participant_id": str(participant_id),
                "text": " ".join(utterances),
            })
        if not rows:
            raise ValueError("Transcript files were found, but no participant speech was extracted.")
        data = pd.DataFrame(rows)

        # Merge a likely split/label metadata CSV only when a participant key is identifiable.
        metadata_candidates = [
            n for n in archive.namelist()
            if n.lower().endswith(".csv")
            and "transcript" not in n.lower()
            and ("split" in n.lower() or "label" in n.lower() or "metadata" in n.lower())
        ]
        if metadata_candidates:
            with archive.open(metadata_candidates[0]) as handle:
                metadata = normalize_columns(pd.read_csv(handle))
            if "participant_id" in metadata.columns:
                data["participant_id"] = data["participant_id"].astype(str)
                metadata["participant_id"] = metadata["participant_id"].astype(str)
                data = data.merge(metadata, on="participant_id", how="left")
    return data

def load_uploaded_file(name: str, raw: bytes) -> pd.DataFrame:
    lower = name.lower()
    if lower.endswith(".csv"):
        return read_csv_bytes(raw)
    if lower.endswith(".zip"):
        return read_daic_zip(raw)
    raise ValueError("Supported formats are CSV and ZIP.")
'''

files["src/preprocessing.py"] = r'''
from __future__ import annotations
import re
import pandas as pd
import numpy as np
from .ingestion import normalize_columns

REQUIRED = {"participant_id", "text", "phq8_binary"}

def prepare_participant_data(df: pd.DataFrame, min_tokens: int = 1) -> tuple[pd.DataFrame, dict]:
    data = normalize_columns(df)
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    if "speaker" in data.columns:
        data = data[data["speaker"].astype(str).str.strip().str.lower().eq("participant")].copy()

    data["participant_id"] = data["participant_id"].astype(str).str.strip()
    data["text"] = data["text"].fillna("").astype(str)
    data["phq8_binary"] = pd.to_numeric(data["phq8_binary"], errors="coerce")
    invalid_labels = int((~data["phq8_binary"].isin([0, 1])).sum())
    data = data[data["phq8_binary"].isin([0, 1])].copy()
    data["token_count_row"] = data["text"].map(lambda x: len(re.findall(r"\b[\w']+\b", x)))

    if "phq8_score" in data.columns:
        data["phq8_score"] = pd.to_numeric(data["phq8_score"], errors="coerce")

    aggregation = {
        "text": lambda values: " ".join(v for v in values.astype(str) if v.strip()),
        "phq8_binary": "first",
        "token_count_row": "sum",
    }
    if "phq8_score" in data.columns:
        aggregation["phq8_score"] = "first"
    if "split" in data.columns:
        aggregation["split"] = "first"

    # Conflicting participant labels are excluded rather than silently choosing one.
    label_counts = data.groupby("participant_id")["phq8_binary"].nunique()
    conflicting_ids = set(label_counts[label_counts > 1].index)
    data = data[~data["participant_id"].isin(conflicting_ids)].copy()
    participant = data.groupby("participant_id", as_index=False).agg(aggregation)
    participant = participant.rename(columns={"token_count_row": "token_count"})
    participant["phq8_binary"] = participant["phq8_binary"].astype(int)
    participant = participant[participant["token_count"] >= int(min_tokens)].copy()
    participant = participant[participant["text"].str.strip().ne("")].copy()

    audit = {
        "input_rows": int(len(df)),
        "retained_participants": int(len(participant)),
        "invalid_or_missing_label_rows": invalid_labels,
        "conflicting_label_participants_excluded": int(len(conflicting_ids)),
        "duplicate_ids_after_aggregation": int(participant["participant_id"].duplicated().sum()),
        "empty_text_participants_excluded": int((participant["text"].str.strip() == "").sum()),
        "analysis_unit": "participant",
    }
    return participant.reset_index(drop=True), audit
'''

files["src/statistics.py"] = r'''
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

def descriptive_survey(data: pd.DataFrame) -> dict:
    out = {"n": int(len(data)), "class_counts": data["phq8_binary"].value_counts().sort_index().to_dict()}
    if "phq8_score" in data.columns:
        scores = pd.to_numeric(data["phq8_score"], errors="coerce").dropna()
        out["phq8_score_summary"] = scores.describe().to_dict()
        if len(scores) >= 1:
            out["phq8_score_n"] = int(len(scores))
    return out

def exploratory_correlations(data: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_columns:
        if col not in data.columns:
            continue
        tmp = data[[col, "phq8_score"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tmp) < 5 or tmp[col].nunique() < 2 or tmp["phq8_score"].nunique() < 2:
            continue
        rho, p = spearmanr(tmp[col], tmp["phq8_score"])
        rows.append({"variable": col, "n": len(tmp), "spearman_rho": rho, "p_raw": p})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = multipletests(result["p_raw"], method="holm")[1]
    return result

def permutation_test_auc(y, probabilities, n_permutations=2000, seed=2026):
    """Permutation test on fixed out-of-fold probabilities; descriptive test, not a replacement for external validation."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("AUC permutation test requires both outcome classes.")
    observed = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = roc_auc_score(rng.permutation(y), p)
    p_value = (1 + np.sum(null >= observed)) / (n_permutations + 1)
    return {"observed_auc": float(observed), "permutation_p": float(p_value),
            "permutations": int(n_permutations), "null_auc_mean": float(null.mean())}
'''

files["src/modelling.py"] = r'''
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix
)

def _n_splits(y, desired):
    counts = pd.Series(y).value_counts()
    if len(counts) != 2:
        raise ValueError("Both binary outcome classes must be present.")
    return min(desired, int(counts.min()))

def make_text_pipeline(max_features=5000):
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                 max_features=max_features, sublinear_tf=True,
                                 strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=2, max_features=max(500, max_features // 3),
                                 sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear",
            random_state=2026
        ))
    ])

def evaluate_nested_group_cv(data: pd.DataFrame, max_features=5000, outer_splits=5,
                             inner_splits=3, seed=2026):
    X = data["text"].astype(str).to_numpy()
    y = data["phq8_binary"].astype(int).to_numpy()
    groups = data["participant_id"].astype(str).to_numpy()

    n_outer = _n_splits(y, outer_splits)
    if n_outer < 2:
        raise ValueError("Insufficient participants per class for grouped cross-validation.")

    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    oof_prob = np.full(len(data), np.nan)
    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups), start=1):
        y_train = y[train_idx]
        n_inner = _n_splits(y_train, inner_splits)
        if n_inner < 2:
            raise ValueError(f"Fold {fold}: insufficient class counts for inner grouped CV.")
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + fold)
        inner_splits_list = list(inner.split(X[train_idx], y_train, groups[train_idx]))

        search = GridSearchCV(
            make_text_pipeline(max_features),
            param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0]},
            scoring="f1_macro",
            cv=inner_splits_list,
            n_jobs=-1,
            refit=True,
            error_score="raise"
        )
        search.fit(X[train_idx], y_train)
        probabilities = search.best_estimator_.predict_proba(X[test_idx])[:, 1]
        oof_prob[test_idx] = probabilities
        fold_records.append({
            "fold": fold,
            "n_test": int(len(test_idx)),
            "best_C": float(search.best_params_["classifier__C"]),
        })

    valid = np.isfinite(oof_prob)
    yt, yp = y[valid], oof_prob[valid]
    pred = (yp >= 0.5).astype(int)
    metrics = {
        "n_oof": int(valid.sum()),
        "accuracy": float(accuracy_score(yt, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
        "brier_score": float(brier_score_loss(yt, yp)),
        "roc_auc": float(roc_auc_score(yt, yp)) if len(np.unique(yt)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(yt, yp)) if len(np.unique(yt)) == 2 else float("nan"),
    }

    # Majority baseline evaluated with same outer fold allocation.
    baseline_prob = np.full(len(data), np.nan)
    for train_idx, test_idx in outer.split(X, y, groups):
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(np.zeros((len(train_idx), 1)), y[train_idx])
        baseline_prob[test_idx] = dummy.predict_proba(np.zeros((len(test_idx), 1)))[:, 1]
    bvalid = np.isfinite(baseline_prob)
    baseline_pred = (baseline_prob[bvalid] >= 0.5).astype(int)
    baseline = {
        "accuracy": float(accuracy_score(y[bvalid], baseline_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[bvalid], baseline_pred)),
        "macro_f1": float(f1_score(y[bvalid], baseline_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y[bvalid], baseline_prob[bvalid])),
    }

    predictions = pd.DataFrame({
        "participant_id": groups[valid],
        "observed": yt,
        "predicted_probability": yp,
        "predicted_at_0_5": pred,
    })
    return {
        "metrics": metrics,
        "baseline": baseline,
        "folds": pd.DataFrame(fold_records),
        "predictions": predictions,
        "classification_report": classification_report(yt, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(yt, pred, labels=[0, 1]),
    }

def participant_bootstrap_auc(predictions: pd.DataFrame, n_boot=1000, seed=2026):
    from sklearn.metrics import roc_auc_score
    ids = predictions["participant_id"].astype(str).unique()
    if len(ids) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = []
    grouped = {pid: predictions[predictions["participant_id"].astype(str) == pid]
               for pid in ids}
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat([grouped[pid] for pid in sampled], ignore_index=True)
        if sample["observed"].nunique() < 2:
            continue
        values.append(roc_auc_score(sample["observed"], sample["predicted_probability"]))
    if not values:
        return (float("nan"), float("nan"))
    return tuple(np.quantile(values, [0.025, 0.975]))
'''

files["src/annotation.py"] = r'''
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd

def krippendorff_alpha_nominal(long_df: pd.DataFrame) -> float:
    required = {"participant_id", "segment_id", "annotator_id", "label"}
    if not required.issubset(long_df.columns):
        raise ValueError("Annotation data must contain participant_id, segment_id, annotator_id, label.")
    frame = long_df.copy()
    frame["unit"] = frame["participant_id"].astype(str) + "::" + frame["segment_id"].astype(str)
    wide = frame.pivot_table(index="unit", columns="annotator_id", values="label", aggfunc="first")
    values = wide.to_numpy(dtype=object)
    coincidence = {}
    for row in values:
        ratings = [v for v in row if pd.notna(v)]
        m = len(ratings)
        if m < 2:
            continue
        for i, j in combinations(range(m), 2):
            a, b = ratings[i], ratings[j]
            coincidence[(a, b)] = coincidence.get((a, b), 0.0) + 1.0 / (m - 1)
            coincidence[(b, a)] = coincidence.get((b, a), 0.0) + 1.0 / (m - 1)
    total = sum(coincidence.values())
    if total == 0:
        return float("nan")
    categories = set()
    for a, b in coincidence:
        categories.update([a, b])
    marginals = {c: sum(v for (a, _), v in coincidence.items() if a == c)
                 for c in categories}
    observed_disagreement = sum(v for (a, b), v in coincidence.items() if a != b) / total
    marginal_total = sum(marginals.values())
    denominator = total * (total - 1)
    if denominator <= 0:
        return float("nan")
    expected_disagreement = sum(
        marginals[c] * (marginal_total - marginals[c]) for c in categories
    ) / denominator
    if expected_disagreement == 0:
        return float("nan")
    return float(1 - observed_disagreement / expected_disagreement)
'''

files["src/reporting.py"] = r'''
from __future__ import annotations
import json
from datetime import datetime, timezone

def build_research_report(dataset_name, audit, metrics=None, baseline=None, notes=None):
    report = {
        "project": "The Neurometric Gaze",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_label": dataset_name,
        "data_audit": audit or {},
        "model_metrics": metrics or None,
        "baseline_metrics": baseline or None,
        "researcher_notes": notes or "",
        "interpretation_boundary": (
            "Associations with PHQ-8 self-report labels do not establish diagnosis, causation, "
            "clinical utility, or transportability to other populations."
        ),
    }
    return json.dumps(report, indent=2, default=str)
'''

files["app.py"] = r'''
from __future__ import annotations
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.ingestion import load_uploaded_file
from src.preprocessing import prepare_participant_data
from src.statistics import descriptive_survey, permutation_test_auc
from src.modelling import evaluate_nested_group_cv, participant_bootstrap_auc
from src.reporting import build_research_report

st.set_page_config(page_title="The Neurometric Gaze", page_icon="◉", layout="wide")
st.title("The Neurometric Gaze")
st.subheader("A computational and humanistic study of psychological distress in digital narratives")
st.caption("Research prototype | Reproducibility-first | Not a diagnostic or clinical decision-support tool")

st.warning(
    "This platform does not diagnose mental illness or infer an individual's true emotional state. "
    "PHQ-8 is a self-reported symptom measure, not a clinical diagnosis. Use only data you are authorized to process."
)

with st.expander("Research question, scope and limitations", expanded=False):
    st.markdown("""
**Primary research question:** To what extent are linguistic patterns in interview narratives associated
with self-reported psychological distress, and what aspects of lived experience remain inaccessible to
computational interpretation?

**Important boundaries**
- DAIC-WOZ interviews use a virtual interviewer (Ellie), not ordinary doctor–patient consultations.
- PHQ-8 >= 10 is a commonly used grouping threshold; it is not a diagnosis.
- Word-emotion lexicons indicate lexical associations, not a speaker's actual emotional state.
- Predictive performance on one dataset is not clinical validation or external validation.
- Never present exploratory findings as causal evidence.
""")

with st.sidebar:
    st.header("Study input")
    uploaded = st.file_uploader("Upload CSV or DAIC-WOZ ZIP", type=["csv", "zip"])
    min_tokens = st.number_input("Minimum participant tokens", min_value=1, max_value=500, value=20)
    max_features = st.select_slider("Maximum TF-IDF features", options=[1000, 2000, 3000, 5000, 8000], value=5000)
    run_eval = st.button("Run nested grouped evaluation", type="primary")
    st.caption("For DAIC-WOZ, verify access terms, metadata, and official split documentation.")

tabs = st.tabs(["Overview & audit", "Survey analysis", "Model laboratory", "Annotation reliability", "Research report"])
data = None
audit = None
dataset_label = uploaded.name if uploaded else "No dataset uploaded"
model_result = None

if uploaded:
    try:
        raw_df = load_uploaded_file(uploaded.name, uploaded.getvalue())
        data, audit = prepare_participant_data(raw_df, min_tokens=min_tokens)
        st.success(f"Successfully prepared {len(data):,} participant-level records.")
    except Exception as exc:
        st.error(f"Data preparation failed: {exc}")

with tabs[0]:
    st.header("Dataset integrity and provenance")
    if data is None:
        st.info("Upload a CSV or DAIC-WOZ ZIP to begin. No demo or synthetic data are generated.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Participants", f"{len(data):,}")
        c2.metric("PHQ-8 group 0", int((data.phq8_binary == 0).sum()))
        c3.metric("PHQ-8 group 1", int((data.phq8_binary == 1).sum()))
        st.json(audit)
        st.subheader("Outcome class distribution")
        counts = data["phq8_binary"].value_counts().sort_index()
        st.bar_chart(counts)
        st.subheader("Participant-level data preview")
        st.dataframe(data.head(20), use_container_width=True)
        if "split" in data.columns:
            st.subheader("Provided split metadata")
            st.dataframe(pd.crosstab(data["split"], data["phq8_binary"]), use_container_width=True)
            st.caption("The app displays supplied split metadata; it does not certify the split as official.")

with tabs[1]:
    st.header("Survey and descriptive analysis")
    if data is None:
        st.info("Upload data to enable survey analysis.")
    else:
        summary = descriptive_survey(data)
        st.json(summary)
        if "phq8_score" in data.columns:
            scores = pd.to_numeric(data["phq8_score"], errors="coerce").dropna()
            if not scores.empty:
                fig, ax = plt.subplots()
                ax.hist(scores, bins=min(12, max(4, scores.nunique())))
                ax.set_xlabel("PHQ-8 total score")
                ax.set_ylabel("Participants")
                ax.set_title("Observed PHQ-8 score distribution")
                st.pyplot(fig)
                st.caption("Distribution describes the uploaded sample only; it is not a population prevalence estimate.")
        else:
            st.info("No PHQ8_Score field detected. Item-level or total-score results will not be invented.")

with tabs[2]:
    st.header("Model laboratory")
    st.markdown("""
**Evaluation design:** participant-level grouping, stratified grouped outer folds, and hyperparameter
selection inside each outer training fold. The output is out-of-fold performance on this dataset.
""")
    if data is None:
        st.info("Upload a dataset with participant_id, text, and PHQ8_Binary.")
    elif run_eval:
        try:
            with st.spinner("Training and evaluating models. This may take several minutes."):
                model_result = evaluate_nested_group_cv(data, max_features=max_features)
            metrics = model_result["metrics"]
            baseline = model_result["baseline"]
            st.subheader("Out-of-fold model performance")
            cols = st.columns(4)
            for col, key in zip(cols, ["macro_f1", "balanced_accuracy", "roc_auc", "pr_auc"]):
                col.metric(key.replace("_", " ").title(), f"{metrics[key]:.3f}")
            st.subheader("Majority/prior baseline comparison")
            st.dataframe(pd.DataFrame([{"model": "TF-IDF + Logistic Regression", **metrics},
                                       {"model": "Prior baseline", **baseline}]), use_container_width=True)
            ci = participant_bootstrap_auc(model_result["predictions"])
            st.write(f"Participant-bootstrap 95% ROC-AUC interval: {ci[0]:.3f}–{ci[1]:.3f}"
                     if pd.notna(ci[0]) else "Bootstrap interval not estimable.")
            st.subheader("Fold configuration")
            st.dataframe(model_result["folds"], use_container_width=True)
            st.subheader("Classification report")
            st.code(model_result["classification_report"])
            cm = model_result["confusion_matrix"]
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
            ax.set_yticks([0, 1], ["Observed 0", "Observed 1"])
            for (i, j), value in __import__("numpy").ndenumerate(cm):
                ax.text(j, i, str(value), ha="center", va="center")
            ax.set_title("Out-of-fold confusion matrix")
            st.pyplot(fig)
            st.download_button("Download out-of-fold predictions",
                               model_result["predictions"].to_csv(index=False),
                               "oof_predictions.csv", "text/csv")
            if st.button("Run exploratory AUC permutation test"):
                try:
                    perm = permutation_test_auc(
                        model_result["predictions"]["observed"],
                        model_result["predictions"]["predicted_probability"]
                    )
                    st.json(perm)
                    st.caption("Permutation p-value is conditional on these fixed out-of-fold predictions; it does not replace an external test cohort.")
                except Exception as exc:
                    st.error(str(exc))
            st.warning("These results are internal dataset-specific estimates, not clinical validation.")
        except Exception as exc:
            st.error(f"Model evaluation could not be completed: {exc}")
            st.caption("Possible causes include insufficient class counts, sparse vocabulary, or too few independent participants.")
    else:
        st.info("Click 'Run nested grouped evaluation' in the sidebar when the dataset audit is complete.")

with tabs[3]:
    st.header("Human annotation reliability")
    st.markdown("""
Upload a long-format CSV with `participant_id`, `segment_id`, `annotator_id`, and `label`.
Annotators must independently label the same units using a pre-specified codebook.
Do not treat model predictions as human annotations.
""")
    annotation_file = st.file_uploader("Upload independent annotation CSV", type=["csv"], key="annotation")
    if annotation_file:
        try:
            from src.annotation import krippendorff_alpha_nominal
            ann = pd.read_csv(annotation_file)
            alpha = krippendorff_alpha_nominal(ann)
            st.metric("Krippendorff's alpha (nominal)", f"{alpha:.3f}" if pd.notna(alpha) else "Not estimable")
            st.dataframe(ann.head(30), use_container_width=True)
            st.caption("Report codebook development, annotator training, category prevalence, missing ratings and disagreements. No universal alpha threshold guarantees validity.")
        except Exception as exc:
            st.error(f"Annotation analysis failed: {exc}")

with tabs[4]:
    st.header("Reproducible research report")
    notes = st.text_area("Researcher notes and interpretation", height=150,
                         placeholder="Describe the sample, data source, deviations from protocol, limitations, and interpretation...")
    metrics_for_report = model_result["metrics"] if model_result else None
    baseline_for_report = model_result["baseline"] if model_result else None
    report = build_research_report(dataset_label, audit, metrics_for_report, baseline_for_report, notes)
    st.download_button("Download JSON research record", report, "research_record.json", "application/json")
    st.markdown("""
### Before making an academic claim
- Cite the exact dataset release, documentation and access conditions.
- Define the outcome, inclusion rules and analysis unit.
- Report all class counts, missingness and exclusions.
- Compare against simple baselines.
- Explain cross-validation, tuning and uncertainty estimation.
- Include error analysis and subgroup limitations.
- Do not infer emotion, diagnosis, causation or clinical usefulness from text patterns alone.
- Seek ethics review or an institutional determination where required.
""")

st.divider()
st.caption("The Neurometric Gaze | Research prototype | No fabricated participants, survey responses, annotations, metrics, or validation claims.")
'''

files["pages/1_Data_Dictionary_and_Protocol.py"] = r'''
import streamlit as st
st.title("Data Dictionary & Protocol")
st.markdown("""
## Recommended normalized data schema

| Field | Required | Meaning |
|---|---:|---|
| participant_id | Yes | Stable pseudonymous participant key |
| text | Yes | Participant-only narrative text |
| PHQ8_Binary | Yes | Documented binary grouping label (0/1) |
| PHQ8_Score | No | Observed PHQ-8 total score |
| split | No | Official documented split label |
| speaker | No | Speaker role, used to retain participant speech |

## Pre-analysis protocol
1. Record dataset release, acquisition date, permission basis and documentation.
2. Define the unit of analysis before examining model results.
3. Define exclusions and missing-data handling.
4. Separate exploratory analysis from confirmatory analysis.
5. Freeze the model-selection plan before reporting final estimates.
6. Keep the original official test partition untouched if it is available and labels are legitimately supplied.
7. Record all deviations from the protocol.

## DAIC-WOZ interpretation
DAIC-WOZ consists of interviews conducted with a virtual interviewer. Avoid presenting it as a sample of ordinary clinical consultations.
""")
'''

files["pages/2_Methodology_and_Limitations.py"] = r'''
import streamlit as st
st.title("Methodology & Limitations")
st.markdown("""
## Statistical safeguards
- Participant-level aggregation prevents utterance-level rows from being treated as independent participants.
- Grouped cross-validation prevents the same participant from appearing in training and test folds.
- Nested tuning reduces optimistic estimates caused by hyperparameter selection on the evaluation folds.
- A baseline is reported alongside the text model.
- Bootstrap intervals communicate sampling uncertainty, not transportability.
- Permutation tests are exploratory and conditional on the fixed out-of-fold predictions.

## Limitations
1. Dataset-specific performance may not generalize to other languages, cultures, populations or interview settings.
2. PHQ-8 is self-report, not diagnosis.
3. Text alone cannot reveal a person's complete psychological state.
4. Word-level affect lexicons are associations, not emotion measurements.
5. Predictive association does not establish causality.
6. Fairness analysis requires adequate subgroup sample sizes and careful interpretation.
7. Human annotation reliability does not establish that the codebook captures the full meaning of a narrative.
""")
'''

files["pages/3_Medical_Humanities_Framework.py"] = r'''
import streamlit as st
st.title("Medical Humanities Framework")
st.markdown("""
## The Limits of the Clinical Gaze

This project treats computational analysis as one interpretive lens rather than an authoritative account of a person's experience.

### Four questions
1. **Narrative agency:** Which parts of a participant's story are retained, compressed or excluded by feature extraction?
2. **Interpretive authority:** Who gets to define what a linguistic pattern means—the participant, annotator, clinician, or model designer?
3. **Context:** What is lost when language is detached from biography, culture, power, and the interview setting?
4. **Ethics:** When does measurement become surveillance, stigma, or an unjustified claim about an individual?

### Suggested scholarly anchors
- Rita Charon — narrative medicine and close reading.
- Arthur W. Frank — illness narratives and the experience of illness.
- Michel Foucault — the clinical gaze and the production of medical knowledge.

Use primary texts and peer-reviewed scholarship in the final paper. Do not cite this page as a substitute for literature review.
""")
'''

files["tests/test_core.py"] = r'''
import pandas as pd
from src.preprocessing import prepare_participant_data
from src.annotation import krippendorff_alpha_nominal

def test_aggregation_and_conflicting_labels():
    raw = pd.DataFrame({
        "participant_id": ["a", "a", "b", "b"],
        "text": ["hello there", "how are you", "good morning", "another line"],
        "PHQ8_Binary": [0, 0, 1, 0],
    })
    result, audit = prepare_participant_data(raw, min_tokens=1)
    assert result["participant_id"].tolist() == ["a"]
    assert audit["conflicting_label_participants_excluded"] == 1

def test_alpha_perfect_agreement():
    ann = pd.DataFrame({
        "participant_id": ["p1", "p1", "p2", "p2"],
        "segment_id": ["s1", "s1", "s2", "s2"],
        "annotator_id": ["a1", "a2", "a1", "a2"],
        "label": ["present", "present", "absent", "absent"],
    })
    alpha = krippendorff_alpha_nominal(ann)
    assert abs(alpha - 1.0) < 1e-9
'''

files["config/research_config.yaml"] = """project_name: The Neurometric Gaze
analysis_unit: participant
outcome:
  field: PHQ8_Binary
  type: binary_self_report_grouping
  interpretation: "Not a clinical diagnosis"
validation:
  outer_cv: StratifiedGroupKFold
  inner_cv: StratifiedGroupKFold
  random_seed: 2026
reporting:
  include_confidence_intervals: true
  include_baseline: true
  prohibit_synthetic_results: true
"""

files["README.md"] = r'''
# The Neurometric Gaze — Complete Research Prototype

A multi-page Streamlit research platform connecting computational language analysis with survey data and Medical Humanities questions.

## 1. Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 2. Input data

### Normalized CSV (recommended)
Required columns:
- `participant_id`
- `text`
- `PHQ8_Binary` (0/1; confirm the label meaning in the source documentation)

Optional:
- `PHQ8_Score`
- `split`
- `speaker`

If `speaker` is present, only rows whose value is exactly `Participant` (case-insensitive) are retained. Multiple rows per participant are aggregated. Participants with conflicting binary labels are excluded and counted in the audit.

### DAIC-WOZ ZIP
The archive should contain files named `*_TRANSCRIPT.csv` with `speaker` and `value` columns, plus a metadata CSV containing participant IDs and PHQ-8 labels. The importer extracts rows whose speaker is `Participant`.

**Important:** Inspect the merged dataset and verify the metadata mapping. The ZIP importer is a convenience parser, not a certification of official DAIC-WOZ split integrity. Do not assume an official test label exists if it is not supplied in the permitted data.

## 3. Modules
- `src/ingestion.py`: CSV and transcript ZIP import.
- `src/preprocessing.py`: participant-level aggregation, label checks and audit.
- `src/statistics.py`: descriptive survey summaries and exploratory tests.
- `src/modelling.py`: nested grouped CV, baseline and participant bootstrap.
- `src/annotation.py`: nominal Krippendorff alpha.
- `src/reporting.py`: reproducible JSON research record.
- `pages/`: protocol, limitations and Medical Humanities framework.
- `tests/`: small unit tests for preprocessing and reliability calculation.

## 4. Evaluation
The predictive workflow uses word/character TF-IDF and logistic regression. Hyperparameter tuning occurs within each outer training fold. Participants are grouped in both loops.

Metrics include accuracy, balanced accuracy, macro F1, ROC-AUC, PR-AUC and Brier score. A prior baseline is also calculated. Bootstrap intervals quantify uncertainty in the observed sample; they do not establish external validity.

## 5. Human annotation
Long-format CSV columns:
`participant_id, segment_id, annotator_id, label`

At least two annotators should independently label the same units with a written codebook. The nominal Krippendorff alpha calculation does not validate the codebook or the construct.

## 6. Scientific and ethical boundaries
- PHQ-8 is a self-report symptom measure, not diagnosis.
- DAIC-WOZ virtual-interviewer sessions are not ordinary doctor–patient consultations.
- Lexical markers do not directly measure emotion.
- Model associations do not prove causality or clinical utility.
- Do not upload identifiable or non-consented data.
- Check dataset licence, access conditions and ethics requirements.
- Do not report metrics from a run that has not actually completed.
- Do not call internal cross-validation "external validation."

## 7. Suggested references to verify and cite
- DAIC-WOZ official portal: https://dcapswoz.ict.usc.edu/
- DAIC-WOZ documentation: https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf
- NRC Emotion Lexicons: https://www.nrc.canada.ca/en/research-development/products-services/technical-advisory-services/sentiment-emotion-lexicons
- TRIPOD+AI: https://www.tripod-statement.org/

Always cite the exact dataset version, documentation and any lexicon release used in your manuscript.
'''

for rel, content in files.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")

# Compile all Python files.
for p in root.rglob("*.py"):
    py_compile.compile(str(p), doraise=True)

# Basic static import test (no Streamlit runtime execution).
import sys
sys.path.insert(0, str(root))
from src.preprocessing import prepare_participant_data
from src.annotation import krippendorff_alpha_nominal
import pandas as pd
sample = pd.DataFrame({
    "participant_id": ["a", "a", "b", "b"],
    "text": ["hello there", "how are you", "good morning", "another line"],
    "PHQ8_Binary": [0, 0, 1, 0],
})
prepared, audit = prepare_participant_data(sample, 1)
assert prepared["participant_id"].tolist() == ["a"]
ann = pd.DataFrame({
    "participant_id": ["p1", "p1", "p2", "p2"],
    "segment_id": ["s1", "s1", "s2", "s2"],
    "annotator_id": ["a1", "a2", "a1", "a2"],
    "label": ["yes", "yes", "no", "no"],
})
assert abs(krippendorff_alpha_nominal(ann) - 1.0) < 1e-9

zip_path = Path("/mnt/data/neurometric_gaze_complete.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for p in root.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            archive.write(p, arcname=f"neurometric_gaze_complete/{p.relative_to(root)}")

print("Created", zip_path)
print("Files:", len([p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]))
print("Compilation and basic preprocessing/annotation checks passed.")
