from pathlib import Path
import textwrap, zipfile, py_compile, os

base = Path("/mnt/data/neurometric_gaze_research_v3")
base.mkdir(parents=True, exist_ok=True)

app_code = r'''
from __future__ import annotations

import io
import zipfile
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score, brier_score_loss
)
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="The Neurometric Gaze | Research Lab", layout="wide")
st.title("The Neurometric Gaze")
st.caption("A reproducible research environment for language patterns, self-reported distress, and the limits of computational interpretation.")

st.warning(
    "Research prototype only. This application does not diagnose, screen, triage, or recommend treatment. "
    "PHQ-8 is a self-report symptom measure, not a clinical diagnosis. Do not upload identifiable or "
    "non-consented clinical data."
)

with st.expander("Research scope and evidence boundaries", expanded=False):
    st.markdown("""
**Primary question:** To what extent are linguistic patterns in interview narratives associated with
self-reported psychological distress, and what aspects of lived experience remain inaccessible to
computational interpretation?

- DAIC-WOZ interviews use a virtual interviewer (Ellie); they are not ordinary doctor–patient consultations.
- PHQ-8 scores are self-reported symptom measures. A threshold of 10 is a conventional grouping threshold,
  not a diagnosis.
- Lexical-emotion dictionaries identify word-level associations, not the speaker's actual emotional state.
- Predictive performance is dataset- and split-specific. It does not establish clinical utility or external validity.
- This app never creates synthetic survey responses, labels, annotators, p-values, or performance metrics.
""")

st.sidebar.header("Study configuration")
uploaded = st.sidebar.file_uploader(
    "Upload data (.csv or DAIC-WOZ .zip)", type=["csv", "zip"],
    help="Preferred normalized CSV columns: participant_id, text, PHQ8_Binary. Optional: PHQ8_Score, split, speaker, segment_id."
)
metadata_file = st.sidebar.file_uploader("Optional metadata CSV (if transcript ZIP)", type=["csv"])
min_tokens = st.sidebar.number_input("Minimum participant tokens per record", min_value=1, max_value=500, value=20)
max_features = st.sidebar.select_slider("TF-IDF maximum features", options=[1000, 2000, 3000, 5000, 8000, 10000], value=5000)
run_model = st.sidebar.button("Run leakage-aware evaluation", type="primary")

def read_uploaded(upload):
    if upload is None:
        return None, "No dataset uploaded."
    raw = upload.getvalue()
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw)), None
    if upload.name.lower().endswith(".zip"):
        frames = []
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                names = zf.namelist()
                transcript_names = [n for n in names if n.lower().endswith("_transcript.csv")]
                if not transcript_names:
                    return None, "ZIP contains no *_TRANSCRIPT.csv files."
                for name in transcript_names:
                    with zf.open(name) as f:
                        t = pd.read_csv(f)
                    pid = Path(name).name.split("_")[0]
                    if not {"speaker", "value"}.issubset(t.columns):
                        continue
                    participant = t[t["speaker"].astype(str).str.lower().str.contains("participant", na=False)]
                    text = " ".join(participant["value"].dropna().astype(str))
                    frames.append({"participant_id": pid, "text": text})
            data = pd.DataFrame(frames)
            meta = None
            if metadata_file is not None:
                meta = pd.read_csv(metadata_file)
            else:
                # Look for common split metadata files in the archive.
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    candidates = [n for n in zf.namelist() if "split" in n.lower() and n.lower().endswith(".csv")]
                    if candidates:
                        with zf.open(candidates[0]) as f:
                            meta = pd.read_csv(f)
            if meta is not None:
                id_col = next((c for c in meta.columns if c.lower() in ["participant_id", "participant", "id"]), None)
                if id_col:
                    meta = meta.rename(columns={id_col: "participant_id"})
                    data["participant_id"] = data["participant_id"].astype(str)
                    meta["participant_id"] = meta["participant_id"].astype(str)
                    data = data.merge(meta, on="participant_id", how="left", suffixes=("", "_meta"))
            return data, None
        except Exception as e:
            return None, f"Could not read ZIP: {e}"
    return None, "Unsupported file."

def normalise_columns(df):
    df = df.copy()
    lower = {c.lower().strip(): c for c in df.columns}
    aliases = {
        "participant_id": ["participant_id", "participant", "id", "subject_id"],
        "text": ["text", "transcript", "narrative", "value"],
        "phq8_binary": ["phq8_binary", "phq_8_binary", "label", "target"],
        "phq8_score": ["phq8_score", "phq_8_score", "phq8"],
        "split": ["split", "set", "partition"],
        "speaker": ["speaker", "role"]
    }
    rename = {}
    for target, opts in aliases.items():
        for opt in opts:
            if opt in lower:
                rename[lower[opt]] = target
                break
    return df.rename(columns=rename)

def prepare_data(df):
    df = normalise_columns(df)
    required = {"participant_id", "text", "phq8_binary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    df["participant_id"] = df["participant_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    if "speaker" in df.columns:
        df = df[df["speaker"].astype(str).str.lower().str.contains("participant", na=False)]
    df["phq8_binary"] = pd.to_numeric(df["phq8_binary"], errors="coerce")
    df = df[df["phq8_binary"].isin([0, 1])].copy()
    df["token_count"] = df["text"].str.findall(r"\b[\w']+\b").str.len()
    df = df[df["token_count"] >= min_tokens]
    # Aggregate multiple transcript rows per participant; participant is the unit of analysis.
    agg = {"text": lambda x: " ".join(x.astype(str)), "phq8_binary": "first", "token_count": "sum"}
    if "phq8_score" in df.columns:
        df["phq8_score"] = pd.to_numeric(df["phq8_score"], errors="coerce")
        agg["phq8_score"] = "first"
    if "split" in df.columns:
        agg["split"] = "first"
    out = df.groupby("participant_id", as_index=False).agg(agg)
    return out

def safe_group_splits(y, groups, desired=5):
    counts = pd.Series(y).value_counts()
    n_splits = min(desired, int(counts.min()), pd.Series(groups).nunique())
    if n_splits < 2:
        raise ValueError("Not enough participants in each class for grouped cross-validation.")
    return n_splits

def build_pipeline(maxf):
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=maxf,
                                sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                max_features=max(500, maxf // 3), sublinear_tf=True))
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"))
    ])

def group_bootstrap_ci(y_true, y_prob, groups, metric_fn, n_boot=1000, seed=17):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(metric_fn(yt, yp))
        except Exception:
            pass
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.quantile(vals, [0.025, 0.975]))

def evaluate_group_cv(data, maxf):
    X = data["text"].values
    y = data["phq8_binary"].astype(int).values
    groups = data["participant_id"].values
    n = safe_group_splits(y, groups)
    outer = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42)
    oof_prob = np.full(len(data), np.nan)
    fold_rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
        ytr = y[tr]
        inner_n = safe_group_splits(ytr, groups[tr], desired=3)
        inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=fold + 100)
        pipe = build_pipeline(maxf)
        search = GridSearchCV(
            pipe,
            {"classifier__C": [0.1, 1.0, 10.0]},
            scoring="f1_macro",
            cv=inner.split(X[tr], ytr, groups[tr]),
            n_jobs=-1,
            refit=True
        )
        search.fit(X[tr], ytr)
        model = search.best_estimator_
        prob = model.predict_proba(X[te])[:, 1]
        oof_prob[te] = prob
        pred = (prob >= 0.5).astype(int)
        fold_rows.append({
            "Fold": fold, "N test": len(te),
            "Macro F1": f1_score(y[te], pred, average="macro", zero_division=0),
            "Balanced accuracy": balanced_accuracy_score(y[te], pred),
            "Best C": search.best_params_["classifier__C"]
        })
    valid = np.isfinite(oof_prob)
    yt, yp = y[valid], oof_prob[valid]
    pred = (yp >= 0.5).astype(int)
    auc = roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else np.nan
    metrics = {
        "Participants": int(valid.sum()),
        "Macro F1": f1_score(yt, pred, average="macro", zero_division=0),
        "Balanced accuracy": balanced_accuracy_score(yt, pred),
        "ROC AUC": auc,
        "Brier score": brier_score_loss(yt, yp),
        "Accuracy": accuracy_score(yt, pred)
    }
    auc_ci = group_bootstrap_ci(yt, yp, groups[valid], roc_auc_score)
    return metrics, pd.DataFrame(fold_rows), yt, yp, pred, auc_ci

def krippendorff_alpha_nominal(matrix):
    # Rows = items, columns = annotators; NaN = missing rating.
    arr = np.asarray(matrix, dtype=object)
    coincidence = {}
    for row in arr:
        vals = [v for v in row if pd.notna(v)]
        m = len(vals)
        if m < 2:
            continue
        for a, b in combinations(range(m), 2):
            x, y = vals[a], vals[b]
            coincidence[(x, y)] = coincidence.get((x, y), 0) + 1 / (m - 1)
            coincidence[(y, x)] = coincidence.get((y, x), 0) + 1 / (m - 1)
    total = sum(coincidence.values())
    if total == 0:
        return np.nan
    categories = set()
    for x, y in coincidence:
        categories.update([x, y])
    marg = {c: sum(v for (x, y), v in coincidence.items() if x == c) for c in categories}
    Do = sum(v for (x, y), v in coincidence.items() if x != y) / total
    denom = total * (total - 1)
    if denom <= 0:
        return np.nan
    De = sum(marg[c] * (sum(marg.values()) - marg[c]) for c in categories) / denom
    return 1 - Do / De if De > 0 else np.nan

tab1, tab2, tab3, tab4 = st.tabs(["Dataset audit", "Predictive evaluation", "Survey analysis", "Human annotation reliability"])

data_raw, err = read_uploaded(uploaded)
data = None
if err:
    st.info(err)
if data_raw is not None:
    try:
        data = prepare_data(data_raw)
        st.success(f"Loaded {len(data):,} participant-level records.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Participants", f"{len(data):,}")
        c2.metric("PHQ-8 group 0", int((data.phq8_binary == 0).sum()))
        c3.metric("PHQ-8 group 1", int((data.phq8_binary == 1).sum()))
        with tab1:
            st.subheader("Data provenance and quality checks")
            st.write("The participant is the analysis unit. Repeated transcript rows are aggregated before modelling.")
            audit = pd.DataFrame({
                "Check": ["Duplicate participant IDs after aggregation", "Empty narratives", "Median token count", "Missing PHQ-8 binary label"],
                "Result": [
                    int(data.participant_id.duplicated().sum()),
                    int((data.text.str.strip() == "").sum()),
                    float(data.token_count.median()),
                    int(data.phq8_binary.isna().sum())
                ]
            })
            st.dataframe(audit, use_container_width=True, hide_index=True)
            st.write("Class distribution")
            st.bar_chart(data.phq8_binary.value_counts().sort_index())
            if "split" in data.columns:
                st.write("Provided split distribution (not assumed to be independent unless verified):")
                st.dataframe(pd.crosstab(data["split"], data["phq8_binary"]), use_container_width=True)
            st.caption("Check the dataset documentation and split definitions yourself. The app does not infer or certify provenance.")
    except Exception as e:
        st.error(str(e))
        data = None

with tab2:
    st.subheader("Leakage-aware model evaluation")
    st.markdown("""
The model uses word- and character-level TF-IDF with logistic regression. Outer folds hold out participants;
hyperparameter selection occurs only inside each training fold. Metrics are based on out-of-fold predictions.
""")
    if data is None:
        st.info("Upload a valid dataset to enable evaluation.")
    else:
        st.write(f"Ready to evaluate **{len(data)} participants**. Both classes must have enough independent participants.")
        if run_model:
            try:
                with st.spinner("Running nested grouped cross-validation..."):
                    metrics, folds, yt, yp, pred, auc_ci = evaluate_group_cv(data, max_features)
                cols = st.columns(5)
                for col, (name, value) in zip(cols, metrics.items()):
                    col.metric(name, f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value))
                st.markdown("**ROC AUC 95% participant-bootstrap interval:** "
                            + (f"{auc_ci[0]:.3f}–{auc_ci[1]:.3f}" if np.isfinite(auc_ci[0]) else "not estimable"))
                st.dataframe(folds, use_container_width=True, hide_index=True)
                st.subheader("Out-of-fold classification report")
                st.code(classification_report(yt, pred, zero_division=0))
                fig, ax = plt.subplots()
                cm = confusion_matrix(yt, pred, labels=[0, 1])
                ax.imshow(cm)
                ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
                ax.set_yticks([0, 1], ["Observed 0", "Observed 1"])
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, str(v), ha="center", va="center")
                ax.set_title("Out-of-fold confusion matrix")
                st.pyplot(fig)
                st.download_button("Download out-of-fold predictions",
                    pd.DataFrame({"observed": yt, "predicted_probability": yp, "predicted_at_0_5": pred}).to_csv(index=False),
                    file_name="oof_predictions.csv", mime="text/csv")
                st.info("These estimates describe this dataset and evaluation design only. They are not clinical validation, "
                        "external validation, or evidence that the model can detect an individual's mental state.")
            except Exception as e:
                st.error(f"Evaluation could not be completed: {e}")
                st.caption("Common causes: too few participants in a class, too few documents per fold, or insufficient vocabulary.")

with tab3:
    st.subheader("Self-report survey analysis")
    if data is None:
        st.info("Upload data to inspect survey variables.")
    else:
        if "phq8_score" in data.columns:
            st.write("PHQ-8 total score distribution")
            st.histogram_chart(data["phq8_score"].dropna())
            st.caption("PHQ-8 score is a self-reported symptom measure, not a clinician-assigned diagnosis.")
            # Descriptive association only; no causal interpretation.
            st.write("Descriptive association between narrative length and PHQ-8 score")
            tmp = data[["token_count", "phq8_score"]].dropna()
            if len(tmp) >= 5:
                rho, p = spearmanr(tmp.token_count, tmp.phq8_score)
                st.write({"Spearman rho": round(float(rho), 3), "p-value": round(float(p), 4), "N": len(tmp)})
                st.caption("Exploratory, unadjusted association. It does not establish causation and may be confounded.")
        else:
            st.info("No PHQ8_Score column found. Add the actual documented score field to enable total-score analysis.")
        item_cols = [c for c in data.columns if c.lower().startswith("phq8_") and c.lower() not in ["phq8_binary", "phq8_score"]]
        if item_cols:
            st.write("Available item-level survey fields:", ", ".join(item_cols))
            st.dataframe(data[item_cols].describe(include="all").T, use_container_width=True)
        else:
            st.caption("No item-level PHQ-8 responses detected. The app will not fabricate survey item results.")

with tab4:
    st.subheader("Human annotation reliability")
    st.markdown("""
Upload a long-format CSV with **participant_id**, **segment_id**, **annotator_id**, and **label**.
Each annotator must independently label the same units using a pre-specified codebook.
Do not use model-generated labels as independent human annotations.
""")
    ann_file = st.file_uploader("Upload independent annotation CSV", type=["csv"], key="ann")
    if ann_file:
        try:
            ann = pd.read_csv(ann_file)
            required = {"participant_id", "segment_id", "annotator_id", "label"}
            if not required.issubset(ann.columns):
                st.error("Required columns: participant_id, segment_id, annotator_id, label.")
            else:
                ann["unit"] = ann["participant_id"].astype(str) + "::" + ann["segment_id"].astype(str)
                wide = ann.pivot_table(index="unit", columns="annotator_id", values="label", aggfunc="first")
                st.write(f"Annotation units: {len(wide)} | Annotators: {wide.shape[1]}")
                alpha = krippendorff_alpha_nominal(wide.to_numpy())
                st.metric("Krippendorff's alpha (nominal)", f"{alpha:.3f}" if np.isfinite(alpha) else "Not estimable")
                st.dataframe(wide.head(20), use_container_width=True)
                st.caption("Interpret alpha in context; there is no universal pass threshold. Report the codebook, annotator training, "
                           "missingness, category prevalence, and disagreements. This calculation assumes nominal categories.")
                st.download_button("Download aligned annotation matrix", wide.reset_index().to_csv(index=False),
                                   file_name="annotation_matrix.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process annotations: {e}")

st.divider()
st.subheader("Required reporting checklist")
st.markdown("""
- [ ] Cite the official dataset source, version, access conditions, and documentation.
- [ ] State the unit of analysis, inclusion/exclusion rules, and exact label definition.
- [ ] Verify that participant-level records do not cross folds.
- [ ] Compare against a majority-class baseline and a simple lexical baseline before making performance claims.
- [ ] Report class counts, missingness, confidence intervals, calibration, and error analysis.
- [ ] Treat all survey associations as observational, not causal.
- [ ] Conduct subgroup and fairness analyses only where sample sizes and permissions justify them.
- [ ] Obtain ethics approval/waiver where required; remove identifiers and follow data-use agreements.
- [ ] Validate on an independent dataset before making transportability or clinical claims.
""")
st.caption("Version 3.0 | Reproducibility-first prototype | No synthetic evidence or automatic clinical conclusions.")
'''
requirements = """streamlit>=1.36
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
"""

readme = r'''
# The Neurometric Gaze — Research Lab v3

## Purpose
A reproducible research prototype for examining associations between interview language and PHQ-8 self-report groupings. It is not a diagnostic or clinical decision-support system.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Input data

### Preferred normalized CSV
Required columns:
- `participant_id`
- `text`
- `PHQ8_Binary` (0/1; confirm its meaning in the dataset documentation)

Optional:
- `PHQ8_Score`
- `split`
- `speaker`
- item-level survey fields such as `PHQ8_NoInterest`

The app aggregates multiple rows per participant before modelling. If `speaker` exists, it retains rows whose speaker value contains `participant`.

### DAIC-WOZ ZIP
The ZIP should include `*_TRANSCRIPT.csv` files with `speaker` and `value` columns. The app extracts participant speech only. Include a split/metadata CSV in the ZIP or upload it separately. Metadata must contain a participant identifier plus the label column. Check the resulting audit carefully.

## Evaluation design
- Participant is the grouping unit.
- StratifiedGroupKFold is used for outer evaluation.
- Hyperparameter selection is nested within each outer training fold.
- Out-of-fold probabilities are used for metrics.
- A participant-level bootstrap interval is calculated for ROC AUC.

This is not a substitute for an externally held-out test cohort. Do not repeatedly tune decisions against the same evaluation data and then describe the result as independent validation.

## Human annotation
Upload long-format CSV with:
`participant_id, segment_id, annotator_id, label`

Two or more annotators should independently annotate the same units using a written codebook. The app calculates nominal Krippendorff's alpha. The statistic is meaningful only if the annotation design is sound.

## Important limitations
- PHQ-8 is self-report, not diagnosis.
- DAIC-WOZ interviews use virtual interviewer Ellie; they are not ordinary clinical consultations.
- Word-level emotion lexicons are not direct measurements of emotion.
- A model's association with labels does not show that it detects an individual's inner state.
- No synthetic data, survey results, p-values, annotator labels, or validation claims are generated.
- Check DAIC-WOZ terms of use and access requirements before use.

## Research references to consult
- DAIC-WOZ official portal: https://dcapswoz.ict.usc.edu/
- DAIC-WOZ documentation: https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf
- NRC Emotion Lexicons information/licensing: https://www.nrc.canada.ca/en/research-development/products-services/technical-advisory-services/sentiment-emotion-lexicons
- TRIPOD+AI reporting guideline: https://www.tripod-statement.org/
- PROBAST+AI risk-of-bias framework: consult the current official publication and documentation.

Always cite the exact dataset version and any lexicon version used in the final paper.
'''
(base/"app.py").write_text(textwrap.dedent(app_code).lstrip(), encoding="utf-8")
(base/"requirements.txt").write_text(requirements, encoding="utf-8")
(base/"README.md").write_text(textwrap.dedent(readme).lstrip(), encoding="utf-8")
py_compile.compile(str(base/"app.py"), doraise=True)
zip_path = Path("/mnt/data/neurometric_gaze_research_v3.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in base.iterdir():
        z.write(p, arcname=f"neurometric_gaze_research_v3/{p.name}")
print("Created:", zip_path, "and", list(base.iterdir()))
