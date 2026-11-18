"""
THE NEUROMETRIC GAZE 3.0
Advanced Computational Narrative Medicine Research Prototype

Important:
- Exploratory research software, not a diagnostic or clinical decision tool.
- Lexical matches are not evidence of a patient's psychological state.
- ML performance is reported only when independently labelled data is supplied.
- Do not upload identifiable or confidential patient data without appropriate
  consent, ethics approval, security controls, and institutional authorization.
"""

from __future__ import annotations

import io
import json
import re
import hashlib
import platform
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_VERSION = "3.0.1"


def get_writable_output_dir() -> Optional[Path]:
    """Return a writable runtime directory without writing into the app source.

    Streamlit Cloud mounts the repository under /mount/src, which is commonly
    read-only. Use the platform temporary directory first and fail gracefully
    if the runtime does not permit filesystem writes.
    """
    candidates = [
        Path(tempfile.gettempdir()) / "neurometric_gaze",
        Path.home() / ".cache" / "neurometric_gaze",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except (OSError, PermissionError):
            continue
    return None


OUTPUT_DIR = get_writable_output_dir()
PROJECT_NAME = "The Neurometric Gaze"
PROJECT_SUBTITLE = (
    "Computational Narrative Medicine & Patient–Physician "
    "Communication Research"
)

st.set_page_config(
    page_title=f"{PROJECT_NAME} | Research Edition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #F5F7FA; }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 15px;
        border-radius: 12px;
    }
    .hero {
        background: linear-gradient(120deg, #10243A, #244C65);
        padding: 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 18px;
    }
    .hero h1 { color: white; font-size: 34px; margin-bottom: 5px; }
    .hero p { color: #DCE8F0; }
    .muted { color: #64748B; font-size: 0.88rem; }
    .research-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 17px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 2. RESEARCH FRAMEWORK AND LEXICON
# ============================================================

RESEARCH_QUESTION = (
    "How can transparent computational language analysis help researchers "
    "study linguistic indicators of communication barriers in "
    "patient–physician narratives?"
)

MARKER_LIBRARY: Dict[str, Dict[str, Any]] = {
    "Hesitation": {
        "description": "Expressions of reluctance or difficulty speaking.",
        "patterns": [
            r"\bi hesitated\b", r"\bi couldn't say\b", r"\bi could not say\b",
            r"\bi was reluctant\b", r"\bi didn't know how to say\b",
            r"\bi did not know how to say\b", r"\bi struggled to tell\b",
            r"\bdeməyə çətinlik çəkdim\b", r"\bdanışmağa çəkindim\b",
            r"\bdeməyə çəkindim\b", r"\bdilə gətirə bilmədim\b",
            r"\bdeyə bilmədim\b", r"\btərəddüd etdim\b",
            r"\bdanışmağa tərəddüd etdim\b",
        ],
    },
    "Fear and Anxiety": {
        "description": "Explicit lexical expressions of fear or worry.",
        "patterns": [
            r"\bi was afraid\b", r"\bi was scared\b", r"\bi feared\b",
            r"\bi felt anxious\b", r"\bi was worried\b",
            r"\bi felt frightened\b", r"\bqorxdum\b", r"\bqorxurdum\b",
            r"\bqorxuram\b", r"\bnarahat idim\b", r"\bnarahat oldum\b",
            r"\btəşvişə düşdüm\b",
        ],
    },
    "Avoidance": {
        "description": "Expressions potentially indicating topic avoidance.",
        "patterns": [
            r"\bit doesn't matter\b", r"\bit is not important\b",
            r"\bnever mind\b", r"\bi ignored it\b", r"\bi avoided\b",
            r"\bi just waited\b", r"\bvacib deyil\b", r"\bönəmli deyil\b",
            r"\bəhəmiyyət vermədim\b", r"\bsadəcə gözlədim\b",
            r"\böz-özünə keçər\b", r"\bboş ver\b",
        ],
    },
    "Self-Censorship": {
        "description": "Expressions of withholding or not disclosing information.",
        "patterns": [
            r"\bi kept it to myself\b", r"\bi hid it\b",
            r"\bi didn't tell the doctor\b", r"\bi did not tell the doctor\b",
            r"\bi kept silent\b", r"\bi remained silent\b",
            r"\bi concealed\b", r"\bi chose not to mention\b",
            r"\bsusdum\b", r"\bgizlətdim\b", r"\bdemədim\b",
            r"\bdanışmadım\b", r"\bheç kimə demədim\b",
            r"\böz içimdə saxladım\b",
        ],
    },
    "Power Imbalance": {
        "description": "References to authority, judgment, or unequal interaction.",
        "patterns": [
            r"\bthe doctor would be angry\b", r"\bthe doctor might judge me\b",
            r"\bi was afraid of the doctor\b", r"\bthe doctor wouldn't listen\b",
            r"\bhe wouldn't believe me\b", r"\bshe wouldn't believe me\b",
            r"\bhəkim əsəbləşər\b", r"\bhəkim məni danlayar\b",
            r"\bmənə inanmaz\b", r"\bhəkimdən qorxdum\b",
            r"\bməni dinləmədi\b",
        ],
    },
    "Dismissal and Minimisation": {
        "description": "Expressions that may minimise symptoms or concerns.",
        "patterns": [
            r"\bit is probably nothing\b", r"\bit's probably nothing\b",
            r"\bi am overreacting\b", r"\bi don't want to bother\b",
            r"\bi didn't want to bother\b", r"\bjust a small problem\b",
            r"\bproblem deyil\b", r"\byəqin ki, heç nə deyil\b",
            r"\bşişirtmək istəmədim\b", r"\bəhəmiyyətsiz hesab etdim\b",
        ],
    },
    "Empathic Communication": {
        "description": "Explicit supportive or validating language.",
        "patterns": [
            r"\bi understand how you feel\b", r"\bi hear your concern\b",
            r"\btake your time\b", r"\bthank you for telling me\b",
            r"\byour concerns matter\b", r"\bsizi başa düşürəm\b",
            r"\bsizi dinləyirəm\b", r"\brahat danışa bilərsiniz\b",
            r"\btəşəkkür edirəm ki, paylaşdınız\b",
        ],
    },
    "Patient Agency": {
        "description": "Expressions of choice, participation, or self-advocacy.",
        "patterns": [
            r"\bi asked for clarification\b", r"\bi explained my concerns\b",
            r"\bi wanted to understand\b", r"\bi made the decision\b",
            r"\bi asked questions\b", r"\bfikrimi bildirdim\b",
            r"\bsual verdim\b", r"\bqərarı özüm verdim\b",
            r"\bnarahatlığımı izah etdim\b",
        ],
    },
}

NEGATION_PATTERNS = [
    r"\bnot\b", r"\bnever\b", r"\bno longer\b", r"\bdidn't\b",
    r"\bdid not\b", r"\bwasn't\b", r"\bwas not\b", r"\bcannot\b",
    r"\bcan't\b", r"\bqorxmadım\b", r"\bqorxmuram\b",
    r"\bnarahat deyiləm\b", r"\bsusmadım\b",
]

# ============================================================
# 3. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "annotation_records": [],
    "analysis_history": [],
    "last_analysis": None,
    "trained_model": None,
    "trained_model_info": None,
    "uploaded_dataset": None,
    "validation_results": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value

# ============================================================
# 4. TEXT PROCESSING
# ============================================================

def normalize_text(text: Any) -> str:
    """Normalize whitespace and case while preserving punctuation."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter; not a full linguistic parser."""
    chunks = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def detect_language_heuristic(text: str) -> str:
    """
    Transparent script/character heuristic.
    This is not a reliable language-identification model.
    """
    text_lower = normalize_text(text)
    az_chars = set("əğıöüşç")
    if any(char in az_chars for char in text_lower):
        return "Azerbaijani-likely"
    if re.search(r"[a-z]", text_lower):
        return "English-or-other-Latin"
    return "Unknown"


def detect_negation(sentence: str, start_position: int) -> bool:
    """Simple local-window negation flag; requires human review."""
    preceding = normalize_text(sentence)[:start_position][-65:]
    return any(re.search(pattern, preceding) for pattern in NEGATION_PATTERNS)


def analyze_sentence(sentence: str) -> List[Dict[str, Any]]:
    normalized = normalize_text(sentence)
    findings: List[Dict[str, Any]] = []

    for category, info in MARKER_LIBRARY.items():
        for pattern in info["patterns"]:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                negated = detect_negation(normalized, match.start())
                findings.append({
                    "category": category,
                    "matched_expression": match.group(0),
                    "sentence": sentence,
                    "negation_flag": bool(negated),
                    "description": info["description"],
                    "interpretation": (
                        "Potentially negated lexical match; human review required."
                        if negated else
                        "Potential lexical indicator; not a psychological or clinical finding."
                    ),
                })

    return findings


def analyze_document(text: str) -> Dict[str, Any]:
    sentences = split_sentences(text)
    sentence_results = []
    all_findings = []

    for index, sentence in enumerate(sentences, start=1):
        findings = analyze_sentence(sentence)
        all_findings.extend(findings)
        sentence_results.append({
            "sentence_id": index,
            "sentence": sentence,
            "marker_count": len(findings),
            "findings": findings,
        })

    return {
        "sentences": sentence_results,
        "findings": all_findings,
        "language_heuristic": detect_language_heuristic(text),
    }


def calculate_descriptive_indicators(text: str, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    word_count = len(re.findall(r"\b[\w'-]+\b", str(text), flags=re.UNICODE))
    total = len(findings)
    negated = sum(bool(item["negation_flag"]) for item in findings)
    non_negated = total - negated
    categories = Counter(item["category"] for item in findings)

    density = (non_negated / word_count * 100) if word_count else 0.0

    return {
        "word_count": word_count,
        "total_matches": total,
        "negated_matches": negated,
        "non_negated_matches": non_negated,
        "marker_density_per_100_words": round(density, 2),
        "category_distribution": dict(categories),
    }


def document_to_prediction(text: str, threshold: int = 1) -> int:
    result = analyze_document(text)
    indicators = calculate_descriptive_indicators(text, result["findings"])
    return int(indicators["non_negated_matches"] >= threshold)


# ============================================================
# 5. STATISTICAL UTILITIES
# ============================================================

def normalize_binary_label(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    positive = {"1", "positive", "yes", "true", "present", "barrier"}
    negative = {"0", "negative", "no", "false", "absent", "no barrier"}
    if value in positive:
        return 1
    if value in negative:
        return 0
    return None


def krippendorff_alpha_nominal(ratings: List[List[Optional[int]]]) -> Optional[float]:
    """
    Krippendorff's alpha for nominal data.
    Rows are units; columns are raters. None denotes missing annotation.
    Returns None when there is insufficient information or zero observed
    disagreement denominator.
    """
    coincidence = {}
    n_units_with_data = 0

    for row in ratings:
        values = [v for v in row if v is not None]
        if len(values) < 2:
            continue
        n_units_with_data += 1
        m = len(values)
        for i in range(m):
            for j in range(m):
                if i != j:
                    pair = (values[i], values[j])
                    coincidence[pair] = coincidence.get(pair, 0.0) + 1.0 / (m - 1)

    if not coincidence or n_units_with_data == 0:
        return None

    categories = {x for pair in coincidence for x in pair}
    total = sum(coincidence.values())
    marginals = {c: 0.0 for c in categories}

    for (a, b), count in coincidence.items():
        marginals[a] += count

    observed_disagreement = 0.0
    for (a, b), count in coincidence.items():
        if a != b:
            observed_disagreement += count
    observed_disagreement /= total

    expected_disagreement = 0.0
    if total <= 1:
        return None

    for a in categories:
        for b in categories:
            if a != b:
                expected_disagreement += marginals[a] * marginals[b]
    expected_disagreement /= (total * (total - 1))

    if expected_disagreement == 0:
        return None

    return float(1 - observed_disagreement / expected_disagreement)


def bootstrap_f1_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap interval for F1; descriptive, not a substitute for design."""
    if len(y_true) < 2:
        return None, None

    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        yt = y_true[indices]
        yp = y_pred[indices]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(f1_score(yt, yp, zero_division=0))

    if not scores:
        return None, None

    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


# ============================================================
# 6. MODEL TRAINING AND EVALUATION
# ============================================================

def build_tfidf_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=30000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def train_supervised_model(
    dataframe: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a transparent TF-IDF + Logistic Regression baseline.
    Requires an independently labelled dataset.
    """
    work = dataframe[[text_column, label_column]].copy()
    work = work.dropna()
    work[text_column] = work[text_column].astype(str)
    work["__label"] = work[label_column].apply(normalize_binary_label)
    work = work.dropna(subset=["__label"])
    work["__label"] = work["__label"].astype(int)

    if len(work) < 12:
        raise ValueError("At least 12 valid labelled records are required for this exploratory split.")
    if work["__label"].nunique() != 2:
        raise ValueError("The gold-label column must contain both positive and negative examples.")
    if work["__label"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two records for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        work[text_column],
        work["__label"],
        test_size=test_size,
        random_state=random_state,
        stratify=work["__label"],
    )

    model = build_tfidf_logistic_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "n_total": int(len(work)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_test, predictions)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
    }

    low, high = bootstrap_f1_interval(
        np.asarray(y_test),
        np.asarray(predictions),
    )
    metrics["f1_bootstrap_95ci"] = [low, high]

    return {
        "model": model,
        "metrics": metrics,
        "y_test": np.asarray(y_test),
        "predictions": np.asarray(predictions),
        "test_text": list(X_test),
        "feature_names": model.named_steps["tfidf"].get_feature_names_out(),
    }


# ============================================================
# 7. DATA EXPORT AND REPORT HELPERS
# ============================================================

def dataframe_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def make_research_report(analysis: Dict[str, Any]) -> str:
    findings = analysis["result"]["findings"]
    indicators = analysis["indicators"]
    categories = indicators["category_distribution"]

    lines = [
        f"# {PROJECT_NAME}",
        "",
        "## Exploratory analysis report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Application version: {APP_VERSION}",
        f"- Heuristic language estimate: {analysis['result']['language_heuristic']}",
        f"- Word count: {indicators['word_count']}",
        f"- Lexical matches: {indicators['total_matches']}",
        f"- Potentially negated matches: {indicators['negated_matches']}",
        f"- Non-negated matches: {indicators['non_negated_matches']}",
        f"- Descriptive marker density per 100 words: {indicators['marker_density_per_100_words']}",
        "",
        "## Category counts",
    ]

    if categories:
        for category, count in categories.items():
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- No lexicon matches detected.")

    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "These results are exploratory lexical matches. They do not establish",
        "a patient's psychological state, intent, diagnosis, or actual",
        "communication behaviour. Contextual human review is required.",
        "",
        "## Detected expressions",
    ])

    for item in findings:
        lines.append(
            f"- **{item['category']}** — `{item['matched_expression']}` "
            f"(negation flag: {item['negation_flag']})"
        )
        lines.append(f"  - Sentence: {item['sentence']}")

    lines.extend([
        "",
        "## Methodological limitations",
        "",
        "- The current lexicon is manually defined and not clinically validated.",
        "- Sentence segmentation and negation detection are heuristic.",
        "- No causal inference is made.",
        "- No clinical risk score is calculated.",
        "- Performance metrics require independent human-labelled data.",
    ])

    return "\n".join(lines)


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.title("Research Configuration")
st.sidebar.caption(f"Version {APP_VERSION}")

analysis_language = st.sidebar.selectbox(
    "Research language",
    ["Azerbaijani", "English", "Mixed / Multilingual"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Theoretical foundations")
st.sidebar.markdown(
    """
    - Narrative Medicine
    - Health Communication
    - Medical Humanities
    - Computational Linguistics
    - Explainable AI
    - Patient-centred care
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Research session")
st.sidebar.write(f"Annotations: {len(st.session_state.annotation_records)}")
st.sidebar.write(f"Analyses: {len(st.session_state.analysis_history)}")
st.sidebar.write(
    "Supervised model: "
    + ("Trained in this session" if st.session_state.trained_model else "Not trained")
)

if st.sidebar.button("Clear session data", use_container_width=True):
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value.copy() if isinstance(value, (list, dict)) else value
    st.rerun()

# ============================================================
# 9. HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <h1>🧠 {PROJECT_NAME}</h1>
        <p><strong>COMPUTATIONAL NARRATIVE MEDICINE & COMMUNICATION INTELLIGENCE</strong></p>
        <hr>
        <p>{PROJECT_SUBTITLE}</p>
        <p><em>{RESEARCH_QUESTION}</em></p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research-only prototype. Do not use for diagnosis, triage, treatment, "
    "or individual patient risk assessment. Do not enter identifiable patient data."
)

# ============================================================
# 10. NAVIGATION
# ============================================================

tabs = st.tabs([
    "Overview",
    "Text Analysis",
    "Explainability",
    "Human Annotation",
    "Research Validation",
    "Model Laboratory",
    "Narrative Comparison",
    "Research Reports",
    "Methodology & Ethics",
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:
    st.header("Research Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Research domains", "6")
    col2.metric("Exploratory marker families", len(MARKER_LIBRARY))
    col3.metric("Application version", APP_VERSION)

    st.markdown("### Research objectives")
    st.markdown(
        """
        1. Explore linguistic indicators that may be relevant to patient–physician communication.
        2. Make text-analysis outputs transparent and reviewable.
        3. Compare computational outputs with independent human annotations.
        4. Examine model performance across language groups when suitable data exist.
        5. Document uncertainty, error patterns, and ethical limitations.
        """
    )

    st.markdown("### Conceptual dimensions")
    dimensions = pd.DataFrame({
        "Dimension": [
            "Hesitation", "Fear and Anxiety", "Avoidance",
            "Self-Censorship", "Power Imbalance",
            "Dismissal and Minimisation", "Empathic Communication",
            "Patient Agency",
        ],
        "Research focus": [
            "Reluctance to speak",
            "Explicit fear or worry language",
            "Potential topic avoidance",
            "Withholding or non-disclosure language",
            "Authority and perceived judgment",
            "Minimisation of concerns",
            "Supportive and validating language",
            "Participation and self-advocacy",
        ],
    })
    st.dataframe(dimensions, use_container_width=True, hide_index=True)

    st.info(
        "The dimensions are an initial research taxonomy. They need a literature-grounded "
        "codebook, expert review, and empirical refinement before research claims are made."
    )

# ============================================================
# TAB 2 — TEXT ANALYSIS
# ============================================================

with tabs[1]:
    st.header("Patient–Physician Narrative Analysis")
    st.write(
        "Run transparent, exploratory lexical analysis on a short, anonymised text sample."
    )

    example_text = (
        "I wanted to tell the doctor about my headaches, "
        "but I was afraid that he would become angry. "
        "I kept silent and simply waited for a prescription. "
        "I thought it was probably nothing."
    )

    patient_text = st.text_area(
        "Enter an anonymised narrative",
        value=example_text,
        height=180,
        key="analysis_input",
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        run_analysis = st.button(
            "Run exploratory analysis",
            type="primary",
            use_container_width=True,
        )
    with c2:
        st.caption(
            "Text is processed in this Streamlit session. Hosting configuration, "
            "logging, and deployment environment may affect data handling."
        )

    if run_analysis:
        if len(patient_text.strip()) < 10:
            st.error("Please provide a longer text sample.")
        else:
            doc_result = analyze_document(patient_text)
            indicators = calculate_descriptive_indicators(
                patient_text, doc_result["findings"]
            )
            analysis_object = {
                "text": patient_text,
                "result": doc_result,
                "indicators": indicators,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            st.session_state.last_analysis = analysis_object
            st.session_state.analysis_history.append({
                "timestamp_utc": analysis_object["timestamp_utc"],
                "word_count": indicators["word_count"],
                "matches": indicators["total_matches"],
                "marker_density": indicators["marker_density_per_100_words"],
            })

    if st.session_state.last_analysis:
        current = st.session_state.last_analysis
        indicators = current["indicators"]

        st.divider()
        st.subheader("Descriptive indicators")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Word count", indicators["word_count"])
        m2.metric("Lexical matches", indicators["total_matches"])
        m3.metric("Negation flags", indicators["negated_matches"])
        m4.metric(
            "Matches / 100 words",
            f"{indicators['marker_density_per_100_words']:.2f}",
        )

        st.caption(
            "Density is a descriptive count, not a severity score, probability, "
            "clinical measure, or validated psychometric instrument."
        )
        st.caption(
            f"Language heuristic: {current['result']['language_heuristic']} "
            "(not a validated language detector)."
        )

        st.subheader("Sentence-level analysis")
        for sentence_data in current["result"]["sentences"]:
            with st.expander(
                f"Sentence {sentence_data['sentence_id']} — "
                f"{sentence_data['marker_count']} matches"
            ):
                st.write(sentence_data["sentence"])
                if sentence_data["findings"]:
                    for finding in sentence_data["findings"]:
                        st.markdown(
                            f"**{finding['category']}** · "
                            f"`{finding['matched_expression']}`"
                        )
                        st.caption(finding["interpretation"])
                else:
                    st.info("No current lexicon matches detected.")

        if indicators["category_distribution"]:
            st.subheader("Category distribution")
            category_df = pd.DataFrame(
                list(indicators["category_distribution"].items()),
                columns=["Category", "Count"],
            ).set_index("Category")
            st.bar_chart(category_df)
        else:
            st.info(
                "No lexicon matches were detected. This does not demonstrate "
                "that communication barriers are absent."
            )

# ============================================================
# TAB 3 — EXPLAINABILITY
# ============================================================

with tabs[2]:
    st.header("Explainability & Traceable Evidence")
    st.write(
        "Inspect the exact lexical expressions and sentences that triggered the rules."
    )

    if not st.session_state.last_analysis:
        st.info("Run a text analysis first.")
    else:
        findings = st.session_state.last_analysis["result"]["findings"]
        if findings:
            explain_df = pd.DataFrame(findings)
            columns = [
                "category", "matched_expression", "sentence",
                "negation_flag", "interpretation",
            ]
            st.dataframe(
                explain_df[columns],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("### Category counts")
            counts = explain_df["category"].value_counts().rename_axis(
                "Category"
            ).to_frame("Occurrences")
            st.bar_chart(counts)
        else:
            st.info("No current lexical rules were triggered.")

    st.markdown("### Explanation boundary")
    st.markdown(
        """
        - This version uses explicit lexical rules, so the triggering expression is visible.
        - The negation detector is a local-window heuristic and can be wrong.
        - A matched phrase is not proof of a patient's intention, emotion, or experience.
        - The current rule-based module is not a trained or clinically validated AI model.
        """
    )

# ============================================================
# TAB 4 — HUMAN ANNOTATION
# ============================================================

with tabs[3]:
    st.header("Human Annotation Workspace")
    st.write(
        "Create a small, transparent annotation record for the currently analysed text."
    )

    if st.session_state.last_analysis:
        st.text_area(
            "Narrative under review",
            value=st.session_state.last_analysis["text"],
            height=110,
            disabled=True,
            key="annotation_text_display",
        )

        label = st.radio(
            "Human assessment of communication-barrier evidence",
            ["Positive", "Negative", "Uncertain"],
            horizontal=True,
            key="annotation_label",
        )

        category = st.selectbox(
            "Primary research category",
            ["Hesitation", "Fear and Anxiety", "Avoidance", "Self-Censorship",
             "Power Imbalance", "Dismissal and Minimisation",
             "Empathic Communication", "Patient Agency",
             "Other / No clear category"],
            key="annotation_category",
        )

        annotator_id = st.text_input(
            "Pseudonymous annotator code",
            placeholder="e.g. RATER_A (do not enter a real name)",
        )
        rationale = st.text_area(
            "Evidence and rationale",
            placeholder="Record the contextual evidence and explain uncertainty.",
        )

        if st.button("Save annotation", type="primary"):
            st.session_state.annotation_records.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "text": st.session_state.last_analysis["text"],
                "human_label": label,
                "primary_category": category,
                "annotator_code": annotator_id.strip() or "UNSPECIFIED",
                "rationale": rationale,
            })
            st.success("Annotation added to this session.")

    else:
        st.info("Run a text analysis before annotating.")

    if st.session_state.annotation_records:
        st.divider()
        annotation_df = pd.DataFrame(st.session_state.annotation_records)
        st.dataframe(annotation_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Export human annotations (CSV)",
            dataframe_csv_bytes(annotation_df),
            file_name="neurometric_human_annotations.csv",
            mime="text/csv",
        )

# ============================================================
# TAB 5 — RESEARCH VALIDATION
# ============================================================

with tabs[4]:
    st.header("Empirical Validation Laboratory")
    st.write(
        "Evaluate rule-based outputs against independently produced human labels."
    )

    st.info(
        "A valid evaluation requires an appropriate sampling strategy, a pre-defined "
        "annotation codebook, independent labels, and separation of development and test data."
    )

    uploaded_validation = st.file_uploader(
        "Upload labelled CSV",
        type=["csv"],
        key="validation_upload",
        help="Required columns: text, gold_label. Optional: rater_1, rater_2.",
    )

    if uploaded_validation is not None:
        try:
            validation_df = pd.read_csv(uploaded_validation)
            st.subheader("Dataset preview")
            st.dataframe(validation_df.head(15), use_container_width=True)

            if "text" not in validation_df.columns or "gold_label" not in validation_df.columns:
                st.error("Required columns are 'text' and 'gold_label'.")
            else:
                threshold = st.slider(
                    "Minimum non-negated lexical matches for positive prediction",
                    1, 8, 1, key="rule_threshold",
                )

                valid_rows = validation_df.dropna(subset=["text", "gold_label"]).copy()
                valid_rows["gold_binary"] = valid_rows["gold_label"].apply(
                    normalize_binary_label
                )
                valid_rows = valid_rows.dropna(subset=["gold_binary"])
                valid_rows["gold_binary"] = valid_rows["gold_binary"].astype(int)
                valid_rows["rule_prediction"] = valid_rows["text"].astype(str).apply(
                    lambda text: document_to_prediction(text, threshold)
                )

                if len(valid_rows) == 0:
                    st.warning("No valid binary gold labels found.")
                else:
                    y_true = valid_rows["gold_binary"].to_numpy()
                    y_pred = valid_rows["rule_prediction"].to_numpy()

                    if len(np.unique(y_true)) < 2:
                        st.warning(
                            "Both gold-label classes are required for meaningful "
                            "binary classification metrics."
                        )
                    else:
                        precision = precision_score(y_true, y_pred, zero_division=0)
                        recall = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        accuracy = accuracy_score(y_true, y_pred)
                        kappa = cohen_kappa_score(y_true, y_pred)
                        ci_low, ci_high = bootstrap_f1_interval(y_true, y_pred)

                        a, b, c, d, e = st.columns(5)
                        a.metric("Precision", f"{precision:.3f}")
                        b.metric("Recall", f"{recall:.3f}")
                        c.metric("F1", f"{f1:.3f}")
                        d.metric("Accuracy", f"{accuracy:.3f}")
                        e.metric("Cohen's κ", f"{kappa:.3f}")

                        if ci_low is not None:
                            st.caption(
                                f"Exploratory bootstrap 95% interval for F1: "
                                f"{ci_low:.3f}–{ci_high:.3f}"
                            )

                        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual negative", "Actual positive"],
                            columns=["Predicted negative", "Predicted positive"],
                        )
                        st.subheader("Confusion matrix")
                        st.dataframe(cm_df)

                        st.subheader("Classification report")
                        st.code(
                            classification_report(
                                y_true, y_pred, zero_division=0
                            )
                        )

                    st.session_state.validation_results = valid_rows.copy()
                    st.download_button(
                        "Download rule-based validation results",
                        dataframe_csv_bytes(valid_rows),
                        file_name="rule_based_validation_results.csv",
                        mime="text/csv",
                    )

                # Inter-rater reliability
                if {"rater_1", "rater_2"}.issubset(validation_df.columns):
                    st.divider()
                    st.subheader("Inter-rater reliability")

                    paired = validation_df[["rater_1", "rater_2"]].copy()
                    paired["r1"] = paired["rater_1"].apply(normalize_binary_label)
                    paired["r2"] = paired["rater_2"].apply(normalize_binary_label)
                    paired = paired.dropna(subset=["r1", "r2"])

                    if len(paired) >= 2:
                        r1 = paired["r1"].astype(int).to_numpy()
                        r2 = paired["r2"].astype(int).to_numpy()
                        if len(np.unique(np.concatenate([r1, r2]))) > 1:
                            kappa_raters = cohen_kappa_score(r1, r2)
                            st.metric("Cohen's Kappa (raters)", f"{kappa_raters:.3f}")
                        else:
                            st.warning(
                                "Cohen's Kappa is not informative when both raters "
                                "use only one category."
                            )
                    else:
                        st.warning("At least two complete paired annotations are required.")

                # Multi-rater alpha
                rater_columns = [
                    column for column in validation_df.columns
                    if re.fullmatch(r"rater_\d+", str(column))
                ]
                if len(rater_columns) >= 2:
                    st.subheader("Krippendorff's Alpha (nominal)")
                    rating_rows = []
                    for _, row in validation_df[rater_columns].iterrows():
                        rating_rows.append([
                            normalize_binary_label(value)
                            for value in row.tolist()
                        ])
                    alpha = krippendorff_alpha_nominal(rating_rows)
                    if alpha is None:
                        st.warning(
                            "Alpha could not be calculated from the supplied ratings."
                        )
                    else:
                        st.metric("Krippendorff's α", f"{alpha:.3f}")
                        st.caption(
                            "Calculated for nominal binary labels using available "
                            "ratings. Interpret in light of prevalence, sample size, "
                            "missingness, and annotation design."
                        )

        except Exception as exc:
            st.error(f"Could not process the uploaded dataset: {exc}")

    st.markdown("### Example CSV schema")
    st.code(
        "text,gold_label,rater_1,rater_2\n"
        '"I was afraid to speak.",1,1,1\n'
        '"I discussed my concerns.",0,0,0\n'
        '"I did not feel afraid.",0,0,0',
        language="csv",
    )
    st.caption("Illustrative schema only. These rows are not research evidence.")

# ============================================================
# TAB 6 — MODEL LABORATORY
# ============================================================

with tabs[5]:
    st.header("Supervised Model Laboratory")
    st.write(
        "Train a TF-IDF + Logistic Regression baseline on your own independently "
        "labelled dataset. This is separate from the lexicon-based rule engine."
    )

    st.warning(
        "Use a dataset with a defensible annotation protocol. A random split is not "
        "sufficient when records from the same participant, source, or conversation "
        "could appear in both train and test sets. In that case, use group-based splitting."
    )

    model_file = st.file_uploader(
        "Upload labelled training CSV",
        type=["csv"],
        key="model_upload",
    )

    if model_file is not None:
        try:
            model_df = pd.read_csv(model_file)
            st.dataframe(model_df.head(10), use_container_width=True)

            col_text, col_label = st.columns(2)
            with col_text:
                text_col = st.selectbox(
                    "Text column",
                    model_df.columns.tolist(),
                    index=(model_df.columns.tolist().index("text")
                           if "text" in model_df.columns else 0),
                )
            with col_label:
                default_label_index = (
                    model_df.columns.tolist().index("gold_label")
                    if "gold_label" in model_df.columns else
                    min(1, len(model_df.columns) - 1)
                )
                label_col = st.selectbox(
                    "Gold-label column",
                    model_df.columns.tolist(),
                    index=default_label_index,
                )

            test_fraction = st.slider(
                "Test set proportion",
                min_value=0.20,
                max_value=0.40,
                value=0.25,
                step=0.05,
            )

            if st.button("Train and evaluate baseline model", type="primary"):
                with st.spinner("Training TF-IDF + Logistic Regression..."):
                    trained = train_supervised_model(
                        model_df,
                        text_col,
                        label_col,
                        test_size=test_fraction,
                    )
                    st.session_state.trained_model = trained["model"]
                    st.session_state.trained_model_info = {
                        "metrics": trained["metrics"],
                        "test_text": trained["test_text"],
                        "y_test": trained["y_test"].tolist(),
                        "predictions": trained["predictions"].tolist(),
                        "feature_names": trained["feature_names"][:100].tolist(),
                        "text_column": text_col,
                        "label_column": label_col,
                    }

            if st.session_state.trained_model_info:
                info = st.session_state.trained_model_info
                metrics = info["metrics"]
                st.divider()
                st.subheader("Held-out test results")
                a, b, c, d = st.columns(4)
                a.metric("Precision", f"{metrics['precision']:.3f}")
                b.metric("Recall", f"{metrics['recall']:.3f}")
                c.metric("F1", f"{metrics['f1']:.3f}")
                d.metric("Accuracy", f"{metrics['accuracy']:.3f}")

                st.caption(
                    f"Records: {metrics['n_total']} total · "
                    f"{metrics['n_train']} train · {metrics['n_test']} test"
                )

                ci = metrics["f1_bootstrap_95ci"]
                if ci[0] is not None:
                    st.caption(f"Bootstrap 95% F1 interval: {ci[0]:.3f}–{ci[1]:.3f}")

                cm = pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=["Actual negative", "Actual positive"],
                    columns=["Predicted negative", "Predicted positive"],
                )
                st.subheader("Confusion matrix")
                st.dataframe(cm)

                st.subheader("Most influential positive/negative linear features")
                classifier = st.session_state.trained_model.named_steps["classifier"]
                vectorizer = st.session_state.trained_model.named_steps["tfidf"]
                feature_names = vectorizer.get_feature_names_out()
                coefficients = classifier.coef_[0]
                feature_table = pd.DataFrame({
                    "term": feature_names,
                    "coefficient": coefficients,
                }).sort_values("coefficient", ascending=False)

                left, right = st.columns(2)
                with left:
                    st.write("Higher positive-class coefficients")
                    st.dataframe(feature_table.head(15), hide_index=True)
                with right:
                    st.write("Higher negative-class coefficients")
                    st.dataframe(
                        feature_table.tail(15).sort_values("coefficient"),
                        hide_index=True,
                    )

                st.caption(
                    "Linear coefficients describe associations in this fitted model. "
                    "They are not causal explanations and may reflect dataset artefacts."
                )

                st.download_button(
                    "Download model evaluation summary",
                    json_bytes(metrics),
                    file_name="model_evaluation_summary.json",
                    mime="application/json",
                )

                st.subheader("Try a new anonymised text")
                prediction_text = st.text_area(
                    "New text for the trained baseline",
                    key="model_prediction_input",
                    height=100,
                )
                if st.button("Predict with trained baseline"):
                    if prediction_text.strip():
                        model = st.session_state.trained_model
                        predicted = int(model.predict([prediction_text])[0])
                        probabilities = model.predict_proba([prediction_text])[0]
                        class_order = list(model.classes_)
                        probability_map = {
                            str(cls): float(prob)
                            for cls, prob in zip(class_order, probabilities)
                        }
                        st.write(
                            "Model output:",
                            "Positive class" if predicted == 1 else "Negative class",
                        )
                        st.json({
                            "predicted_class": predicted,
                            "class_probabilities": probability_map,
                            "warning": (
                                "Model probabilities are not calibrated clinical "
                                "probabilities and must not be interpreted as an "
                                "individual's psychological state."
                            ),
                        })

        except Exception as exc:
            st.error(f"Model training failed: {exc}")

    else:
        st.info(
            "Upload a CSV with text and binary gold_label columns to train a baseline. "
            "The platform does not ship with a fabricated pretrained clinical model."
        )

# ============================================================
# TAB 7 — NARRATIVE COMPARISON
# ============================================================

with tabs[6]:
    st.header("Narrative Comparison Laboratory")
    st.write(
        "Compare two anonymised narratives using transparent lexical counts. "
        "This is descriptive comparison, not a test of which patient is more distressed."
    )

    left_text = st.text_area(
        "Narrative A",
        "I wanted to explain my symptoms, but I hesitated.",
        height=130,
        key="comparison_a",
    )
    right_text = st.text_area(
        "Narrative B",
        "I explained my concerns and asked the doctor several questions.",
        height=130,
        key="comparison_b",
    )

    if st.button("Compare narratives", type="primary"):
        if not left_text.strip() or not right_text.strip():
            st.error("Enter both narratives.")
        else:
            a_result = analyze_document(left_text)
            b_result = analyze_document(right_text)
            a_metrics = calculate_descriptive_indicators(
                left_text, a_result["findings"]
            )
            b_metrics = calculate_descriptive_indicators(
                right_text, b_result["findings"]
            )

            comparison = pd.DataFrame([
                {
                    "Narrative": "A",
                    "Words": a_metrics["word_count"],
                    "Lexical matches": a_metrics["total_matches"],
                    "Negation flags": a_metrics["negated_matches"],
                    "Matches per 100 words": a_metrics["marker_density_per_100_words"],
                },
                {
                    "Narrative": "B",
                    "Words": b_metrics["word_count"],
                    "Lexical matches": b_metrics["total_matches"],
                    "Negation flags": b_metrics["negated_matches"],
                    "Matches per 100 words": b_metrics["marker_density_per_100_words"],
                },
            ])
            st.dataframe(comparison, use_container_width=True, hide_index=True)

            category_union = sorted(
                set(a_metrics["category_distribution"])
                | set(b_metrics["category_distribution"])
            )
            if category_union:
                cat_compare = pd.DataFrame({
                    "Narrative A": [
                        a_metrics["category_distribution"].get(cat, 0)
                        for cat in category_union
                    ],
                    "Narrative B": [
                        b_metrics["category_distribution"].get(cat, 0)
                        for cat in category_union
                    ],
                }, index=category_union)
                st.bar_chart(cat_compare)

            st.caption(
                "Differences may reflect text length, vocabulary, context, language, "
                "or lexicon coverage. They do not establish differences in actual "
                "patient experience or communication quality."
            )

# ============================================================
# TAB 8 — REPORTS AND EXPORTS
# ============================================================

with tabs[7]:
    st.header("Research Reports & Reproducibility")

    if st.session_state.last_analysis:
        current = st.session_state.last_analysis
        report_text = make_research_report(current)

        st.subheader("Current analysis report")
        st.download_button(
            "Download Markdown research report",
            data=report_text.encode("utf-8"),
            file_name="neurometric_exploratory_report.md",
            mime="text/markdown",
        )

        export_payload = {
            "project": PROJECT_NAME,
            "version": APP_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": current,
            "interpretation_notice": (
                "Exploratory lexical analysis only; not a clinical assessment."
            ),
        }
        st.download_button(
            "Download analysis as JSON",
            data=json_bytes(export_payload),
            file_name="neurometric_analysis.json",
            mime="application/json",
        )

        findings_df = pd.DataFrame(current["result"]["findings"])
        if not findings_df.empty:
            st.download_button(
                "Download detected markers as CSV",
                dataframe_csv_bytes(findings_df),
                file_name="neurometric_detected_markers.csv",
                mime="text/csv",
            )
    else:
        st.info("Run a text analysis to generate a report.")

    st.divider()
    st.subheader("Session analysis history")
    if st.session_state.analysis_history:
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Export session history",
            dataframe_csv_bytes(history_df),
            file_name="neurometric_session_history.csv",
            mime="text/csv",
        )
    else:
        st.caption("No analyses recorded in this session.")

    st.divider()
    st.subheader("Environment information")
    environment = {
        "application_version": APP_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
    }
    st.json(environment)
    st.download_button(
        "Export environment metadata",
        json_bytes(environment),
        file_name="neurometric_environment.json",
        mime="application/json",
    )

# ============================================================
# TAB 9 — METHODOLOGY AND ETHICS
# ============================================================

with tabs[8]:
    st.header("Methodology, Theory & Ethics")

    with st.expander("1. Narrative Medicine", expanded=True):
        st.markdown(
            """
            Narrative Medicine examines how illness is experienced, narrated,
            interpreted, and responded to. Rita Charon's work on narrative
            competence is a relevant theoretical foundation. This platform
            operationalises only selected linguistic features; it does not
            claim to measure narrative competence or empathy in full.
            """
        )

    with st.expander("2. The Clinical Gaze"):
        st.markdown(
            """
            Michel Foucault's analysis of the clinical gaze can serve as a
            critical lens for examining how institutional medical practices
            privilege certain forms of knowledge. The software does not
            computationally measure Foucauldian power relations; it explores
            selected language patterns that researchers may interpret in context.
            """
        )

    with st.expander("3. Explainable NLP"):
        st.markdown(
            """
            The lexical module provides direct traceability by showing the
            expression and rule that triggered a match. The TF-IDF model
            provides inspectable linear coefficients. Neither form of
            explanation proves that the model's interpretation is correct.
            """
        )

    with st.expander("4. Validation protocol"):
        st.markdown(
            """
            A defensible study should pre-register or document the research
            question, inclusion criteria, sampling procedure, annotation
            handbook, annotator training, adjudication process, data split,
            primary metrics, and error analysis.

            If several records come from the same participant, interview,
            clinic, or source, split by group rather than randomly by row.
            A held-out test set should not be used repeatedly for model tuning.
            """
        )

    with st.expander("5. Ethical and privacy safeguards"):
        st.markdown(
            """
            - Do not enter names, addresses, contact details, record numbers,
              or other identifying information.
            - Obtain appropriate consent and ethics review before collecting
              or analysing human-subject narratives.
            - Use data minimisation, access controls, retention limits, and
              secure storage.
            - Do not infer diagnosis, deception, trauma, mental state, or
              treatment risk from lexical markers.
            - Do not use outputs for triage, clinical decisions, or performance
              assessment of individual clinicians or patients.
            - Check applicable institutional and local data-protection rules.
            """
        )

    with st.expander("6. Current limitations"):
        st.markdown(
            """
            - The bilingual lexicon is manually defined and incomplete.
            - Language identification is a heuristic.
            - Negation handling is local and can misclassify scope.
            - The sentence splitter is lightweight.
            - Sarcasm, metaphor, code-switching, dialects, and pragmatic context
              are not reliably handled.
            - The supervised model is only as reliable as its training labels,
              sampling, and evaluation design.
            - No clinical validation has been performed by this application.
            """
        )

    with st.expander("7. Proposed research extensions"):
        st.markdown(
            """
            1. Develop and publish a versioned annotation codebook.
            2. Build an ethically sourced, consented, de-identified corpus.
            3. Include independent annotators and adjudication.
            4. Compare lexicon, classical ML, and appropriately fine-tuned
               multilingual transformer models.
            5. Use participant- or source-level data splits.
            6. Evaluate calibration, subgroup performance, robustness,
               and out-of-domain generalisation.
            7. Conduct qualitative review of disagreement cases.
            8. Publish a model card, data statement, limitations, and
               reproducible analysis scripts.
            """
        )

    st.markdown("### Research integrity statement")
    st.info(
        "No empirical performance, reliability coefficient, clinical validity, "
        "or statistical significance is claimed unless it is calculated from "
        "appropriate, documented research data."
    )

# ============================================================
# 11. FOOTER
# ============================================================

st.divider()
st.markdown(
    f"""
    <div style="text-align:center; color:#64748B; padding:14px;">
        <strong>{PROJECT_NAME}</strong><br>
        {PROJECT_SUBTITLE}<br>
        <small>Version {APP_VERSION} · Experimental research software · Not a medical device</small>
    </div>
    """,
    unsafe_allow_html=True,
)
