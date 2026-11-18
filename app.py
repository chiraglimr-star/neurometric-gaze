
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
from datetime import datetime
from collections import Counter

from sklearn.metrics import (
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="The Neurometric Gaze | Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FA;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #E3E8EF;
        padding: 18px;
        border-radius: 12px;
    }

    .hero {
        background: linear-gradient(120deg, #10243A, #244C65);
        padding: 35px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 35px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #D5E4EE;
        font-size: 15px;
    }

    .section-card {
        background: white;
        border: 1px solid #E3E8EF;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .small-note {
        color: #64748B;
        font-size: 12px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. RESEARCH IDENTITY
# ============================================================

PROJECT_NAME = "The Neurometric Gaze"

PROJECT_DESCRIPTION = (
    "Computational Narrative Medicine and Patient–Physician "
    "Communication Research"
)

RESEARCH_QUESTION = (
    "How can explainable computational language analysis "
    "help researchers identify linguistic indicators of "
    "communication barriers in patient–physician interactions?"
)

st.markdown("""
<div class="hero">
    <h1>THE NEUROMETRIC GAZE</h1>
    <p>COMPUTATIONAL NARRATIVE MEDICINE & COMMUNICATION INTELLIGENCE</p>
    <hr>
    <p>
    An exploratory research platform examining linguistic indicators
    of hesitation, fear, avoidance, and self-censorship in healthcare
    communication.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "annotation_records" not in st.session_state:
    st.session_state.annotation_records = []

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None


# ============================================================
# 4. LINGUISTIC MARKER FRAMEWORK
# ============================================================

# IMPORTANT:
# These are exploratory lexical patterns.
# They are NOT clinically validated indicators.
# Context and negation must be considered by human reviewers.

MARKER_LIBRARY = {

    "Hesitation": {
        "description": "Expressions of difficulty or reluctance to speak.",
        "patterns": [
            r"\bi hesitated\b",
            r"\bi couldn't say\b",
            r"\bi could not say\b",
            r"\bi was reluctant\b",
            r"\bi didn't know how to say\b",
            r"\bi did not know how to say\b",
            r"\bdeməyə çətinlik çəkdim\b",
            r"\bdanışmağa çəkindim\b",
            r"\bdeməyə çəkindim\b",
            r"\bdilə gətirə bilmədim\b",
            r"\bdeyə bilmədim\b",
            r"\btərəddüd etdim\b"
        ]
    },

    "Fear and Anxiety": {
        "description": "Explicit expressions of fear or anxiety.",
        "patterns": [
            r"\bi was afraid\b",
            r"\bi was scared\b",
            r"\bi feared\b",
            r"\bi felt anxious\b",
            r"\bi was worried\b",
            r"\bqorxdum\b",
            r"\bqorxurdum\b",
            r"\bqorxuram\b",
            r"\bnarahat idim\b",
            r"\bnarahat oldum\b",
            r"\btəşvişə düşdüm\b"
        ]
    },

    "Avoidance": {
        "description": "Expressions potentially indicating avoidance or minimisation.",
        "patterns": [
            r"\bit doesn't matter\b",
            r"\bit is not important\b",
            r"\bnever mind\b",
            r"\bi ignored it\b",
            r"\bi avoided\b",
            r"\bi just waited\b",
            r"\bvacib deyil\b",
            r"\bönəmli deyil\b",
            r"\bəhəmiyyət vermədim\b",
            r"\bsadəcə gözlədim\b",
            r"\böz-özünə keçər\b",
            r"\bboş ver\b"
        ]
    },

    "Self-Censorship": {
        "description": "Expressions of withholding or concealing information.",
        "patterns": [
            r"\bi kept it to myself\b",
            r"\bi hid it\b",
            r"\bi didn't tell the doctor\b",
            r"\bi did not tell the doctor\b",
            r"\bi kept silent\b",
            r"\bi remained silent\b",
            r"\bi concealed\b",
            r"\bI chose not to mention\b",
            r"\bsusdum\b",
            r"\bgizlətdim\b",
            r"\bdemədim\b",
            r"\bdanışmadım\b",
            r"\bheç kimə demədim\b",
            r"\böz içimdə saxladım\b"
        ]
    },

    "Power Imbalance": {
        "description": "Possible references to authority or unequal interaction.",
        "patterns": [
            r"\bthe doctor would be angry\b",
            r"\bthe doctor might judge me\b",
            r"\bi was afraid of the doctor\b",
            r"\bthe doctor wouldn't listen\b",
            r"\bhe wouldn't believe me\b",
            r"\bshe wouldn't believe me\b",
            r"\bhəkim əsəbləşər\b",
            r"\bhəkim məni danlayar\b",
            r"\bmənə inanmaz\b",
            r"\bhəkimdən qorxdum\b",
            r"\bməni dinləmədi\b"
        ]
    },

    "Dismissal and Minimisation": {
        "description": "Potential minimisation of personal symptoms or concerns.",
        "patterns": [
            r"\bit is probably nothing\b",
            r"\bit's probably nothing\b",
            r"\bi am overreacting\b",
            r"\bi don't want to bother\b",
            r"\bi didn't want to bother\b",
            r"\bjust a small problem\b",
            r"\bproblem deyil\b",
            r"\byəqin ki, heç nə deyil\b",
            r"\bşişirtmək istəmədim\b",
            r"\bəhəmiyyətsiz hesab etdim\b"
        ]
    }
}


# ============================================================
# 5. TEXT PREPROCESSING
# ============================================================

def normalize_text(text):
    """Normalize text without removing meaningful linguistic content."""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text):
    """Basic multilingual sentence segmentation."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# 6. NEGATION AND CONTEXT HANDLING
# ============================================================

NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bno longer\b",
    r"\bdidn't\b",
    r"\bdid not\b",
    r"\bwasn't\b",
    r"\bwas not\b",
    r"\bcannot\b",
    r"\bcan't\b",
    r"\bnot afraid\b",
    r"\bqorxmadım\b",
    r"\bqorxmuram\b",
    r"\bnarahat deyiləm\b",
    r"\bsusmadım\b",
    r"\bdemək istəmirəm\b"
]


def detect_negation(sentence, start_position):
    """
    Check a limited window before a matched expression.
    This is a basic heuristic, not a syntactic negation model.
    """

    preceding_text = sentence[:start_position].lower()

    window = preceding_text[-55:]

    return any(
        re.search(pattern, window)
        for pattern in NEGATION_PATTERNS
    )


# ============================================================
# 7. EXPLAINABLE TEXT ANALYSIS
# ============================================================

def analyze_sentence(sentence):

    normalized = normalize_text(sentence)

    findings = []

    for category, information in MARKER_LIBRARY.items():

        for pattern in information["patterns"]:

            matches = list(
                re.finditer(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE
                )
            )

            for match in matches:

                negated = detect_negation(
                    normalized,
                    match.start()
                )

                findings.append({
                    "category": category,
                    "matched_expression": match.group(),
                    "sentence": sentence,
                    "negation_flag": negated,
                    "description": information["description"],
                    "interpretation": (
                        "Possible negated expression; requires review"
                        if negated
                        else "Potential linguistic indicator; requires review"
                    )
                })

    return findings


def analyze_document(text):

    sentences = split_sentences(text)

    all_findings = []

    sentence_results = []

    for index, sentence in enumerate(sentences, start=1):

        findings = analyze_sentence(sentence)

        all_findings.extend(findings)

        sentence_results.append({
            "sentence_id": index,
            "sentence": sentence,
            "marker_count": len(findings),
            "findings": findings
        })

    return {
        "sentences": sentence_results,
        "findings": all_findings
    }


# ============================================================
# 8. EXPLORATORY INDICATOR CALCULATION
# ============================================================

def calculate_exploratory_indicators(text, findings):

    word_count = len(text.split())

    total_markers = len(findings)

    categories = Counter(
        item["category"]
        for item in findings
    )

    negated_count = sum(
        item["negation_flag"]
        for item in findings
    )

    # Exclude potentially negated matches from the exploratory count.
    non_negated_count = total_markers - negated_count

    # Transparent descriptive density.
    marker_density = (
        non_negated_count / word_count * 100
        if word_count > 0 else 0
    )

    # This is NOT a clinical score or a probability.
    return {
        "word_count": word_count,
        "total_markers": total_markers,
        "negated_matches": negated_count,
        "non_negated_matches": non_negated_count,
        "marker_density": round(marker_density, 2),
        "category_distribution": dict(categories)
    }


# ============================================================
# 9. RESEARCH SIDEBAR
# ============================================================

st.sidebar.title("Research Configuration")

language = st.sidebar.selectbox(
    "Analysis language",
    ["Azerbaijani", "English", "Mixed / Multilingual"]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Methodological Framework")

st.sidebar.markdown("""
- Narrative Medicine
- Health Communication
- Computational Linguistics
- Explainable NLP
- Human-Centred AI
""")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Version 2.0 | Exploratory research prototype"
)


# ============================================================
# 10. MAIN NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Text Analysis",
    "Explainability",
    "Human Annotation",
    "Research Validation",
    "Methodology & Ethics"
])


# ============================================================
# 11. TAB 1 — TEXT ANALYSIS
# ============================================================

with tab1:

    st.header("Patient–Physician Communication Analysis")

    st.write(
        "Examine possible linguistic indicators of communication "
        "barriers in anonymised narrative samples."
    )

    st.warning(
        "Do not enter names, addresses, medical record numbers, "
        "or identifiable patient information."
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
        key="main_patient_text"
    )

    analyze_button = st.button(
        "Run Exploratory Analysis",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if len(patient_text.strip()) < 10:

            st.error("Please provide a longer text sample.")

        else:

            result = analyze_document(patient_text)

            indicators = calculate_exploratory_indicators(
                patient_text,
                result["findings"]
            )

            st.session_state.last_analysis = {
                "text": patient_text,
                "result": result,
                "indicators": indicators
            }

            st.session_state.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "word_count": indicators["word_count"],
                "marker_count": indicators["total_markers"],
                "marker_density": indicators["marker_density"]
            })

    if st.session_state.last_analysis is not None:

        analysis = st.session_state.last_analysis

        indicators = analysis["indicators"]

        st.divider()

        st.subheader("Descriptive Research Indicators")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Word Count",
            indicators["word_count"]
        )

        c2.metric(
            "Detected Matches",
            indicators["total_markers"]
        )

        c3.metric(
            "Potential Negated Matches",
            indicators["negated_matches"]
        )

        c4.metric(
            "Marker Density",
            f'{indicators["marker_density"]}%'
        )

        st.caption(
            "Marker density = non-negated lexical matches / total words × 100. "
            "It is descriptive, not a clinical severity measure."
        )

        st.divider()

        st.subheader("Sentence-Level Results")

        for sentence_result in analysis["result"]["sentences"]:

            with st.expander(
                f'Sentence {sentence_result["sentence_id"]} '
                f'| {sentence_result["marker_count"]} potential matches'
            ):

                st.write(sentence_result["sentence"])

                if sentence_result["findings"]:

                    for finding in sentence_result["findings"]:

                        st.markdown(
                            f'**{finding["category"]}** — '
                            f'`{finding["matched_expression"]}`'
                        )

                        st.caption(
                            finding["interpretation"]
                        )

                else:

                    st.info(
                        "No expressions from the current exploratory "
                        "lexicon were detected."
                    )


# ============================================================
# 12. TAB 2 — EXPLAINABILITY
# ============================================================

with tab2:

    st.header("Explainable Linguistic Analysis")

    st.write(
        "Inspect the expressions responsible for each detected marker."
    )

    if st.session_state.last_analysis is None:

        st.info("Run a text analysis first.")

    else:

        findings = st.session_state.last_analysis[
            "result"
        ]["findings"]

        if not findings:

            st.info(
                "No current lexicon matches were identified. "
                "This does not establish that communication barriers "
                "are absent."
            )

        else:

            explanation_df = pd.DataFrame(findings)

            display_columns = [
                "category",
                "matched_expression",
                "sentence",
                "negation_flag",
                "interpretation"
            ]

            st.dataframe(
                explanation_df[display_columns],
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Marker Distribution")

            category_counts = (
                explanation_df["category"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Occurrences")
            )

            st.bar_chart(
                category_counts.set_index("Category")
            )

            st.subheader("Interpretation")

            st.markdown("""
            The system provides traceable lexical matches rather than
            hidden model predictions.

            Each result includes:

            - The original sentence.
            - The matched linguistic expression.
            - Its assigned exploratory category.
            - A basic negation flag.
            - An explanation of why the expression was flagged.

            Human review is necessary before assigning any research label.
            """)


# ============================================================
# 13. TAB 3 — HUMAN ANNOTATION
# ============================================================

with tab3:

    st.header("Human Annotation Workspace")

    st.write(
        "Create a human-reviewed dataset for subsequent evaluation."
    )

    st.markdown("""
    Human annotation is important because a lexical match does not
    necessarily represent fear, avoidance, or self-censorship.
    """)

    if st.session_state.last_analysis is not None:

        current_text = st.session_state.last_analysis["text"]

        st.text_area(
            "Text currently under review",
            value=current_text,
            height=100,
            disabled=True
        )

        st.markdown("### Researcher's assessment")

        human_label = st.radio(
            "Does this narrative contain a communication barrier?",
            [
                "Positive",
                "Negative",
                "Uncertain"
            ],
            horizontal=True,
            key="human_label"
        )

        human_category = st.selectbox(
            "Primary category",
            [
                "Hesitation",
                "Fear and Anxiety",
                "Avoidance",
                "Self-Censorship",
                "Power Imbalance",
                "Dismissal and Minimisation",
                "Other / No clear category"
            ]
        )

        annotation_note = st.text_area(
            "Researcher's justification",
            placeholder=(
                "Explain the linguistic and contextual evidence "
                "supporting your annotation."
            )
        )

        if st.button("Save Human Annotation"):

            st.session_state.annotation_records.append({
                "timestamp": datetime.now().isoformat(),
                "text": current_text,
                "human_label": human_label,
                "primary_category": human_category,
                "researcher_note": annotation_note
            })

            st.success(
                "Annotation added to the current session."
            )

    else:

        st.info(
            "Analyse a narrative before opening the annotation workspace."
        )

    if st.session_state.annotation_records:

        st.divider()

        st.subheader("Annotation Records")

        annotation_df = pd.DataFrame(
            st.session_state.annotation_records
        )

        st.dataframe(
            annotation_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = annotation_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Export Annotation Dataset",
            data=csv_data,
            file_name="human_annotations.csv",
            mime="text/csv"
        )


# ============================================================
# 14. TAB 4 — RESEARCH VALIDATION
# ============================================================

with tab4:

    st.header("Empirical Validation & Reliability")

    st.write(
        "Evaluate the exploratory classifier against independently "
        "human-labelled research data."
    )

    st.info(
        "Validation metrics are displayed only when suitable labelled "
        "data is supplied. No reliability coefficient is assumed."
    )

    uploaded_file = st.file_uploader(
        "Upload a research CSV dataset",
        type=["csv"],
        help=(
            "Required column: text. Optional columns: gold_label, "
            "rater_1, rater_2."
        )
    )

    if uploaded_file is not None:

        try:

            dataset = pd.read_csv(uploaded_file)

            st.subheader("Dataset Preview")

            st.dataframe(
                dataset.head(10),
                use_container_width=True
            )

            if "text" not in dataset.columns:

                st.error(
                    "The dataset must contain a 'text' column."
                )

            else:

                st.write(
                    f"Dataset size: {len(dataset)} records."
                )

                validation_threshold = st.slider(
                    "Exploratory classifier threshold",
                    min_value=1,
                    max_value=5,
                    value=1,
                    help=(
                        "Minimum number of non-negated lexical matches "
                        "required for a positive prediction."
                    )
                )

                prediction_results = []

                for text_sample in dataset["text"].fillna(""):

                    doc = analyze_document(str(text_sample))

                    metrics = calculate_exploratory_indicators(
                        str(text_sample),
                        doc["findings"]
                    )

                    prediction_results.append(
                        int(
                            metrics["non_negated_matches"]
                            >= validation_threshold
                        )
                    )

                dataset["exploratory_prediction"] = (
                    prediction_results
                )

                st.subheader("Classifier Output")

                st.dataframe(
                    dataset,
                    use_container_width=True
                )

                # ---------------------------------------------
                # GOLD STANDARD EVALUATION
                # ---------------------------------------------

                if "gold_label" in dataset.columns:

                    st.markdown("### Evaluation Against Gold Labels")

                    def convert_binary_label(value):

                        if pd.isna(value):
                            return np.nan

                        value = str(value).strip().lower()

                        positive = {
                            "1", "positive", "yes",
                            "true", "present"
                        }

                        negative = {
                            "0", "negative", "no",
                            "false", "absent"
                        }

                        if value in positive:
                            return 1

                        if value in negative:
                            return 0

                        return np.nan

                    y_true = dataset["gold_label"].apply(
                        convert_binary_label
                    )

                    valid = y_true.notna()

                    y_true = y_true[valid].astype(int)

                    y_pred = dataset.loc[
                        valid,
                        "exploratory_prediction"
                    ].astype(int)

                    if len(y_true) > 0:

                        precision = precision_score(
                            y_true,
                            y_pred,
                            zero_division=0
                        )

                        recall = recall_score(
                            y_true,
                            y_pred,
                            zero_division=0
                        )

                        f1 = f1_score(
                            y_true,
                            y_pred,
                            zero_division=0
                        )

                        accuracy = accuracy_score(
                            y_true,
                            y_pred
                        )

                        a, b, c, d = st.columns(4)

                        a.metric(
                            "Precision",
                            f"{precision:.3f}"
                        )

                        b.metric(
                            "Recall",
                            f"{recall:.3f}"
                        )

                        c.metric(
                            "F1 Score",
                            f"{f1:.3f}"
                        )

                        d.metric(
                            "Accuracy",
                            f"{accuracy:.3f}"
                        )

                        st.caption(
                            f"Evaluated on {len(y_true)} labelled records."
                        )

                        st.markdown("#### Confusion Matrix")

                        cm = confusion_matrix(
                            y_true,
                            y_pred,
                            labels=[0, 1]
                        )

                        cm_df = pd.DataFrame(
                            cm,
                            index=[
                                "Actual Negative",
                                "Actual Positive"
                            ],
                            columns=[
                                "Predicted Negative",
                                "Predicted Positive"
                            ]
                        )

                        st.dataframe(cm_df)

                        st.text(
                            classification_report(
                                y_true,
                                y_pred,
                                zero_division=0
                            )
                        )

                    else:

                        st.warning(
                            "No valid binary gold labels were found."
                        )

                # ---------------------------------------------
                # INTER-RATER RELIABILITY
                # ---------------------------------------------

                if (
                    "rater_1" in dataset.columns
                    and "rater_2" in dataset.columns
                ):

                    st.markdown("### Inter-Rater Reliability")

                    rater_data = dataset[
                        ["rater_1", "rater_2"]
                    ].copy()

                    rater_data = rater_data.dropna()

                    rater1 = rater_data["rater_1"].apply(
                        convert_binary_label
                    )

                    rater2 = rater_data["rater_2"].apply(
                        convert_binary_label
                    )

                    valid_raters = (
                        rater1.notna() & rater2.notna()
                    )

                    rater1 = rater1[valid_raters].astype(int)
                    rater2 = rater2[valid_raters].astype(int)

                    if len(rater1) > 0:

                        kappa = cohen_kappa_score(
                            rater1,
                            rater2
                        )

                        st.metric(
                            "Cohen's Kappa",
                            f"{kappa:.3f}"
                        )

                        st.caption(
                            f"Independent annotation pairs: {len(rater1)}"
                        )

                        st.write(
                            "This coefficient measures agreement "
                            "between the two supplied raters beyond "
                            "chance agreement, subject to the assumptions "
                            "and limitations of Cohen's Kappa."
                        )

                    else:

                        st.warning(
                            "No valid paired binary annotations."
                        )

                st.download_button(
                    "Download Validation Results",
                    data=dataset.to_csv(
                        index=False
                    ).encode("utf-8-sig"),
                    file_name="validation_results.csv",
                    mime="text/csv"
                )

        except Exception as error:

            st.error(
                f"Dataset processing error: {error}"
            )

    else:

        st.markdown("""
        **Recommended dataset structure**

        | text | gold_label | rater_1 | rater_2 |
        |---|---:|---:|---:|
        | I was afraid to speak. | 1 | 1 | 1 |
        | I discussed my concerns. | 0 | 0 | 0 |
        | I did not feel afraid. | 0 | 0 | 0 |

        This is an illustrative format only, not an empirical dataset.
        Use independently annotated, ethically sourced samples for
        actual validation.
        """)


# ============================================================
# 15. TAB 5 — METHODOLOGY AND ETHICS
# ============================================================

with tab5:

    st.header("Research Methodology & Ethical Framework")

    st.subheader("1. Theoretical Foundations")

    with st.expander("Narrative Medicine", expanded=True):

        st.write("""
        Narrative Medicine examines how patients communicate their
        experiences of illness and how healthcare professionals
        interpret those narratives.

        This project draws on the work of Rita Charon and the
        importance of narrative competence in healthcare.
        """)

    with st.expander("The Clinical Gaze"):

        st.write("""
        Michel Foucault's concept of the clinical gaze offers a
        theoretical lens for examining how institutional medical
        practices may privilege observable clinical information
        over patients' subjective experiences.

        The project uses this concept as a critical framework,
        not as a computationally established causal mechanism.
        """)

    with st.expander("Explainable Artificial Intelligence"):

        st.write("""
        The current implementation uses transparent lexical rules.
        It exposes each detected expression and its category.

        It does not currently implement SHAP, LIME, transformer
        attention analysis, or a trained language model.
        """)

    st.subheader("2. Research Workflow")

    st.markdown("""
    1. Collect ethically sourced and anonymised narratives.
    2. Define an annotation protocol.
    3. Have independent human annotators label the material.
    4. Apply the exploratory linguistic analysis.
    5. Compare predictions with human annotations.
    6. Calculate precision, recall and F1.
    7. Calculate inter-rater reliability where appropriate.
    8. Analyse false positives and false negatives.
    9. Revise the linguistic framework.
    """)

    st.subheader("3. Ethical Safeguards")

    st.markdown("""
    - No automated medical diagnosis.
    - No inference of an individual's psychological condition.
    - No claims that silence proves deception or concealment.
    - No patient-identifiable information.
    - Human review of all interpretations.
    - Transparent disclosure of model limitations.
    - Secure handling of any research dataset.
    """)

    st.subheader("4. Current Technical Limitations")

    st.warning("""
    This prototype uses manually defined lexical patterns.
    It does not understand language in the full contextual sense.

    It may fail to recognise:
    - Sarcasm and irony.
    - Complex negation.
    - Cultural and linguistic variation.
    - Implicit expressions of distress.
    - Metaphors and indirect communication.
    - Context-dependent meanings.

    The current system has not undergone clinical validation.
    """)

    st.subheader("5. Future Research Development")

    st.markdown("""
    **Proposed extensions:**

    - Contextual multilingual NLP models.
    - Azerbaijani-specific linguistic resources.
    - Human-in-the-loop annotation.
    - Cross-cultural communication comparisons.
    - More advanced negation and discourse analysis.
    - External validation on independently collected datasets.
    - Reproducible research documentation.
    """)

    st.divider()

    st.caption(
        "The Neurometric Gaze | Computational Narrative Medicine "
        "| Exploratory Research Prototype"
    )


# ============================================================
# 16. SESSION EXPORT
# ============================================================

st.sidebar.divider()

st.sidebar.markdown("### Research Session")

st.sidebar.write(
    f"Saved annotations: {len(st.session_state.annotation_records)}"
)

st.sidebar.write(
    f"Analyses in session: {len(st.session_state.analysis_history)}"
)

if st.sidebar.button("Clear Current Session"):

    st.session_state.annotation_records = []
    st.session_state.analysis_history = []
    st.session_state.last_analysis = None

    st.rerun()


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B; padding:15px;">
        <strong>THE NEUROMETRIC GAZE</strong><br>
        Computational Narrative Medicine Research Platform<br>
        <small>Experimental tool | Not a medical diagnostic system</small>
    </div>
    """,
    unsafe_allow_html=True
)
