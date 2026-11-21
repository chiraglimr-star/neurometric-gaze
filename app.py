"""
THE NEUROMETRIC GAZE — BEYOND THE UNHEARD
Interdisciplinary Research & Media Platform for Narrative Medicine,
Psychological Discourse Analysis, Stigma, and Public Health Communication.

SCIENTIFIC & ETHICAL INTEGRITY STATEMENT:
- This application is a research and educational prototype.
- It is NOT a clinical diagnostic, psychiatric, or screening instrument.
- It does NOT compute fabricated psychometric scores, clinical risk scores,
  or synthetic p-values / inter-rater reliability coefficients.
- Lexical and thematic indicators reflect observed text characteristics in the
  supplied dataset and do not directly measure internal emotions, clinical
  diagnosis, acoustic silence, or individual mental health status.
- Always use only data for which you hold lawful authorization, ethical
  approval, and informed consent where applicable.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import warnings
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except Exception:
    scipy_stats = None
    HAS_SCIPY = False

try:
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
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# 1. APPLICATION & EDITORIAL DESIGN CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="The Neurometric Gaze — Beyond the Unheard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

EDITORIAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #F5F2EB;
    color: #14243A;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 2.0rem;
    padding-bottom: 4.0rem;
    max-width: 1280px;
}

h1, h2, h3 {
    font-family: 'Instrument Serif', Georgia, serif;
    color: #14243A;
    font-weight: 400;
    letter-spacing: -0.3px;
}

h1 {
    font-size: 3.15rem;
    line-height: 1.06;
    margin-bottom: 0.4rem;
}

h2 {
    font-size: 2.05rem;
    line-height: 1.15;
    margin-top: 0.35rem;
    margin-bottom: 0.65rem;
}

h3 {
    font-size: 1.45rem;
    line-height: 1.25;
}

p, li, label, .stMarkdown {
    color: #1E314B;
}

[data-testid="stSidebar"] {
    background-color: #ECE7DC;
    border-right: 1px solid rgba(20, 36, 58, 0.14);
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid rgba(20, 36, 58, 0.13);
    border-top: 3px solid #5C8581;
    padding: 15px 18px;
    border-radius: 10px;
}

div[data-testid="stMetric"] label {
    color: #485C76 !important;
    font-size: 0.82rem !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #14243A !important;
    font-weight: 600 !important;
}

.kicker {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.76rem;
    font-weight: 700;
    color: #5C8581;
    margin-bottom: 0.35rem;
}

.scholar-card {
    background-color: #FFFFFF;
    border: 1px solid rgba(20, 36, 58, 0.13);
    border-left: 4px solid #D6A56C;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.0rem;
}

.teal-card {
    background-color: #FFFFFF;
    border: 1px solid rgba(20, 36, 58, 0.13);
    border-left: 4px solid #5C8581;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.0rem;
}

.pull-quote {
    background-color: #EFECE4;
    border-left: 4px solid #14243A;
    padding: 0.95rem 1.25rem;
    border-radius: 0 8px 8px 0;
    font-style: italic;
    color: #14243A;
    margin: 0.75rem 0 1.0rem 0;
}

.badge {
    display: inline-block;
    background-color: rgba(92, 133, 129, 0.14);
    color: #14243A;
    border: 1px solid rgba(92, 133, 129, 0.35);
    border-radius: 999px;
    padding: 0.18rem 0.65rem;
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}

.amber-badge {
    display: inline-block;
    background-color: rgba(214, 165, 108, 0.22);
    color: #14243A;
    border: 1px solid rgba(214, 165, 108, 0.55);
    border-radius: 999px;
    padding: 0.18rem 0.65rem;
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-bottom: 0.35rem;
}
</style>
"""

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)

# =====================================================================
# 2. VERIFIED PSYCHOLOGICAL & COMMUNICATION FRAMEWORKS
# =====================================================================

PSYCHOLOGICAL_FRAMEWORKS: List[Dict[str, Any]] = [
    {
        "id": "narrative_medicine",
        "title": "Narrative Medicine & Illness Accounts",
        "theorists": "Rita Charon (2006); Arthur Kleinman (1988)",
        "core_premise": (
            "Distinguishes between disease (biomedical pathology) and illness "
            "(the lived human experience of suffering, meaning-making, and silence). "
            "Attentive reading of narratives examines what is voiced, what is "
            "hesitated over, and how patients situate themselves in care settings."
        ),
        "text_application": (
            "Tracks first-person positioning, hedging/uncertainty phrasing, and "
            "expressions of non-disclosure or restraint in narratives without "
            "treating text markers as clinical diagnoses."
        ),
        "limitation": (
            "Written transcripts lack prosody, facial cues, eye gaze, and "
            "interactional timing; narrative themes do not verify clinical pathology."
        ),
    },
    {
        "id": "self_determination",
        "title": "Self-Determination Theory (SDT)",
        "theorists": "Edward L. Deci & Richard M. Ryan (1985, 2000)",
        "core_premise": (
            "Posits three basic psychological needs—autonomy (volition and "
            "self-endorsement), competence (perceived capability and mastery), and "
            "relatedness (connection, belonging, and being understood)—that "
            "support psychological functioning and engagement."
        ),
        "text_application": (
            "Examines explicit language around choice/constraint (autonomy), "
            "coping/capability/helplessness (competence), and connection/isolation "
            "or being heard/dismissed (relatedness)."
        ),
        "limitation": (
            "Keyword/lexicon counts are descriptive indicators of topic mention, "
            "not validated psychometric scales of need satisfaction."
        ),
    },
    {
        "id": "stigma_help_seeking",
        "title": "Mental Health Stigma & Help-Seeking Frameworks",
        "theorists": "Patrick W. Corrigan (2004); Sara Clement et al. (2015); Graham Thornicroft",
        "core_premise": (
            "Differentiates public stigma (stereotypes, prejudice, and "
            "discrimination held by others) from self-stigma (internalization of "
            "devaluing beliefs such as shame or perceived weakness), both of which "
            "can impede help-seeking."
        ),
        "text_application": (
            "Identifies explicit mentions of judgment, shame, secrecy, fear of "
            "being labeled, and care-seeking or avoidance in narratives and "
            "media texts."
        ),
        "limitation": (
            "Attitudes and intentions reported in surveys or interviews do not "
            "guarantee future behavioral help-seeking."
        ),
    },
    {
        "id": "social_cognitive",
        "title": "Social Cognitive Theory & Media Modeling",
        "theorists": "Albert Bandura (1986, 2001)",
        "core_premise": (
            "Explains human adaptation through triadic reciprocal causation "
            "(personal factors, behavior, and environment), emphasizing perceived "
            "self-efficacy and symbolic modeling through mass and digital media."
        ),
        "text_application": (
            "Evaluates how media stories model realistic coping, agency, and "
            "support pathways versus fatalism, sensationalism, or avoidance."
        ),
        "limitation": (
            "Media exposure interacts with cultural, socioeconomic, and "
            "interpersonal contexts; media framing alone is not a sole cause."
        ),
    },
    {
        "id": "framing_dramaturgy",
        "title": "Media Framing & Dramaturgical Analysis",
        "theorists": "Robert M. Entman (1993); Erving Goffman (1959)",
        "core_premise": (
            "Framing selects and makes salient certain aspects of a perceived "
            "reality to define problems, diagnose causes, make moral judgments, "
            "and suggest remedies. Goffman's dramaturgy highlights front-stage "
            "presentation versus back-stage restraint."
        ),
        "text_application": (
            "Structures qualitative/quantitative coding of how public narratives "
            "and personal accounts manage impressions, disclosure, and concealment."
        ),
        "limitation": (
            "Requires transparent codebooks and independent human inter-rater "
            "agreement checks to avoid analyst confirmation bias."
        ),
    },
    {
        "id": "cognitive_affective_nlp",
        "title": "Affective, Cognitive & Function-Word Linguistics",
        "theorists": "James W. Pennebaker et al. (2003, 2015) — Linguistic Inquiry Tradition",
        "core_premise": (
            "Shows that pronoun usage, negation, tentativeness, and affective "
            "words correlate modestly in empirical studies with attentional focus, "
            "cognitive processing, and social orientation."
        ),
        "text_application": (
            "Computes transparent token proportions (first-person singular, "
            "negation/restraint terms, tentative/hedging markers, affective words) "
            "with explicit sample-size and non-diagnostic caveats."
        ),
        "limitation": (
            "Effect sizes in language-psychopathology research are typically small "
            "to moderate and vary across languages, genres, and samples."
        ),
    },
]

# Transparent, auditable lexical categories (English + Azerbaijani tokens)
# Used purely for descriptive text exploration—never for clinical diagnosis.
LEXICON_CATEGORIES: Dict[str, List[str]] = {
    "first_person_singular": [
        "i", "me", "my", "mine", "myself",
        "mən", "məni", "mənə", "mənim", "məndə", "məndən", "özüm", "özümü", "özümə"
    ],
    "restraint_and_non_disclosure": [
        "quiet", "silent", "silence", "hid", "hide", "hidden", "held", "hold",
        "withheld", "hesitate", "hesitated", "hesitation", "reluctant", "avoid",
        "avoided", "avoidance", "downplay", "brush", "unspoken", "secret",
        "susdum", "sükut", "gizlətdim", "gizli", "çəkindim", "deyəmədim", "demədim",
        "saxladım", "qaçındım", "tərəddüd", "daxilimdə"
    ],
    "stigma_and_judgment": [
        "stigma", "stigmatized", "shame", "ashamed", "embarrassed", "embarrassment",
        "weak", "weakness", "judge", "judged", "judgment", "label", "labeled",
        "taboo", "whispered", "mock", "dramatic",
        "damğa", "damğalanma", "utanmaq", "utandım", "ayıb", "zəif", "zəiflik",
        "qınaq", "qınamaq", "mühakimə", "tabu", "gülünc"
    ],
    "help_seeking_and_care": [
        "help", "counselor", "therapist", "therapy", "support", "talk", "shared",
        "openly", "listen", "listened", "care", "reach", "ask", "advice",
        "kömək", "dəstək", "psixoloq", "həkim", "danışmaq", "bölüşmək",
        "dinləmək", "müraciət", "qayğı"
    ],
    "tentative_and_hedging": [
        "maybe", "perhaps", "might", "seem", "seems", "seemed", "sort", "kind",
        "probably", "unsure", "uncertain", "sometimes", "often",
        "bəlkə", "ola", "bilər", "deyəsən", "elə", "gəlir", "bilmirəm",
        "əmin", "deyiləm", "bəzən", "tez-tez"
    ],
    "sdt_autonomy_competence_relatedness": [
        "choice", "choose", "forced", "pressure", "control", "capable", "able",
        "coping", "alone", "lonely", "connected", "belong", "understood", "heard",
        "seçim", "məcbur", "təzyiq", "nəzarət", "bacarmaq", "tək", "tənha",
        "bağlılıq", "anlaşılan", "eşidilən"
    ],
}

CURATED_STORIES: List[Dict[str, Any]] = [
    {
        "id": "case_01",
        "title": "The Quiet Waiting Room",
        "setting": "Urban clinical encounter · Anonymized composite narrative",
        "frameworks": ["Narrative Medicine", "Self-Determination Theory", "Stigma & Help-Seeking"],
        "quote": "I came in for persistent exhaustion and sleeplessness, yet when asked how I was coping, I said 'just busy with work' and waited for the prescription.",
        "account": (
            "A young adult describes entering a short medical appointment with "
            "ongoing fatigue and sleep disruption. When a broad opening question "
            "is asked, a brief 'just busy' response closes down further exploration "
            "of work strain and emotional overload. Non-disclosure here functions "
            "as a social self-protective move within time-constrained interaction."
        ),
        "media_cue": "Podcast Episode 02 segment (03:18) — 'Voices of Hesitation: Why young adults edit distress in brief encounters.'",
        "research_prompt": "How do consultation pacing, perceived authority, and stigma expectations shape whether a person elaborates on distress or gives a minimal answer?",
    },
    {
        "id": "case_02",
        "title": "Labels That Travel Faster Than Care",
        "setting": "University & family context · Azerbaijan narrative framing",
        "frameworks": ["Stigma & Help-Seeking", "Social Cognitive Theory", "Media Framing"],
        "quote": "In everyday conversation, seeking counseling is still whispered about as 'weakness' or 'a serious flaw,' even by people who care deeply.",
        "account": (
            "Students and young professionals often report tension between wanting "
            "emotional relief and fearing how peers, family, or media stereotypes "
            "interpret psychological help-seeking. Sensational or overly simplistic "
            "media depictions can reinforce fear of labeling rather than modeling "
            "constructive support."
        ),
        "media_cue": "Podcast Episode 01 segment (04:05) — 'Beyond the Label: Cultural scripts and stigma around mental health in Azerbaijan.'",
        "research_prompt": "Which media frames reduce self-stigma among young adults without trivializing genuine psychological distress?",
    },
    {
        "id": "case_03",
        "title": "Finding Words Without Clinical Jargon",
        "setting": "Peer & public storytelling · Help-seeking transition",
        "frameworks": ["Narrative Medicine", "Social Cognitive Theory"],
        "quote": "I didn't know how to explain what I was going through until I heard someone else describe burnout in plain, human language.",
        "account": (
            "Hearing relatable, non-stigmatizing accounts helps individuals name "
            "experiences such as chronic exhaustion, social withdrawal, or "
            "performance pressure in everyday terms, supporting earlier, more "
            "informed conversations with trusted people or professionals."
        ),
        "media_cue": "Podcast Episode 03 segment (02:54) — 'Opening the Door: Plain-language storytelling and pathways to support.'",
        "research_prompt": "How can short documentary films and podcasts pair lived stories with accurate, non-alarmist public health information?",
    },
]

EVIDENCE_CATALOG: List[Dict[str, str]] = [
    {
        "claim": "Mental health stigma (public and self-stigma) is consistently associated with reduced or delayed help-seeking in empirical and review literature.",
        "framework": "Stigma & Help-Seeking (Corrigan, 2004; Clement et al., 2015)",
        "evidence_type": "Systematic Review & Conceptual Framework",
        "boundary": "Association and barrier identification; context-dependent across cultures and age groups.",
    },
    {
        "claim": "Narrative medicine emphasizes attention to patients' stories, silences, and social context alongside biomedical assessment.",
        "framework": "Narrative Medicine (Charon, 2006; Kleinman, 1988)",
        "evidence_type": "Clinical Humanities & Theoretical Framework",
        "boundary": "Enhances interpretive understanding and communication; does not replace clinical evaluation.",
    },
    {
        "claim": "Basic psychological needs (autonomy, competence, relatedness) shape motivation, engagement, and well-being across domains.",
        "framework": "Self-Determination Theory (Deci & Ryan, 2000)",
        "evidence_type": "Empirical Psychological Theory",
        "boundary": "Lexicon inspection in text is descriptive only and is not a validated SDT psychometric scale.",
    },
    {
        "claim": "Media framing and symbolic modeling influence public understanding and perceived self-efficacy toward health behaviors.",
        "framework": "Framing (Entman, 1993) & Social Cognitive Theory (Bandura, 2001)",
        "evidence_type": "Communication & Psychological Theory",
        "boundary": "Media narratives shape perceptions and norms; audience reception varies and requires pre/post evaluation.",
    },
    {
        "claim": "Text transcripts alone cannot measure acoustic silence duration, vocal prosody, gaze, or clinical diagnosis.",
        "framework": "Methodological & Ethical Governance",
        "evidence_type": "Research Scope Constraint",
        "boundary": "Strict non-diagnostic boundary; independent human coding required for qualitative/thematic claims.",
    },
]

DEFAULT_RESEARCH_CORPUS = pd.DataFrame(
    [
        {
            "participant_id": "N01",
            "theme": "Silence & Restraint",
            "text": "I stayed quiet during the visit and held back what was bothering me because I did not want to sound weak.",
            "phq8_binary": 1,
            "phq8_score": 14,
            "rater_1": "restraint",
            "rater_2": "restraint",
        },
        {
            "participant_id": "N02",
            "theme": "Stigma & Judgment",
            "text": "People whisper and judge when someone mentions therapy, so there is a lot of shame and secrecy around stress.",
            "phq8_binary": 1,
            "phq8_score": 15,
            "rater_1": "stigma",
            "rater_2": "stigma",
        },
        {
            "participant_id": "N03",
            "theme": "Help-Seeking & Support",
            "text": "I talked openly with a counselor and got practical support that helped me choose healthier routines.",
            "phq8_binary": 0,
            "phq8_score": 4,
            "rater_1": "help_seeking",
            "rater_2": "help_seeking",
        },
        {
            "participant_id": "N04",
            "theme": "Silence & Restraint",
            "text": "Mən daxilimdəki narahatlığı gizlətdim, çəkindim və sadəcə yaxşıyam deyə susdum.",
            "phq8_binary": 1,
            "phq8_score": 13,
            "rater_1": "restraint",
            "rater_2": "restraint",
        },
        {
            "participant_id": "N05",
            "theme": "Help-Seeking & Support",
            "text": "I shared my concerns clearly, felt heard, and found useful guidance on managing pressure.",
            "phq8_binary": 0,
            "phq8_score": 3,
            "rater_1": "help_seeking",
            "rater_2": "help_seeking",
        },
        {
            "participant_id": "N06",
            "theme": "Stigma & Judgment",
            "text": "Some media stories frame struggles around taboo or shame rather than clear, supportive information.",
            "phq8_binary": 1,
            "phq8_score": 12,
            "rater_1": "stigma",
            "rater_2": "restraint",
        },
        {
            "participant_id": "N07",
            "theme": "Help-Seeking & Support",
            "text": "Bəlkə də əvvəl tərəddüd edirdim, amma dəstək almaq və danışmaq mənə aydınlıq gətirdi.",
            "phq8_binary": 0,
            "phq8_score": 5,
            "rater_1": "help_seeking",
            "rater_2": "help_seeking",
        },
        {
            "participant_id": "N08",
            "theme": "Silence & Restraint",
            "text": "I felt pressure and uncertain about whether anyone would listen, so I kept my worries to myself.",
            "phq8_binary": 1,
            "phq8_score": 16,
            "rater_1": "restraint",
            "rater_2": "restraint",
        },
    ]
)

# =====================================================================
# 3. DATA INGESTION, CLEANING & VALIDATION UTILITIES
# =====================================================================

def clean_text(value: Any) -> str:
    """Normalize whitespace and safely convert scalar/missing values to str."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_col_name(col: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")


def find_matching_column(columns: Any, candidates: List[str]) -> Optional[str]:
    col_map = {normalize_col_name(c): c for c in columns}
    for cand in candidates:
        key = normalize_col_name(cand)
        if key in col_map:
            return col_map[key]
    return None


def read_csv_safely(raw_bytes: bytes, filename: str = "dataset.csv") -> pd.DataFrame:
    errors: List[str] = []
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise ValueError(f"Unable to parse CSV '{filename}'. Details: {' | '.join(errors[-2:])}")


def normalize_uploaded_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize flexible CSV column names into a transparent research schema."""
    text_col = find_matching_column(
        df.columns,
        ["text", "transcript", "utterance", "narrative", "speech", "statement", "response"],
    )
    if text_col is None:
        raise ValueError(
            "No text/narrative column found. Include a column named "
            "'text', 'transcript', 'narrative', or 'utterance'."
        )

    id_col = find_matching_column(
        df.columns,
        ["participant_id", "doc_id", "id", "subject_id", "participant", "case_id"],
    )
    theme_col = find_matching_column(
        df.columns,
        ["theme", "category", "topic", "domain", "frame"],
    )
    bin_col = find_matching_column(
        df.columns,
        ["phq8_binary", "phq_8_binary", "distress_flag", "binary_label", "label"],
    )
    score_col = find_matching_column(
        df.columns,
        ["phq8_score", "phq_8_score", "phq8", "score", "scale_score"],
    )
    r1_col = find_matching_column(
        df.columns,
        ["rater_1", "coder_1", "annotator_1", "code_a"],
    )
    r2_col = find_matching_column(
        df.columns,
        ["rater_2", "coder_2", "annotator_2", "code_b"],
    )

    out = pd.DataFrame(index=df.index)
    if id_col is not None:
        raw_ids = df[id_col].map(clean_text)
        out["participant_id"] = [v if v else f"item_{i+1:03d}" for i, v in enumerate(raw_ids)]
    else:
        out["participant_id"] = [f"item_{i+1:03d}" for i in range(len(df))]

    if theme_col is not None:
        raw_themes = df[theme_col].map(clean_text)
        out["theme"] = [v if v else "Unspecified Theme" for v in raw_themes]
    else:
        out["theme"] = "Unspecified Theme"

    out["text"] = df[text_col].map(clean_text)
    out["phq8_binary"] = pd.to_numeric(df[bin_col], errors="coerce") if bin_col is not None else np.nan
    out["phq8_score"] = pd.to_numeric(df[score_col], errors="coerce") if score_col is not None else np.nan
    out["rater_1"] = df[r1_col].map(clean_text) if r1_col is not None else ""
    out["rater_2"] = df[r2_col].map(clean_text) if r2_col is not None else ""

    return out.reset_index(drop=True)


def parse_transcript_csv_bytes(raw_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Parse transcript CSVs (including DAIC-style speaker/value rows) safely."""
    sub = read_csv_safely(raw_bytes, filename)
    text_col = find_matching_column(sub.columns, ["value", "text", "utterance", "transcript", "narrative"])
    if text_col is None:
        return None

    speaker_col = find_matching_column(sub.columns, ["speaker", "role", "speaker_id"])
    if speaker_col is not None:
        speakers = sub[speaker_col].map(clean_text).str.lower()
        mask = speakers.str.contains(r"participant|patient|subject", regex=True, na=False)
        rows = sub.loc[mask, text_col] if mask.any() else sub[text_col]
    else:
        rows = sub[text_col]

    joined = " ".join(cleaned for item in rows if (cleaned := clean_text(item)))
    if not joined:
        return None

    pid_match = re.search(r"(\d+)", filename)
    pid = pid_match.group(1) if pid_match else filename
    return {
        "participant_id": str(pid),
        "theme": "Transcript Archive",
        "text": joined,
        "phq8_binary": np.nan,
        "phq8_score": np.nan,
        "rater_1": "",
        "rater_2": "",
    }


def parse_zip_archive(raw_bytes: bytes) -> Tuple[pd.DataFrame, int]:
    """Extract CSV transcripts and optional PHQ-8 metadata from a ZIP archive."""
    transcripts: List[Dict[str, Any]] = []
    skipped_files = 0
    meta_frames: List[pd.DataFrame] = []

    try:
        zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("Uploaded ZIP file is corrupted or invalid.") from exc

    with zf:
        for member in zf.namelist():
            if member.endswith("/") or "__MACOSX" in member:
                continue
            base = member.rsplit("/", 1)[-1]
            if not base.lower().endswith(".csv"):
                continue
            content = zf.read(member)
            if "transcript" in base.lower():
                parsed = parse_transcript_csv_bytes(content, base)
                if parsed is not None:
                    transcripts.append(parsed)
                else:
                    skipped_files += 1
            else:
                try:
                    meta_frames.append(read_csv_safely(content, base))
                except Exception:
                    skipped_files += 1

    if not transcripts:
        raise ValueError("No valid transcript CSV files were found inside the ZIP archive.")

    df = pd.DataFrame(transcripts)

    for mdf in meta_frames:
        id_col = find_matching_column(mdf.columns, ["participant_id", "participant", "subject_id", "id"])
        bin_col = find_matching_column(mdf.columns, ["phq8_binary", "phq_8_binary", "phq8binary"])
        if id_col is not None and bin_col is not None:
            meta = pd.DataFrame()
            raw_ids = mdf[id_col].map(clean_text)
            extracted = raw_ids.str.extract(r"(\d+)", expand=False)
            meta["participant_id"] = extracted.where(extracted.notna(), raw_ids)
            meta["phq8_binary_meta"] = pd.to_numeric(mdf[bin_col], errors="coerce")
            score_col = find_matching_column(mdf.columns, ["phq8_score", "phq_8_score", "phq8"])
            if score_col is not None:
                meta["phq8_score_meta"] = pd.to_numeric(mdf[score_col], errors="coerce")
            else:
                meta["phq8_score_meta"] = np.nan
            meta = meta.drop_duplicates(subset=["participant_id"], keep="first")
            df = df.merge(meta, on="participant_id", how="left")
            df["phq8_binary"] = df["phq8_binary_meta"].combine_first(df["phq8_binary"])
            df["phq8_score"] = df["phq8_score_meta"].combine_first(df["phq8_score"])
            df = df.drop(columns=["phq8_binary_meta", "phq8_score_meta"], errors="ignore")
            break

    return df, skipped_files


def aggregate_by_participant(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple rows per participant without fabricating missing values."""
    rows: List[Dict[str, Any]] = []
    for pid, grp in df.groupby("participant_id", dropna=False, sort=False):
        texts = [clean_text(x) for x in grp["text"]]
        joined_text = " ".join(t for t in texts if t)

        themes = [clean_text(x) for x in grp["theme"] if clean_text(x) and clean_text(x) != "Unspecified Theme"]
        theme_val = themes[0] if len(themes) == 1 else ("Multiple / Mixed" if len(themes) > 1 else "Unspecified Theme")

        bins = pd.to_numeric(grp["phq8_binary"], errors="coerce").dropna().unique()
        bin_val = float(bins[0]) if len(bins) == 1 else np.nan

        scores = pd.to_numeric(grp["phq8_score"], errors="coerce").dropna().unique()
        score_val = float(scores[0]) if len(scores) == 1 else np.nan

        r1_vals = [clean_text(x) for x in grp["rater_1"] if clean_text(x)]
        r2_vals = [clean_text(x) for x in grp["rater_2"] if clean_text(x)]

        rows.append(
            {
                "participant_id": str(pid),
                "theme": theme_val,
                "text": joined_text,
                "phq8_binary": bin_val,
                "phq8_score": score_val,
                "rater_1": r1_vals[0] if len(r1_vals) == 1 else "",
                "rater_2": r2_vals[0] if len(r2_vals) == 1 else "",
            }
        )

    out = pd.DataFrame(rows)
    out["token_count"] = out["text"].map(lambda t: len(re.findall(r"\b[\w'-]+\b", t)))
    out["sentence_count"] = out["text"].map(
        lambda t: max(1, len([s for s in re.split(r"[.!?]+", t) if s.strip()])) if t.strip() else 0
    )
    out["avg_sentence_length"] = np.where(
        out["sentence_count"] > 0,
        out["token_count"] / out["sentence_count"],
        0.0,
    )
    return out


# =====================================================================
# 4. TRANSPARENT LINGUISTIC & PSYCHOLOGICAL-DISCOURSE METRICS
# =====================================================================

def compute_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transparent, non-diagnostic lexical category proportions
    (matches per 100 tokens) for each record.
    """
    enriched = df.copy()
    cat_cols: List[str] = []

    for cat_name, keywords in LEXICON_CATEGORIES.items():
        kw_set = {k.lower() for k in keywords}
        col_rate = f"{cat_name}_per100"
        col_raw = f"{cat_name}_hits"
        cat_cols.append(col_rate)

        raw_hits: List[int] = []
        rates_per_100: List[float] = []

        for txt in enriched["text"].astype(str):
            tokens = [tok.lower() for tok in re.findall(r"\b[\w'-]+\b", txt)]
            n_tok = len(tokens)
            hits = sum(1 for tok in tokens if tok in kw_set)
            raw_hits.append(hits)
            rates_per_100.append(round((hits / n_tok) * 100.0, 2) if n_tok > 0 else 0.0)

        enriched[col_raw] = raw_hits
        enriched[col_rate] = rates_per_100

    enriched.attrs["rate_columns"] = cat_cols
    return enriched


def compute_cohens_kappa(r1: pd.Series, r2: pd.Series) -> Dict[str, Any]:
    """
    Compute real Cohen's kappa ONLY when two non-empty rater columns exist.
    Never fabricates or hardcodes an agreement score.
    """
    pair = pd.DataFrame({"r1": r1.map(clean_text), "r2": r2.map(clean_text)})
    pair = pair[(pair["r1"] != "") & (pair["r2"] != "")]
    n = len(pair)
    if n < 2:
        return {
            "available": False,
            "n_paired": n,
            "kappa": None,
            "observed_agreement": None,
            "note": "At least 2 paired human annotations (rater_1 and rater_2) are required to compute Cohen's kappa.",
        }

    categories = sorted(set(pair["r1"]).union(set(pair["r2"])))
    if len(categories) < 2:
        po = float((pair["r1"] == pair["r2"]).mean())
        return {
            "available": True,
            "n_paired": n,
            "kappa": None,
            "observed_agreement": round(po, 4),
            "note": "Only one unique label appears across both raters; chance agreement prevents standard kappa division.",
        }

    po = float((pair["r1"] == pair["r2"]).mean())
    pe = 0.0
    for c in categories:
        p1 = float((pair["r1"] == c).mean())
        p2 = float((pair["r2"] == c).mean())
        pe += p1 * p2

    denom = 1.0 - pe
    kappa = float((po - pe) / denom) if abs(denom) > 1e-12 else None

    return {
        "available": True,
        "n_paired": n,
        "kappa": round(kappa, 4) if kappa is not None else None,
        "observed_agreement": round(po, 4),
        "expected_agreement": round(pe, 4),
        "note": "Computed directly from supplied rater_1 and rater_2 columns.",
    }


def compute_exploratory_associations(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute genuine non-parametric/parametric statistics only when data supports them."""
    out: Dict[str, Any] = {}

    score_sub = df[["token_count", "phq8_score"]].dropna()
    if len(score_sub) >= 3 and score_sub["token_count"].nunique() > 1 and score_sub["phq8_score"].nunique() > 1:
        if HAS_SCIPY and scipy_stats is not None:
            rho, pval = scipy_stats.spearmanr(score_sub["token_count"], score_sub["phq8_score"])
            out["spearman"] = {
                "n": len(score_sub),
                "rho": round(float(rho), 4),
                "p_value": round(float(pval), 4),
            }
        else:
            rho = float(score_sub["token_count"].corr(score_sub["phq8_score"], method="spearman"))
            out["spearman"] = {
                "n": len(score_sub),
                "rho": round(rho, 4),
                "p_value": None,
            }
    else:
        out["spearman"] = None

    bin_sub = df[df["phq8_binary"].isin([0, 1])].copy()
    g0 = bin_sub.loc[bin_sub["phq8_binary"] == 0, "token_count"]
    g1 = bin_sub.loc[bin_sub["phq8_binary"] == 1, "token_count"]
    if len(g0) >= 2 and len(g1) >= 2 and HAS_SCIPY and scipy_stats is not None:
        u_stat, u_pval = scipy_stats.mannwhitneyu(g0, g1, alternative="two-sided")
        out["mann_whitney_tokens"] = {
            "n_0": len(g0),
            "n_1": len(g1),
            "u_stat": round(float(u_stat), 2),
            "p_value": round(float(u_pval), 4),
        }
    else:
        out["mann_whitney_tokens"] = None

    return out


# =====================================================================
# 5. OPTIONAL SECONDARY BASELINE NLP EVALUATION (EXPLORATORY ONLY)
# =====================================================================

def run_exploratory_tfidf_cv(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Secondary exploratory TF-IDF + LogisticRegression out-of-fold evaluation
    when two binary classes (0/1) each have >= 2 records.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is not installed in the current environment.")

    sub = df[df["phq8_binary"].isin([0, 1]) & df["text"].str.len().gt(0)].copy()
    X = sub["text"].astype(str).to_numpy()
    y = sub["phq8_binary"].astype(int).to_numpy()

    counts = pd.Series(y).value_counts()
    if len(counts) != 2 or int(counts.min()) < 2:
        raise ValueError(
            "Exploratory cross-validation requires both binary classes (0 and 1) "
            "with at least 2 records per class."
        )

    n_splits = min(5, int(counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    vec_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=8000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    vec_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=8000,
        sublinear_tf=True,
    )
    pipe = Pipeline(
        [
            ("feats", FeatureUnion([("word", vec_word), ("char", vec_char)])),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=42)),
        ]
    )

    probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)

    dummy = DummyClassifier(strategy="prior")
    dummy_preds = cross_val_predict(dummy, X.reshape(-1, 1), y, cv=cv, method="predict")

    roc_val = float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else float("nan")
    pr_val = float(average_precision_score(y, probs))

    return {
        "n_records": int(len(y)),
        "folds": int(n_splits),
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "roc_auc": roc_val,
        "pr_auc": pr_val,
        "brier_score": float(brier_score_loss(y, probs)),
        "baseline_accuracy": float(accuracy_score(y, dummy_preds)),
        "baseline_macro_f1": float(f1_score(y, dummy_preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, preds, labels=[0, 1]),
        "predictions": pd.DataFrame(
            {
                "participant_id": sub["participant_id"].values,
                "actual_label": y,
                "predicted_probability": np.round(probs, 4),
                "predicted_label": preds,
            }
        ),
    }


# =====================================================================
# 6. SIDEBAR NAVIGATION (8 SECTIONS)
# =====================================================================

st.sidebar.markdown('<div class="kicker">THE NEUROMETRIC GAZE</div>', unsafe_allow_html=True)
st.sidebar.markdown("### Beyond the Unheard")
st.sidebar.caption("Narrative Medicine · Psychological Discourse · Media Intervention")

page = st.sidebar.radio(
    "Section",
    [
        "01 · Home",
        "02 · The Problem",
        "03 · Human Stories",
        "04 · Research Lab",
        "05 · Evidence Library",
        "06 · The Intervention",
        "07 · Impact",
        "08 · About & Ethics",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Research & media prototype aligned with Harvard MS-MMH themes "
    "(storytelling, social medicine, health communication, and public impact)."
)

# =====================================================================
# 7. SECTION 01 — HOME
# =====================================================================

if page == "01 · Home":
    st.markdown('<div class="kicker">STORIES · EVIDENCE · UNDERSTANDING · ACTION</div>', unsafe_allow_html=True)
    st.title("Beyond the Unheard")
    st.markdown(
        "### Understanding mental health narratives, stigma, and help-seeking "
        "through grounded research and human-centered media"
    )
    st.markdown(
        "Mental health is shaped not only by clinical symptoms, but also by cultural "
        "scripts, fear of judgment, unspoken silences, and how public media frames "
        "psychological distress. This platform integrates **human stories**, "
        "**transparent research analysis**, and **media intervention** with an "
        "emphasis on young adults in Azerbaijan."
    )

    st.info(
        "**Research Question:** How do media narratives shape young adults' "
        "understanding of mental health stigma and help-seeking in Azerbaijan?"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="scholar-card">
                <div class="kicker">PILLAR 01 · HUMAN STORIES</div>
                <h4>Lived Experiences & Silence</h4>
                <p>Anonymized narratives examining restraint, stigma, and how people find words to describe distress and seek care.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="teal-card">
                <div class="kicker">PILLAR 02 · RESEARCH & EVIDENCE</div>
                <h4>Theory-Grounded Inquiry</h4>
                <p>Connects Narrative Medicine, Self-Determination Theory, Stigma frameworks, and transparent text metrics without fabricated scores.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="scholar-card">
                <div class="kicker">PILLAR 03 · MEDIA INTERVENTION</div>
                <h4>Documentary, Podcast & Action</h4>
                <p>Translates findings into a short documentary, a 3-episode podcast series, awareness assets, and pre/post audience evaluation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Project Architecture & 10-Week Execution Roadmap")
    roadmap_df = pd.DataFrame(
        [
            {"Weeks": "Week 1–2", "Phase": "Research Question & Literature", "Deliverable": "Refined Azerbaijan mental health narrative & stigma scope; literature synthesis"},
            {"Weeks": "Week 3–4", "Phase": "Ethics, Sampling & Corpus", "Deliverable": "Consent/anonymization protocol; media/interview/survey dataset & coding rubric"},
            {"Weeks": "Week 5–6", "Phase": "Analysis & Narrative Arc", "Deliverable": "Thematic findings, transparent text metrics, and documentary/podcast structure"},
            {"Weeks": "Week 7–8", "Phase": "Media & Platform Production", "Deliverable": "Interactive platform, 5–8 min documentary cut, 3-part podcast & campaign assets"},
            {"Weeks": "Week 9–10", "Phase": "Pilot Impact & Portfolio", "Deliverable": "Pre/post audience test, limitations audit, and Harvard MS-MMH portfolio package"},
        ]
    )
    st.dataframe(roadmap_df, use_container_width=True, hide_index=True)

# =====================================================================
# 8. SECTION 02 — THE PROBLEM
# =====================================================================

elif page == "02 · The Problem":
    st.markdown('<div class="kicker">SOCIAL MEDICINE & COMMUNICATION BARRIERS</div>', unsafe_allow_html=True)
    st.header("The Problem: Silence, Stigma, and Narrative Gaps")
    st.markdown(
        "In many social contexts, discussions of psychological distress remain "
        "constrained by stigma, fear of social judgment, and narrow media tropes. "
        "When public discourse treats emotional strain as a taboo or relies on "
        "sensationalism, young adults may delay asking for help or minimize "
        "symptoms in everyday and clinical conversations."
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            """
            <div class="scholar-card">
                <div class="amber-badge">Stigma & Social Risk</div>
                <h4>Why People Stay Quiet</h4>
                <p>Fear of being judged, misunderstood by family or peers, or labeled 'weak' can lead individuals to conceal distress and appear 'fine' on the surface.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            """
            <div class="teal-card">
                <div class="badge">Media & Public Framing</div>
                <h4>How Narratives Shape Norms</h4>
                <p>Media coverage and social norms influence whether mental health is approached with empathy and practical support pathways or stigma and avoidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Scope Boundaries")
    st.warning(
        "This platform studies narratives, stigma framing, and public health "
        "communication. It does not diagnose conditions, infer private mental "
        "states from text alone, or replace licensed clinical care."
    )

# =====================================================================
# 9. SECTION 03 — HUMAN STORIES
# =====================================================================

elif page == "03 · Human Stories":
    st.markdown('<div class="kicker">INTERACTIVE STORYTELLING</div>', unsafe_allow_html=True)
    st.header("Human Stories: Voices and Silences")
    st.markdown(
        "Explore anonymized composite narratives illustrating how restraint, "
        "social stigma, and supportive communication appear in lived accounts."
    )

    titles = [item["title"] for item in CURATED_STORIES]
    chosen = st.selectbox("Select a narrative study", titles)
    selected_story = next(item for item in CURATED_STORIES if item["title"] == chosen)

    badges_html = "".join(
        f'<span class="badge">{fw}</span>' for fw in selected_story["frameworks"]
    )
    st.markdown(
        f"""
        <div class="scholar-card">
            <div class="kicker">{selected_story["setting"]}</div>
            <h3>{selected_story["title"]}</h3>
            <div>{badges_html}</div>
            <div class="pull-quote">“{selected_story["quote"]}”</div>
            <p>{selected_story["account"]}</p>
            <p><strong>Media Cue:</strong> {selected_story["media_cue"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Analytical Reflection Question")
    st.write(selected_story["research_prompt"])

# =====================================================================
# 10. SECTION 04 — RESEARCH LAB (DEEP, TRANSPARENT, NON-FABRICATED)
# =====================================================================

elif page == "04 · Research Lab":
    st.markdown('<div class="kicker">EMPIRICAL TEXT & PSYCHOLOGICAL-DISCOURSE LAB</div>', unsafe_allow_html=True)
    st.header("Research Lab: Transparent Narrative & Text Analysis")
    st.markdown(
        "Analyze narrative records using transparent structural and lexical metrics "
        "grounded in psychological and communication frameworks. All numbers are "
        "computed directly from the active dataset—no synthetic scores or "
        "unverified reliability values are inserted."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Research Lab Data Controls")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or ZIP archive",
        type=["csv", "zip"],
        help="Upload a CSV with a text/narrative column or a ZIP containing transcript CSVs.",
    )
    use_default_sample = st.sidebar.checkbox(
        "Load built-in illustrative research corpus",
        value=(uploaded_file is None),
        help="Uses 8 illustrative bilingual (EN/AZ) narrative rows when no file is uploaded.",
    )
    min_tokens = st.sidebar.slider(
        "Minimum token count filter",
        min_value=0,
        max_value=300,
        value=1,
        step=1,
    )

    if uploaded_file is None and not use_default_sample:
        st.info("Upload a CSV/ZIP file or enable the built-in illustrative corpus in the sidebar.")
        st.stop()

    try:
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            source_hash = hashlib.sha256(raw_bytes).hexdigest()[:12]
            source_label = f"Uploaded: {uploaded_file.name} ({source_hash})"
            if uploaded_file.name.lower().endswith(".zip"):
                raw_df, skipped_zip = parse_zip_archive(raw_bytes)
                if skipped_zip > 0:
                    st.caption(f"Note: {skipped_zip} non-matching or empty CSV file(s) inside ZIP were skipped.")
            else:
                csv_df = read_csv_safely(raw_bytes, uploaded_file.name)
                raw_df = normalize_uploaded_csv(csv_df)
        else:
            raw_df = DEFAULT_RESEARCH_CORPUS.copy()
            source_label = "Built-in Illustrative Research Corpus (8 rows, EN/AZ)"

        agg_df = aggregate_by_participant(raw_df)
        filtered_df = agg_df[
            agg_df["text"].str.len().gt(0) & agg_df["token_count"].ge(min_tokens)
        ].reset_index(drop=True)

        if filtered_df.empty:
            st.warning("No records match the current text and minimum token filter.")
            st.stop()

        analyzed_df = compute_lexical_features(filtered_df)
        rate_cols: List[str] = analyzed_df.attrs.get("rate_columns", [])

        st.caption(f"**Active Source:** {source_label} · **Eligible Records:** {len(analyzed_df)}")

        # --- Sub-panel A: Structural & Lexical Overview ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Records Analyzed", f"{len(analyzed_df)}")
        m2.metric("Mean Tokens / Record", f"{analyzed_df['token_count'].mean():.1f}")
        m3.metric("Median Tokens / Record", f"{analyzed_df['token_count'].median():.1f}")
        m4.metric("Mean Sentence Length", f"{analyzed_df['avg_sentence_length'].mean():.1f} tok")

        st.markdown("#### Lexical Category Rates (Occurrences per 100 Tokens)")
        st.caption(
            "Descriptive word-list rates only. High or low rates do not diagnose "
            "psychological states or prove causal mechanisms."
        )
        if rate_cols:
            mean_rates = analyzed_df[rate_cols].mean().sort_values(ascending=False)
            st.bar_chart(mean_rates, use_container_width=True)

        # --- Sub-panel B: Inter-Rater Agreement (Only if rater_1 & rater_2 exist) ---
        st.markdown("#### Human Coding & Inter-Rater Reliability Check")
        kappa_res = compute_cohens_kappa(analyzed_df["rater_1"], analyzed_df["rater_2"])
        if kappa_res["available"]:
            k1, k2, k3 = st.columns(3)
            k1.metric("Paired Coded Items", f"{kappa_res['n_paired']}")
            k2.metric("Observed Agreement", f"{kappa_res['observed_agreement']:.4f}")
            k3.metric(
                "Cohen's Kappa (κ)",
                f"{kappa_res['kappa']:.4f}" if kappa_res["kappa"] is not None else "N/A",
            )
            st.caption(kappa_res["note"])
        else:
            st.info(kappa_res["note"])

        # --- Sub-panel C: Exploratory Non-Clinical Correlations (If present) ---
        assoc_res = compute_exploratory_associations(analyzed_df)
        with st.expander("Exploratory Quantitative Associations (If Score/Binary Columns Supplied)"):
            if assoc_res["spearman"] is not None:
                sp = assoc_res["spearman"]
                p_txt = f", p = {sp['p_value']}" if sp["p_value"] is not None else ""
                st.write(f"- **Spearman correlation (token count vs. numeric score, n={sp['n']}):** r_s = {sp['rho']}{p_txt}")
            else:
                st.write("- **Spearman correlation:** Not computed (requires at least 3 non-missing numeric score rows with variance).")

            if assoc_res["mann_whitney_tokens"] is not None:
                mw = assoc_res["mann_whitney_tokens"]
                st.write(
                    f"- **Mann–Whitney U test on token count (group 0 n={mw['n_0']} vs. group 1 n={mw['n_1']}):** "
                    f"U = {mw['u_stat']}, p = {mw['p_value']} (two-sided, descriptive only)."
                )
            else:
                st.write("- **Mann–Whitney U test:** Not computed (requires at least 2 records in both binary groups 0 and 1).")

        # --- Sub-panel D: Optional Secondary TF-IDF Cross-Validation ---
        with st.expander("Secondary Methodological Check: Out-of-Fold TF-IDF + Logistic Regression"):
            st.caption(
                "Exploratory text-classification benchmark only. Internal cross-validation "
                "does not establish external validity or clinical utility."
            )
            if st.button("Run Out-of-Fold Cross-Validation", type="primary"):
                try:
                    cv_res = run_exploratory_tfidf_cv(analyzed_df)
                    st.session_state["cv_res"] = cv_res
                except Exception as cv_err:
                    st.session_state.pop("cv_res", None)
                    st.error(str(cv_err))

            if "cv_res" in st.session_state:
                r = st.session_state["cv_res"]
                c_a, c_b, c_c, c_d = st.columns(4)
                c_a.metric("Balanced Accuracy", f"{r['balanced_accuracy']:.4f}")
                c_b.metric("Macro F1", f"{r['macro_f1']:.4f}")
                c_c.metric("ROC AUC", f"{r['roc_auc']:.4f}" if not np.isnan(r["roc_auc"]) else "N/A")
                c_d.metric("Brier Score", f"{r['brier_score']:.4f}")
                st.dataframe(r["predictions"], use_container_width=True, hide_index=True)

        # --- Sub-panel E: Detailed Record Table & Export ---
        st.markdown("#### Record-Level Analytical Table")
        display_cols = [
            "participant_id",
            "theme",
            "token_count",
            "avg_sentence_length",
            "first_person_singular_per100",
            "restraint_and_non_disclosure_per100",
            "stigma_and_judgment_per100",
            "help_seeking_and_care_per100",
            "tentative_and_hedging_per100",
            "sdt_autonomy_competence_relatedness_per100",
            "text",
        ]
        present_cols = [c for c in display_cols if c in analyzed_df.columns]
        st.dataframe(analyzed_df[present_cols], use_container_width=True, hide_index=True)

        export_payload = {
            "project": "The Neurometric Gaze — Beyond the Unheard",
            "source": source_label,
            "records_analyzed": int(len(analyzed_df)),
            "mean_token_count": round(float(analyzed_df["token_count"].mean()), 2),
            "lexical_rates_per_100_tokens_mean": {
                col: round(float(analyzed_df[col].mean()), 3) for col in rate_cols
            },
            "cohens_kappa": kappa_res,
            "non_diagnostic_caveat": (
                "Descriptive lexical/structural metrics only; not a clinical "
                "diagnosis or direct measure of internal emotions."
            ),
        }

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download Analyzed Table (CSV)",
                analyzed_df[present_cols].to_csv(index=False).encode("utf-8"),
                file_name="neurometric_gaze_analysis.csv",
                mime="text/csv",
            )
        with d2:
            st.download_button(
                "Download Research Summary (JSON)",
                json.dumps(export_payload, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name="neurometric_gaze_summary.json",
                mime="application/json",
            )

    except Exception as exc:
        st.error(f"Dataset processing error: {exc}")

# =====================================================================
# 11. SECTION 05 — EVIDENCE LIBRARY & PSYCHOLOGICAL FRAMEWORKS
# =====================================================================

elif page == "05 · Evidence Library":
    st.markdown('<div class="kicker">THEORETICAL & EMPIRICAL FOUNDATIONS</div>', unsafe_allow_html=True)
    st.header("Evidence Library & Psychological Frameworks")
    st.markdown(
        "Explore established frameworks from psychology, narrative medicine, "
        "and health communication alongside their explicit analytical boundaries."
    )

    tab_fw, tab_ev = st.tabs(["Psychological & Communication Theories", "Verified Claim & Boundary Ledger"])

    with tab_fw:
        fw_names = [f["title"] for f in PSYCHOLOGICAL_FRAMEWORKS]
        chosen_fw_name = st.selectbox("Inspect framework", fw_names)
        fw = next(item for item in PSYCHOLOGICAL_FRAMEWORKS if item["title"] == chosen_fw_name)

        st.markdown(
            f"""
            <div class="teal-card">
                <div class="kicker">{fw["theorists"]}</div>
                <h3>{fw["title"]}</h3>
                <p><strong>Core Premise:</strong> {fw["core_premise"]}</p>
                <p><strong>Application in This Prototype:</strong> {fw["text_application"]}</p>
                <p><strong>Methodological Boundary:</strong> {fw["limitation"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_ev:
        domains = ["All"] + sorted({e["evidence_type"] for e in EVIDENCE_CATALOG})
        sel_dom = st.selectbox("Filter by evidence type", domains)
        items = EVIDENCE_CATALOG if sel_dom == "All" else [e for e in EVIDENCE_CATALOG if e["evidence_type"] == sel_dom]

        for ev in items:
            st.markdown(
                f"""
                <div class="scholar-card">
                    <span class="badge">{ev["evidence_type"]}</span>
                    <span class="amber-badge">{ev["framework"]}</span>
                    <h4>{ev["claim"]}</h4>
                    <p><strong>Analytical Boundary:</strong> {ev["boundary"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =====================================================================
# 12. SECTION 06 — THE INTERVENTION (MEDIA & COMMUNICATION)
# =====================================================================

elif page == "06 · The Intervention":
    st.markdown('<div class="kicker">PUBLIC HEALTH & MEDIA INTERVENTION</div>', unsafe_allow_html=True)
    st.header("The Intervention: Documentary, Podcast & Campaign")
    st.markdown(
        "Designed for **Harvard MS-MMH** alignment: combining empirical and "
        "narrative inquiry with concrete media artifacts aimed at reducing stigma "
        "and supporting informed help-seeking."
    )

    t1, t2, t3 = st.tabs(["Short Documentary (5–8 min)", "3-Episode Podcast Series", "Awareness Campaign Planner"])

    with t1:
        st.markdown("### Short Documentary Proposal: *Beyond the Unheard*")
        st.markdown(
            """
            - **Target Length:** 6–7 minutes.
            - **Structure:**
              1. *Opening:* Everyday quiet and unspoken strain around stress and expectations.
              2. *Context:* How stigma and media framing shape whether young people speak up.
              3. *Perspective:* Lived voices, domain context, and constructive pathways toward support.
            - **Ethical Safeguard:** Informed consent, anonymized identities where needed, and non-sensational framing.
            """
        )

    with t2:
        st.markdown("### 3-Part Podcast Blueprint")
        pod_df = pd.DataFrame(
            [
                {"Episode": "Episode 01", "Focus": "Stigma & Media Framing", "Core Idea": "How cultural scripts and media stories shape perceived shame or openness."},
                {"Episode": "Episode 02", "Focus": "Silence & Non-Disclosure", "Core Idea": "Why people hold back in conversations and how attentive listening matters."},
                {"Episode": "Episode 03", "Focus": "Help-Seeking & Support", "Core Idea": "Clear, stigma-aware steps for reaching out and supporting peers."},
            ]
        )
        st.dataframe(pod_df, use_container_width=True, hide_index=True)

    with t3:
        st.markdown("### Interactive Campaign Brief Generator")
        pillar = st.selectbox(
            "Select communication pillar",
            [
                "Normalize Early Help-Seeking",
                "Counter Stigma Without Sensationalism",
                "Encourage Attentive Listening",
            ],
        )
        audience = st.selectbox(
            "Primary audience segment",
            ["University Students & Young Adults", "Educators & Campus Communities", "General Public Readers"],
        )
        st.markdown(
            f"""
            <div class="teal-card">
                <div class="kicker">CAMPAIGN BRIEF · {audience.upper()}</div>
                <h4>Pillar: {pillar}</h4>
                <p><strong>Key Message:</strong> Seeking guidance or talking openly about distress is a constructive human response, not a personal failing.</p>
                <p><strong>Planned Assets:</strong> 1 campaign key visual, 6 short-form social media stories/reels, and a clear local support resource list.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =====================================================================
# 13. SECTION 07 — IMPACT (PRE / POST AUDIENCE EVALUATION)
# =====================================================================

elif page == "07 · Impact":
    st.markdown('<div class="kicker">AUDIENCE EVALUATION</div>', unsafe_allow_html=True)
    st.header("Impact Measurement: Pre- and Post-Engagement Check")
    st.markdown(
        "Use this module to record **actual pre- and post-exposure survey or "
        "pilot responses** from your audience. No synthetic audience scores are generated."
    )

    c_pre, c_post = st.columns(2)
    with c_pre:
        st.markdown("#### Pre-Exposure Mean / Selected Rating (1–7)")
        pre_stigma = st.slider("Perceived stigma around talking about mental health (1 = Low, 7 = High)", 1, 7, 5, key="pre_s")
        pre_comfort = st.slider("Comfort discussing help-seeking options (1 = Low, 7 = High)", 1, 7, 3, key="pre_c")

    with c_post:
        st.markdown("#### Post-Exposure Mean / Selected Rating (1–7)")
        post_stigma = st.slider("Perceived stigma after media exposure (1 = Low, 7 = High)", 1, 7, 3, key="post_s")
        post_comfort = st.slider("Comfort discussing help-seeking options after exposure (1 = Low, 7 = High)", 1, 7, 5, key="post_c")

    stigma_diff = post_stigma - pre_stigma
    comfort_diff = post_comfort - pre_comfort

    m1, m2 = st.columns(2)
    m1.metric("Perceived Stigma Delta (Post − Pre)", f"{stigma_diff:+d}", "Negative indicates lower rating")
    m2.metric("Support Comfort Delta (Post − Pre)", f"{comfort_diff:+d}", "Positive indicates higher rating")

    chart_df = pd.DataFrame(
        {"Pre": [pre_stigma, pre_comfort], "Post": [post_stigma, post_comfort]},
        index=["Perceived Stigma", "Support Comfort"],
    )
    st.bar_chart(chart_df, use_container_width=True)

    st.caption(
        "For formal evaluation, collect paired pre/post responses across a defined "
        "pilot sample and report sample size (n), dispersion, and potential selection bias."
    )

# =====================================================================
# 14. SECTION 08 — ABOUT, ETHICS & METHODOLOGICAL GUARDRAILS
# =====================================================================

elif page == "08 · About & Ethics":
    st.markdown('<div class="kicker">ETHICS, GOVERNANCE & LIMITATIONS</div>', unsafe_allow_html=True)
    st.header("About, Ethics & Methodological Guardrails")
    st.markdown(
        """
        - **Non-Clinical Boundary:** This prototype is strictly for research, education, and media communication. It does not diagnose, screen, or assign individual clinical risk.
        - **No Fabricated Metrics:** Every statistic shown in the Research Lab is computed directly from supplied data or explicitly marked as not available when required columns/samples are absent.
        - **Informed Consent & Privacy:** Any interviews, surveys, or community testing require clear informed consent, data minimization, and de-identification/anonymization prior to analysis or sharing.
        - **Support Resources:** If you or someone you know needs support, please consult a qualified mental health professional or local support service.
        """
    )
