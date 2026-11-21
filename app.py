from __future__ import annotations

import io
import json
import re
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# PAGE CONFIGURATION & EDITORIAL DESIGN SYSTEM
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="The Neurometric Gaze — Beyond the Unheard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #F5F2EB;
        color: #14243A;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }
    h1, h2, h3 {
        font-family: 'Instrument Serif', Georgia, serif;
        color: #14243A;
        letter-spacing: -0.3px;
    }
    h1 {
        font-size: 3.1rem;
        line-height: 1.08;
        font-weight: 400;
    }
    h2 {
        font-size: 2.05rem;
        line-height: 1.15;
        margin-top: 0.4rem;
    }
    p, li, label, .stCaption {
        color: #1F324D;
    }
    [data-testid="stSidebar"] {
        background-color: #ECE7DC;
        border-right: 1px solid rgba(20, 36, 58, 0.12);
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid rgba(20, 36, 58, 0.12);
        border-top: 3px solid #5C8581;
        padding: 16px 18px;
        border-radius: 10px;
    }
    div[data-testid="stMetric"] label {
        color: #4A5E78 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #14243A !important;
        font-weight: 600;
    }
    .editorial-kicker {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.78rem;
        font-weight: 600;
        color: #5C8581;
        margin-bottom: 0.4rem;
    }
    .editorial-card {
        background-color: #FFFFFF;
        border: 1px solid rgba(20, 36, 58, 0.12);
        border-left: 4px solid #D6A56C;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }
    .quote-box {
        background-color: #EFECE4;
        border-left: 4px solid #14243A;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        color: #14243A;
        margin: 0.8rem 0;
    }
    .pill-tag {
        display: inline-block;
        background-color: rgba(92, 133, 129, 0.14);
        color: #14243A;
        border: 1px solid rgba(92, 133, 129, 0.35);
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# CURATED RESEARCH & MEDIA DATA (ILLUSTRATIVE / RESEARCH STRUCTURE)
# ---------------------------------------------------------------------

STORIES = [
    {
        "id": "story_01",
        "title": "The Quiet Waiting Room",
        "context": "Young adult · Urban clinic visit · Anonymized composite narrative",
        "theme": "Silence & Stigma",
        "silence_index": 78,
        "quote": "I came in for persistent exhaustion and sleeplessness, yet when asked how I was coping, I said 'just busy with work' and waited for the prescription.",
        "narrative": (
            "During a ten-minute consultation, clinical questions focused quickly on physical checklists. "
            "Emotional strain and social pressure to appear strong remained unspoken. In narrative terms, "
            "silence functioned not as an absence of feeling, but as a protective strategy against judgment."
        ),
        "audio_note": "Podcast Episode 02 excerpt (03:18) — Voices of Hesitation: Why young adults edit their distress before speaking.",
        "reflection": "What shifts in a clinical or public conversation when silence is treated as a signal worth asking about rather than a blank space?",
    },
    {
        "id": "story_02",
        "title": "Labels That Travel Faster Than Care",
        "context": "University student · Family & media framing · Anonymized narrative",
        "theme": "Social Stigma",
        "silence_index": 66,
        "quote": "In everyday conversation, seeking counseling is still whispered about as 'weakness' or 'a serious flaw,' even by people who care deeply.",
        "narrative": (
            "Media headlines and casual remarks often swing between sensationalism and dismissive clichés. "
            "Young people navigating academic and family expectations describe hesitating to name anxiety or burnout "
            "out of concern for reputation and misunderstanding."
        ),
        "audio_note": "Podcast Episode 01 excerpt (04:05) — Beyond the Label: How cultural narratives shape stigma in Azerbaijan.",
        "reflection": "How do everyday metaphors around mental health shape whether someone reaches out early or waits until distress becomes acute?",
    },
    {
        "id": "story_03",
        "title": "Finding Words Without Clinical Jargon",
        "context": "Early-career professional · Help-seeking journey · Anonymized narrative",
        "theme": "Help-Seeking",
        "silence_index": 44,
        "quote": "I didn't know how to explain what I was going through until I heard someone else describe burnout in plain, human language.",
        "narrative": (
            "Accessible storytelling and relatable lived accounts helped bridge the gap between vague discomfort "
            "and constructive help-seeking. Clear, non-stigmatizing information made asking for support feel "
            "ordinary rather than intimidating."
        ),
        "audio_note": "Podcast Episode 03 excerpt (02:54) — Opening the Door: Practical, human pathways to care and support.",
        "reflection": "Which formats—short documentary stories, peer conversations, or clear guides—best lower barriers to help-seeking?",
    },
]

EVIDENCE_ITEMS = [
    {
        "claim": "Self-stigma and fear of social judgment are major barriers to mental health help-seeking among young adults.",
        "domain": "Stigma & Help-Seeking",
        "source": "Clement et al. (2015) / Thornicroft et al. — Systematic review literature on mental health-related stigma",
        "evidence_level": "Systematic Review / Conceptual Evidence",
        "status": "Verified Research Theme",
    },
    {
        "claim": "Attention to patients' stories, silences, and lived context strengthens clinical understanding and empathic engagement.",
        "domain": "Narrative Medicine",
        "source": "Charon, R. (2006) — Narrative Medicine: Honoring the Stories of Illness",
        "evidence_level": "Foundational Clinical Humanities Framework",
        "status": "Core Theoretical Anchor",
    },
    {
        "claim": "Framing mental health through human stories and stigma-aware communication improves public comprehension compared with fear- or blame-heavy framing.",
        "domain": "Health & Media Communication",
        "source": "WHO / Public Health & Media Framing Literature on Stigma Reduction",
        "evidence_level": "Public Health Communication Synthesis",
        "status": "Intervention Basis",
    },
    {
        "claim": "Text transcripts alone cannot measure acoustic pauses, tone, facial expression, or clinical diagnosis, and require strict ethical guardrails.",
        "domain": "Computational & Ethical Scope",
        "source": "Project Methodological & Ethics Protocol (Neurometric Gaze)",
        "evidence_level": "Methodological Boundary",
        "status": "Mandatory Guardrail",
    },
]

SAMPLE_NARRATIVES = pd.DataFrame(
    [
        {"participant_id": "N01", "theme": "Silence & Stigma", "text": "I stayed quiet during the visit because I did not want to sound weak or dramatic.", "distress_flag": 1, "phq8_score": 14},
        {"participant_id": "N02", "theme": "Social Stigma", "text": "People around me treat therapy as something shameful so I hide how stressed I feel.", "distress_flag": 1, "phq8_score": 15},
        {"participant_id": "N03", "theme": "Help-Seeking", "text": "Talking openly with a counselor helped me put my stress into words and sleep better.", "distress_flag": 0, "phq8_score": 4},
        {"participant_id": "N04", "theme": "Silence & Stigma", "text": "I kept everything inside and just said I was fine when asked how things were going.", "distress_flag": 1, "phq8_score": 13},
        {"participant_id": "N05", "theme": "Help-Seeking", "text": "I shared my concerns clearly and felt supported by clear, practical guidance.", "distress_flag": 0, "phq8_score": 3},
        {"participant_id": "N06", "theme": "Social Stigma", "text": "Media stories often exaggerate or mock mental struggles instead of explaining them.", "distress_flag": 1, "phq8_score": 12},
    ]
)


def clean_text(val: Any) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(val)).strip()


def extract_zip_transcripts(raw: bytes) -> pd.DataFrame:
    records = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in zf.namelist():
            if name.endswith("/") or "__MACOSX" in name:
                continue
            base = name.rsplit("/", 1)[-1]
            if not base.lower().endswith(".csv"):
                continue
            try:
                df_sub = pd.read_csv(io.BytesIO(zf.read(name)))
            except Exception:
                continue
            text_col = next(
                (c for c in df_sub.columns if c.lower() in {"text", "transcript", "utterance", "value", "narrative"}),
                None,
            )
            if text_col is None:
                continue
            if "speaker" in [c.lower() for c in df_sub.columns]:
                sp_col = next(c for c in df_sub.columns if c.lower() == "speaker")
                mask = df_sub[sp_col].astype(str).str.lower().str.contains("participant|patient|subject", regex=True, na=False)
                rows = df_sub.loc[mask, text_col] if mask.any() else df_sub[text_col]
            else:
                rows = df_sub[text_col]
            joined = " ".join(clean_text(x) for x in rows if clean_text(x))
            if joined:
                pid_match = re.search(r"(\d+)", base)
                pid = pid_match.group(1) if pid_match else base
                records.append({"participant_id": pid, "theme": "Uploaded Transcript", "text": joined, "distress_flag": np.nan, "phq8_score": np.nan})
    if not records:
        raise ValueError("No readable CSV transcripts found inside the ZIP archive.")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION (8 CORE SECTIONS)
# ---------------------------------------------------------------------

st.sidebar.markdown('<div class="editorial-kicker">The Neurometric Gaze</div>', unsafe_allow_html=True)
st.sidebar.markdown("### Beyond the Unheard")
st.sidebar.caption("Research · Evidence · Human Stories · Media Intervention")

section = st.sidebar.radio(
    "Navigate platform",
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
    "Prototype for Harvard MS-MMH portfolio exploration. "
    "Not a medical or diagnostic tool."
)

# ---------------------------------------------------------------------
# 01 · HOME
# ---------------------------------------------------------------------

if section == "01 · Home":
    st.markdown('<div class="editorial-kicker">STORIES · EVIDENCE · UNDERSTANDING · ACTION</div>', unsafe_allow_html=True)
    st.title("Beyond the Unheard")
    st.markdown(
        "### Understanding mental health narratives, stigma, and help-seeking through research and human-centered media"
    )
    st.markdown(
        "Mental health is not only a clinical category—it is also shaped by what people feel able to say, "
        "what remains unspoken, and how media and culture frame distress. This platform brings together "
        "**human stories**, **research evidence**, and **media intervention** with a focus on young adults in Azerbaijan."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-kicker">01 · Human Stories</div>
                <h4>Lived Experiences</h4>
                <p>Anonymized narratives exploring how people talk about distress, stay silent, navigate stigma, and seek support.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-kicker">02 · Research & Evidence</div>
                <h4>Empirical & Media Analysis</h4>
                <p>Structured examination of media framing, communication barriers, thematic patterns, and verified literature.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="editorial-card">
                <div class="editorial-kicker">03 · Media Intervention</div>
                <h4>Public Health Storytelling</h4>
                <p>Short documentary, 3-part podcast series, interactive narratives, and stigma-aware public communication.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "**Core Research Question:** How do media narratives shape young adults' understanding of mental health stigma and help-seeking in Azerbaijan?"
    )

# ---------------------------------------------------------------------
# 02 · THE PROBLEM
# ---------------------------------------------------------------------

elif section == "02 · The Problem":
    st.markdown('<div class="editorial-kicker">CONTEXT & STIGMA</div>', unsafe_allow_html=True)
    st.header("The Problem: Silence, Stigma, and Fragmented Narratives")
    st.markdown(
        "Across many communities, conversations around psychological distress are constrained by social stigma, "
        "fear of misjudgment, and narrow media portrayals. When public narratives treat emotional struggle as taboo "
        "or purely sensational, young adults often delay seeking support or edit out key experiences during conversations."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Primary Lens", "Narrative & Stigma", "Social medicine framing")
    m2.metric("Target Audience", "Young Adults", "Azerbaijan context")
    m3.metric("Core Output", "Media + Research", "Evidence-backed intervention")

    st.markdown("### Key Communication Friction Points")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="editorial-card">
                <h4>Cultural & Social Hesitation</h4>
                <p>Concern over family expectations, social reputation, or being labeled 'weak' can turn everyday stress into private silence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="editorial-card">
                <h4>Media & Information Gaps</h4>
                <p>Fragmented or stigmatizing media coverage rarely models constructive, empathetic, and realistic help-seeking pathways.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# 03 · HUMAN STORIES
# ---------------------------------------------------------------------

elif section == "03 · Human Stories":
    st.markdown('<div class="editorial-kicker">INTERACTIVE STORYTELLING</div>', unsafe_allow_html=True)
    st.header("Human Stories: What Remains Unheard")
    st.markdown(
        "Explore anonymized, composite narratives illustrating how silence, social stigma, and supportive communication appear in everyday life."
    )

    selected_title = st.selectbox(
        "Select a story to read and inspect",
        [s["title"] for s in STORIES],
    )
    story = next(s for s in STORIES if s["title"] == selected_title)

    st.markdown(
        f"""
        <div class="editorial-card">
            <span class="pill-tag">{story["theme"]}</span>
            <span class="pill-tag">{story["context"]}</span>
            <h3>{story["title"]}</h3>
            <div class="quote-box">“{story["quote"]}”</div>
            <p>{story["narrative"]}</p>
            <p><strong>Audio & Documentary Cue:</strong> {story["audio_note"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Reader Reflection Prompt")
    st.write(story["reflection"])

# ---------------------------------------------------------------------
# 04 · RESEARCH LAB
# ---------------------------------------------------------------------

elif section == "04 · Research Lab":
    st.markdown('<div class="editorial-kicker">RESEARCH ANALYTICS & THEMATIC CODING</div>', unsafe_allow_html=True)
    st.header("Research Lab: Narrative Patterns & Data Exploration")
    st.markdown(
        "Inspect thematic patterns and word frequencies across anonymized narrative notes or upload your own authorized CSV / ZIP dataset."
    )

    uploaded = st.file_uploader("Upload CSV or ZIP transcript archive (optional)", type=["csv", "zip"])

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        if uploaded.name.lower().endswith(".zip"):
            df_lab = extract_zip_transcripts(raw_bytes)
        else:
            df_lab = pd.read_csv(io.BytesIO(raw_bytes))
            text_col = next(
                (c for c in df_lab.columns if c.lower() in {"text", "transcript", "narrative", "utterance"}),
                None,
            )
            if text_col is None:
                st.error("CSV must contain a text, transcript, narrative, or utterance column.")
                st.stop()
            df_lab = df_lab.copy()
            df_lab["text"] = df_lab[text_col].map(clean_text)
            if "theme" not in df_lab.columns:
                df_lab["theme"] = "Uploaded Narrative"
            if "participant_id" not in df_lab.columns:
                df_lab["participant_id"] = [f"P{i+1:02d}" for i in range(len(df_lab))]
    else:
        df_lab = SAMPLE_NARRATIVES.copy()
        st.caption("Showing illustrative research notes. Upload your own authorized dataset above to explore custom texts.")

    df_lab["word_count"] = df_lab["text"].astype(str).apply(lambda t: len(re.findall(r"\b\w+\b", t)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Records Analyzed", f"{len(df_lab)}")
    c2.metric("Mean Word Count", f"{df_lab['word_count'].mean():.1f}")
    c3.metric("Themes Represented", f"{df_lab['theme'].nunique()}")

    st.markdown("#### Thematic & Text Table")
    st.dataframe(df_lab, use_container_width=True, hide_index=True)

    # Simple transparent keyword/theme frequency inspection
    tokens = re.findall(r"\b[a-zA-ZəğıöşüçƏĞIİÖŞÜÇ]{4,}\b", " ".join(df_lab["text"].astype(str)).lower())
    stopwords = {"that", "this", "with", "from", "have", "were", "what", "when", "they", "them", "into", "about", "just", "some", "more", "than", "because", "would", "could", "said", "asked"}
    freq = pd.Series([w for w in tokens if w not in stopwords]).value_counts().head(12)

    if not freq.empty:
        st.markdown("#### Top Non-Stopword Terms (Descriptive Word Frequency)")
        st.bar_chart(freq, use_container_width=True)

    with st.expander("Optional Secondary Module: Exploratory PHQ-8 Text Association (Non-Clinical)"):
        st.caption(
            "Kept as a secondary methodological module only. PHQ-8 is a self-report symptom scale, "
            "not a clinical diagnosis, and lexical patterns do not measure emotions or individual risk."
        )
        if "phq8_score" in df_lab.columns and df_lab["phq8_score"].notna().sum() >= 3:
            sub = df_lab[["word_count", "phq8_score"]].dropna()
            if len(sub) >= 3 and sub["phq8_score"].nunique() > 1:
                corr = sub.corr(method="spearman").iloc[0, 1]
                st.metric("Exploratory Spearman Correlation (Word Count vs PHQ-8)", f"{corr:.3f}")
        else:
            st.write("Include a numeric `phq8_score` column (at least 3 valid rows) to inspect descriptive correlation.")

# ---------------------------------------------------------------------
# 05 · EVIDENCE LIBRARY
# ---------------------------------------------------------------------

elif section == "05 · Evidence Library":
    st.markdown('<div class="editorial-kicker">SOURCE VERIFICATION & LITERATURE</div>', unsafe_allow_html=True)
    st.header("Evidence Library")
    st.markdown(
        "Every core claim in the project is paired with its literature anchor, evidence type, and analytical boundary."
    )

    domain_filter = st.selectbox(
        "Filter by research domain",
        ["All"] + sorted({item["domain"] for item in EVIDENCE_ITEMS}),
    )

    filtered_evidence = (
        EVIDENCE_ITEMS
        if domain_filter == "All"
        else [item for item in EVIDENCE_ITEMS if item["domain"] == domain_filter]
    )

    for item in filtered_evidence:
        st.markdown(
            f"""
            <div class="editorial-card">
                <span class="pill-tag">{item["domain"]}</span>
                <span class="pill-tag">{item["evidence_level"]}</span>
                <span class="pill-tag">{item["status"]}</span>
                <h4>{item["claim"]}</h4>
                <p><strong>Reference / Basis:</strong> {item["source"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# 06 · THE INTERVENTION
# ---------------------------------------------------------------------

elif section == "06 · The Intervention":
    st.markdown('<div class="editorial-kicker">CAMPAIGN STUDIO & MEDIA FORMATS</div>', unsafe_allow_html=True)
    st.header("The Intervention: Documentary, Podcast & Public Campaign")
    st.markdown(
        "Rather than presenting raw findings alone, the intervention translates research themes into accessible media formats designed to reduce stigma and encourage constructive help-seeking."
    )

    tab1, tab2, tab3 = st.tabs(["Short Documentary (5–8 min)", "3-Episode Podcast Series", "Public Awareness Campaign"])

    with tab1:
        st.markdown("### Documentary Concept: *Voices Behind the Pause*")
        st.write("- **Runtime:** 6–7 minutes · **Format:** Observational storytelling, expert commentary, and archival/editorial visuals.")
        st.write("- **Arc:** Opens with everyday silence around stress, examines stigma and media framing, and closes with human pathways toward support.")

    with tab2:
        st.markdown("### 3-Part Podcast Blueprint")
        st.write("1. **Episode 1 — Naming the Unspoken:** How cultural scripts and media language shape stigma.")
        st.write("2. **Episode 2 — Silence in the Room:** Why young adults hold back—and how attentive listening changes dialogue.")
        st.write("3. **Episode 3 — Moving Toward Support:** Practical, stigma-free steps for reaching out and supporting peers.")

    with tab3:
        st.markdown("### Awareness Campaign Asset Planner")
        topic_choice = st.selectbox(
            "Choose a campaign pillar",
            ["Normalize Asking for Help", "Challenge Stigma Without Blame", "Listen Beyond Quick Answers"],
        )
        st.markdown(
            f"""
            <div class="editorial-card">
                <div class="editorial-kicker">Campaign Brief · {topic_choice}</div>
                <p><strong>Core Message:</strong> Seeking support or talking openly about strain is a practical step toward wellbeing, not a personal failure.</p>
                <p><strong>Deliverables:</strong> 6 short social media visuals/reels, 1 key campaign poster, and a shareable resource card with local support pathways.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# 07 · IMPACT
# ---------------------------------------------------------------------

elif section == "07 · Impact":
    st.markdown('<div class="editorial-kicker">PRE / POST EVALUATION</div>', unsafe_allow_html=True)
    st.header("Impact Measurement: Pre- and Post-Intervention Check")
    st.markdown(
        "Measure how exposure to a story or media piece shifts perceived stigma and willingness to seek or recommend support."
    )

    col_pre, col_post = st.columns(2)
    with col_pre:
        st.markdown("#### Before Viewing / Listening (Pre)")
        pre_stigma = st.slider("Perceived stigma around talking about mental health (1 = Low, 7 = High)", 1, 7, 5)
        pre_help = st.slider("Comfort discussing support options openly (1 = Low, 7 = High)", 1, 7, 3)

    with col_post:
        st.markdown("#### After Viewing / Listening (Post)")
        post_stigma = st.slider("Perceived stigma after media engagement (1 = Low, 7 = High)", 1, 7, 3)
        post_help = st.slider("Comfort discussing support options after engagement (1 = Low, 7 = High)", 1, 7, 5)

    stigma_delta = post_stigma - pre_stigma
    help_delta = post_help - pre_help

    m1, m2 = st.columns(2)
    m1.metric("Stigma Score Change", f"{stigma_delta:+d} pts", "Negative = lower stigma")
    m2.metric("Support Comfort Change", f"{help_delta:+d} pts", "Positive = higher comfort")

    impact_df = pd.DataFrame(
        {
            "Pre-Intervention": [pre_stigma, pre_help],
            "Post-Intervention": [post_stigma, post_help],
        },
        index=["Stigma Perception", "Support Comfort"],
    )
    st.bar_chart(impact_df, use_container_width=True)

    summary_payload = {
        "project": "The Neurometric Gaze — Beyond the Unheard",
        "pre": {"stigma": pre_stigma, "support_comfort": pre_help},
        "post": {"stigma": post_stigma, "support_comfort": post_help},
        "deltas": {"stigma_change": stigma_delta, "support_comfort_change": help_delta},
    }
    st.download_button(
        "Export Impact Snapshot (JSON)",
        json.dumps(summary_payload, indent=2).encode("utf-8"),
        file_name="neurometric_gaze_impact_snapshot.json",
        mime="application/json",
    )

# ---------------------------------------------------------------------
# 08 · ABOUT & ETHICS
# ---------------------------------------------------------------------

elif section == "08 · About & Ethics":
    st.markdown('<div class="editorial-kicker">GOVERNANCE, CONSENT & LIMITATIONS</div>', unsafe_allow_html=True)
    st.header("About, Methodology & Ethical Guardrails")
    st.markdown(
        """
        - **Not a clinical or diagnostic tool:** This prototype supports research, storytelling, and public communication—never medical diagnosis, screening, or individual risk scoring.
        - **Informed consent & anonymization:** Any real interviews or audience responses require explicit consent, clear purpose disclosure, and anonymization prior to analysis or publication.
        - **Methodological transparency:** Text and media analysis capture narratives and perceptions; they do not establish clinical causality or replace professional care.
        - **Support notice:** If you or someone you know is going through a difficult time, please reach out to a qualified healthcare professional or local support service.
        """
    )
