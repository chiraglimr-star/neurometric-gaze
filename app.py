"""
THE NEUROMETRIC GAZE — Beyond the Unheard
=========================================================================
A research and methodological prototype on the influence of media narratives
on mental health stigma, literacy, and help-seeking intentions among young
adults in Azerbaijan.

Research question
-----------------
How do media narratives influence mental health stigma, social attitudes, and
help-seeking intentions among young adults in Azerbaijan, and can evidence-
based storytelling improve these outcomes?

Scientific honesty statement
----------------------------
Every empirical number in this application is SYNTHETIC / ILLUSTRATIVE and is
labeled as such. No real participants have been recruited at the time of this
prototype. Real findings require ethical approval and data collection.

Design principles
-----------------
1. Use only validated instruments (ATMHPS-AZ, RIBS, PSOSH, MHSAS, MHLS).
2. Never claim diagnosis. Never use PHQ-8 as a central feature.
3. Never fabricate clinical indices (e.g., "silence_index") without validation.
4. Report N, effect size, and 95% CI alongside every statistical claim.
5. Distinguish preprints from peer-reviewed sources.
6. Anonymity, informed consent, and data minimization are non-negotiable.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# =============================================================================
# 1. PAGE CONFIGURATION AND BRAND TOKENS
# =============================================================================

st.set_page_config(
    page_title="The Neurometric Gaze — Beyond the Unheard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BRAND = {
    "ivory": "#F5F2EB",
    "navy": "#14243A",
    "teal": "#5C8581",
    "amber": "#D6A56C",
    "muted": "#EBE6DC",
    "danger": "#B85450",
    "success": "#5C8581",
    "warn": "#D6A56C",
}

RNG_SEED = 20260218


# =============================================================================
# 2. GLOBAL CSS (editorial design system)
# =============================================================================

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', sans-serif;
            color: {BRAND['navy']};
            background-color: {BRAND['ivory']};
        }}
        .block-container {{
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}
        h1, h2, h3, h4 {{
            font-family: 'Playfair Display', serif;
            letter-spacing: -0.015em;
            color: {BRAND['navy']};
        }}
        h1 {{ font-size: 2.7rem; line-height: 1.15; }}
        h2 {{ font-size: 1.9rem; line-height: 1.25; }}
        h3 {{ font-size: 1.3rem; }}

        section[data-testid="stSidebar"] {{
            background-color: {BRAND['navy']} !important;
            border-right: 1px solid rgba(214,165,108,0.2);
        }}
        section[data-testid="stSidebar"] * {{ color: {BRAND['ivory']} !important; }}
        section[data-testid="stSidebar"] hr {{ border-color: rgba(214,165,108,0.25); }}

        .hero {{
            text-align: center;
            padding: 3rem 1rem 2rem 1rem;
            border-bottom: 1px solid {BRAND['muted']};
            margin-bottom: 2rem;
        }}
        .hero-kicker {{
            color: {BRAND['teal']};
            letter-spacing: 0.3em;
            font-size: 0.72rem;
            font-weight: 500;
        }}
        .hero-title {{
            font-family: 'Playfair Display', serif;
            font-size: 3.6rem;
            line-height: 1.05;
            margin: 1rem 0;
        }}
        .hero-sub {{
            font-size: 1.1rem;
            color: {BRAND['teal']};
            max-width: 680px;
            margin: 0 auto;
            line-height: 1.6;
        }}
        .hero-tags {{
            margin-top: 1.8rem;
            color: {BRAND['amber']};
            font-weight: 600;
            letter-spacing: 0.15em;
            font-size: 0.78rem;
        }}

        .synth-banner {{
            background: {BRAND['amber']};
            color: white;
            padding: 0.7rem 1.2rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 1rem 0;
            letter-spacing: 0.02em;
        }}
        .info-box {{
            background: {BRAND['muted']};
            border-left: 4px solid {BRAND['amber']};
            padding: 1.1rem 1.4rem;
            border-radius: 8px;
            margin: 1.2rem 0;
            font-size: 0.92rem;
            line-height: 1.6;
        }}
        .danger-box {{
            background: #F8E5E3;
            border-left: 4px solid {BRAND['danger']};
            padding: 1.1rem 1.4rem;
            border-radius: 8px;
            margin: 1.2rem 0;
            font-size: 0.92rem;
            line-height: 1.6;
        }}
        .success-box {{
            background: #E6F0EE;
            border-left: 4px solid {BRAND['teal']};
            padding: 1.1rem 1.4rem;
            border-radius: 8px;
            margin: 1.2rem 0;
            font-size: 0.92rem;
            line-height: 1.6;
        }}

        .pillar {{
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1.7rem 1.5rem;
            border: 1px solid {BRAND['muted']};
            height: 100%;
            transition: 0.25s;
        }}
        .pillar:hover {{
            border-color: {BRAND['teal']};
            box-shadow: 0 8px 28px rgba(20,36,58,0.08);
        }}
        .pillar .num {{
            font-family: 'Playfair Display', serif;
            color: {BRAND['amber']};
            font-weight: 700;
            font-size: 0.95rem;
        }}
        .pillar h3 {{ margin: 0.5rem 0 0.7rem 0; font-size: 1.25rem; }}
        .pillar p {{ color: #4a4a4a; line-height: 1.6; font-size: 0.93rem; }}

        .stat-card {{
            background: #FFFFFF;
            border-radius: 10px;
            padding: 1.3rem 1.4rem;
            border: 1px solid {BRAND['muted']};
            margin-bottom: 0.9rem;
        }}
        .stat-card .label {{
            color: {BRAND['teal']};
            font-size: 0.74rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .stat-card .value {{
            font-family: 'Playfair Display', serif;
            font-size: 2rem;
            margin: 0.3rem 0;
        }}
        .stat-card .note {{
            color: #777;
            font-size: 0.82rem;
            line-height: 1.5;
        }}

        .codebook-row {{
            background: #FFFFFF;
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 0.6rem;
            border-left: 3px solid {BRAND['teal']};
        }}
        .codebook-row .code {{
            font-family: 'Inter', monospace;
            color: {BRAND['amber']};
            font-weight: 700;
            font-size: 0.8rem;
        }}
        .codebook-row .name {{
            font-weight: 600;
            margin: 0.2rem 0;
        }}
        .codebook-row .desc {{
            font-size: 0.86rem;
            color: #555;
            line-height: 1.5;
        }}

        .ref-card {{
            background: #FFFFFF;
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.9rem;
            border: 1px solid {BRAND['muted']};
        }}
        .ref-card .num {{
            display: inline-block;
            background: {BRAND['navy']};
            color: white;
            width: 28px; height: 28px;
            border-radius: 50%;
            text-align: center;
            line-height: 28px;
            font-weight: 600;
            font-size: 0.8rem;
            margin-right: 0.7rem;
        }}
        .ref-card .title {{ font-weight: 600; font-size: 0.98rem; }}
        .ref-card .authors {{ color: #555; font-size: 0.86rem; margin-top: 0.35rem; }}
        .ref-card .meta {{ color: {BRAND['teal']}; font-size: 0.8rem; margin-top: 0.5rem; }}
        .ref-card .role {{
            display: inline-block;
            background: {BRAND['muted']};
            color: {BRAND['teal']};
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 0.72rem;
            margin-top: 0.6rem;
            font-weight: 600;
        }}

        .pull-quote {{
            font-family: 'Playfair Display', serif;
            font-size: 1.45rem;
            line-height: 1.5;
            color: {BRAND['teal']};
            border-left: 3px solid {BRAND['amber']};
            padding-left: 1.6rem;
            margin: 2rem 0;
            font-style: italic;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 11px;
            border-radius: 12px;
            font-size: 0.72rem;
            font-weight: 600;
        }}
        .badge-high {{ background: {BRAND['teal']}; color: white; }}
        .badge-med  {{ background: {BRAND['amber']}; color: white; }}
        .badge-low  {{ background: {BRAND['danger']}; color: white; }}

        .stButton > button {{
            background-color: {BRAND['navy']};
            color: {BRAND['ivory']};
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.3rem;
            font-weight: 500;
        }}
        .stButton > button:hover {{ background-color: {BRAND['teal']}; color: white; }}

        .stTabs [data-baseweb="tab"] {{
            color: {BRAND['teal']};
            padding: 0.55rem 1.1rem;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: #FFFFFF;
            color: {BRAND['navy']};
            border-bottom: 2px solid {BRAND['amber']};
        }}

        hr {{ border-color: {BRAND['muted']}; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =============================================================================
# 3. STATISTICAL FUNCTIONS (real implementations, not placeholders)
# =============================================================================

def cronbach_alpha(items: pd.DataFrame) -> float | None:
    """Cronbach's alpha for internal consistency.

    Parameters
    ----------
    items : pd.DataFrame
        Rows = participants, columns = items (numeric, same scale direction).

    Returns
    -------
    float or None
        Alpha value; None if undefined (fewer than 2 items, zero total variance).
    """
    df = items.dropna()
    k = df.shape[1]
    if k < 2 or len(df) < 2:
        return None
    item_variances = df.var(axis=0, ddof=1)
    total_variance = df.sum(axis=1).var(ddof=1)
    if total_variance <= 0:
        return None
    return float((k / (k - 1)) * (1 - item_variances.sum() / total_variance))


def mcdonald_omega_approx(items: pd.DataFrame) -> float | None:
    """Approximate McDonald's omega for a single-factor model via PCA.

    This is a *demonstration approximation*. For publication, use a proper
    CFA-based estimator (e.g., `semopy`, `lavaan`) with a theoretically
    justified factor structure.
    """
    df = items.dropna()
    if df.shape[1] < 3 or len(df) < 5:
        return None
    corr = df.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    loadings = eigvecs[:, 0] * np.sqrt(max(eigvals[0], 0))
    sum_l = loadings.sum()
    sum_u = (1 - loadings**2).sum()
    denom = sum_l**2 + sum_u
    if denom <= 0:
        return None
    return float((sum_l**2) / denom)


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float | None:
    """Cohen's d for independent groups (pooled SD)."""
    a = np.asarray([x for x in group_a if not pd.isna(x)], dtype=float)
    b = np.asarray([x for x in group_b if not pd.isna(x)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return None
    n1, n2 = len(a), len(b)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return None
    return float((a.mean() - b.mean()) / pooled)


def paired_cohens_d(before: Sequence[float], after: Sequence[float]) -> float | None:
    """Cohen's d for paired samples (d_z)."""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return None
    diff = b - a
    sd = diff.std(ddof=1)
    if sd == 0:
        return None
    return float(diff.mean() / sd)


def krippendorff_alpha_nominal(data: np.ndarray) -> float | None:
    """Krippendorff's alpha for nominal data (units x coders, NaN = missing)."""
    data = np.asarray(data, dtype=float)
    values = sorted({v for row in data for v in row if not np.isnan(v)})
    if len(values) < 2:
        return 1.0
    idx = {v: i for i, v in enumerate(values)}
    V = len(values)
    coincidence = np.zeros((V, V))
    for row in data:
        codings = [v for v in row if not np.isnan(v)]
        m = len(codings)
        if m < 2:
            continue
        for i, v1 in enumerate(codings):
            for j, v2 in enumerate(codings):
                if i != j:
                    coincidence[idx[v1], idx[v2]] += 1 / (m - 1)
    n_c = coincidence.sum(axis=1)
    n_total = n_c.sum()
    if n_total <= 1:
        return None
    observed = (coincidence.sum() - np.trace(coincidence)) / n_total
    expected = 0.0
    for c in range(V):
        for k in range(V):
            if c != k:
                expected += n_c[c] * n_c[k]
    expected = expected / (n_total * (n_total - 1))
    if expected == 0:
        return 1.0
    return float(1 - observed / expected)


def sample_size_two_arm(d: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-arm sample size for two-sample t-test (Cohen's approximation)."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(math.ceil(((z_a + z_b) ** 2) * 2 / (d**2)))


def power_curve(d: float, ns: Iterable[int], alpha: float = 0.05) -> pd.DataFrame:
    """Power as a function of per-arm n for a two-sample t-test."""
    rows = []
    z_a = stats.norm.ppf(1 - alpha / 2)
    for n in ns:
        ncp = d * math.sqrt(n / 2)
        power = 1 - stats.norm.cdf(z_a - ncp) + stats.norm.cdf(-z_a - ncp)
        rows.append({"n_per_arm": n, "power": power})
    return pd.DataFrame(rows)


def interpret_d(d: float | None) -> str:
    if d is None:
        return "—"
    a = abs(d)
    if a < 0.2:
        return "trivial"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def interpret_alpha(a: float | None) -> str:
    if a is None:
        return "—"
    if a < 0.5:
        return "unacceptable"
    if a < 0.6:
        return "poor"
    if a < 0.7:
        return "questionable"
    if a < 0.8:
        return "acceptable"
    if a < 0.9:
        return "good"
    return "excellent"


# =============================================================================
# 4. VALIDATED INSTRUMENTS — reference metadata only
# =============================================================================

INSTRUMENTS = [
    {
        "code": "ATMHPS-AZ",
        "name": "Attitudes Towards Mental Health Problems Scale (Azerbaijani)",
        "items": 35,
        "factors": 7,
        "measures": "Cultural attitudes and stigma towards mental health problems",
        "source": "Shahhuseynbayova et al. (2026)",
        "status": "OSF preprint — check peer-review status before use",
        "reported_alpha": "0.954 (reported)",
    },
    {
        "code": "RIBS",
        "name": "Reported and Intended Behaviour Scale",
        "items": 8,
        "factors": 2,
        "measures": "Reported and intended social behaviour toward people with mental illness",
        "source": "Evans-Lacko et al. (2011)",
        "status": "Peer-reviewed",
        "reported_alpha": "see original",
    },
    {
        "code": "PSOSH",
        "name": "Perceptions of Stigmatization by Others for Seeking Help",
        "items": 5,
        "factors": 1,
        "measures": "Anticipated stigmatization from others when seeking psychological help",
        "source": "Vogel et al. (2009)",
        "status": "Peer-reviewed",
        "reported_alpha": "0.91 (reported)",
    },
    {
        "code": "MHSAS",
        "name": "Mental Help Seeking Attitudes Scale",
        "items": 9,
        "factors": 1,
        "measures": "General attitudes toward seeking professional psychological help",
        "source": "Hammer et al. (2018)",
        "status": "Peer-reviewed",
        "reported_alpha": "see original",
    },
]


# =============================================================================
# 5. MEDIA CODEBOOK (M01–M10) — PICMIN-inspired
# =============================================================================

CODEBOOK = [
    ("M01", "Violence frame", "Is mental illness linked with violence in headline or body?"),
    ("M02", "Dangerousness", "Is the person presented as dangerous to self or others?"),
    ("M03", "Blame / weakness", "Is the problem framed as personal weakness or moral failure?"),
    ("M04", "Sensationalism", "Do headline and body use sensationalist language?"),
    ("M05", "Recovery", "Are recovery, treatment, or improvement possibilities shown?"),
    ("M06", "Human experience", "Is the lived experience and voice of the person present?"),
    ("M07", "Agency", "Is the person depicted as an agent making their own decisions?"),
    ("M08", "Help-seeking", "Is professional help-seeking encouraged or represented?"),
    ("M09", "Support", "Are family or social support networks shown?"),
    ("M10", "Respectful language", "Is person-first, non-stigmatizing language used?"),
]


# =============================================================================
# 6. EVIDENCE LIBRARY — the 10 priority references from the plan
# =============================================================================

REFERENCES = [
    {
        "n": 1,
        "title": "Interventions to Reduce Mental Health Stigma in Young People",
        "authors": "Crockett, M. A., et al.",
        "year": 2025,
        "meta": "Systematic review · 97 RCTs · 43,852 participants",
        "role": "Design basis for intervention",
        "level": "high",
        "key_finding": "Contact-based and lived-experience interventions outperform information-only approaches for behavioural stigma outcomes.",
    },
    {
        "n": 2,
        "title": "Psychometric Validation of the Azerbaijani Version of ATMHPS",
        "authors": "Shahhuseynbayova, N., Bunyatova, A., Mustafayev, M., Kotera, Y.",
        "year": 2026,
        "meta": "OSF preprint · N = 946 Azerbaijani university students",
        "role": "Primary Azerbaijani-language stigma instrument",
        "level": "med",
        "key_finding": "Highest subscale: disclosure concerns. Lowest: family attitudes and family burden.",
    },
    {
        "n": 3,
        "title": "Effect of Web-Based Educational Interventions on Mental Health Literacy, Stigma, Help-Seeking",
        "authors": "Nazari, A., et al.",
        "year": 2023,
        "meta": "Meta-analysis · 20 studies (10 in pooled analysis)",
        "role": "Baseline for what information-only can/cannot achieve",
        "level": "high",
        "key_finding": "Literacy improves (SMD = 0.70); stigma and help-seeking intentions do not (95% CI crosses zero).",
    },
    {
        "n": 4,
        "title": "Selfie Videos to Reduce Stigma and Increase Treatment Seeking",
        "authors": "Amsalem, D., et al.",
        "year": 2023,
        "meta": "Two RCTs · N = 1,532 · 58–102 s videos",
        "role": "Short-form, lived-experience video design",
        "level": "high",
        "key_finding": "Cohen's d = 0.31–0.76 (Study 1) and 0.13–0.47 (Study 2) for stigma reduction.",
    },
    {
        "n": 5,
        "title": "Perceived Barriers and Facilitators to Mental Health Help-Seeking",
        "authors": "Gulliver, A., Griffiths, K. M., Christensen, H.",
        "year": 2010,
        "meta": "Systematic review · 22 studies",
        "role": "Help-seeking barrier taxonomy",
        "level": "high",
        "key_finding": "Top barriers: stigma, low symptom recognition, preference for self-reliance, uncertainty about sources.",
    },
    {
        "n": 6,
        "title": "The Mental Health Literacy Scale (MHLS)",
        "authors": "O'Connor, M., Casey, L.",
        "year": 2015,
        "meta": "Instrument development · 35 items",
        "role": "Literacy measurement",
        "level": "high",
        "key_finding": "Strong internal and test-retest reliability reported.",
    },
    {
        "n": 7,
        "title": "Development and Psychometric Properties of RIBS",
        "authors": "Evans-Lacko, S., et al.",
        "year": 2011,
        "meta": "Instrument development · 8 items",
        "role": "Social distance and behavioural intention",
        "level": "high",
        "key_finding": "Two-factor structure (reported behaviour; intended behaviour).",
    },
    {
        "n": 8,
        "title": "Mass Media Interventions for Reducing Mental Health-Related Stigma",
        "authors": "Clement, S., et al.",
        "year": 2013,
        "meta": "Systematic review · Cochrane-style",
        "role": "Evidence base for media interventions",
        "level": "high",
        "key_finding": "Mixed evidence; effects strongest for targeted, contact-based media rather than mass awareness.",
    },
    {
        "n": 9,
        "title": "Development of the PICMIN Instrument (Picture of Mental Illness in Newspapers)",
        "authors": "PICMIN Consortium",
        "year": 2012,
        "meta": "Codebook · Krippendorff's α > 0.75 across 622 articles (3 countries)",
        "role": "Media content coding framework",
        "level": "high",
        "key_finding": "10 analytic categories with strong inter-coder reliability.",
    },
    {
        "n": 10,
        "title": "A New Podcast for Reducing Stigma Against People Living with Mental Illness (Co-Design)",
        "authors": "Carrotte, E. R., et al.",
        "year": 2022,
        "meta": "Co-design study",
        "role": "Podcast format and co-design methodology",
        "level": "med",
        "key_finding": "Co-designed lived-experience podcasts show strong acceptability and perceived impact.",
    },
]


# =============================================================================
# 7. SYNTHETIC DATA GENERATORS (clearly labeled as illustrative)
# =============================================================================

@st.cache_data(show_spinner=False)
def synth_survey(n: int = 200) -> pd.DataFrame:
    """Synthetic survey data mimicking the study design.

    IMPORTANT: this is *not* real data. It exists only to demonstrate
    the analysis pipeline. Item scales and directions are approximate.
    """
    rng = np.random.default_rng(RNG_SEED)
    n_half = n // 2

    age_group = rng.choice(["18–21", "22–25", "26–29"], size=n, p=[0.35, 0.4, 0.25])
    education = rng.choice(["Undergraduate", "Graduate", "Other"], size=n, p=[0.6, 0.3, 0.1])
    region = rng.choice(["Baku", "Ganja", "Sumqayit", "Other"], size=n, p=[0.55, 0.15, 0.15, 0.15])
    group = np.array(["Intervention"] * n_half + ["Control"] * (n - n_half))
    rng.shuffle(group)
    wave = rng.choice(["T0", "T1", "T2"], size=n, p=[0.34, 0.33, 0.33])

    def gen_items(n, k, lo, hi, sd_shift=0.0, group_shift=None):
        base = rng.normal(0, 1, size=(n, 1))
        items = base + rng.normal(0, sd_shift, size=(n, k))
        if group_shift is not None:
            items = items + group_shift
        scaled = np.clip(np.round(items + (lo + hi) / 2), lo, hi)
        return scaled.astype(int)

    df = pd.DataFrame({
        "response_id": [f"R{i:04d}" for i in range(1, n + 1)],
        "age_group": age_group,
        "education": education,
        "region": region,
        "group": group,
        "survey_wave": wave,
    })

    # 35 ATMHPS items (0–4)
    atm = gen_items(n, 35, 0, 4, sd_shift=0.5)
    for i in range(35):
        df[f"atm{i+1:02d}"] = atm[:, i]

    # 8 RIBS items (1–5)
    ribs = gen_items(n, 8, 1, 5, sd_shift=0.6)
    for i in range(8):
        df[f"ribs{i+1:02d}"] = ribs[:, i]

    # 5 PSOSH items (1–5), slightly higher for control group
    psosh = gen_items(n, 5, 1, 5, sd_shift=0.5,
                      group_shift=np.where(group == "Control", 0.2, 0))
    for i in range(5):
        df[f"psosh{i+1:02d}"] = psosh[:, i]

    # 9 MHSAS items (1–7)
    mhsas = gen_items(n, 9, 1, 7, sd_shift=0.6)
    for i in range(9):
        df[f"mhsas{i+1:02d}"] = mhsas[:, i]

    return df


@st.cache_data(show_spinner=False)
def synth_media(n_articles: int = 60, n_coders: int = 2) -> pd.DataFrame:
    """Synthetic media articles coded by two independent coders."""
    rng = np.random.default_rng(RNG_SEED + 1)
    sources = ["525.az", "Qafqazinfo", "APA", "Report", "Trend", "News.az"]
    rows = []
    for i in range(1, n_articles + 1):
        article = {
            "article_id": f"A{i:03d}",
            "source": rng.choice(sources),
            "publication_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 365))),
        }
        for code, _, _ in CODEBOOK:
            # Two coders with ~80% agreement
            base = rng.integers(0, 2)
            coder1 = int(base)
            coder2 = int(base) if rng.random() < 0.82 else int(1 - base)
            article[f"{code}_coder1"] = coder1
            article[f"{code}_coder2"] = coder2
        rows.append(article)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def synth_intervention(n: int = 220) -> pd.DataFrame:
    """Synthetic pre/post/follow-up data for a two-arm trial."""
    rng = np.random.default_rng(RNG_SEED + 2)
    n_half = n // 2
    group = np.array(["Intervention"] * n_half + ["Control"] * (n - n_half))
    rng.shuffle(group)

    # PSOSH — higher = more anticipated stigma; intervention should lower it
    baseline = rng.normal(3.4, 0.7, size=n)
    post = baseline + np.where(group == "Intervention", rng.normal(-0.35, 0.5, size=n),
                               rng.normal(-0.05, 0.5, size=n))
    follow = post + rng.normal(0.05, 0.4, size=n)

    return pd.DataFrame({
        "response_id": [f"P{i:04d}" for i in range(1, n + 1)],
        "group": group,
        "psosh_baseline": np.clip(baseline, 1, 5),
        "psosh_post": np.clip(post, 1, 5),
        "psosh_followup": np.clip(follow, 1, 5),
    })


# =============================================================================
# 8. UI COMPONENTS
# =============================================================================

def synth_banner(text: str = "SYNTHETIC / ILLUSTRATIVE DATA — not real participants") -> None:
    st.markdown(f'<div class="synth-banner">⚠ {text}</div>', unsafe_allow_html=True)


def info_box(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def danger_box(text: str) -> None:
    st.markdown(f'<div class="danger-box">{text}</div>', unsafe_allow_html=True)


def success_box(text: str) -> None:
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)


def pull_quote(text: str) -> None:
    st.markdown(f'<div class="pull-quote">“{text}”</div>', unsafe_allow_html=True)


def stat_card(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="stat-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def evidence_badge(level: str) -> str:
    cls = {"high": "badge-high", "med": "badge-med", "low": "badge-low"}.get(level, "badge-med")
    label = {"high": "Peer-reviewed", "med": "Preprint / mixed", "low": "Low"}.get(level, level)
    return f'<span class="badge {cls}">{label}</span>'


def style_plotly(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font_family="Inter",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        colorway=[BRAND["teal"], BRAND["amber"], BRAND["navy"], BRAND["danger"]],
    )
    return fig


# =============================================================================
# 9. PAGES
# =============================================================================

def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">B E Y O N D &nbsp; T H E &nbsp; U N H E A R D</div>
            <div class="hero-title">What remains<br>unheard?</div>
            <p class="hero-sub">
                Mental health is not only a medical issue. It is also a human story — of silence,
                fear, hope, and the courage to speak. This platform is a research and media
                prototype investigating how narratives shape stigma and help-seeking in Azerbaijan.
            </p>
            <div class="hero-tags">RESEARCH · MEDIA · EVIDENCE · ETHICS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Research question")
    st.markdown(
        "> *How do media narratives influence mental health stigma, social attitudes, "
        "and help-seeking intentions among young adults in Azerbaijan, and can "
        "evidence-based storytelling improve these outcomes?*"
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            '<div class="pillar"><div class="num">01 — MODULE A</div>'
            '<h3>Media Content Analysis</h3>'
            '<p>How does Azerbaijani media frame mental health? We apply a PICMIN-based '
            'codebook with 10 categories and compute inter-coder reliability.</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="pillar"><div class="num">02 — MODULE B</div>'
            '<h3>Youth Survey</h3>'
            '<p>Validated instruments (ATMHPS-AZ, RIBS, PSOSH, MHSAS) measure stigma, '
            'social distance, anticipated stigma, and help-seeking attitudes.</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="pillar"><div class="num">03 — MODULE C</div>'
            '<h3>Intervention Evaluation</h3>'
            '<p>Two-arm design (intervention vs. control) with T0 / T1 / T2 waves, '
            'effect size, and 95% CI reporting.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    pull_quote("The project is not about building a prettier dashboard. "
               "It is about testing whether a story can move a needle.")

    st.divider()

    st.markdown("### Design principles")
    col1, col2 = st.columns(2)
    with col1:
        success_box(
            "<strong>What we do:</strong><br>"
            "• Use validated instruments with published psychometrics<br>"
            "• Report N, effect size, and 95% CI for every finding<br>"
            "• Label every synthetic number as SYNTHETIC<br>"
            "• Distinguish preprints from peer-reviewed evidence<br>"
            "• Anonymize all participant stories and quotes"
        )
    with col2:
        danger_box(
            "<strong>What we do NOT do:</strong><br>"
            "• No clinical diagnosis, no PHQ-8 as a central feature<br>"
            "• No fabricated indices (e.g., 'silence_index')<br>"
            "• No fabricated cross-sectional correlations posing as causal<br>"
            "• No participant risk scores shown to individuals<br>"
            "• No 'successful impact' claims without a control group"
        )

    st.caption(
        f"© 2026 The Neurometric Gaze · Research prototype · "
        f"Last updated {datetime.now().strftime('%d %b %Y')} · "
        "Not a clinical service. If you or someone you know is struggling, "
        "please contact a licensed mental health professional."
    )


def render_research_question() -> None:
    st.title("Research Question & Framework")

    st.markdown("### Central research question")
    info_box(
        "<strong>How do media narratives influence mental health stigma, social attitudes, "
        "and help-seeking intentions among young adults in Azerbaijan, and can evidence-"
        "based storytelling improve these outcomes?</strong>"
    )

    st.markdown("### Three research objectives")
    st.markdown(
        "**Objective 1 — Media Content Analysis (Module A)**<br>"
        "Identify stigmatizing and anti-stigmatizing narratives in Azerbaijani media coverage "
        "of mental health using a validated codebook.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Objective 2 — Youth Survey (Module B)**<br>"
        "Measure stigma, mental health literacy, social distance, anticipated stigma, and "
        "help-seeking attitudes among 18–29 year-old young adults using validated instruments.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Objective 3 — Intervention Evaluation (Module C)**<br>"
        "Assess the effect of an evidence-based, narrative-driven media intervention on the "
        "same outcomes, using a two-arm design with T0 / T1 / T2 measurement waves.",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Theoretical grounding")
    st.markdown(
        "The project draws on three converging evidence streams:"
    )
    st.markdown(
        "1. **Contact hypothesis** — direct or mediated contact with people with lived "
        "experience reduces prejudice more effectively than information alone "
        "(Crockett et al., 2025; Amsalem et al., 2023).\n"
        "2. **Behavioural vs. attitudinal change** — knowledge and attitudes change more "
        "easily than behaviour; the reporting plan treats these as separate outcomes "
        "(Nazari et al., 2023).\n"
        "3. **Help-seeking as a social decision** — disclosure concerns and anticipated "
        "stigma are strong predictors of help-seeking, especially in collectivist contexts "
        "(Shahhuseynbayova et al., 2026; Gulliver et al., 2010)."
    )

    st.divider()
    st.markdown("### Why not an AI diagnostic tool?")
    danger_box(
        "A diagnostic-AI framing (e.g., predicting depression from text) is scientifically "
        "infeasible at the sample sizes achievable in this project, ethically fraught, and "
        "misaligned with the research question. The question is about <em>media and society</em>, "
        "not about individual diagnosis. We therefore deliberately exclude PHQ-8 and any "
        "individual-level risk score."
    )


def render_media_analysis() -> None:
    st.title("Module A — Media Content Analysis")
    synth_banner()

    st.markdown(
        "This module applies a PICMIN-inspired codebook (10 categories) to Azerbaijani media "
        "articles about mental health. Each article is independently coded by two researchers; "
        "inter-coder reliability is computed with Krippendorff's alpha (nominal)."
    )

    with st.expander("📖 Codebook (M01–M10)", expanded=True):
        for code, name, desc in CODEBOOK:
            st.markdown(
                f'<div class="codebook-row">'
                f'<div class="code"><code>{code}</code></div>'
                f'<div class="name">{name}</div>'
                f'<div class="desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Inter-coder reliability (synthetic demonstration)")
    df = synth_media()

    alphas = []
    for code, name, _ in CODEBOOK:
        arr = df[[f"{code}_coder1", f"{code}_coder2"]].values.astype(float)
        a = krippendorff_alpha_nominal(arr)
        alphas.append({"code": code, "name": name, "alpha": a})

    alpha_df = pd.DataFrame(alphas)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.bar(
            alpha_df, x="code", y="alpha",
            color="alpha",
            color_continuous_scale=["#EBE6DC", "#5C8581", "#14243A"],
            text=alpha_df["alpha"].round(2),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 1.05],
                          yaxis_title="Krippendorff's α")
        fig.add_hline(y=0.75, line_dash="dash", line_color=BRAND["amber"],
                      annotation_text="α = 0.75 (acceptable threshold)")
        st.plotly_chart(style_plotly(fig, 400), use_container_width=True)
    with col2:
        mean_alpha = float(alpha_df["alpha"].mean())
        stat_card(
            "Mean α (10 codes)",
            f"{mean_alpha:.3f}",
            "Illustrative only. Real reliability requires actual double-coded sample.",
        )

    st.markdown("### Frequency of each code (synthetic)")
    freq_rows = []
    for code, name, _ in CODEBOOK:
        pct = df[f"{code}_coder1"].mean() * 100
        freq_rows.append({"code": code, "name": name, "prevalence_pct": pct})
    freq_df = pd.DataFrame(freq_rows).sort_values("prevalence_pct", ascending=True)

    fig2 = px.bar(
        freq_df, x="prevalence_pct", y="name", orientation="h",
        color="prevalence_pct",
        color_continuous_scale=["#EBE6DC", "#5C8581", "#14243A"],
        text=freq_df["prevalence_pct"].round(1),
    )
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    fig2.update_layout(coloraxis_showscale=False, xaxis_title="% of articles")
    st.plotly_chart(style_plotly(fig2, 420), use_container_width=True)

    st.divider()
    st.markdown("### What this analysis will report (once real data exist)")
    st.markdown(
        "- Krippendorff's α per code (target: ≥ 0.75)<br>"
        "- Frequency of each code across the corpus<br>"
        "- Cross-tabulation by outlet / article type / region<br>"
        "- Qualitative excerpts with anonymized source IDs<br>"
        "- Explicit coding notes and coder IDs for audit".replace("\n", "\n"),
        unsafe_allow_html=True,
    )


def render_survey() -> None:
    st.title("Module B — Youth Survey")
    synth_banner()

    st.markdown(
        "The survey uses **only validated instruments**. We do not invent new scales. "
        "The table below lists the four primary instruments."
    )

    for inst in INSTRUMENTS:
        with st.expander(f"📋 {inst['code']} — {inst['name']}", expanded=False):
            st.markdown(
                f"**Items:** {inst['items']}  \n"
                f"**Factors:** {inst['factors']}  \n"
                f"**Measures:** {inst['measures']}  \n"
                f"**Source:** {inst['source']}  \n"
                f"**Status:** {inst['status']}  \n"
                f"**Reported reliability:** {inst['reported_alpha']}"
            )

    st.divider()
    st.markdown("### Psychometric demonstration (synthetic data)")
    info_box(
        "The values below are computed live on **synthetic** responses. "
        "Their purpose is to demonstrate that the analysis pipeline works, "
        "not to report real psychometric findings."
    )

    survey = synth_survey()

    # Split into instrument blocks
    atm_items = survey[[c for c in survey.columns if c.startswith("atm")]]
    ribs_items = survey[[c for c in survey.columns if c.startswith("ribs")]]
    psosh_items = survey[[c for c in survey.columns if c.startswith("psosh")]]
    mhsas_items = survey[[c for c in survey.columns if c.startswith("mhsas")]]

    blocks = [
        ("ATMHPS-AZ", atm_items),
        ("RIBS", ribs_items),
        ("PSOSH", psosh_items),
        ("MHSAS", mhsas_items),
    ]

    rows = []
    for name, block in blocks:
        a = cronbach_alpha(block)
        w = mcdonald_omega_approx(block)
        rows.append({
            "Instrument": name,
            "Items": block.shape[1],
            "N (complete)": int(block.dropna().shape[0]),
            "Cronbach's α": a,
            "McDonald's ω (approx.)": w,
            "α interpretation": interpret_alpha(a),
        })
    rel_df = pd.DataFrame(rows)

    st.dataframe(
        rel_df.style.format({
            "Cronbach's α": "{:.3f}",
            "McDonald's ω (approx.)": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Cronbach's α and approximate McDonald's ω are computed live. "
        "For publication, McDonald's ω should be estimated from a CFA-fitted "
        "model (e.g., `semopy` or `lavaan`) with a pre-registered factor structure."
    )

    st.divider()
    st.markdown("### Composite score distributions (synthetic)")

    comp = pd.DataFrame({
        "ATMHPS total (mean)": atm_items.mean(axis=1),
        "RIBS total (mean)": ribs_items.mean(axis=1),
        "PSOSH total (mean)": psosh_items.mean(axis=1),
        "MHSAS total (mean)": mhsas_items.mean(axis=1),
        "Group": survey["group"],
    })

    tab1, tab2, tab3, tab4 = st.tabs(["ATMHPS", "RIBS", "PSOSH", "MHSAS"])
    for tab, col in zip(
        [tab1, tab2, tab3, tab4],
        ["ATMHPS total (mean)", "RIBS total (mean)", "PSOSH total (mean)", "MHSAS total (mean)"],
    ):
        with tab:
            fig = px.histogram(
                comp, x=col, color="Group", nbins=20,
                color_discrete_map={"Intervention": BRAND["teal"], "Control": BRAND["amber"]},
                barmode="overlay", opacity=0.75,
            )
            fig.update_layout(xaxis_title=col, yaxis_title="count")
            st.plotly_chart(style_plotly(fig, 320), use_container_width=True)

    st.divider()
    st.markdown("### Data dictionary")
    st.markdown(
        "| Variable | Type | Description |\n"
        "|---|---|---|\n"
        "| `response_id` | string | Anonymous unique identifier |\n"
        "| `age_group` | category | 18–21, 22–25, 26–29 |\n"
        "| `education` | category | Undergraduate / Graduate / Other |\n"
        "| `region` | category | Broad geographic region |\n"
        "| `group` | category | Intervention / Control |\n"
        "| `survey_wave` | category | T0 / T1 / T2 |\n"
        "| `atm01`–`atm35` | int | ATMHPS-AZ items (0–4; check reverse-coding) |\n"
        "| `ribs01`–`ribs08` | int | RIBS items (1–5) |\n"
        "| `psosh01`–`psosh05` | int | PSOSH items (1–5; higher = more anticipated stigma) |\n"
        "| `mhsas01`–`mhsas09` | int | MHSAS items (1–7; higher = more favourable) |"
    )


def render_intervention() -> None:
    st.title("Module C — Intervention Evaluation")
    synth_banner()

    st.markdown(
        "The intervention is evaluated with a **two-arm design** with T0 (baseline), "
        "T1 (immediately post), and T2 (4-week follow-up). The primary outcome is "
        "**anticipated stigma (PSOSH)**; secondary outcomes include RIBS, MHSAS, and "
        "ATMHPS subscales."
    )

    st.markdown("### Design")
    col1, col2 = st.columns(2)
    with col1:
        success_box(
            "<strong>Intervention arm</strong><br>"
            "• Lived-experience short film (5–8 min) + 60–100 s clips<br>"
            "• 3-episode co-designed podcast<br>"
            "• Signposting to validated mental health resources"
        )
    with col2:
        info_box(
            "<strong>Control arm</strong><br>"
            "• Neutral, fact-based information leaflet<br>"
            "• Signposting to the same resources<br>"
            "• Same duration and format to balance contact time"
        )

    st.divider()
    st.markdown("### Power analysis (interactive)")
    st.markdown(
        "We pre-specify the target effect size and required power. The sample-size "
        "calculator uses a two-sample t-test approximation."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        d = st.slider("Expected effect size (Cohen's d)", 0.10, 0.80, 0.30, 0.05)
    with col2:
        power_target = st.slider("Target power", 0.70, 0.95, 0.80, 0.01)
    with col3:
        attrition = st.slider("Attrition (to inflate N)", 0.0, 0.35, 0.15, 0.01)

    n_per_arm = sample_size_two_arm(d, alpha=0.05, power=power_target)
    n_total_raw = n_per_arm * 2
    n_total_inflated = int(math.ceil(n_total_raw / (1 - attrition)))

    col1, col2, col3 = st.columns(3)
    with col1:
        stat_card("Per arm", f"{n_per_arm}", "primary analysis sample")
    with col2:
        stat_card("Total (raw)", f"{n_total_raw}", "before attrition")
    with col3:
        stat_card("Total (inflated)", f"{n_total_inflated}", f"+{int(attrition*100)}% attrition")

    ns = np.arange(30, max(500, n_per_arm * 2), 20)
    curve = power_curve(d, ns)
    fig = px.line(curve, x="n_per_arm", y="power", markers=False)
    fig.add_hline(y=power_target, line_dash="dash", line_color=BRAND["amber"],
                  annotation_text=f"Target power = {power_target:.2f}")
    fig.add_vline(x=n_per_arm, line_dash="dash", line_color=BRAND["teal"],
                  annotation_text=f"n = {n_per_arm}")
    fig.update_layout(xaxis_title="n per arm", yaxis_title="power",
                      yaxis_range=[0, 1.02])
    st.plotly_chart(style_plotly(fig, 380), use_container_width=True)

    st.caption(
        "For a definitive sample-size calculation, use G*Power or an R-based "
        "approach (`pwr::pwr.t.test`) with the primary outcome, ICC (if clustered), "
        "and expected attrition clearly specified."
    )

    st.divider()
    st.markdown("### Pre/post / follow-up analysis (synthetic demo)")
    df = synth_intervention()

    fig = go.Figure()
    for grp, color in [("Intervention", BRAND["teal"]), ("Control", BRAND["amber"])]:
        sub = df[df["group"] == grp]
        means = [sub["psosh_baseline"].mean(), sub["psosh_post"].mean(), sub["psosh_followup"].mean()]
        sems = [sub["psosh_baseline"].sem(), sub["psosh_post"].sem(), sub["psosh_followup"].sem()]
        fig.add_trace(go.Scatter(
            x=["T0", "T1", "T2"], y=means, mode="lines+markers", name=grp,
            line=dict(color=color, width=3),
            marker=dict(size=10),
            error_y=dict(type="data", array=sems, visible=True),
        ))
    fig.update_layout(yaxis_title="PSOSH mean (higher = more anticipated stigma)",
                      xaxis_title="Measurement wave")
    st.plotly_chart(style_plotly(fig, 380), use_container_width=True)

    # Paired effect within intervention arm
    int_arm = df[df["group"] == "Intervention"]
    d_z = paired_cohens_d(int_arm["psosh_baseline"], int_arm["psosh_post"])
    t_stat, p_val = stats.ttest_rel(int_arm["psosh_baseline"], int_arm["psosh_post"])

    col1, col2, col3 = st.columns(3)
    with col1:
        stat_card("Paired t", f"{t_stat:.2f}", f"p = {p_val:.4f}")
    with col2:
        stat_card("Cohen's d_z", f"{d_z:.2f}" if d_z is not None else "—",
                  interpret_d(d_z))
    with col3:
        stat_card("N (intervention)", f"{len(int_arm)}", "synthetic")

    st.caption(
        "This is a within-arm pre/post test on synthetic data. In the real "
        "analysis, the primary test is an ANCOVA comparing arms at T1 with T0 "
        "as a covariate, per the pre-registered analysis plan."
    )


def render_stats_lab() -> None:
    st.title("Statistical Analysis Lab")
    synth_banner()

    st.markdown(
        "An interactive workspace for verifying each statistical primitive used "
        "elsewhere in the platform."
    )

    tabs = st.tabs(["Reliability", "Effect size", "Inter-rater", "Correlation", "Power"])

    with tabs[0]:
        st.markdown("#### Cronbach's α and McDonald's ω")
        st.markdown("Enter a small matrix of item responses (rows = participants).")
        default_data = pd.DataFrame(
            [[3, 4, 3, 5], [2, 3, 4, 3], [4, 4, 5, 4], [3, 2, 3, 3], [5, 5, 4, 5]],
            columns=["Item1", "Item2", "Item3", "Item4"],
        )
        edited = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
        a = cronbach_alpha(edited)
        w = mcdonald_omega_approx(edited)
        col1, col2 = st.columns(2)
        with col1:
            stat_card("Cronbach's α", f"{a:.3f}" if a is not None else "—",
                      interpret_alpha(a))
        with col2:
            stat_card("McDonald's ω (approx.)", f"{w:.3f}" if w is not None else "—",
                      "single-factor PCA approximation")

    with tabs[1]:
        st.markdown("#### Cohen's d — two independent groups")
        rng = np.random.default_rng(1)
        g1 = st.text_area("Group A", "4, 5, 3, 4, 5, 4, 3, 5", height=80)
        g2 = st.text_area("Group B", "3, 3, 4, 2, 3, 4, 3, 2", height=80)
        try:
            a_vals = [float(x) for x in g1.replace(",", " ").split()]
            b_vals = [float(x) for x in g2.replace(",", " ").split()]
            d = cohens_d(a_vals, b_vals)
            if d is not None:
                stat_card("Cohen's d", f"{d:.3f}", interpret_d(d))
            else:
                st.warning("Not enough data to compute d.")
        except ValueError:
            st.error("Please enter comma- or space-separated numbers.")

    with tabs[2]:
        st.markdown("#### Krippendorff's α (nominal)")
        st.markdown("Enter codings — rows = units, columns = coders. Blank = missing.")
        default_coding = pd.DataFrame({
            "Coder1": [1, 0, 1, 1, 0, 1],
            "Coder2": [1, 0, 0, 1, 0, 1],
        })
        coding = st.data_editor(default_coding, num_rows="dynamic", use_container_width=True)
        arr = coding.replace("", np.nan).astype(float).values
        alpha = krippendorff_alpha_nominal(arr)
        stat_card("Krippendorff's α", f"{alpha:.3f}" if alpha is not None else "—",
                  "target ≥ 0.75")

    with tabs[3]:
        st.markdown("#### Pearson correlation")
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 50)
        y = 0.5 * x + rng.normal(0, 1, 50)
        r, p = stats.pearsonr(x, y)
        stat_card("r", f"{r:.3f}", f"p = {p:.4g}  ·  N = 50 (simulated)")

    with tabs[4]:
        st.markdown("#### Power curve for a chosen d")
        d_choice = st.slider("Cohen's d", 0.10, 1.00, 0.30, 0.05, key="lab_power")
        ns = np.arange(20, 400, 20)
        curve = power_curve(d_choice, ns)
        fig = px.line(curve, x="n_per_arm", y="power")
        fig.add_hline(y=0.80, line_dash="dash", line_color=BRAND["amber"])
        fig.update_layout(xaxis_title="n per arm", yaxis_title="power",
                          yaxis_range=[0, 1.02])
        st.plotly_chart(style_plotly(fig, 340), use_container_width=True)


def render_results() -> None:
    st.title("Results — Reporting Template")
    info_box(
        "No real data have been collected at the time of this prototype. "
        "The layout below is the pre-registered reporting template. Every field "
        "marked '—' is intentional: the platform refuses to display empirical "
        "results without actual data."
    )

    st.markdown("### Study status")
    col1, col2, col3 = st.columns(3)
    with col1:
        stat_card("Participants", "—", "No real data yet")
    with col2:
        stat_card("Survey completion", "—", "No real data yet")
    with col3:
        stat_card("Intervention effect", "—", "Requires control arm data")

    st.divider()
    st.markdown("### What will be reported per outcome")
    st.markdown(
        "| Outcome | Instrument | Primary analysis | Effect size | 95% CI | p-value | Missing |\n"
        "|---|---|---|---|---|---|---|\n"
        "| Anticipated stigma | PSOSH | ANCOVA (T1 ~ arm + T0) | Cohen's d (adjusted) | yes | yes | yes |\n"
        "| Social distance | RIBS | ANCOVA (T1 ~ arm + T0) | Cohen's d | yes | yes | yes |\n"
        "| Help-seeking attitude | MHSAS | ANCOVA (T1 ~ arm + T0) | Cohen's d | yes | yes | yes |\n"
        "| Stigma attitudes | ATMHPS-AZ | ANCOVA (T1 ~ arm + T0) | Cohen's d | yes | yes | yes |"
    )

    st.divider()
    st.markdown("### Reporting rules (mandatory)")
    danger_box(
        "1. Never report p-values without N, effect size, and 95% CI.<br>"
        "2. Never interpret a cross-sectional correlation as causal.<br>"
        "3. Never report an intervention effect without a control arm.<br>"
        "4. Never hide missing data or drop non-completers silently.<br>"
        "5. Never run exploratory factor analysis on n < 100.<br>"
        "6. Never treat a preprint finding as peer-reviewed evidence."
    )

    st.divider()
    st.markdown("### Expected output artifacts")
    st.markdown(
        "- `analysis_summary.csv` — one row per outcome, columns: N, effect, 95% CI, p\n"
        "- `figures/forest_plot.png` — forest plot of primary and secondary outcomes\n"
        "- `figures/reliability.png` — reliability comparison per instrument\n"
        "- `research_report.json` — machine-readable, versioned research report"
    )


def render_evidence() -> None:
    st.title("Evidence Library")

    st.markdown(
        "The 10 priority references that anchor the methodology. "
        "Preprints and peer-reviewed sources are distinguished."
    )

    for ref in REFERENCES:
        st.markdown(
            f'<div class="ref-card">'
            f'<span class="num">{ref["n"]}</span>'
            f'<span class="title">{ref["title"]}</span>'
            f'<div class="authors">{ref["authors"]} ({ref["year"]})</div>'
            f'<div class="meta">{ref["meta"]}</div>'
            f'<div style="margin-top:0.7rem; font-size:0.9rem; color:#333; line-height:1.55;">'
            f'<strong>Key finding:</strong> {ref["key_finding"]}</div>'
            f'<span class="role">Role: {ref["role"]}</span> &nbsp; {evidence_badge(ref["level"])}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Preprints are explicitly marked. Presence in this library does not "
        "imply endorsement by a peer-reviewed outlet. Verify each reference "
        "independently before using it in a submission."
    )


def render_ethics() -> None:
    st.title("Ethics & Methodology")

    st.markdown("### What this project does")
    success_box(
        "• Uses validated instruments with published psychometrics<br>"
        "• Reports N, effect size, and 95% CI for every empirical claim<br>"
        "• Uses informed consent and anonymity for every participant<br>"
        "• Keeps data collection to the minimum required for the research question<br>"
        "• Labels all synthetic data as SYNTHETIC everywhere it appears<br>"
        "• Distinguishes preprints from peer-reviewed evidence"
    )

    st.markdown("### What this project does not do")
    danger_box(
        "• Does not diagnose any individual<br>"
        "• Does not display personal risk scores to participants<br>"
        "• Does not include PHQ-8 or any clinical screener in its core<br>"
        "• Does not fabricate indices such as 'silence_index'<br>"
        "• Does not present cross-sectional correlations as causal findings<br>"
        "• Does not claim intervention effects without a control arm<br>"
        "• Does not collect suicide-risk information without a formal safety protocol"
    )

    st.divider()
    st.markdown("### Data protection")
    st.markdown(
        "- Data are stored on encrypted servers, access-limited to named researchers.\n"
        "- Identifiable data (names, emails, phone) are not collected in the survey wave.\n"
        "- Interview transcripts are de-identified; identifiable details are removed.\n"
        "- Composite narratives are labeled `composite` — never presented as verbatim accounts.\n"
        "- Analysis scripts and data dictionaries are versioned."
    )

    st.divider()
    st.markdown("### Limitations (pre-registered)")
    st.markdown(
        "1. Pilot sample sizes are small; early-phase findings will be descriptive.\n"
        "2. Control arm is not matched on all covariates; ANCOVA-adjusted estimates.\n"
        "3. Media sample is limited to a defined set of outlets; not representative of all Azerbaijani media.\n"
        "4. Four-week follow-up captures short-term change only; long-term effects not measured.\n"
        "5. Cross-cultural validity of the instruments in the Azerbaijani context is documented but still evolving."
    )

    st.divider()
    st.markdown("### Versioning")
    st.markdown(
        f"- Platform version: **1.0** (scientific edition)\n"
        f"- Last updated: **{datetime.now().strftime('%d %b %Y')}**\n"
        "- Repository: see README in the project root\n"
        "- Pre-registration: template available in `/preregistration/`"
    )

    st.divider()
    st.markdown("### Contact")
    st.markdown(
        "📧 `research@neurometricgaze.example`  \n"
        "🔗 `github.com/your-handle/neurometric-gaze`  \n"
        "📍 Baku, Azerbaijan"
    )


# =============================================================================
# 10. SIDEBAR AND ROUTER
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.5rem 0;">
            <div style="color:#D6A56C; font-size:0.68rem; letter-spacing:0.25em; font-weight:600;">
                THE NEUROMETRIC GAZE
            </div>
            <div style="color:#F5F2EB; font-family:'Playfair Display',serif; font-size:1.05rem; margin-top:0.4rem;">
                Beyond the Unheard
            </div>
            <div style="color:rgba(245,242,235,0.55); font-size:0.72rem; margin-top:0.2rem;">
                Research prototype · v1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "❓ Research Question",
            "📰 Module A — Media Analysis",
            "📋 Module B — Youth Survey",
            "🎬 Module C — Intervention",
            "🧮 Statistical Analysis Lab",
            "📊 Results (Template)",
            "📚 Evidence Library",
            "⚖️ Ethics & Methodology",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:0.7rem; color:rgba(245,242,235,0.65); line-height:1.7;">
            <strong style="color:#D6A56C;">Status:</strong> Prototype<br>
            <strong style="color:#D6A56C;">Empirical data:</strong> None yet<br>
            <strong style="color:#D6A56C;">Updated:</strong> {datetime.now().strftime('%d %b %Y')}<br>
            <strong style="color:#D6A56C;">License:</strong> CC-BY-NC-4.0
        </div>
        """,
        unsafe_allow_html=True,
    )

if page == "🏠 Home":
    render_home()
elif page == "❓ Research Question":
    render_research_question()
elif page == "📰 Module A — Media Analysis":
    render_media_analysis()
elif page == "📋 Module B — Youth Survey":
    render_survey()
elif page == "🎬 Module C — Intervention":
    render_intervention()
elif page == "🧮 Statistical Analysis Lab":
    render_stats_lab()
elif page == "📊 Results (Template)":
    render_results()
elif page == "📚 Evidence Library":
    render_evidence()
elif page == "⚖️ Ethics & Methodology":
    render_ethics()
