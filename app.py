"""
THE NEUROMETRIC GAZE — Beyond the Unheard
=========================================================================
A Global Mental Health Storytelling, Research & Awareness Platform.

RESEARCH QUESTION
-----------------
How do media narratives shape mental health stigma, public understanding,
and help-seeking intentions across different cultures, and how can
evidence-based storytelling create positive change?

MISSION
-------
Reduce misconceptions about mental health, amplify lived experience, make
scientific evidence accessible, and encourage help-seeking — across
countries, languages, and cultures.

CORE SCIENTIFIC PRINCIPLES
--------------------------
1. All numbers are sourced or explicitly labeled SYNTHETIC.
2. Preprints are distinguished from peer-reviewed evidence.
3. Every statistical result includes N, effect size, and 95% CI.
4. Cross-cultural measurement requires invariance testing before
   comparison (Global Core + Local Context framework).
5. No clinical diagnosis, no individual risk scores, no PHQ-8.
6. Anonymity, informed consent, and data minimization are non-negotiable.
"""

from __future__ import annotations

import io
import math
import re
import warnings
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except Exception:
    sp_stats = None
    HAS_SCIPY = False

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# 1. PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="The Neurometric Gaze — Beyond the Unheard",
    page_icon="◈",
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
    "sidebar": "#ECE7DC",
}

RNG_SEED = 20260218


# =============================================================================
# 2. EDITORIAL DESIGN SYSTEM
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #F5F2EB;
        color: #14243A;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 4.0rem;
        max-width: 1280px;
    }
    h1, h2, h3, h4 {
        font-family: 'Instrument Serif', Georgia, serif;
        color: #14243A;
        font-weight: 400;
        letter-spacing: -0.3px;
    }
    h1 { font-size: 3.0rem; line-height: 1.07; margin-bottom: 0.4rem; }
    h2 { font-size: 2.0rem; line-height: 1.15; margin-top: 0.4rem; margin-bottom: 0.7rem; }
    h3 { font-size: 1.4rem; line-height: 1.25; }
    p, li, label, .stMarkdown { color: #1E314B; }

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
    .card-amber {
        background-color: #FFFFFF;
        border: 1px solid rgba(20, 36, 58, 0.13);
        border-left: 4px solid #D6A56C;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.0rem;
    }
    .card-teal {
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
        padding: 1rem 1.3rem;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        color: #14243A;
        margin: 0.9rem 0 1.1rem 0;
        font-family: 'Instrument Serif', serif;
        font-size: 1.15rem;
    }
    .tag-teal {
        display: inline-block;
        background-color: rgba(92, 133, 129, 0.15);
        color: #14243A;
        border: 1px solid rgba(92, 133, 129, 0.38);
        border-radius: 999px;
        padding: 0.18rem 0.65rem;
        font-size: 0.74rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .tag-amber {
        display: inline-block;
        background-color: rgba(214, 165, 108, 0.24);
        color: #14243A;
        border: 1px solid rgba(214, 165, 108, 0.58);
        border-radius: 999px;
        padding: 0.18rem 0.65rem;
        font-size: 0.74rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .info-box {
        background: #EBE6DC;
        border-left: 4px solid #D6A56C;
        padding: 0.95rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.93rem;
        line-height: 1.6;
    }
    .danger-box {
        background: #F8E5E3;
        border-left: 4px solid #B85450;
        padding: 0.95rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.93rem;
        line-height: 1.6;
    }
    .success-box {
        background: #E6F0EE;
        border-left: 4px solid #5C8581;
        padding: 0.95rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.93rem;
        line-height: 1.6;
    }
    .hero {
        text-align: center;
        padding: 3rem 1rem 2.5rem 1rem;
        border-bottom: 1px solid #EBE6DC;
        margin-bottom: 2rem;
    }
    .hero-kicker {
        color: #5C8581;
        letter-spacing: 0.3em;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .hero-title {
        font-family: 'Instrument Serif', serif;
        font-size: 4rem;
        line-height: 1.05;
        margin: 1rem 0;
        color: #14243A;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #5C8581;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .story-card {
        background: #FFFFFF;
        border-left: 4px solid #5C8581;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(20, 36, 58, 0.04);
    }
    .story-card.amber {
        border-left-color: #D6A56C;
    }
    .stButton > button {
        background-color: #14243A;
        color: #F5F2EB;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.3rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #5C8581;
        color: white;
    }
    hr { border-color: #EBE6DC; }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 3. SESSION STATE
# =============================================================================

def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "quiz_answers": {},
        "quiz_completed": False,
        "quiz_results": None,
        "user_stories": [],
        "story_draft": "",
        "last_story_analysis": None,
        "coding_index": 0,
        "coding_answers": {},
        "coding_completed": False,
        "intervention_before": None,
        "intervention_after": None,
        "intervention_stage": "before",
        "intervention_choice": None,
        "current_section": "01 · Home",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()


# =============================================================================
# 4. STATISTICAL FUNCTIONS (REAL IMPLEMENTATIONS)
# =============================================================================

def cronbach_alpha(df_items: pd.DataFrame) -> Tuple[Optional[float], int, int]:
    """Cronbach's alpha on complete rows of numeric scale items."""
    num = df_items.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    k, n = num.shape[1], num.shape[0]
    if k < 2 or n < 3:
        return None, n, k
    item_vars = num.var(axis=0, ddof=1)
    total_var = num.sum(axis=1).var(ddof=1)
    if total_var <= 0 or np.isnan(total_var):
        return None, n, k
    alpha = (k / (k - 1.0)) * (1.0 - float(item_vars.sum()) / float(total_var))
    return round(float(alpha), 4), int(n), int(k)


def mcdonald_omega_approx(df_items: pd.DataFrame) -> Optional[float]:
    """Approximate McDonald's omega via single-factor PCA (demonstration only)."""
    num = df_items.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if num.shape[1] < 3 or num.shape[0] < 5:
        return None
    corr = num.corr().values
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    loadings = eigvecs[:, 0] * np.sqrt(max(eigvals[0], 0))
    sum_l = loadings.sum()
    sum_u = (1 - loadings**2).sum()
    denom = sum_l**2 + sum_u
    return round(float((sum_l**2) / denom), 4) if denom > 0 else None


def cohens_d_ci(g1: np.ndarray, g2: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Cohen's d for two independent groups with 95% normal-approx CI."""
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return None, None, None
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled <= 0:
        return None, None, None
    d = float((np.mean(g1) - np.mean(g2)) / pooled)
    se = math.sqrt((n1 + n2) / (n1 * n2) + (d ** 2) / (2.0 * (n1 + n2)))
    return round(d, 4), round(d - 1.96 * se, 4), round(d + 1.96 * se, 4)


def paired_cohens_d(before: np.ndarray, after: np.ndarray) -> Optional[float]:
    """Cohen's dz for paired samples."""
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    mask = ~(np.isnan(b) | np.isnan(a))
    b, a = b[mask], a[mask]
    if len(b) < 2:
        return None
    diff = a - b
    sd = diff.std(ddof=1)
    return round(float(diff.mean() / sd), 4) if sd > 0 else None


def cohens_kappa(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    """Cohen's kappa from paired categorical ratings."""
    df = pd.DataFrame({"a": a.astype(str).str.strip(), "b": b.astype(str).str.strip()})
    df = df[(df["a"] != "") & (df["b"] != "")]
    n = len(df)
    if n < 2:
        return {"available": False, "n": n, "kappa": None, "po": None, "reason": "Need ≥ 2 paired ratings."}
    cats = sorted(set(df["a"]).union(set(df["b"])))
    po = float((df["a"] == df["b"]).mean())
    if len(cats) < 2:
        return {"available": True, "n": n, "kappa": None, "po": round(po, 4), "reason": "Single category present."}
    pe = sum(float((df["a"] == c).mean()) * float((df["b"] == c).mean()) for c in cats)
    denom = 1.0 - pe
    kappa = (po - pe) / denom if abs(denom) > 1e-12 else None
    return {
        "available": True, "n": n,
        "kappa": round(kappa, 4) if kappa is not None else None,
        "po": round(po, 4), "pe": round(pe, 4),
        "reason": "Computed from paired ratings.",
    }


def krippendorff_alpha_nominal(data: np.ndarray) -> Optional[float]:
    """Krippendorff's α for nominal data (units × coders; NaN = missing)."""
    data = np.asarray(data, dtype=float)
    values = sorted({v for row in data for v in row if not np.isnan(v)})
    if len(values) < 2:
        return 1.0
    idx_map = {v: i for i, v in enumerate(values)}
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
                    coincidence[idx_map[v1], idx_map[v2]] += 1 / (m - 1)
    n_c = coincidence.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return None
    observed = (coincidence.sum() - np.trace(coincidence)) / n
    expected = 0.0
    for c in range(V):
        for k in range(V):
            if c != k:
                expected += n_c[c] * n_c[k]
    expected = expected / (n * (n - 1))
    if expected == 0:
        return 1.0
    return round(float(1 - observed / expected), 4)


def welch_ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if not HAS_SCIPY or len(a) < 2 or len(b) < 2:
        return {"available": False}
    t, p = sp_stats.ttest_ind(a, b, equal_var=False)
    return {"available": True, "t": round(float(t), 4), "p": round(float(p), 4)}


def mann_whitney(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if not HAS_SCIPY or len(a) < 2 or len(b) < 2:
        return {"available": False}
    try:
        u, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"available": True, "U": round(float(u), 2), "p": round(float(p), 4)}
    except ValueError:
        return {"available": False}


def power_curve(d: float, ns: List[int], alpha: float = 0.05) -> pd.DataFrame:
    z_a = sp_stats.norm.ppf(1 - alpha / 2) if HAS_SCIPY else 1.96
    rows = []
    for n in ns:
        ncp = d * math.sqrt(n / 2)
        if HAS_SCIPY:
            power = 1 - sp_stats.norm.cdf(z_a - ncp) + sp_stats.norm.cdf(-z_a - ncp)
        else:
            power = float("nan")
        rows.append({"n_per_arm": n, "power": float(power)})
    return pd.DataFrame(rows)


def sample_size_two_arm(d: float, alpha: float = 0.05, power: float = 0.80) -> int:
    if not HAS_SCIPY:
        return 0
    z_a = sp_stats.norm.ppf(1 - alpha / 2)
    z_b = sp_stats.norm.ppf(power)
    return int(math.ceil(((z_a + z_b) ** 2) * 2 / (d ** 2)))


def apply_reverse_scoring(df: pd.DataFrame, cols: List[str], smin: int, smax: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (smin + smax) - pd.to_numeric(out[c], errors="coerce")
    return out


def interpret_d(d: Optional[float]) -> str:
    if d is None:
        return "—"
    a = abs(d)
    if a < 0.2: return "trivial"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def interpret_alpha(a: Optional[float]) -> str:
    if a is None: return "—"
    if a < 0.5: return "unacceptable"
    if a < 0.6: return "poor"
    if a < 0.7: return "questionable"
    if a < 0.8: return "acceptable"
    if a < 0.9: return "good"
    return "excellent"


# =============================================================================
# 5. HELPERS
# =============================================================================

def safe_str(val: Any) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(val)).strip()


def load_csv_bytes(raw: bytes, label: str = "upload.csv") -> pd.DataFrame:
    last_err = ""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as exc:
            last_err = f"{enc}: {exc}"
    raise ValueError(f"Could not read {label} ({last_err}).")


def info_box(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def danger_box(text: str) -> None:
    st.markdown(f'<div class="danger-box">{text}</div>', unsafe_allow_html=True)


def success_box(text: str) -> None:
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)


def pull_quote(text: str) -> None:
    st.markdown(f'<div class="pull-quote">“{text}”</div>', unsafe_allow_html=True)


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


def synth_banner(text: str = "SYNTHETIC / ILLUSTRATIVE DATA — not real participants") -> None:
    st.markdown(
        f'<div style="background:#D6A56C; color:white; padding:0.65rem 1.1rem; '
        f'border-radius:8px; font-size:0.84rem; font-weight:600; margin:1rem 0;">⚠ {text}</div>',
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")


# =============================================================================
# 6. GLOBAL CONSTANTS
# =============================================================================

INSTRUMENTS = [
    {
        "code": "MHLS",
        "name": "Mental Health Literacy Scale",
        "items": 35, "factors": 1,
        "measures": "Knowledge and recognition of mental health problems",
        "source": "O'Connor & Casey (2015)",
        "status": "Peer-reviewed",
        "alpha": "See original",
    },
    {
        "code": "RIBS",
        "name": "Reported and Intended Behaviour Scale",
        "items": 8, "factors": 2,
        "measures": "Reported and intended social behaviour toward people with mental illness",
        "source": "Evans-Lacko et al. (2011)",
        "status": "Peer-reviewed",
        "alpha": "See original",
    },
    {
        "code": "PSOSH",
        "name": "Perceptions of Stigmatization by Others for Seeking Help",
        "items": 5, "factors": 1,
        "measures": "Anticipated stigmatization from others when seeking help",
        "source": "Vogel et al. (2009)",
        "status": "Peer-reviewed",
        "alpha": "0.91 (reported)",
    },
    {
        "code": "MHSAS",
        "name": "Mental Help Seeking Attitudes Scale",
        "items": 9, "factors": 1,
        "measures": "General attitudes toward seeking professional help",
        "source": "Hammer et al. (2018)",
        "status": "Peer-reviewed",
        "alpha": "See original",
    },
    {
        "code": "ISMI",
        "name": "Internalized Stigma of Mental Illness Scale",
        "items": 29, "factors": 5,
        "measures": "Internalized stigma (alienation, stereotype endorsement, discrimination, withdrawal, resistance)",
        "source": "Boyd Ritsher et al. (2003)",
        "status": "Peer-reviewed",
        "alpha": "See original",
    },
    {
        "code": "ATMHPS-AZ",
        "name": "Attitudes Towards Mental Health Problems Scale (Azerbaijani)",
        "items": 35, "factors": 7,
        "measures": "Regional adaptation example — Azerbaijani context",
        "source": "Shahhuseynbayova et al. (2026)",
        "status": "OSF PREPRINT — verify status before use",
        "alpha": "0.954 (reported)",
    },
]

MEDIA_CODEBOOK = [
    {"code": "M01", "name": "Violence frame", "focus": "Stigma", "definition": "Associates mental health with violence or aggression."},
    {"code": "M02", "name": "Dangerousness", "focus": "Stigma", "definition": "Portrays individuals as dangerous or unpredictable."},
    {"code": "M03", "name": "Blame / weakness", "focus": "Stigma", "definition": "Frames distress as personal weakness or moral failure."},
    {"code": "M04", "name": "Sensationalism", "focus": "Style", "definition": "Uses alarmist, exaggerated, or clickbait framing."},
    {"code": "M05", "name": "Recovery", "focus": "Anti-stigma", "definition": "Depicts recovery, management, or constructive adaptation."},
    {"code": "M06", "name": "Human experience", "focus": "Anti-stigma", "definition": "Centers lived experience and personal voice."},
    {"code": "M07", "name": "Agency", "focus": "Anti-stigma", "definition": "Shows individuals exercising agency and choice."},
    {"code": "M08", "name": "Help-seeking", "focus": "Help-seeking", "definition": "Provides realistic information about support options."},
    {"code": "M09", "name": "Support", "focus": "Help-seeking", "definition": "Highlights family, peer, or institutional support."},
    {"code": "M10", "name": "Respectful language", "focus": "Anti-stigma", "definition": "Uses respectful, person-centered terminology."},
]

EVIDENCE_LIBRARY = [
    {
        "n": 1,
        "title": "Interventions to Reduce Mental Health Stigma in Young People",
        "authors": "Crockett, M. A., et al.",
        "year": 2025,
        "meta": "Systematic review · 97 RCTs · 43,852 participants (mean age 18.7)",
        "role": "Global evidence base for youth stigma interventions",
        "level": "high",
        "key_finding": "Contact-based and lived-experience interventions outperform information-only approaches. Stigma knowledge (SMD 0.66), attitudes (0.38), behaviour (0.29), help-seeking intention (0.14).",
    },
    {
        "n": 2,
        "title": "Does Storytelling Reduce Stigma? A Meta-Analytic View of Narrative Persuasion",
        "authors": "Multiple authors",
        "year": 2022,
        "meta": "Meta-analysis · 46 effect sizes from 40 articles",
        "role": "Direct evidence for narrative intervention",
        "level": "high",
        "key_finding": "Narratives produce a small but positive reduction in stigma. First-person narratives are more effective than third-person accounts.",
    },
    {
        "n": 3,
        "title": "Digital Empowerment in Mental Health: Meta-Analysis of Digital Interventions",
        "authors": "Multiple authors",
        "year": 2024,
        "meta": "Meta-analysis · 29 studies · 11,582 participants",
        "role": "Evidence for digital platforms",
        "level": "high",
        "key_finding": "Knowledge improves post-test (Hedges' g = 0.459) and at follow-up (0.487); stigma decreases (−0.332); help-seeking attitudes (0.168) and intentions (0.135) show small effects. Gains may not persist without reinforcement.",
    },
    {
        "n": 4,
        "title": "The Picture of Mental Health/Illness in Printed Media in Three Central European Countries",
        "authors": "PICMIN Consortium",
        "year": 2012,
        "meta": "Content analysis · 450 articles · 3 countries",
        "role": "Cross-cultural media framing evidence",
        "level": "high",
        "key_finding": "Over half of articles contained stigmatizing framing. Psychosis coverage more stigmatizing than other conditions. Significant cross-country differences — a message that works in one country may not work in another.",
    },
    {
        "n": 5,
        "title": "Co-Design of Baatcheet: A Peer-Supported Web-Based Storytelling Platform (India)",
        "authors": "Multiple authors",
        "year": 2024,
        "meta": "Co-design study · Young People's Advisory Group (n=11) · stakeholder group (n=20) · focus groups (n=42)",
        "role": "Youth co-design methodology",
        "level": "med",
        "key_finding": "Documents the co-design process for a web-based storytelling platform with peer support for youth 16–24. Does not prove effectiveness but provides a strong methodological template.",
    },
    {
        "n": 6,
        "title": "Selfie Videos to Reduce Stigma and Increase Treatment Seeking Among Youths",
        "authors": "Amsalem, D., et al.",
        "year": 2023,
        "meta": "Two RCTs · N = 1,532 · 58–102 s videos",
        "role": "Short-form lived-experience video design",
        "level": "high",
        "key_finding": "Cohen's d = 0.31–0.76 (Study 1) and 0.13–0.47 (Study 2) for stigma outcomes.",
    },
    {
        "n": 7,
        "title": "Perceived Barriers and Facilitators to Mental Health Help-Seeking",
        "authors": "Gulliver, A., Griffiths, K. M., Christensen, H.",
        "year": 2010,
        "meta": "Systematic review · 22 studies",
        "role": "Help-seeking barrier taxonomy",
        "level": "high",
        "key_finding": "Top barriers: stigma, low symptom recognition, self-reliance preference, uncertainty about sources. Facilitators: social support, encouragement, prior positive experiences.",
    },
    {
        "n": 8,
        "title": "Effect of Web-Based Educational Interventions on Mental Health Literacy, Stigma, Help-Seeking",
        "authors": "Nazari, A., et al.",
        "year": 2023,
        "meta": "Meta-analysis · 20 studies (10 pooled)",
        "role": "Baseline for what information-only achieves",
        "level": "high",
        "key_finding": "Literacy improves (SMD 0.70); stigma (SMD −0.20) and help-seeking (SMD 0.48) 95% CIs cross zero — knowledge alone is insufficient.",
    },
    {
        "n": 9,
        "title": "PICMIN Media Coding Instrument (Picture of Mental Illness in Newspapers)",
        "authors": "PICMIN Consortium",
        "year": 2012,
        "meta": "Codebook · Krippendorff's α > 0.75 across 622 articles",
        "role": "Media content coding framework",
        "level": "high",
        "key_finding": "10 analytic categories with strong inter-coder reliability — the basis for the M01–M10 codebook in this platform.",
    },
    {
        "n": 10,
        "title": "Co-Designed Podcast for Reducing Stigma",
        "authors": "Carrotte, E. R., et al.",
        "year": 2022,
        "meta": "Co-design study",
        "role": "Podcast format design",
        "level": "med",
        "key_finding": "Co-designed lived-experience podcasts show strong acceptability and perceived impact. Format matters as much as content.",
    },
    {
        "n": 11,
        "title": "Mass Media Interventions for Reducing Mental Health-Related Stigma",
        "authors": "Clement, S., et al.",
        "year": 2013,
        "meta": "Systematic review",
        "role": "Media intervention evidence base",
        "level": "high",
        "key_finding": "Mixed evidence overall; effects strongest for targeted, contact-based media rather than broad awareness campaigns.",
    },
    {
        "n": 12,
        "title": "Psychometric Validation of the Azerbaijani ATMHPS (n = 946)",
        "authors": "Shahhuseynbayova, N., et al.",
        "year": 2026,
        "meta": "OSF preprint · N = 946 Azerbaijani university students",
        "role": "Regional adaptation example",
        "level": "med",
        "key_finding": "Reported 7-factor structure (α = 0.954, ω = 0.956). Highest subscale: disclosure concerns. Demonstrates how a global instrument can be localized — but preprint status requires verification.",
    },
]

QUIZ_ITEMS = [
    {"id": "q1", "text": "I hesitate to tell others when I am struggling emotionally.", "dim": "disclosure"},
    {"id": "q2", "text": "If I saw a therapist, my family might not accept it.", "dim": "family"},
    {"id": "q3", "text": "I would be comfortable working with someone who has a mental health condition.", "dim": "social_distance", "reverse": True},
    {"id": "q4", "text": "Seeking psychological help is a sign of weakness.", "dim": "help_attitude", "reverse": True},
    {"id": "q5", "text": "I feel I know enough about mental health conditions.", "dim": "literacy", "reverse": True},
    {"id": "q6", "text": "Media mostly presents mental health in a negative light.", "dim": "media_awareness"},
    {"id": "q7", "text": "Society is prejudiced against people with mental health conditions.", "dim": "media_awareness"},
    {"id": "q8", "text": "If I struggled, I would seek professional help.", "dim": "help_attitude", "reverse": True},
]

QUIZ_SCALE = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
QUIZ_VALUES = [1, 2, 3, 4, 5]

QUIZ_DIMENSIONS = {
    "disclosure": "Disclosure concern",
    "family": "Family concern",
    "social_distance": "Social distance",
    "help_attitude": "Help-seeking attitude",
    "literacy": "Mental health literacy",
    "media_awareness": "Media awareness",
}

STORY_THEME_KEYWORDS = {
    "Silence / Disclosure": ["silence", "silent", "didn't tell", "never told", "kept quiet", "hid", "hide", "couldn't say", "withheld", "secret", "afraid to say"],
    "Family pressure": ["family", "father", "mother", "parents", "brother", "sister", "home", "my mom", "my dad", "household"],
    "Stigma": ["stigma", "shame", "ashamed", "judged", "laughed", "mocked", "society", "rejected", "humiliated", "disgrace", "weak"],
    "Help-seeking": ["therapist", "therapy", "help", "psychologist", "psychiatrist", "counselor", "clinic", "support", "treatment", "doctor", "reached out"],
    "Media influence": ["social media", "instagram", "facebook", "news", "headline", "post", "story", "media", "tiktok", "youtube", "twitter"],
    "Work / School": ["work", "job", "university", "college", "school", "exam", "stress", "burnout", "focus", "class", "teacher", "boss"],
    "Physical symptoms": ["sleep", "can't sleep", "appetite", "headache", "tired", "exhausted", "energy", "pain", "heart", "breath", "panic"],
    "Hope / Recovery": ["better", "recovered", "recovery", "hope", "improved", "support", "started", "light", "safe", "healing", "forward"],
}

POSITIVE_WORDS = ["good", "happy", "hope", "love", "support", "recovered", "strong", "light", "solution", "calm", "safe", "grateful", "better", "healing"]
NEGATIVE_WORDS = ["bad", "pain", "fear", "afraid", "alone", "lonely", "hard", "heavy", "anxious", "sad", "hopeless", "tired", "shame", "stuck"]

CODING_HEADLINES = [
    {
        "id": "h1",
        "title": "“Schizophrenic attacks neighbor — is anyone safe?”",
        "source": "Example headline (English translation)",
        "gold": {"M01": 1, "M02": 1, "M03": 0, "M04": 1, "M05": 0, "M06": 0, "M07": 0, "M08": 0, "M09": 0, "M10": 0},
        "explanation": "Violence + dangerousness + sensationalism. No personal voice, no recovery narrative.",
    },
    {
        "id": "h2",
        "title": "“Living with depression: ‘Asking for help is courage’”",
        "source": "Example headline (English translation)",
        "gold": {"M01": 0, "M02": 0, "M03": 0, "M04": 0, "M05": 1, "M06": 1, "M07": 1, "M08": 1, "M09": 0, "M10": 1},
        "explanation": "Personal voice, help-seeking encouraged, respectful language.",
    },
    {
        "id": "h3",
        "title": "“Stressed youth told: ‘Just pull yourself together’”",
        "source": "Example headline (English translation)",
        "gold": {"M01": 0, "M02": 0, "M03": 1, "M04": 0, "M05": 0, "M06": 0, "M07": 0, "M08": 0, "M09": 0, "M10": 0},
        "explanation": "Problem framed as personal weakness (blame frame).",
    },
    {
        "id": "h4",
        "title": "“University opens mental health support center for students”",
        "source": "Example headline (English translation)",
        "gold": {"M01": 0, "M02": 0, "M03": 0, "M04": 0, "M05": 1, "M06": 0, "M07": 0, "M08": 1, "M09": 1, "M10": 1},
        "explanation": "Recovery, help-seeking, and support frames.",
    },
    {
        "id": "h5",
        "title": "“Famous actor dies by suicide: family declines to comment”",
        "source": "Example headline (English translation)",
        "gold": {"M01": 0, "M02": 0, "M03": 0, "M04": 1, "M05": 0, "M06": 0, "M07": 0, "M08": 0, "M09": 0, "M10": 0},
        "explanation": "Sensationalism; no personal voice; no help resources mentioned. Suicide reporting requires specific media guidelines.",
    },
    {
        "id": "h6",
        "title": "“Panic attack survivor: ‘If I had support, I wouldn't have been so scared’”",
        "source": "Example headline (English translation)",
        "gold": {"M01": 0, "M02": 0, "M03": 0, "M04": 0, "M05": 1, "M06": 1, "M07": 1, "M08": 0, "M09": 1, "M10": 1},
        "explanation": "Personal voice, recovery, support — respectful language.",
    },
    {
        "id": "h7",
        "title": "“Anxiety rising among youth: what do experts say?”",
        "source": "Example headline (English translation)",
        "gold": {"M01": 0, "M02": 0, "M03": 0, "M04": 0, "M05": 0, "M06": 0, "M07": 0, "M08": 1, "M09": 0, "M10": 1},
        "explanation": "Balanced tone, help-seeking encouraged, respectful language.",
    },
    {
        "id": "h8",
        "title": "“Mentally ill man causes another accident — is the danger ongoing?”",
        "source": "Example headline (English translation)",
        "gold": {"M01": 1, "M02": 1, "M03": 0, "M04": 1, "M05": 0, "M06": 0, "M07": 0, "M08": 0, "M09": 0, "M10": 0},
        "explanation": "Violence, dangerousness, sensationalism, stigmatizing language.",
    },
]

COMPOSITE_STORIES = [
    {
        "title": "The silence at home",
        "theme": "Silence / Disclosure",
        "label": "Composite narrative — not a real individual",
        "text": "I never told my family. For them, it was just ‘being moody’. My grades were fine, but I couldn't sleep at night. One day a teacher asked, ‘Are you okay?’ I didn't answer. A year later, I learned the university had a counseling service. After my first appointment, I realized I had been carrying this alone for years.",
    },
    {
        "title": "Stigma at work",
        "theme": "Stigma",
        "label": "Composite narrative — not a real individual",
        "text": "When I admitted I was burned out, my manager said, ‘It's just stress, take a break.’ But it wasn't just stress. I couldn't sleep for weeks, I lost focus. One day I wrote to HR. They connected me with a counselor. My family still doesn't know — I'm afraid they'd call me ‘weak’.",
    },
    {
        "title": "Media and self-recognition",
        "theme": "Media influence",
        "label": "Composite narrative — not a real individual",
        "text": "On social media I shared ‘positive vibes only’, but at night I couldn't sleep. One day an acquaintance shared her own story — she had felt the same way. For the first time I realized I wasn't alone. That post was the biggest support I had ever received. Eventually, I started speaking out too.",
    },
    {
        "title": "After telling my mother",
        "theme": "Family pressure",
        "label": "Composite narrative — not a real individual",
        "text": "My father used to say, ‘Men don't cry’. So I learned not to cry — inside. At 26 I told my mother for the first time that I needed help. She was silent. A week later she called: ‘Let's go somewhere together.’ That day I opened a therapist's door for the first time. My father still doesn't know, but my mother does — and for me, that's enough.",
    },
]


# =============================================================================
# 7. SYNTHETIC DATA
# =============================================================================

def get_synthetic_survey_df() -> pd.DataFrame:
    """Explicitly synthetic survey data for pipeline demonstration."""
    return pd.DataFrame([
        {"response_id": "SYN_001", "age_group": "18-24", "country": "AZ", "survey_wave": "T0", "group": "Intervention", "stigma_score": 4.4, "help_intent_score": 3.2},
        {"response_id": "SYN_002", "age_group": "18-24", "country": "AZ", "survey_wave": "T0", "group": "Intervention", "stigma_score": 4.8, "help_intent_score": 2.9},
        {"response_id": "SYN_003", "age_group": "25-29", "country": "TR", "survey_wave": "T0", "group": "Intervention", "stigma_score": 4.1, "help_intent_score": 3.5},
        {"response_id": "SYN_004", "age_group": "18-24", "country": "GE", "survey_wave": "T0", "group": "Intervention", "stigma_score": 4.6, "help_intent_score": 3.0},
        {"response_id": "SYN_005", "age_group": "18-24", "country": "AZ", "survey_wave": "T0", "group": "Control", "stigma_score": 4.3, "help_intent_score": 3.3},
        {"response_id": "SYN_006", "age_group": "25-29", "country": "TR", "survey_wave": "T0", "group": "Control", "stigma_score": 4.5, "help_intent_score": 3.1},
        {"response_id": "SYN_007", "age_group": "18-24", "country": "GE", "survey_wave": "T0", "group": "Control", "stigma_score": 4.2, "help_intent_score": 3.4},
        {"response_id": "SYN_008", "age_group": "25-29", "country": "AZ", "survey_wave": "T0", "group": "Control", "stigma_score": 4.7, "help_intent_score": 2.8},
        {"response_id": "SYN_001", "age_group": "18-24", "country": "AZ", "survey_wave": "T1", "group": "Intervention", "stigma_score": 3.6, "help_intent_score": 4.1},
        {"response_id": "SYN_002", "age_group": "18-24", "country": "AZ", "survey_wave": "T1", "group": "Intervention", "stigma_score": 3.9, "help_intent_score": 3.8},
        {"response_id": "SYN_003", "age_group": "25-29", "country": "TR", "survey_wave": "T1", "group": "Intervention", "stigma_score": 3.5, "help_intent_score": 4.2},
        {"response_id": "SYN_004", "age_group": "18-24", "country": "GE", "survey_wave": "T1", "group": "Intervention", "stigma_score": 3.8, "help_intent_score": 3.9},
        {"response_id": "SYN_005", "age_group": "18-24", "country": "AZ", "survey_wave": "T1", "group": "Control", "stigma_score": 4.2, "help_intent_score": 3.4},
        {"response_id": "SYN_006", "age_group": "25-29", "country": "TR", "survey_wave": "T1", "group": "Control", "stigma_score": 4.4, "help_intent_score": 3.2},
        {"response_id": "SYN_007", "age_group": "18-24", "country": "GE", "survey_wave": "T1", "group": "Control", "stigma_score": 4.2, "help_intent_score": 3.4},
        {"response_id": "SYN_008", "age_group": "25-29", "country": "AZ", "survey_wave": "T1", "group": "Control", "stigma_score": 4.6, "help_intent_score": 2.9},
    ])


# =============================================================================
# 8. STORY ANALYSIS ENGINE (heuristic, labeled as such)
# =============================================================================

def analyze_story_text(text: str) -> Dict[str, Any]:
    """Heuristic keyword-based analysis — NOT a validated psychometric tool."""
    text_lower = text.lower()
    words = re.findall(r"\b[\w']+\b", text_lower, flags=re.UNICODE)
    n_words = len(words)
    n_chars = len(text)
    sentences = re.split(r"[.!?]+\s+", text.strip())
    n_sentences = max(1, len([s for s in sentences if s.strip()]))

    theme_hits: Dict[str, int] = {}
    for theme, keywords in STORY_THEME_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            theme_hits[theme] = hits

    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total_tone = pos + neg
    if total_tone == 0:
        tone = "neutral"
        tone_score = 50.0
    else:
        tone_score = round((pos / total_tone) * 100, 1)
        if tone_score >= 60:
            tone = "positive"
        elif tone_score <= 40:
            tone = "negative"
        else:
            tone = "mixed"

    avg_words_per_sentence = round(n_words / n_sentences, 1) if n_sentences else 0
    reading_seconds = max(15, int((n_words / 200) * 60))

    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "n_unique": len(set(words)),
        "n_sentences": n_sentences,
        "avg_words_per_sentence": avg_words_per_sentence,
        "themes": theme_hits,
        "tone": tone,
        "tone_score": tone_score,
        "reading_seconds": reading_seconds,
        "positive_hits": pos,
        "negative_hits": neg,
    }


# =============================================================================
# 9. MODULE — HOME
# =============================================================================

def render_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">GLOBAL · MENTAL HEALTH · STORYTELLING · RESEARCH</div>
            <div class="hero-title">What remains<br>unheard?</div>
            <p class="hero-sub">
                Mental health is not only a medical issue. It is also a human story —
                of silence, fear, hope, and the courage to speak.
                <br><br>
                This platform is interactive. Write your story, receive a real-time
                analysis, learn from evidence, and take part in research-shaped
                reflection. Every visitor gets a personalized experience.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What would you like to do?")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="card-teal">'
            '<div class="kicker">01 · 3 min</div>'
            '<h4>Write Your Story</h4>'
            '<p>Anonymous. Local only. Real-time theme and tone analysis.</p>'
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Write →", key="home_to_story", use_container_width=True):
            st.session_state["current_section"] = "02 · Write Your Story"
            st.rerun()
    with c2:
        st.markdown(
            '<div class="card-amber">'
            '<div class="kicker">02 · 2 min</div>'
            '<h4>See Your Profile</h4>'
            '<p>8-item self-reflection. Radar profile across 6 dimensions.</p>'
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Profile →", key="home_to_quiz", use_container_width=True):
            st.session_state["current_section"] = "03 · Your Profile"
            st.rerun()
    with c3:
        st.markdown(
            '<div class="card-teal">'
            '<div class="kicker">03 · 5 min</div>'
            '<h4>Media Coding Game</h4>'
            '<p>Code 8 headlines. Get real Krippendorff\'s α feedback.</p>'
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Play →", key="home_to_coding", use_container_width=True):
            st.session_state["current_section"] = "04 · Media Coding Game"
            st.rerun()
    with c4:
        st.markdown(
            '<div class="card-amber">'
            '<div class="kicker">04 · 3 min</div>'
            '<h4>Intervention Simulator</h4>'
            '<p>Experience a media intervention. See your own pre/post shift.</p>'
            '</div>', unsafe_allow_html=True,
        )
        if st.button("Simulate →", key="home_to_sim", use_container_width=True):
            st.session_state["current_section"] = "06 · Intervention Simulator"
            st.rerun()

    st.divider()

    st.markdown("### The research question")
    info_box(
        "<strong>How do media narratives shape mental health stigma, public "
        "understanding, and help-seeking intentions across different cultures, "
        "and how can evidence-based storytelling create positive change?</strong>"
    )

    pull_quote("Stigma lives not in the numbers, but in the silences.")

    st.markdown("### What this platform is — and is not")
    col1, col2 = st.columns(2)
    with col1:
        success_box(
            "<strong>What we do:</strong><br>"
            "• Ground the platform in validated instruments (MHLS, RIBS, PSOSH, MHSAS, ISMI)<br>"
            "• Report N, effect size, and 95% CI for every statistical claim<br>"
            "• Label all synthetic data clearly as SYNTHETIC<br>"
            "• Distinguish preprints from peer-reviewed evidence<br>"
            "• Anonymize and de-identify participant stories"
        )
    with col2:
        danger_box(
            "<strong>What we do NOT do:</strong><br>"
            "• No clinical diagnosis, no PHQ-8<br>"
            "• No fabricated indices<br>"
            "• No causal claims from cross-sectional data<br>"
            "• No intervention effects without a control group<br>"
            "• No 'one-size-fits-all' global survey without cultural adaptation"
        )

    st.divider()
    st.markdown("### This session at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stories written", len(st.session_state["user_stories"]))
    c2.metric("Quiz completed", "Yes" if st.session_state["quiz_completed"] else "No")
    c3.metric("Coding game", "Yes" if st.session_state["coding_completed"] else "No")
    c4.metric("Session start", datetime.now().strftime("%H:%M"))


# =============================================================================
# 10. MODULE — WRITE YOUR STORY (the core interactive feature)
# =============================================================================

def render_story_builder() -> None:
    section_header(
        "INTERACTIVE · ANONYMOUS · LOCAL-ONLY",
        "Write Your Story",
        "Your story stays in your browser. It is not sent to any server. No names, emails, or phones collected."
    )

    info_box(
        "<strong>How it works:</strong> Write your story → press 'Analyze' → see themes, "
        "tone, word statistics, related evidence, and suggested resources. "
        "Then, if you wish, share it anonymously within this session only."
    )

    danger_box(
        "<strong>⚠ Please do NOT include:</strong> real names, phone numbers, emails, "
        "addresses, or any other identifiable information. Stories are NOT automatically "
        "anonymized — you must write anonymously yourself."
    )

    draft = st.text_area(
        "Your story (in any language — analysis is keyword-based in English)",
        value=st.session_state["story_draft"],
        height=220,
        placeholder="Example: 'I never told my family. For them, it was just being moody...'",
        key="story_textarea",
    )
    st.session_state["story_draft"] = draft

    with st.expander("📋 Optional metadata (aggregate only)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            age_group = st.selectbox(
                "Age group (optional)",
                ["—", "18–21", "22–25", "26–29", "30+"],
                key="story_age",
            )
        with c2:
            theme_hint = st.selectbox(
                "Primary theme (optional)",
                ["—"] + list(STORY_THEME_KEYWORDS.keys()),
                key="story_theme_hint",
            )

    if draft.strip():
        live = analyze_story_text(draft)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Words", live["n_words"])
        c2.metric("Unique words", live["n_unique"])
        c3.metric("Reading time", f"~{live['reading_seconds']}s")
        c4.metric("Tone", live["tone"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if len(draft.strip()) < 30:
                st.warning("Please write at least 30 characters.")
            else:
                analysis = analyze_story_text(draft)
                st.session_state["last_story_analysis"] = analysis
                st.rerun()
    with col_b:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state["story_draft"] = ""
            st.session_state["last_story_analysis"] = None
            st.rerun()

    analysis = st.session_state.get("last_story_analysis")
    if analysis is None:
        return

    st.divider()
    st.markdown("### 📊 Analysis result")
    info_box(
        "This analysis is <strong>keyword-based heuristic</strong>. "
        "It is not a validated psychometric tool. Interpreting the meaning "
        "of your story requires a human reader — this is only an initial guide."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words", analysis["n_words"])
    c2.metric("Sentences", analysis["n_sentences"])
    c3.metric("Tone", analysis["tone"], f"score: {analysis['tone_score']}/100")
    c4.metric("Themes detected", len(analysis["themes"]))

    if analysis["themes"]:
        st.markdown("#### Themes detected")
        theme_df = pd.DataFrame([
            {"Theme": k, "Hits": v}
            for k, v in sorted(analysis["themes"].items(), key=lambda x: -x[1])
        ])
        fig = px.bar(
            theme_df, x="Hits", y="Theme", orientation="h",
            color="Hits",
            color_continuous_scale=["#EBE6DC", "#5C8581", "#14243A"],
            text="Hits",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_plotly(fig, 340), use_container_width=True)
    else:
        st.warning(
            "No recognized themes were detected. This does not mean your story is "
            "'wrong' — it simply does not match our keyword dictionary. Try adding "
            "more detail if you wish."
        )

    st.markdown("#### Emotional tone")
    pos = analysis["positive_hits"]
    neg = analysis["negative_hits"]
    neutral = max(0, analysis["n_words"] - pos - neg)
    tone_df = pd.DataFrame({
        "Tone": ["Positive", "Negative", "Neutral / other"],
        "Words": [pos, neg, neutral],
    })
    fig2 = px.pie(
        tone_df, names="Tone", values="Words",
        color="Tone",
        color_discrete_map={
            "Positive": "#5C8581",
            "Negative": "#B85450",
            "Neutral / other": "#EBE6DC",
        },
        hole=0.55,
    )
    fig2.update_traces(textposition="inside", textinfo="percent")
    st.plotly_chart(style_plotly(fig2, 340), use_container_width=True)

    st.markdown("#### Related scientific sources")
    if analysis["themes"]:
        top_theme = max(analysis["themes"], key=analysis["themes"].get)
        related_map = {
            "Silence / Disclosure": [2, 7, 12],
            "Family pressure": [2, 7, 12],
            "Stigma": [1, 2, 4, 11],
            "Help-seeking": [1, 3, 7],
            "Media influence": [3, 4, 8],
            "Work / School": [1, 7],
            "Physical symptoms": [7],
            "Hope / Recovery": [1, 2, 10],
        }
        ref_ids = related_map.get(top_theme, [1, 3])
        for rid in ref_ids:
            ref = next((r for r in EVIDENCE_LIBRARY if r["n"] == rid), None)
            if ref:
                st.markdown(
                    f'<div class="card-teal">'
                    f'<div class="kicker">Source #{ref["n"]} · {ref["level"].upper()}</div>'
                    f'<strong>{ref["title"]}</strong><br>'
                    f'<span style="color:#5C8581; font-size:0.86rem;">'
                    f'{ref["authors"]} ({ref["year"]}) · {ref["meta"]}</span>'
                    f'<p style="margin-top:0.6rem;">{ref["key_finding"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("#### Suggested resources")
    info_box(
        "If your story reflects distress that you are currently experiencing, "
        "please consider reaching out to a licensed mental health professional in "
        "your country. This platform is educational and does not provide crisis "
        "support. In many countries, mental health helplines are free and confidential."
    )

    st.markdown("#### Comparison with other session stories")
    session_stories = st.session_state["user_stories"]
    if len(session_stories) == 0:
        st.caption(
            "No other stories in this session yet. If you share yours first, "
            "a comparison will be available."
        )
    else:
        session_tones = [s["analysis"]["tone"] for s in session_stories if s.get("analysis")]
        if session_tones:
            counts = Counter(session_tones)
            comp_df = pd.DataFrame([{"Tone": k, "Count": v} for k, v in counts.items()])
            fig3 = px.bar(
                comp_df, x="Tone", y="Count", color="Tone",
                color_discrete_map={
                    "positive": "#5C8581",
                    "negative": "#B85450",
                    "mixed": "#D6A56C",
                    "neutral": "#EBE6DC",
                },
                text="Count",
            )
            fig3.update_traces(textposition="outside")
            st.plotly_chart(style_plotly(fig3, 300), use_container_width=True)

    st.divider()
    st.markdown("### Share within this session?")
    info_box(
        "Sharing is entirely voluntary. Your story is shown only in this session — "
        "not sent to a server, database, or another user. It is deleted when you close the page."
    )
    if st.button("✓ Share anonymously in this session", use_container_width=True):
        st.session_state["user_stories"].append({
            "text": draft,
            "analysis": analysis,
            "age_group": age_group if age_group != "—" else "not specified",
            "theme_hint": theme_hint if theme_hint != "—" else None,
            "timestamp": datetime.now().isoformat(),
        })
        st.session_state["story_draft"] = ""
        st.session_state["last_story_analysis"] = None
        st.success("✓ Story added to this session. View it in the Story Library.")
        st.balloons()


# =============================================================================
# 11. MODULE — SELF-REFLECTION PROFILE
# =============================================================================

def render_quiz() -> None:
    section_header(
        "8 ITEMS · 2 MIN · ANONYMOUS",
        "Your Stigma Profile",
        "This is not a clinical diagnosis. It is a self-reflection profile."
    )

    info_box(
        "<strong>Important:</strong> These 8 items are <em>not a validated instrument</em>. "
        "They are simple self-reflection statements inspired by ATMHPS, PSOSH, and MHSAS. "
        "Real research uses the full validated instruments. Your results are yours alone "
        "and are not sent to any server."
    )

    for item in QUIZ_ITEMS:
        st.markdown(
            f'<div class="card-teal" style="padding:1rem 1.2rem;">'
            f'<div style="font-size:1.0rem; margin-bottom:0.5rem;">{item["text"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        current = st.session_state["quiz_answers"].get(item["id"])
        idx = QUIZ_SCALE.index(current) if current in QUIZ_SCALE else 2
        choice = st.radio(
            "Answer", QUIZ_SCALE, index=idx,
            key=f"quiz_{item['id']}",
            label_visibility="collapsed",
            horizontal=True,
        )
        st.session_state["quiz_answers"][item["id"]] = choice

    st.divider()

    if st.button("📊 Show My Results", type="primary", use_container_width=True):
        if len(st.session_state["quiz_answers"]) < len(QUIZ_ITEMS):
            st.warning("Please answer all questions.")
        else:
            st.session_state["quiz_completed"] = True
            compute_quiz_results()

    if st.session_state["quiz_completed"] and st.session_state.get("quiz_results"):
        render_quiz_results(st.session_state["quiz_results"])


def compute_quiz_results() -> None:
    answers = st.session_state["quiz_answers"]
    raw = {k: QUIZ_VALUES[QUIZ_SCALE.index(v)] for k, v in answers.items()}
    adjusted = {}
    for item in QUIZ_ITEMS:
        v = raw[item["id"]]
        adjusted[item["id"]] = (6 - v) if item.get("reverse") else v

    dims = {
        "disclosure": [adjusted["q1"]],
        "family": [adjusted["q2"]],
        "social_distance": [adjusted["q3"]],
        "help_attitude": [adjusted["q4"], adjusted["q8"]],
        "literacy": [adjusted["q5"]],
        "media_awareness": [adjusted["q6"], adjusted["q7"]],
    }
    dim_scores = {k: float(np.mean(v)) for k, v in dims.items()}
    norm = {k: round((v - 1) / 4 * 100, 1) for k, v in dim_scores.items()}
    st.session_state["quiz_results"] = norm


def render_quiz_results(norm: Dict[str, float]) -> None:
    st.markdown("### Your profile")
    st.caption("Higher score = higher on this dimension. Not a clinical meaning.")

    radar_df = pd.DataFrame({
        "Dimension": list(norm.keys()),
        "Score": list(norm.values()),
    })
    fig = go.Figure(go.Scatterpolar(
        r=radar_df["Score"],
        theta=[QUIZ_DIMENSIONS[d] for d in radar_df["Dimension"]],
        fill="toself",
        line=dict(color=BRAND["teal"], width=3),
        fillcolor="rgba(92,133,129,0.25)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=460,
        font_family="Inter", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric(QUIZ_DIMENSIONS["disclosure"], f"{norm['disclosure']:.0f}")
    c2.metric(QUIZ_DIMENSIONS["family"], f"{norm['family']:.0f}")
    c3.metric(QUIZ_DIMENSIONS["social_distance"], f"{norm['social_distance']:.0f}")
    c4, c5, c6 = st.columns(3)
    c4.metric(QUIZ_DIMENSIONS["help_attitude"], f"{norm['help_attitude']:.0f}")
    c5.metric(QUIZ_DIMENSIONS["literacy"], f"{norm['literacy']:.0f}")
    c6.metric(QUIZ_DIMENSIONS["media_awareness"], f"{norm['media_awareness']:.0f}")

    st.markdown("### What this might mean")
    insights = []
    if norm["disclosure"] > 60:
        insights.append(
            "**High disclosure concern** — this aligns with findings from multiple "
            "cultures (Shahhuseynbayova et al., 2026; Gulliver et al., 2010) that "
            "disclosure concerns are among the strongest barriers to help-seeking."
        )
    if norm["family"] > 60:
        insights.append(
            "**High family concern** — common in collectivist cultural contexts, "
            "where family reactions carry significant social weight."
        )
    if norm["help_attitude"] < 50:
        insights.append(
            "**Negative attitude toward professional help** — one of the strongest "
            "predictors of not seeking help (Gulliver et al., 2010)."
        )
    if norm["literacy"] < 50:
        insights.append(
            "**Low mental health literacy** — but remember: information alone does "
            "not reduce stigma (Nazari et al., 2023). Contact and storytelling matter more."
        )
    if norm["media_awareness"] > 60:
        insights.append(
            "**High media awareness** — a marker of critical media literacy."
        )
    if not insights:
        insights.append("Balanced profile. Explore other modules for more insight.")

    for ins in insights:
        st.markdown(f"- {ins}")

    st.divider()
    st.markdown("### Next step")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✍️ Write your story now →", use_container_width=True):
            st.session_state["current_section"] = "02 · Write Your Story"
            st.rerun()
    with c2:
        if st.button("📚 Explore the evidence library", use_container_width=True):
            st.session_state["current_section"] = "08 · Evidence Library"
            st.rerun()


# =============================================================================
# 12. MODULE — MEDIA CODING GAME
# =============================================================================

def render_media_coding() -> None:
    section_header(
        "8 HEADLINES · REAL KRIPPENDORFF'S α",
        "Media Coding Game",
        "Code real-style headlines with the M01–M10 framework — just like research coders do."
    )

    info_box(
        "In real research, two coders independently code the same material and "
        "Krippendorff's α is computed. Here you are the second coder — compared "
        "against a gold standard. α ≥ 0.75 is the acceptable threshold."
    )

    with st.expander("📖 Codebook (M01–M10)", expanded=False):
        st.dataframe(pd.DataFrame(MEDIA_CODEBOOK), use_container_width=True, hide_index=True)

    st.divider()

    idx = st.session_state["coding_index"]
    if idx >= len(CODING_HEADLINES):
        render_coding_results()
        return

    hl = CODING_HEADLINES[idx]
    st.markdown(
        f'<div class="card-amber">'
        f'<div class="kicker">HEADLINE {idx+1} / {len(CODING_HEADLINES)}</div>'
        f'<div style="font-family:Instrument Serif, serif; font-size:1.5rem; margin-top:0.6rem; line-height:1.4;">'
        f'{hl["title"]}</div>'
        f'<div style="color:#5C8581; font-size:0.82rem; margin-top:0.5rem;">{hl["source"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.caption("Which codes are present in this headline? You may select more than one.")

    selection: Dict[str, bool] = {}
    cols = st.columns(2)
    for i, item in enumerate(MEDIA_CODEBOOK):
        with cols[i % 2]:
            selection[item["code"]] = st.checkbox(
                f"`{item['code']}` — {item['name']}",
                key=f"code_{idx}_{item['code']}",
            )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("✓ Save & Next", type="primary", use_container_width=True):
            st.session_state["coding_answers"][hl["id"]] = {k: int(v) for k, v in selection.items()}
            st.session_state["coding_index"] += 1
            st.rerun()
    with col2:
        if st.button("⏭ Skip", use_container_width=True):
            st.session_state["coding_index"] += 1
            st.rerun()

    st.progress(idx / len(CODING_HEADLINES))


def render_coding_results() -> None:
    st.session_state["coding_completed"] = True
    answers = st.session_state["coding_answers"]

    st.markdown("### Your results")

    if not answers:
        st.info("No answers yet.")
        if st.button("Restart"):
            st.session_state["coding_index"] = 0
            st.session_state["coding_answers"] = {}
            st.session_state["coding_completed"] = False
            st.rerun()
        return

    codes = [item["code"] for item in MEDIA_CODEBOOK]
    rows = []
    for hl_id, sel in answers.items():
        gold = next(h["gold"] for h in CODING_HEADLINES if h["id"] == hl_id)
        for c in codes:
            rows.append({"headline": hl_id, "code": c, "your": sel[c], "gold": gold[c]})
    df = pd.DataFrame(rows)
    df["match"] = (df["your"] == df["gold"]).astype(int)
    accuracy = df["match"].mean() * 100

    alpha_data = df[["your", "gold"]].values.astype(float)
    kalpha = krippendorff_alpha_nominal(alpha_data)

    c1, c2, c3 = st.columns(3)
    c1.metric("Headlines coded", f"{len(answers)}", f"of {len(CODING_HEADLINES)}")
    c2.metric("Accuracy", f"{accuracy:.1f}%")
    c3.metric(
        "Krippendorff's α",
        f"{kalpha:.3f}" if kalpha is not None else "—",
        "target ≥ 0.75",
    )

    st.markdown("### Per-headline breakdown")
    for hl_id in answers:
        hl = next(h for h in CODING_HEADLINES if h["id"] == hl_id)
        user_sel = answers[hl_id]
        gold = hl["gold"]
        matches = sum(1 for c in codes if user_sel[c] == gold[c])
        total = len(codes)
        pct = matches / total * 100
        misses = [c for c in codes if user_sel[c] != gold[c]]

        with st.expander(f"{hl['title'][:60]}… · {matches}/{total} ({pct:.0f}%)", expanded=False):
            st.markdown(f"**Gold standard:** {', '.join([c for c in codes if gold[c]]) or 'none'}")
            st.markdown(f"**Your selection:** {', '.join([c for c in codes if user_sel[c]]) or 'none'}")
            if misses:
                st.markdown(f"**Mismatched codes:** {', '.join(misses)}")
            st.info(f"**Explanation:** {hl['explanation']}")

    st.divider()
    if st.button("🔄 Play again", use_container_width=True):
        st.session_state["coding_index"] = 0
        st.session_state["coding_answers"] = {}
        st.session_state["coding_completed"] = False
        st.rerun()


# =============================================================================
# 13. MODULE — STORY LIBRARY
# =============================================================================

def render_story_library() -> None:
    section_header(
        "SESSION + COMPOSITE NARRATIVES",
        "Story Library",
        "Stories shared in this session, and composite illustrative narratives."
    )

    user_stories = st.session_state["user_stories"]

    st.markdown(f"### Stories shared in this session ({len(user_stories)})")
    if not user_stories:
        info_box(
            "No stories in this session yet. Write and share the first one from the "
            "'Write Your Story' section. Stories are visible only in this session."
        )
    else:
        for i, story in enumerate(user_stories, 1):
            analysis = story.get("analysis", {})
            themes = analysis.get("themes", {})
            top_theme = max(themes, key=themes.get) if themes else "—"
            st.markdown(
                f'<div class="story-card">'
                f'<span class="tag-teal">Story {i}</span>'
                f'<span class="tag-amber">{top_theme}</span>'
                f'<span class="tag-teal">Tone: {analysis.get("tone", "—")}</span>'
                f'<p style="margin-top:0.8rem; line-height:1.7;">{story["text"]}</p>'
                f'<div style="color:#5C8581; font-size:0.8rem; margin-top:0.6rem;">'
                f'{story.get("age_group", "—")} · {story["timestamp"][:10]}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Composite illustrative narratives")
    info_box(
        "The stories below are <strong>composite</strong> — not the story of any single "
        "real person. They are illustrative examples written for research and media "
        "materials. They do not belong to any real participant."
    )

    for story in COMPOSITE_STORIES:
        st.markdown(
            f'<div class="story-card amber">'
            f'<span class="tag-amber">{story["label"]}</span>'
            f'<span class="tag-teal">{story["theme"]}</span>'
            f'<h4 style="margin-top:0.6rem;">{story["title"]}</h4>'
            f'<p style="line-height:1.7;">{story["text"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# 14. MODULE — INTERVENTION SIMULATOR
# =============================================================================

def render_intervention_simulator() -> None:
    section_header(
        "PRE / POST · SUGGESTED INTERVENTION",
        "Intervention Simulator",
        "Test: does a media intervention shift your own attitudes?"
    )

    info_box(
        "This simulator is <strong>not a scientific RCT</strong> — it allows you to "
        "observe your own attitude shift after exposure to a media intervention. "
        "Results must not be presented as scientific evidence."
    )

    stage = st.session_state["intervention_stage"]

    if stage == "before":
        st.markdown("### Step 1 — Baseline measurement (T0)")
        st.markdown("Answer the 3 items on a 1–7 scale (1 = strongly disagree, 7 = strongly agree):")

        q1 = st.slider("1. Seeking psychological help is acceptable to me.", 1, 7, 4, key="int_q1")
        q2 = st.slider("2. If I struggled, I would seek professional help.", 1, 7, 4, key="int_q2")
        q3 = st.slider("3. I would not hesitate to work with someone who has a mental health condition.", 1, 7, 4, key="int_q3")

        if st.button("✓ Save T0 → Go to Intervention", type="primary", use_container_width=True):
            st.session_state["intervention_before"] = {"q1": q1, "q2": q2, "q3": q3}
            st.session_state["intervention_stage"] = "material"
            st.rerun()

    elif stage == "material":
        st.markdown("### Step 2 — Media intervention")
        info_box(
            "Choose one material and 'watch' it. Note: these are not real participant "
            "stories — they are composite narratives."
        )

        material = st.radio(
            "Choose material",
            [
                "Short documentary (5–8 min) — composite narrative",
                "Podcast episode (24 min) — silence and disclosure",
                "Short video (60–100 s) — hope and help",
                "Neutral information (control group) — facts only",
            ],
            key="int_material_choice",
        )

        with st.container(border=True):
            st.markdown(f"#### ▶ {material}")
            if "documentary" in material.lower():
                st.markdown(
                    "*Visual narrative: A young person talks about the difficulty of "
                    "finding psychological support at university. They fear their "
                    "family's reaction, but with a friend's support they take the "
                    "first step. At the end, they describe the meeting with the "
                    "therapist as a positive experience.*"
                )
                st.caption("Composite narrative — not a real participant's story.")
            elif "podcast" in material.lower():
                st.markdown(
                    "*Audio: A podcast episode on 'Why do we stay silent?' — two young "
                    "people discuss stigma, family pressure, and their first experience "
                    "reaching out. At the end, practical resources are shared.*"
                )
                st.caption("Composite narrative — not a real participant's story.")
            elif "video" in material.lower():
                st.markdown(
                    "*Video (80 s): A young person speaks to the camera: 'Asking for help "
                    "is courage. I used to be afraid, but now I know I'm not alone.' "
                    "At the end, help resources appear on screen.*"
                )
                st.caption("Composite narrative — not a real participant's story.")
            else:
                st.markdown(
                    "*Neutral information: Statistics on the prevalence of mental health "
                    "conditions, treatment options, and pathways to care. No emotional "
                    "narrative.*"
                )
                st.caption("Neutral control material.")

        if st.button("✓ 'Watched' → Go to T1", type="primary", use_container_width=True):
            st.session_state["intervention_choice"] = material
            st.session_state["intervention_stage"] = "after"
            st.rerun()

    elif stage == "after":
        st.markdown("### Step 3 — Post-measurement (T1)")
        st.markdown("Now answer the same 3 items again:")

        q1_after = st.slider("1. Seeking psychological help is acceptable to me.", 1, 7, 4, key="int_q1_after")
        q2_after = st.slider("2. If I struggled, I would seek professional help.", 1, 7, 4, key="int_q2_after")
        q3_after = st.slider("3. I would not hesitate to work with someone who has a mental health condition.", 1, 7, 4, key="int_q3_after")

        if st.button("✓ See My Result", type="primary", use_container_width=True):
            st.session_state["intervention_after"] = {"q1": q1_after, "q2": q2_after, "q3": q3_after}
            st.session_state["intervention_stage"] = "result"
            st.rerun()

    elif stage == "result":
        before = st.session_state["intervention_before"]
        after = st.session_state["intervention_after"]
        material = st.session_state.get("intervention_choice", "—")

        st.markdown("### Your result")
        info_box(
            f"Material: <strong>{material}</strong>. "
            f"This is the simulation result of <em>one person</em> (you). "
            "Not a scientific result."
        )

        labels = ["Help acceptance", "Help-seeking intention", "Social acceptance"]
        before_vals = [before["q1"], before["q2"], before["q3"]]
        after_vals = [after["q1"], after["q2"], after["q3"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="T0 (before)", x=labels, y=before_vals, marker_color="#D6A56C"))
        fig.add_trace(go.Bar(name="T1 (after)", x=labels, y=after_vals, marker_color="#5C8581"))
        fig.update_layout(barmode="group", yaxis_title="Score (1–7)", yaxis_range=[0, 7.5])
        st.plotly_chart(style_plotly(fig, 380), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        for col, label, b, a in zip([c1, c2, c3], labels, before_vals, after_vals):
            delta = a - b
            sign = "+" if delta >= 0 else ""
            col.metric(label, f"{a}", f"{sign}{delta} change")

        total_before = sum(before_vals)
        total_after = sum(after_vals)
        delta_total = total_after - total_before

        if abs(delta_total) >= 2:
            success_box(
                f"<strong>Simulation result:</strong> A total change of {delta_total:+.0f}. "
                "This is a single-person simulation — statistical significance cannot be computed."
            )
        else:
            info_box(
                f"<strong>Simulation result:</strong> A total change of {delta_total:+.0f}. "
                "A small change. This does not mean the material is ineffective — "
                "it is a single-person simulation."
            )

        st.divider()
        if st.button("🔄 Try again", use_container_width=True):
            st.session_state["intervention_before"] = None
            st.session_state["intervention_after"] = None
            st.session_state["intervention_stage"] = "before"
            st.rerun()


# =============================================================================
# 15. MODULE — GLOBAL PULSE
# =============================================================================

def render_global_pulse() -> None:
    section_header(
        "SESSION AGGREGATE",
        "Global Pulse",
        "Aggregate view of all data collected in this session."
    )

    user_stories = st.session_state["user_stories"]
    quiz_completed = st.session_state["quiz_completed"]
    quiz_results = st.session_state.get("quiz_results")
    coding_completed = st.session_state["coding_completed"]
    coding_answers = st.session_state["coding_answers"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stories", len(user_stories))
    c2.metric("Quiz", "✓" if quiz_completed else "—")
    c3.metric("Coding", f"{len(coding_answers)}/8" if coding_answers else "—")
    c4.metric("Time", datetime.now().strftime("%H:%M"))

    st.divider()

    st.markdown("### Most common themes in session stories")
    if not user_stories:
        info_box("No stories yet. Start from 'Write Your Story'.")
    else:
        all_themes: Dict[str, int] = {}
        for s in user_stories:
            for theme, count in s.get("analysis", {}).get("themes", {}).items():
                all_themes[theme] = all_themes.get(theme, 0) + count

        if all_themes:
            theme_df = pd.DataFrame([
                {"Theme": k, "Hits": v}
                for k, v in sorted(all_themes.items(), key=lambda x: -x[1])
            ])
            fig = px.bar(
                theme_df, x="Hits", y="Theme", orientation="h",
                color="Hits",
                color_continuous_scale=["#EBE6DC", "#5C8581", "#14243A"],
                text="Hits",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_plotly(fig, 380), use_container_width=True)
        else:
            st.info("No themes detected in session stories.")

    st.divider()

    st.markdown("### Your profile")
    if quiz_results:
        quiz_df = pd.DataFrame([
            {"Dimension": QUIZ_DIMENSIONS[k], "Score": v} for k, v in quiz_results.items()
        ])
        fig2 = px.bar(
            quiz_df, x="Score", y="Dimension", orientation="h",
            color="Score",
            color_continuous_scale=["#EBE6DC", "#5C8581", "#14243A"],
            text="Score",
        )
        fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig2.update_layout(
            coloraxis_showscale=False,
            xaxis_range=[0, 105],
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(style_plotly(fig2, 340), use_container_width=True)
    else:
        st.info("Quiz not completed yet.")

    st.divider()

    st.markdown("### Coding performance")
    if coding_completed and coding_answers:
        rows = []
        for hl_id, sel in coding_answers.items():
            gold = next(h["gold"] for h in CODING_HEADLINES if h["id"] == hl_id)
            matches = sum(1 for c in sel if sel[c] == gold[c])
            rows.append({"headline": hl_id, "matches": matches, "total": len(sel)})
        df = pd.DataFrame(rows)
        st.metric("Mean agreement", f"{df['matches'].mean():.1f}/{df['total'].iloc[0]}")
    else:
        st.info("Coding game not completed yet.")

    st.divider()

    st.markdown("### Download session metadata")
    if user_stories:
        export_df = pd.DataFrame([
            {
                "story_id": i + 1,
                "text_length": len(s["text"]),
                "n_words": s["analysis"]["n_words"],
                "tone": s["analysis"]["tone"],
                "n_themes": len(s["analysis"]["themes"]),
                "timestamp": s["timestamp"],
            }
            for i, s in enumerate(user_stories)
        ])
        st.dataframe(export_df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download story metadata (CSV)",
            export_df.to_csv(index=False).encode("utf-8"),
            file_name="session_stories_metadata.csv",
            mime="text/csv",
        )

    st.caption(
        "All data shown on this page exists only in this session. "
        "It is deleted when you close the page or press 'Reset'. "
        "It is not sent to any server."
    )


# =============================================================================
# 16. MODULE — RESEARCH LAB (with real CSV upload)
# =============================================================================

def render_research_lab() -> None:
    section_header(
        "REAL PSYCHOMETRIC ANALYSIS",
        "Research Lab",
        "Upload your own CSV, or use the synthetic demo, to run real statistical analysis."
    )

    st.markdown("### Data source")
    uploaded = st.file_uploader("Upload CSV (survey data)", type=["csv"], key="lab_csv")
    use_synth = st.checkbox(
        "Use synthetic demo data",
        value=(uploaded is None),
        help="Synthetic data is for demonstrating the workflow only.",
    )

    if uploaded is not None:
        try:
            df = load_csv_bytes(uploaded.getvalue(), uploaded.name)
            st.success(f"✓ {len(df)} rows loaded")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return
    elif use_synth:
        df = get_synthetic_survey_df()
        synth_banner("Synthetic demo data — not real participant data")
    else:
        info_box("Upload a CSV or use the synthetic demo data.")
        return

    with st.expander("Data preview"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))
    ]

    if len(numeric_cols) < 2:
        st.warning("The data must contain at least 2 numeric columns.")
        return

    st.markdown("### Cronbach's α — internal consistency")
    c1, c2 = st.columns([2, 1])
    with c1:
        alpha_cols = st.multiselect(
            "Select items (≥ 2)",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))],
        )
    with c2:
        reverse = st.multiselect("Reverse-scored items", alpha_cols, default=[])
        smax = st.number_input("Scale max", min_value=2, max_value=10, value=7)

    if len(alpha_cols) >= 2:
        scored = apply_reverse_scoring(df[alpha_cols], reverse, 1, int(smax))
        alpha, n, k = cronbach_alpha(scored)
        omega = mcdonald_omega_approx(scored)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cronbach's α", f"{alpha:.4f}" if alpha else "—", interpret_alpha(alpha))
        c2.metric("McDonald's ω (approx.)", f"{omega:.4f}" if omega else "—")
        c3.metric("N (complete)", n)
        c4.metric("Items (k)", k)
        if alpha is None:
            st.caption("For α, ≥ 3 complete rows and non-zero variance are required.")

    st.divider()

    st.markdown("### Group comparison — Cohen's d + 95% CI")
    if "group" not in df.columns:
        st.info("For group comparison, the data must include a `group` column.")
        return

    groups = sorted(df["group"].dropna().astype(str).unique())
    if len(groups) < 2:
        st.info("At least 2 groups are required.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        g_a = st.selectbox("Group A", groups, index=0)
    with c2:
        g_b = st.selectbox("Group B", groups, index=1)
    with c3:
        outcome = st.selectbox("Outcome variable", numeric_cols)

    y = pd.to_numeric(df[outcome], errors="coerce")
    arr_a = y[df["group"].astype(str) == g_a].to_numpy(dtype=float)
    arr_b = y[df["group"].astype(str) == g_b].to_numpy(dtype=float)

    d, lo, hi = cohens_d_ci(arr_a, arr_b)
    welch = welch_ttest(arr_a, arr_b)
    mw = mann_whitney(arr_a, arr_b)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Mean ({g_a})", f"{np.nanmean(arr_a):.3f}" if len(arr_a[~np.isnan(arr_a)]) > 0 else "—")
    c2.metric(f"Mean ({g_b})", f"{np.nanmean(arr_b):.3f}" if len(arr_b[~np.isnan(arr_b)]) > 0 else "—")
    c3.metric("Cohen's d", f"{d:.3f}" if d else "—", interpret_d(d))
    c4.metric("95% CI", f"[{lo:.3f}, {hi:.3f}]" if lo else "—")

    if welch.get("available"):
        st.markdown(f"- **Welch's t:** t = {welch['t']}, p = {welch['p']}")
    if mw.get("available"):
        st.markdown(f"- **Mann–Whitney U:** U = {mw['U']}, p = {mw['p']}")

    st.caption(
        "⚠ If N is small, results may be unreliable. Effect size and 95% CI "
        "are always presented together."
    )


# =============================================================================
# 17. MODULE — EVIDENCE LIBRARY
# =============================================================================

def render_evidence_library() -> None:
    section_header(
        "12 GLOBAL SOURCES",
        "Evidence Library",
        "The key scientific sources anchoring the platform's methodology."
    )

    info_box(
        "Preprints and peer-reviewed sources are marked with different labels. "
        "Presence in this library does not imply peer-reviewed publication — "
        "verify each source independently before citing."
    )

    level_filter = st.radio(
        "Evidence level",
        ["All", "Peer-reviewed", "Preprint / mixed"],
        horizontal=True,
        key="evidence_filter",
    )

    filtered = EVIDENCE_LIBRARY
    if level_filter == "Peer-reviewed":
        filtered = [r for r in EVIDENCE_LIBRARY if r["level"] == "high"]
    elif level_filter == "Preprint / mixed":
        filtered = [r for r in EVIDENCE_LIBRARY if r["level"] == "med"]

    for ref in filtered:
        level_color = {"high": "#5C8581", "med": "#D6A56C", "low": "#B85450"}[ref["level"]]
        level_text = {"high": "Peer-reviewed", "med": "Preprint / mixed", "low": "Low"}[ref["level"]]

        st.markdown(
            f'<div class="card-teal">'
            f'<div style="display:flex; justify-content:space-between; align-items:start;">'
            f'<span class="tag-teal">#{ref["n"]}</span>'
            f'<span style="background:{level_color}; color:white; padding:2px 10px; '
            f'border-radius:10px; font-size:0.72rem; font-weight:600;">{level_text}</span>'
            f'</div>'
            f'<h4 style="margin-top:0.6rem;">{ref["title"]}</h4>'
            f'<div style="color:#5C8581; font-size:0.88rem;">{ref["authors"]} ({ref["year"]})</div>'
            f'<div style="color:#777; font-size:0.82rem; margin-top:0.3rem;">{ref["meta"]}</div>'
            f'<p style="margin-top:0.7rem;"><strong>Key finding:</strong> {ref["key_finding"]}</p>'
            f'<div style="margin-top:0.5rem;"><span class="tag-amber">Role: {ref["role"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.download_button(
        "📥 Download all sources (CSV)",
        pd.DataFrame(EVIDENCE_LIBRARY).to_csv(index=False).encode("utf-8"),
        file_name="neurometric_gaze_evidence.csv",
        mime="text/csv",
    )


# =============================================================================
# 18. MODULE — GLOBAL RESEARCH FRAMEWORK
# =============================================================================

def render_global_framework() -> None:
    section_header(
        "METHODOLOGY · ADAPTATION · GOVERNANCE",
        "Global Research Framework",
        "How a truly global study is designed — beyond one country, one language, one culture."
    )

    st.markdown("### Why a global study is not one-size-fits-all")
    info_box(
        "Applying the same questionnaire in every country does not automatically "
        "make the results comparable. For example, the term 'mental illness' in "
        "one language may carry a very different social meaning than its literal "
        "translation in another. Cross-cultural comparison requires a two-level "
        "data model."
    )

    st.markdown("### Two-level data model")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="card-teal">'
            '<div class="kicker">LEVEL 1</div>'
            '<h4>Global Core</h4>'
            '<p>Identical core questions, validated instruments, and methodology '
            'across all countries. Enables cross-cultural comparison — but only '
            'after measurement invariance has been tested.</p>'
            '</div>', unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="card-amber">'
            '<div class="kicker">LEVEL 2</div>'
            '<h4>Local Context</h4>'
            '<p>Language, culture, help systems, and local social characteristics. '
            'Regional adaptations like ATMHPS-AZ (Azerbaijan) exemplify this layer.</p>'
            '</div>', unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### Validated instruments")
    st.dataframe(
        pd.DataFrame(INSTRUMENTS),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Adaptation protocol")
    st.markdown(
        """
        1. **Forward translation** by a bilingual subject-matter expert.
        2. **Back-translation** by an independent translator who has not seen the original.
        3. **Expert committee review** — reconcile discrepancies, preserve conceptual (not just literal) meaning.
        4. **Cognitive interviews** with 5–10 members of the target population.
        5. **Pilot testing** with 30–50 participants to check clarity and acceptability.
        6. **Psychometric validation** — confirm factor structure via EFA/CFA; compute Cronbach's α and McDonald's ω.
        7. **Measurement invariance** — configural, metric, scalar invariance across cultural groups.
        8. **Comparison** — only after invariance has been established.
        """
    )

    st.divider()

    st.markdown("### Intervention design framework")
    st.markdown(
        """
        **Evidence anchors:**
        - Storytelling meta-analysis (2022): first-person narratives outperform third-person.
        - Crockett et al. (2025): contact-based interventions beat information-only.
        - Digital empowerment (2024): knowledge effects persist, but help-seeking effects are small.
        - PICMIN (2012): cross-country media framing differs substantially — localization is essential.

        **Design principles:**
        - **Co-design with youth** (Baatcheet model, 2024): Young People's Advisory Group (n=11), stakeholder group (n=20), focus groups (n=42).
        - **First-person, short-form** (Amsalem et al., 2023): 58–102 second videos show measurable stigma reduction.
        - **Co-designed podcast** (Carrotte et al., 2022): format matters as much as content.
        - **Follow-up measurement** (Digital empowerment, 2024): gains often fade — reinforcement matters.
        """
    )

    st.divider()

    st.markdown("### Phased research plan")
    phases = [
        ("Phase 1", "Global Evidence Review", "Systematic reviews, meta-analyses, RCTs, validated instruments from multiple countries."),
        ("Phase 2", "Global Platform Prototype", "Multi-language interface, open evidence library, anonymous survey, Media Observatory."),
        ("Phase 3", "International Pilot", "At least 3 countries and cultural contexts. Countries chosen based on real participant access and local partners."),
        ("Phase 4", "Media Intervention", "Real stories, videos, podcasts, awareness materials — co-designed with youth."),
        ("Phase 5", "Global Impact Evaluation", "Pre/post/follow-up outcomes, between-group comparison, careful cross-country analysis — only after invariance."),
    ]
    for phase, title, desc in phases:
        st.markdown(
            f'<div class="card-teal">'
            f'<span class="tag-amber">{phase}</span>'
            f'<strong style="font-size:1.05rem; margin-left:0.5rem;">{title}</strong>'
            f'<p style="margin-top:0.6rem;">{desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### Power analysis (interactive)")
    info_box(
        "Before launching any country pilot, we estimate required sample size based "
        "on the smallest effect size we consider meaningful."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        d = st.slider("Expected effect size (Cohen's d)", 0.10, 0.80, 0.30, 0.05, key="framework_d")
    with c2:
        power_target = st.slider("Target power", 0.70, 0.95, 0.80, 0.01, key="framework_power")
    with c3:
        attrition = st.slider("Attrition (to inflate N)", 0.0, 0.35, 0.15, 0.01, key="framework_attrition")

    n_per_arm = sample_size_two_arm(d, alpha=0.05, power=power_target)
    n_total_raw = n_per_arm * 2
    n_total_inflated = int(math.ceil(n_total_raw / (1 - attrition))) if n_total_raw > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Per arm", n_per_arm)
    c2.metric("Total (raw)", n_total_raw)
    c3.metric("Total (inflated)", n_total_inflated, f"+{int(attrition*100)}% attrition")

    if HAS_SCIPY:
        ns = list(range(30, max(500, n_per_arm * 2), 20))
        curve = power_curve(d, ns)
        fig = px.line(curve, x="n_per_arm", y="power")
        fig.add_hline(y=power_target, line_dash="dash", line_color=BRAND["amber"],
                      annotation_text=f"Target power = {power_target:.2f}")
        if n_per_arm > 0:
            fig.add_vline(x=n_per_arm, line_dash="dash", line_color=BRAND["teal"],
                          annotation_text=f"n = {n_per_arm}")
        fig.update_layout(xaxis_title="n per arm", yaxis_title="power", yaxis_range=[0, 1.02])
        st.plotly_chart(style_plotly(fig, 340), use_container_width=True)

    st.caption(
        "For definitive sample-size calculation, use G*Power or `pwr::pwr.t.test` "
        "with the primary outcome and expected attrition clearly specified."
    )


# =============================================================================
# 19. MODULE — ETHICS & METHODOLOGY
# =============================================================================

def render_ethics() -> None:
    section_header(
        "ETHICS · METHODOLOGY · LIMITATIONS",
        "Ethics & Methodology",
        "The platform's ethical principles, methodological framework, and limitations."
    )

    st.markdown("### What we do")
    success_box(
        "• Work with validated instruments (MHLS, RIBS, PSOSH, MHSAS, ISMI)<br>"
        "• Report N, effect size, and 95% CI for every statistical result<br>"
        "• Label all synthetic data as SYNTHETIC<br>"
        "• Distinguish preprints from peer-reviewed sources<br>"
        "• Require participant consent, anonymity, and data minimization<br>"
        "• Adapt instruments culturally before comparison (Global Core + Local Context)"
    )

    st.markdown("### What we do NOT do")
    danger_box(
        "• No clinical diagnosis, no PHQ-8, no individual risk scores<br>"
        "• No fabricated indices (e.g., 'silence_index')<br>"
        "• No causal claims from cross-sectional correlations<br>"
        "• No 'intervention effect' claims without a control group<br>"
        "• No suicide-risk data collection without a formal safety protocol<br>"
        "• No cross-cultural comparison without measurement invariance testing"
    )

    st.divider()

    st.markdown("### Methodological framework")
    st.markdown(
        """
        **Module A — Media Content Analysis**
        - M01–M10 codebook (PICMIN-inspired)
        - Two independent coders; Krippendorff's α ≥ 0.75 target
        - Contextual human verification (not naive keyword counting)
        - Cross-country comparison: only after sufficient corpus per country

        **Module B — Global Youth Survey (ages 18–29)**
        - Validated instruments: MHLS, RIBS, PSOSH, MHSAS, ISMI
        - Anonymous, voluntary, data minimization
        - Sample size: power analysis based (n ≈ 400+ target per country)
        - Two-level model: Global Core + Local Context
        - Measurement invariance testing before any cross-country comparison

        **Module C — Intervention Evaluation**
        - Two-arm RCT (intervention vs. control)
        - T0 / T1 / T2 measurement waves (baseline, post, follow-up)
        - Primary outcome: PSOSH; secondary: RIBS, MHSAS, MHLS
        - ANCOVA (T1 ~ arm + T0), Cohen's d + 95% CI
        - Co-design with youth advisory group
        """
    )

    st.divider()

    st.markdown("### Seven ethical principles")
    principles = [
        ("Voluntary participation", "No one is forced to participate. The right to stop at any time is guaranteed."),
        ("Informed consent", "Purpose, procedures, risks, and rights are explained before participation begins."),
        ("Anonymity", "Names, phones, emails, and addresses are not collected. Interview transcripts are de-identified."),
        ("Age 18+ scope", "The youth survey only covers the 18–29 age range in the initial phase."),
        ("Data minimization", "Only data necessary to answer the research question is collected."),
        ("Transparency", "Methodology, limitations, and funding sources are openly disclosed."),
        ("Security", "Sensitive data is stored on encrypted servers with restricted access."),
    ]
    for i, (title, desc) in enumerate(principles, 1):
        st.markdown(f"**{i}. {title}** — {desc}")

    st.divider()

    st.markdown("### Limitations (pre-registered)")
    st.markdown(
        """
        1. Pilot sample sizes are small; early-phase findings are descriptive only.
        2. Control arm is not matched on all covariates; ANCOVA-adjusted estimates apply.
        3. Media sample is limited to defined outlets; it does not represent all media in any country.
        4. Four-week follow-up captures only short-term change; long-term effects are not measured.
        5. Cross-cultural validity of instruments is documented but still evolving.
        6. Digital interventions show smaller effects on help-seeking than on knowledge (Digital Empowerment, 2024).
        7. Storytelling effects are small in magnitude even when statistically significant (Storytelling meta-analysis, 2022).
        """
    )

    st.divider()

    st.markdown("### Contact & versioning")
    st.markdown(
        f"- Platform version: **2.0 (Global Interactive Edition)**\n"
        f"- Last updated: **{datetime.now().strftime('%d %b %Y')}**\n"
        f"- Repository: see project README\n"
        f"- Pre-registration: template available"
    )

    st.caption(
        "© 2026 The Neurometric Gaze. This platform is for research and education. "
        "Not a clinical service. If you or someone you know is struggling, please "
        "contact a licensed mental health professional."
    )


# =============================================================================
# 20. SIDEBAR + ROUTER
# =============================================================================

with st.sidebar:
    st.markdown(
        '<div style="padding:0.5rem 0 1.2rem 0;">'
        '<div class="kicker">THE NEUROMETRIC GAZE</div>'
        '<div style="font-family:Instrument Serif, serif; font-size:1.15rem; margin-top:0.3rem;">'
        'Beyond the Unheard'
        '</div>'
        '<div style="color:#5C8581; font-size:0.72rem; margin-top:0.3rem;">'
        'Global Interactive Edition · v2.0'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    section = st.radio(
        "Navigation",
        [
            "01 · Home",
            "02 · Write Your Story",
            "03 · Your Profile",
            "04 · Media Coding Game",
            "05 · Story Library",
            "06 · Intervention Simulator",
            "07 · Global Pulse",
            "08 · Evidence Library",
            "09 · Research Lab",
            "10 · Global Framework",
            "11 · Ethics & Methodology",
        ],
        index=0,
        key="current_section",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.7rem; color:#777; line-height:1.7;">'
        f'<strong>Status:</strong> Prototype<br>'
        f'<strong>Real data:</strong> None yet<br>'
        f'<strong>Updated:</strong> {datetime.now().strftime("%d %b %Y")}<br>'
        f'<strong>License:</strong> CC-BY-NC-4.0'
        f'</div>',
        unsafe_allow_html=True,
    )

if section == "01 · Home":
    render_home()
elif section == "02 · Write Your Story":
    render_story_builder()
elif section == "03 · Your Profile":
    render_quiz()
elif section == "04 · Media Coding Game":
    render_media_coding()
elif section == "05 · Story Library":
    render_story_library()
elif section == "06 · Intervention Simulator":
    render_intervention_simulator()
elif section == "07 · Global Pulse":
    render_global_pulse()
elif section == "08 · Evidence Library":
    render_evidence_library()
elif section == "09 · Research Lab":
    render_research_lab()
elif section == "10 · Global Framework":
    render_global_framework()
elif section == "11 · Ethics & Methodology":
    render_ethics()
