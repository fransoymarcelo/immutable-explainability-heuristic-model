# text_emotion/model.py
"""
Text emotion model (deterministic, lightweight MVP) — extended version.

Highlights:
- Lemmatization (when available) plus negation/intensifier heuristics.
- Lexical mapping to valence and emotional labels for more robust scores.
- Rich metadata for auditability and explainability.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import logging
import unicodedata
import os
from pathlib import Path
from collections import Counter
import csv

from common.types import EmotionResult, VAD
from common.utils import normalize_text
from utils.logging import get_logger
from config_labels import LABELS_CANONICAL, LABEL_MAP
from config.settings import SETTINGS

import spacy
nlp = spacy.load("es_core_news_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return lemmas


logger = get_logger("text_emotion")

# Attempt to import external preprocessing utilities (recommended).
# Provide lightweight fallbacks when unavailable.
try:
    from text_emotion.preproc import lemmatize_tokens, detect_negation_scope, detect_intensifiers
    HAS_PREPROC = True
except Exception:
    HAS_PREPROC = False

    def lemmatize_tokens(text: str) -> List[Tuple[str, str]]:
        # Fallback: token simple + lemma = token_lower
        toks = []
        for t in text.split():
            clean = t.strip().lower()
            toks.append((t, clean))
        return toks

    NEGATION_TOKENS = {"no", "nunca", "jamás", "sin", "nadie", "ninguno", "ni"}
    INTENSIFIERS = {
        "muy": 1.5, "extremadamente": 2.0, "sumamente": 1.8, "totalmente": 1.6,
        "algo": 0.8, "un": 0.9, "poco": 0.6, "poco": 0.6
    }

    def detect_negation_scope(tokens: List[Tuple[str, str]], window: int = 3) -> List[bool]:
        flags = [False] * len(tokens)
        for i, (tok, lemma) in enumerate(tokens):
            if lemma in NEGATION_TOKENS:
                for j in range(i + 1, min(len(tokens), i + 1 + window)):
                    flags[j] = True
        return flags

    def detect_intensifiers(tokens: List[Tuple[str, str]]) -> List[float]:
        mults = [1.0] * len(tokens)
        for i, (tok, lemma) in enumerate(tokens):
            if lemma in INTENSIFIERS and i + 1 < len(tokens):
                mults[i + 1] *= INTENSIFIERS[lemma]
        return mults


# Emociones base (fijas en todo el proyecto)
EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "neutral"]

# Basic Spanish stopword list (ASCII) to filter low-information tokens.
STOPWORDS = {
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra",
    "de", "del", "desde", "donde", "durante", "e", "el", "ella", "ellas", "ellos",
    "en", "entre", "era", "eran", "eres", "es", "esa", "ese", "eso", "esta", "estaba",
    "estaban", "estar", "este", "esto", "estos", "fue", "fueron", "ha", "habia", "haber",
    "hace", "hacer", "han", "hasta", "hay", "he", "la", "las", "le", "les", "lo", "los",
    "mas", "me", "mi", "mis", "muy", "no", "nos", "nosotras", "nosotros", "nuestra",
    "nuestro", "nuestras", "nuestros", "o", "os", "otra", "otras", "otro", "otros",
    "para", "pero", "poco", "por", "porque", "que", "quien", "quienes", "se", "ser",
    "si", "sin", "sobre", "so", "su", "sus", "tambien", "te", "tener", "ti", "tu", "tus",
    "un", "una", "uno", "unos", "unas", "ya", "y", "yo"
}

# --- Original lexicon (kept for compatibility) ---
# --- Expanded bilingual lexicon ---
LEXICON = {
    "joy": [
        "happy","feliz","alegre","contento","encantado","entusiasmado","me encanta",
        "sonrisa","satisfecho","alegría","maravilloso"
    ],
    "anger": [
        "angry","enojado","furioso","molesto","rabia","enojo","irritado","fastidio","ira"
    ],
    "disgust": [
        "disgusted","disgust","asco","repugnante","repudio","asqueroso"
    ],
    "sadness": [
        "sad","triste","deprimido","abatido","melancólico","infeliz"
    ],
    "fear": [
        "fear","miedo","asustado","temor","aterrorizado","preocupado"
    ],
    "neutral": []
}


INTENSIFIERS = {"muy": 1.5, "súper": 1.6, "mucho": 1.3, "bastante": 1.2}
DEINTENSIFIERS = {"poco": 0.8, "algo": 0.9}
NEGATIONS = {"no", "nunca", "jamás", "ni"}

# Baseline VA mapping per emotion.
# Valence en [-1.0, 1.0] (negativo a positivo)
# Arousal en [0.0, 1.0] (bajo a alto)
VA_LOOKUP = {
    "joy":     (0.8, 0.7),
    "anger":   (-0.7, 0.8),
    "disgust": (-0.6, 0.6),
    "sadness": (-0.8, 0.3),
    "fear":    (-0.6, 0.9),
    "neutral": (0.0, 0.1),
}

# Heuristic polarity maps per emotion (values in [-1,1])
_EMO_VALENCE_PRIOR = {
    "joy": 0.90,
    "sadness": -0.85,
    "anger": -0.8,
    "fear": -0.7,
    "disgust": -0.9
}

_EMO_WEIGHT_ADJUST = {
    "joy": 1.0,
    "sadness": 1.08,
    "anger": 1.15,
    "fear": 1.12,
    "disgust": 0.6,
}

# Construimos un mapa lemma -> {"valence": float, "emotion": str}
def _canon_token(token: str) -> str:
    """Normalize tokens for lexicographic comparisons (lowercase, no accents)."""
    if not token:
        return ""
    token = unicodedata.normalize("NFD", token.strip().lower())
    token = "".join(ch for ch in token if ch.isalpha())
    return unicodedata.normalize("NFC", token)


LEXICON_MAP: Dict[str, Dict[str, Any]] = {}
for emo, words in LEXICON.items():
    prior = _EMO_VALENCE_PRIOR.get(emo, 0.0)
    for w in words:
        lemma = _canon_token(w)
        if not lemma:
            continue
        # si una palabra aparece en varios grupos, priorizamos la primera ocurrencia
        if lemma not in LEXICON_MAP:
            LEXICON_MAP[lemma] = {"valence": float(prior), "emotion": emo}


# --- Dynamic lexicon derived from the labeled corpus ---
_DYNAMIC_LEXICON_CACHE: Dict[str, Dict[str, Any]] | None = None
_LEXICON_CORPUS_ENV = "TEXT_EMO_DYNAMIC_LEXICON"
_TEXT_CFG = SETTINGS.text
_TEXT_DYNAMIC_CFG = _TEXT_CFG.dynamic_lexicon
_PATHS_CFG = SETTINGS.paths


def _default_corpus_path() -> Path | None:
    configured = _PATHS_CFG.dynamic_lexicon
    if configured:
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "data" / "spanish-mea-corpus-2023" / "spanish-meacorpus-2023-dataset.csv"


def _ensure_dynamic_lexicon() -> Dict[str, Dict[str, Any]]:
    global _DYNAMIC_LEXICON_CACHE
    if _DYNAMIC_LEXICON_CACHE is not None:
        return _DYNAMIC_LEXICON_CACHE

    _DYNAMIC_LEXICON_CACHE = {}

    corpus_path: Path | None = None
    corpus_path_str = os.getenv(_LEXICON_CORPUS_ENV)
    if corpus_path_str:
        corpus_path = Path(corpus_path_str).expanduser()
    else:
        corpus_path = _default_corpus_path()

    if corpus_path is None or not corpus_path.exists():
        return _DYNAMIC_LEXICON_CACHE

    counts = {emo: Counter() for emo in EMOTIONS if emo != "neutral"}
    totals = Counter()

    try:
        with corpus_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not row:
                    continue
                split = (row.get("split") or "train").strip().lower()
                if split in {"test", "holdout"}:
                    continue
                label_raw = (row.get("label") or "").strip().lower()
                label = LABEL_MAP.get(label_raw, label_raw)
                if label not in counts:
                    continue
                text = row.get("transcription") or ""
                norm = normalize_text(text)
                if not norm:
                    continue
                tokens = []
                for tok in norm.split():
                    if len(tok) <= 2:
                        continue
                    canon = _canon_token(tok)
                    if not canon or canon in STOPWORDS:
                        continue
                    tokens.append(canon)
                if not tokens:
                    continue
                for tok in tokens:
                    counts[label][tok] += 1
                    totals[tok] += 1
    except Exception:
        return _DYNAMIC_LEXICON_CACHE

    MIN_FREQ = int(_TEXT_DYNAMIC_CFG.min_freq)
    MIN_RATIO = float(_TEXT_DYNAMIC_CFG.min_ratio)

    for token, total in totals.items():
        if total < MIN_FREQ:
            continue
        best_emo = None
        best_freq = 0
        second = 0
        for emo, counter in counts.items():
            freq = counter.get(token, 0)
            if freq > best_freq:
                second = best_freq
                best_freq = freq
                best_emo = emo
            elif freq > second:
                second = freq
        if best_emo is None or best_freq == 0:
            continue
        ratio = best_freq / max(1, second if second else 1)
        if ratio < MIN_RATIO:
            continue
        valence = _EMO_VALENCE_PRIOR.get(best_emo, 0.0)
        _DYNAMIC_LEXICON_CACHE[token] = {
            "valence": float(valence),
            "emotion": best_emo,
            "source": "dynamic",
        }

    return _DYNAMIC_LEXICON_CACHE


def _lookup_lexicon_entry(lemma: str) -> Tuple[Dict[str, Any] | None, str | None]:
    entry = LEXICON_MAP.get(lemma)
    if entry is not None:
        return entry, "static"
    dynamic_map = _ensure_dynamic_lexicon()
    if dynamic_map:
        dyn = dynamic_map.get(lemma)
        if dyn is not None:
            return dyn, "dynamic"
    return None, None


# Positive/negative word sets for the legacy valence method
POSITIVE = {"feliz", "contento", "alegre", "genial", "excelente", "bien", "buenísimo", "me encanta", "gracias"}
NEGATIVE = {"triste", "mal", "enojo", "bronca", "fear", "temor", "disgust", "repugnante", "asqueroso", "deprimido", "odio"}


def _score_by_lexicon_simple(lemmas: List[str]) -> Dict[str, float]:
    """
    Backward compatibility with the previous method: count occurrences per emotion.
    """
    scores = {e: 0.0 for e in EMOTIONS if e != "neutral"}
    for emo, words in LEXICON.items():
        wset = {_canon_token(w) for w in words}
        s = sum(1.0 for l in lemmas if l in wset)
        scores[emo] = s
    return scores


def _estimate_valence_arousal(raw_text: str, toks_lemma: List[str]) -> VAD:
    """
    Legacy heuristic estimate (retained for compatibility).
    - Valence: (#pos - #neg) / (#pos + #neg + 1)
    - Arousal: exclamation count + uppercase tokens
    """
    pos = sum(1 for t in toks_lemma if t in POSITIVE)
    neg = sum(1 for t in toks_lemma if t in NEGATIVE)
    val_raw = (pos - neg) / float(pos + neg + 1)
    valence = max(-1.0, min(1.0, val_raw))

    excls = raw_text.count("!")
    upper_tokens = sum(1 for w in raw_text.split() if (len(w) >= 2 and w.isupper()))
    ar_raw = min(1.0, 0.2 * excls + 0.1 * upper_tokens)
    arousal = max(0.0, min(1.0, ar_raw))

    dominance = 0.5
    return VAD(valence=valence, arousal=arousal, dominance=dominance)


def predict(text: str) -> EmotionResult:
    """
    Predict emotion from TEXT. Stable signature for the orchestrator.

    Args:
        text: raw user-provided text.

    Returns:
        EmotionResult containing emotion probabilities, VAD, confidence, and metadata.
    """
    raw = text or ""
    norm = normalize_text(raw)
    # Lemmatize (always produce a list of (token, lemma); fall back when preprocessing is absent)
    token_pairs: List[Tuple[str, str]] = lemmatize_tokens(norm)
    lemmas_raw: List[str] = [lemma for (_tok, lemma) in token_pairs]
    lemmas: List[str] = [_canon_token(lemma) for lemma in lemmas_raw]

    # No tokens? return pure neutral
    if len(lemmas) == 0:
        probs = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
        return EmotionResult(probs=probs, vad=VAD(0.0, 0.5, 0.5), confidence=0.4, top_label="neutral", metadata={})

    # Detect negation and intensifiers (functions defined in preproc or fallback)
    neg_flags: List[bool] = detect_negation_scope(token_pairs, window=3)
    intens_mults: List[float] = detect_intensifiers(token_pairs)

    # 1) Scoring using LEXICON_MAP (fine-grained)
    valence_scores: List[float] = []
    emotion_counts: Dict[str, float] = {e: 0.0 for e in EMOTIONS if e != "neutral"}
    matched = 0
    matched_sources: List[str] = []
    considered_indices = [idx for idx, lemma in enumerate(lemmas) if lemma and lemma not in STOPWORDS]

    for i in considered_indices:
        lemma = lemmas[i]
        entry, source = _lookup_lexicon_entry(lemma)
        if entry is None:
            continue
        matched += 1
        v = float(entry.get("valence", 0.0))  # prior [-1..1]
        # Apply negation when applicable (attenuated inversion)
        if neg_flags[i]:
            v = -v * 0.9
        # aplicar intensificador
        v = max(-1.0, min(1.0, v * (intens_mults[i] if i < len(intens_mults) else 1.0)))
        valence_scores.append(v)
        emo = entry.get("emotion")
        if emo:
            intens = intens_mults[i] if i < len(intens_mults) else 1.0
            weight = abs(v) * _EMO_WEIGHT_ADJUST.get(emo, 1.0)
            if intens > 1.05:
                weight *= min(1.4, 1.0 + (intens - 1.0) * 0.6)
            if emo in {"anger", "fear"} and intens > 1.15:
                weight *= 1.1
            if emo == "disgust":
                if intens <= 1.05:
                    weight *= 0.75
                elif intens <= 1.2:
                    weight *= 0.85
                if neg_flags[i]:
                    weight *= 0.6
            emotion_counts[emo] = emotion_counts.get(emo, 0.0) + weight
        if source:
            matched_sources.append(source)

    scores_simple = _score_by_lexicon_simple(lemmas)

    # 2) Fallback simple si no hay coincidencias en LEXICON_MAP:
    if matched == 0:
        # Use the simple method based on lemma counts in LEXICON
        total_signal = sum(scores_simple.values())
        # convertimos a probs como antes (suavizado)
        k = 0.5
        base_neutro = float(_TEXT_CFG.fallback_neutral) if total_signal == 0 else 0.30
        emo_probs = {emo: scores_simple.get(emo, 0.0) + k for emo in EMOTIONS if emo != "neutral"}
        s = sum(emo_probs.values()) or 1.0
        emo_probs = {emo: v / s * (1.0 - base_neutro) for emo, v in emo_probs.items()}
        probs: Dict[str, float] = {**emo_probs, "neutral": base_neutro}
        vad = _estimate_valence_arousal(raw, lemmas)
        coverage = 0.0
        metadata = {
            "lemmas": lemmas,
            "lemmas_raw": lemmas_raw,
            "neg_flags": neg_flags,
            "intensifiers_mult": intens_mults,
            "valence_scores": valence_scores,
            "coverage": coverage,
            "fallback": "simple_lexicon",
            "matched_sources": matched_sources,
        }
        confidence = min(0.9, 0.4 + 0.1 * total_signal)
        top = max(probs, key=probs.get)
        logger.info("text_emotion.predict (fallback): coverage=0.0", extra={"extra_fields": {"matched": matched}})
        return EmotionResult(probs=probs, vad=vad, confidence=float(confidence), top_label=top, metadata=metadata)

    # 3) Consolidation when LEXICON_MAP matches were found
    coverage = matched / max(1, len(considered_indices))

    if coverage < 0.22:
        emotion_counts["disgust"] *= 0.6
    elif coverage < 0.32:
        emotion_counts["disgust"] *= 0.75
    elif coverage < 0.45:
        emotion_counts["disgust"] *= 0.9

    if coverage < 0.25:
        # Blend with the simple lexicon gently to avoid losing signal with few matches
        blend = 0.2 if coverage < 0.15 else 0.1
        for emo, score in scores_simple.items():
            if score > 0:
                emotion_counts[emo] = emotion_counts.get(emo, 0.0) + blend * score

    # Valence promedio de tokens emparejados
    valence = sum(valence_scores) / len(valence_scores) if valence_scores else 0.0

    # Convert emotion_counts into a probability distribution (smoothed)
    k = 0.5
    emo_probs = {emo: emotion_counts.get(emo, 0.0) + k for emo in emotion_counts}
    s = sum(emo_probs.values()) or 1.0
    base_neutro = float(_TEXT_CFG.fallback_neutral)
    total_signal = sum(emotion_counts.values())
    if total_signal > 0:
        if coverage >= 0.25:
            base_neutro = min(base_neutro, 0.28)
        elif coverage >= 0.15:
            base_neutro = min(base_neutro, 0.38)
        else:
            base_neutro = min(base_neutro, 0.45)
    if abs(valence) >= 0.6 and coverage >= 0.25:
        base_neutro = min(base_neutro, 0.24)
    elif abs(valence) >= 0.45 and coverage >= 0.25:
        base_neutro = min(base_neutro, 0.26)

    top_neg = None
    top_neg_score = 0.0
    for emo in ("anger", "fear", "sadness"):
        score = emotion_counts.get(emo, 0.0)
        if score > top_neg_score:
            top_neg = emo
            top_neg_score = score

    if top_neg and coverage >= 0.2 and total_signal > 0:
        base_neutro = min(base_neutro, 0.22)

    emo_probs = {emo: v / s * (1.0 - base_neutro) for emo, v in emo_probs.items()}
    probs: Dict[str, float] = {**emo_probs, "neutral": base_neutro}

    # VAD: usamos valence calculado; dejamos arousal legacy (exclamaciones / mayus) para compatibilidad
    vad_legacy = _estimate_valence_arousal(raw, lemmas)
    vad = VAD(valence=valence, arousal=vad_legacy.arousal, dominance=0.5)

    # Heuristic confidence based on coverage
    confidence = max(0.2, min(0.98, 0.2 + 0.8 * coverage))

    if top_neg and coverage >= 0.2:
        gap = probs[top_neg] - probs["neutral"]
        if gap < 0.12 and probs["neutral"] > 0.15:
            delta = min(0.12 - gap, probs["neutral"] * 0.2)
            probs[top_neg] += delta
            probs["neutral"] -= delta

    if top_neg and confidence >= 0.55 and coverage >= 0.22:
        boost = min(0.06, probs["neutral"] * 0.25)
        probs[top_neg] += boost
        probs["neutral"] -= boost

    metadata = {
        "lemmas": lemmas,
        "lemmas_raw": lemmas_raw,
        "neg_flags": neg_flags,
        "intensifiers_mult": intens_mults,
        "valence_scores": valence_scores,
        "coverage": coverage,
        "matched_sources": matched_sources,
        "coverage_adjusted_neutral": base_neutro,
    }

    top = max(probs, key=probs.get)
    return EmotionResult(probs=probs, vad=vad, confidence=float(confidence), top_label=top, metadata=metadata)
