# text_emotion/preproc.py
from typing import List, Tuple, Dict
try:
    import spacy
    nlp = spacy.load("es_core_news_sm", disable=["parser"])  # enable parser if you need dependency features
    HAS_SPACY = True
except Exception:
    nlp = None
    HAS_SPACY = False

# Basic dictionaries for negation and intensifier terms
NEGATION_TOKENS = {"no", "nunca", "jamás", "sin", "nadie", "ninguno", "ni"}
INTENSIFIERS = {
    "muy": 1.5, "extremadamente": 2.0, "sumamente": 1.8, "totalmente": 1.6,
    "algo": 0.8, "un poco": 0.7, "poco": 0.6
}

def lemmatize_tokens(text: str) -> List[Tuple[str,str]]:
    """
    Devuelve lista (token, lemma). Usa spacy si está disponible; si no, devuelve tokens/raw.
    """
    if HAS_SPACY and nlp is not None:
        doc = nlp(text)
        return [(t.text, t.lemma_.lower()) for t in doc]
    # fallback simple: split y lower
    return [(t, t.lower()) for t in text.split()]

def detect_negation_scope(tokens: List[Tuple[str,str]], window: int = 3) -> List[bool]:
    """
    Marca qué tokens caen dentro del scope de una negación sencilla.
    Heurística: si aparece token de negación, marca los siguientes `window` tokens.
    """
    flags = [False] * len(tokens)
    for i, (token, lemma) in enumerate(tokens):
        if lemma in NEGATION_TOKENS:
            for j in range(i+1, min(len(tokens), i+1+window)):
                flags[j] = True
    return flags

def detect_intensifiers(tokens: List[Tuple[str,str]]) -> List[float]:
    """
    Devuelve multiplicador por token (1.0 por defecto). Si el token es intensificador,
    apply its multiplier to the following token.
    """
    mults = [1.0] * len(tokens)
    for i, (token, lemma) in enumerate(tokens):
        if lemma in INTENSIFIERS:
            if i+1 < len(tokens):
                mults[i+1] *= INTENSIFIERS[lemma]
    return mults
