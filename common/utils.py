"""
common/utils.py

🧹 Utilidades generales para preprocesamiento de texto y normalización.

Actualmente se usa en:
- text_emotion/model.py  → limpieza y uniformización del texto
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Limpia y normaliza texto en español para análisis lingüístico.

    Pasos:
    1. Pasa a minúsculas.
    2. Normaliza acentos y caracteres Unicode (NFD).
    3. Elimina símbolos, números y puntuación no relevante.
    4. Reduce espacios múltiples a uno solo.

    Args:
        text (str): Texto original (puede incluir mayúsculas, tildes, signos, etc.)

    Returns:
        str: Texto normalizado, limpio y en minúsculas.
    """
    if not text:
        return ""
    # Step 1: lowercase
    text = text.lower()

    # Paso 2: normalizar tildes y caracteres Unicode
    text = unicodedata.normalize("NFD", text)

    # Step 3: drop symbols, digits, and non-alphabetic chars (keep ñ and accented vowels)
    text = re.sub(r"[^a-záéíóúüñ ]+", "", text)

    # Step 4: collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
