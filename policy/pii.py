# policy/pii.py
import re

from config.settings import SETTINGS

PII_PATTERNS = SETTINGS.PII_PATTERNS or {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"\b(\+?54\s?9?\s?)?(?:(0\d{2,4}|\(?\d{2,4}\)?)\s?)?(15\s?)?(\d{6,8}|\d{4}[\s\-]\d{4})\b",
}

def redact(text: str) -> str:
    if not text:
        return text
    out = text
    for name, pat in PII_PATTERNS.items():
        out = re.sub(pat, f"[REDACTED_{name.upper()}]", out)
    return out
