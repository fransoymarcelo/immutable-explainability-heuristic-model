# audit/service.py
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Tuple
from pathlib import Path
import os
from config.settings import SETTINGS

# Archivo local donde se guardan los eventos redaccionados (off-chain)
_AUDIT_DIR = SETTINGS.audit_dir
_AUDIT_FILE = SETTINGS.audit_file
_AUDIT_PATH = Path(_AUDIT_DIR) / _AUDIT_FILE
_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def compute_hash(redacted_event: Dict) -> str:
    payload = json.dumps(redacted_event, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def record_event(redacted_event: Dict) -> Tuple[str, Dict]:
    """
    Guarda el evento redaccionado en un JSONL local y devuelve:
      - txid: hash SHA-256 del evento (para anclar en blockchain en el futuro)
      - stored_event: el evento enriquecido con timestamp y txid
    """
    evt = {
        "ts": now_iso(),
        **redacted_event
    }
    txid = compute_hash(evt)
    stored = {"txid": txid, **evt}
    _append_jsonl(_AUDIT_PATH, stored)
    return txid, stored
