# asr/whisper_asr.py
import os
import threading
from typing import List, Optional
from common.types import ASRResult, ASRWord
from config.settings import SETTINGS

try:
    import whisper  # paquete 'openai-whisper'
except Exception as e:
    whisper = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# --- Simple configuration (no Pydantic dependency yet) ---
# Choose the model size without touching code: tiny|base|small|medium|large
os.environ.setdefault("WHISPER_MODEL_SIZE", SETTINGS.models.whisper_size)
# --- In-memory model cache (load once) ---
_model = None
_model_lock = threading.Lock()
_CURRENT_SIZE: Optional[str] = os.environ.get("WHISPER_MODEL_SIZE")


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # Fallback to /tmp if the preferred path cannot be created
        try:
            tmp = "/tmp/.cache"
            os.makedirs(tmp, exist_ok=True)
        except Exception:
            pass

def get_model():
    global _model, _CURRENT_SIZE
    """
    Return the cached Whisper model.
    Enforce a writable download_root (preferably via WHISPER_DOWNLOAD_ROOT env var).
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        # 1) Decide download_root: priority ENV -> /app/.cache -> /tmp/.cache
        download_root = os.environ.get("WHISPER_DOWNLOAD_ROOT")
        if not download_root:
            # Prefer /app/.cache because containers typically allow writing there
            cwd = os.getcwd()
            download_root = os.path.join(cwd, ".cache")
        _ensure_dir(download_root)

        # 2) Also set XDG_CACHE_HOME in case any dependency expects it
        os.environ.setdefault("XDG_CACHE_HOME", download_root)

        # 3) Load the model, passing download_root when supported
        model_size = os.environ.get("WHISPER_MODEL_SIZE") or SETTINGS.models.whisper_size
        try:
            # Newer versions of whisper accept download_root
            _model = whisper.load_model(model_size, download_root=download_root)
        except TypeError:
            # Fallback when older load_model versions reject download_root
            _model = whisper.load_model(model_size)
        except Exception:
            # Re-raise so the service logs the full error
            raise

        _CURRENT_SIZE = model_size
        return _model

def _to_words(result) -> List[ASRWord]:
    """
    Convert whisper output into an ASRWord list.
    Use word_timestamps when available; otherwise fall back to segments.
    """
    words: List[ASRWord] = []
    # Recent versions expose word-level timestamps via result["segments"]
    for seg in result.get("segments", []) or []:
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                words.append(
                    ASRWord(
                        word=w.get("word", "").strip(),
                        start=float(w.get("start", 0.0)),
                        end=float(w.get("end", 0.0)),
                        confidence=float(w.get("probability", 0.5)),
                    )
                )
        else:
            # Fallback: when words are absent, treat the full segment as a "word"
            words.append(
                ASRWord(
                    word=seg.get("text", "").strip(),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    confidence=float(seg.get("avg_logprob", -0.5)),  # not a true probability, but a useful signal
                )
            )
    return words

def transcribe(audio_path: str) -> ASRResult:
    """
    Stable API matching the previous version.
    - Load the model once (cache)
    - Request word timestamps when the model/version supports them
    """
    model = get_model()
    result = model.transcribe(audio_path, word_timestamps=True)
    text = result.get("text", "").strip()
    words = _to_words(result)
    # Simple global confidence: average the word confidences whenever available
    confs = [w.confidence for w in words if isinstance(w.confidence, (int, float))]
    confidence = float(sum(confs) / len(confs)) if confs else 0.55
    return ASRResult(text=text, words=words, confidence=confidence)


def set_model_size(size: str) -> None:
    """
    Adjust the active Whisper model size and reset the cache whenever it changes.
    """
    if not size:
        return
    global _model, _CURRENT_SIZE
    os.environ["WHISPER_MODEL_SIZE"] = size
    with _model_lock:
        if _CURRENT_SIZE == size and _model is not None:
            return
        _model = None
        _CURRENT_SIZE = size


def get_model_size() -> str:
    """
    Return the active or configured model size.
    """
    if _CURRENT_SIZE:
        return _CURRENT_SIZE
    return os.environ.get("WHISPER_MODEL_SIZE") or SETTINGS.models.whisper_size