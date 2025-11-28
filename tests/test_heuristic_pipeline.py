import math
import os
import struct
import tempfile
import wave

import pytest

pytest.importorskip("spacy")

from app.orchestrator import check_guardrails
from audio_emotion.model import predict as audio_predict
from config.settings import SETTINGS
from fusion import fuse_simple
from fuzzy.engine import FuzzyEngine
from policy.pii import redact
from text_emotion.model import predict as text_predict


def _generate_wav(duration_sec: float = 0.5, sample_rate: int = 16000, freq: float = 220.0) -> str:
    num_samples = int(sample_rate * duration_sec)
    amplitude = 0.2
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            frames = [
                struct.pack('<h', int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sample_rate)))
                for i in range(num_samples)
            ]
            wf.writeframes(b"".join(frames))
        return tmp.name


def test_text_model_detects_positive_emotion():
    result = text_predict("Estoy muy feliz y agradecido por el apoyo")
    assert isinstance(result.probs, dict)
    assert pytest.approx(sum(result.probs.values()), rel=1e-6, abs=1e-6) == 1.0
    assert result.top_label in result.probs
    assert result.metadata.get("lemmas_raw") is not None


def test_audio_model_returns_vad_and_probs():
    wav_path = _generate_wav()
    try:
        result = audio_predict(wav_path, use_mfcc=False)
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass
    assert 0.0 <= result.vad.arousal <= 1.0
    assert -1.0 <= result.vad.valence <= 1.0
    assert pytest.approx(sum(result.probs.values()), rel=1e-6, abs=1e-6) == 1.0


def test_fusion_simple_prefers_text_when_confident():
    cfg = SETTINGS.fusion.fusion_simple
    prob_text = {"joy": 0.9, "neutral": 0.1}
    prob_audio = {"joy": 0.2, "neutral": 0.8}
    fused, details = fuse_simple(
        prob_text,
        prob_audio,
        asr_conf=cfg.asr_confidence_thresh + 0.1,
        ar_audio=cfg.audio_arousal_thresh / 2,
    )
    assert details["fired_rule"] == "R_text_strong"
    assert pytest.approx(sum(fused.values()), rel=1e-6, abs=1e-6) == 1.0


def test_fuzzy_engine_outputs_weight_between_0_and_1():
    engine = FuzzyEngine(SETTINGS.fusion.fuzzy_ruleset)
    result = engine.infer_w_text(asr_conf=0.8, arousal=0.6, valence=0.1)
    assert 0.0 <= result["w_text"] <= 1.0
    assert "details" in result
    assert isinstance(result["details"].get("fired_rules", []), list)


def test_guardrails_detects_self_harm_language():
    probs = {"fear": 0.9, "neutral": 0.1}
    stored_event = {"vad": {"arousal": 0.85, "valence": -0.5}}
    action, escalation, _ = check_guardrails(
        probs_fused=probs,
        text="me quiero quitar la vida ahora",
        stored_event=stored_event,
        run_id="test_run",
        txid_parent=None,
    )
    assert action == "suppress_and_escalate"
    assert escalation is not None


def test_pii_redaction_uses_configured_patterns():
    redacted = redact("Mi correo es test@example.com")
    assert "REDACTED_EMAIL" in redacted


