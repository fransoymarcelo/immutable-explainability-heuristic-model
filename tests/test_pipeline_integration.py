# tests/test_pipeline_integration.py
import os
import pytest
from app.orchestrator import run_pipeline
import app.orchestrator as orchestrator

# fixtures: monkeypatch external dependencies to deterministic stubs
class DummyASR:
    def __init__(self, text="hola"):
        self.text = text
        self.confidence = 0.9
        self.words = []

class DummyEmotion:
    def __init__(self, probs=None, vad=None, confidence=0.5, metadata=None):
        self.probs = probs or {"neutral": 1.0}
        self.vad = vad or type("V", (), {"valence": 0.0, "arousal": 0.0, "dominance": 0.5})
        self.confidence = confidence
        self.metadata = metadata or {}

def fake_record_event(event):
    # return fake txid and stored_event (echo)
    return "txid_fake", event

def test_run_pipeline_guardrails_trigger(monkeypatch, tmp_path):
    # monkeypatch transcribe to return text with suicidal phrase to trigger guardrails
    monkeypatch.setattr(orchestrator, "transcribe", lambda p: DummyASR(text="me quiero quitar la vida"), raising=False)
    # audio emotion: neutral -- permitimos crear el atributo si no existe (raising=False)
    monkeypatch.setattr(orchestrator, "predict_audio_emotion", lambda p, use_mfcc=False: DummyEmotion(probs={"neutral":1.0}, vad={"valence":0.0,"arousal":0.0}), raising=False)
    # text emotion: neutral
    monkeypatch.setattr(orchestrator, "predict_text_emotion", lambda t: DummyEmotion(probs={"neutral":1.0}, vad={"valence":0.0,"arousal":0.0}), raising=False)
    # record_event stub
    monkeypatch.setattr(orchestrator, "record_event", lambda e: ("txid_stub", e), raising=False)
    # create a small dummy wav file for the API (not needed by our stubs but run_pipeline expects a path)
    dummy_audio = tmp_path / "dummy.wav"
    dummy_audio.write_bytes(b"RIFF----WAVEfmt ")  # minimal garbage - not opened by stubs
    out = run_pipeline(str(dummy_audio), metrics_port=None, run_id="test_run")
    # assert response is safe template (since action should be suppress_and_escalate)
    assert "Lo siento" in out["response"] or "serio" in out["response"] or out["audit"]["stored_event"].get("escalation") is not None
    # ensure audit stored_event exists
    assert "audit" in out and "stored_event" in out["audit"]
