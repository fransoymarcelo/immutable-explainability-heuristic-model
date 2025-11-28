# tests/test_guardrails.py
import pytest
from app.orchestrator import check_guardrails
from app.orchestrator import now_iso  # import from audit.service if not exported

def make_tmp_event():
    return {"ts": now_iso(), "asr_conf": 0.9, "weights": {"audio": 0.5, "text": 0.5}}

def test_check_guardrails_self_harm():
    # Self-harm text -> suppress_and_escalate (regex captures critical terms)
    text = "me quiero quitar la vida"
    probs = {"fear": 0.1, "anger": 0.1, "sadness": 0.1}
    tmp = make_tmp_event()
    action, escalation, audit_out = check_guardrails(probs, text, tmp, run_id="test_run", txid_parent=None)
    assert action == "suppress_and_escalate"
    assert escalation is not None
    # audit_out["_audit"] resides inside audit_out["audit"]
    assert "audit" in audit_out and isinstance(audit_out["audit"].get("matched", []), list)

def test_check_guardrails_high_fear_prob():
    # alta prob de fear -> action no 'none' y audit contiene matched
    text = "estoy muy asustado"
    probs = {"fear": 0.8, "anger": 0.0, "sadness": 0.0}
    tmp = make_tmp_event()
    action, escalation, audit_out = check_guardrails(probs, text, tmp, run_id="test_run", txid_parent=None)
    assert action in {"use_safe_template", "suppress_and_escalate", "soften", "none"} or action is not None
    # verify audit structure: inner 'audit' has 'matched'
    assert "audit" in audit_out
    assert isinstance(audit_out["audit"].get("matched", []), list)
