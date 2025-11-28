# app/orchestrator.py
import dataclasses
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache
from fuzzy.engine import FuzzyEngine
import os
import re
import requests

# 1) Import existing components
from asr import whisper_asr
from audio_emotion.model import predict as predict_audio_emotion

from text_emotion.model import predict as predict_text_emotion

# 2) Security and auditing layer (MVP)
from policy.pii import redact
from audit.service import record_event, now_iso
from utils.logging import get_logger
from config.settings import SETTINGS
import uuid

from utils.visuals import fired_rules_to_matrix, export_fired_rules_json, save_heatmap_from_matrix
from blockchain.service import anchor_txid as blockchain_anchor_txid


# Prometheus metrics
# Keep metrics-related imports at the module header
from utils.metrics import (
    init_metrics,
    timeit,
    asr_latency,
    audio_emotion_latency,
    text_emotion_latency,
    fusion_latency,
    pii_redactions_total,
    pipeline_errors_total,
    asr_confidence_gauge,
    audio_snr_db_gauge,
    inc_counter,
    set_gauge,
    inc_guardrails,
    cross_modal_coherence_gauge,
)

AUDIT_FIRED_DIR = os.path.join(SETTINGS.paths.audit_dir, "fired_rules")

# loggers
logger = get_logger("pipeline")
guard_logger = get_logger("guardrails")

def _plan_response(text: str, fused_probs: Dict[str, float], response_hint: Optional[str] = None, action: Optional[str] = None) -> str:
    """
    Minimal empathetic response planner. Guardrail hints take precedence.
    """
    # honour guardrail hints first
    if action == "suppress_and_escalate" and response_hint:
        return response_hint
    if action == "use_safe_template" and response_hint:
        return response_hint
    if action == "soften" and response_hint:
        return response_hint

    # fallback
    if not text.strip():
        return "Puedo escucharte; cuando estés listo, contame más."
    top = max(fused_probs, key=fused_probs.get) if fused_probs else "neutral"
    if top == "sadness":
        return "Siento que esto te afecta; ¿querés que lo desarmemos paso a paso?"
    if top == "anger":
        return "Entiendo tu frustración; respiremos un segundo y lo resolvemos juntos."
    if top == "fear":
        return "Es normal sentir incertidumbre; estoy acá para ayudarte con claridad."
    if top == "joy":
        return "¡Excelente! Sigamos aprovechando ese envión positivo."
    return "Te sigo; ¿podés ampliar un poco así te ayudo mejor?"

# ---------------------- Guardrails helper ----------------------
def check_guardrails(probs_fused: Dict[str, float],
                     text: str,
                     stored_event: Dict[str, Any],
                     run_id: str,
                     txid_parent: Optional[str] = None
                     ) -> Tuple[str, Optional[dict], dict]:
    """
    Evaluate guardrail risk conditions.
    - Returns (action, escalation_info_or_None, audit_info)
      where action ∈ {"none","soften","use_safe_template","suppress_and_escalate"}.
    - May mutate stored_event (adds an 'escalation' key).
    """
    action = "none"
    escalation = None
    audit: Dict[str, Any] = {"matched": [], "params": {}}

    # Configurable thresholds
    thresholds = SETTINGS.guardrails.thresholds
    THRESH_MIEDO = thresholds.fear
    THRESH_IRA = thresholds.anger
    THRESH_TRISTEZA = thresholds.sadness
    THRESH_AROUSAL = thresholds.arousal
    THRESH_VALENCE_NEG = thresholds.valence_neg

    audit["params"] = {
        "THRESH_MIEDO": THRESH_MIEDO,
        "THRESH_IRA": THRESH_IRA,
        "THRESH_TRISTEZA": THRESH_TRISTEZA,
        "THRESH_AROUSAL": THRESH_AROUSAL,
        "THRESH_VALENCE_NEG": THRESH_VALENCE_NEG
    }

    # Sensitive keywords
    risk_keywords = SETTINGS.guardrails.risk_keywords

    text_lower = (text or "").lower()

    # 1) High emotion probabilities
    miedo_p = float(probs_fused.get("fear", 0.0))
    ira_p = float(probs_fused.get("anger", 0.0))
    tristeza_p = float(probs_fused.get("sadness", 0.0))

    if miedo_p >= THRESH_MIEDO:
        audit["matched"].append({"type": "prob", "label": "fear", "value": miedo_p})
    if ira_p >= THRESH_IRA:
        audit["matched"].append({"type": "prob", "label": "anger", "value": ira_p})
    if tristeza_p >= THRESH_TRISTEZA:
        audit["matched"].append({"type": "prob", "label": "sadness", "value": tristeza_p})

    # 2) Sensitive text (store matched patterns)
    matches = [k for k in risk_keywords if re.search(k, text_lower)]
    for m in matches:
        audit["matched"].append({"type": "text", "pattern": m})

    # 3) VAD risk if present in stored_event
    vad = stored_event.get("vad") or stored_event.get("metadata", {}).get("vad", {})
    arousal = float(vad.get("arousal", 0.0)) if vad else 0.0
    valence = float(vad.get("valence", 0.0)) if vad else 0.0
    if arousal >= THRESH_AROUSAL and valence <= THRESH_VALENCE_NEG:
        audit["matched"].append({"type": "vad", "desc": f"high_arousal_low_valence", "arousal": arousal, "valence": valence})

    # ----------------- Decision flow (priority order) -----------------
    # 1) Critical self-harm textual patterns -> suppress_and_escalate (PRIORITY)
    critical_indicators = ("suicid", "quitar", "quitarme", "no quiero vivir")
    if any(isinstance(m, dict) and m.get("type") == "text" and any(ci in m.get("pattern", "") for ci in critical_indicators) for m in audit["matched"]):
        action = "suppress_and_escalate"
        escalation = {"level": "critical", "reason": "self_harm_terms", "matched": audit["matched"]}
    # 2) Explicit violent language + high anger -> use_safe_template + escalate
    elif any(isinstance(m, dict) and m.get("type") == "text" and ("matar" in m.get("pattern", "") or "hacer daño" in m.get("pattern", "")) for m in audit["matched"]) and ira_p > 0.6:
        action = "use_safe_template"
        escalation = {"level": "high", "reason": "violence_terms_high_anger", "matched": audit["matched"]}
    # 3) High probability of extreme emotion -> use_safe_template
    elif miedo_p >= 0.6 or ira_p >= 0.7 or tristeza_p >= 0.75:
        action = "use_safe_template"
        escalation = {"level": "high", "reason": "high_emotion_probs", "matched": audit["matched"]}
    # 4) Any other matched textual or VAD condition -> soften
    elif audit["matched"]:
        action = "soften"
        escalation = {"level": "notice", "reason": "sensitive_terms_or_mid_probs", "matched": audit["matched"]}
    else:
        action = "none"
        escalation = None

    # Persist escalation information when applicable
    if escalation:
        escalation_record = {
            "escalation_ts": now_iso(),
            "run_id": run_id,
            "txid_parent": txid_parent,
            "probs": {k: float(v) for k, v in probs_fused.items()},
            "text_snippet": (text or "")[:512],
            "info": escalation
        }
        stored_event["escalation"] = escalation_record
        guard_logger.warning("escalation:triggered", extra={"extra_fields": escalation_record})
        webhook = SETTINGS.paths.escalation_webhook
        if webhook:
            try:
                requests.post(webhook, json=escalation_record, timeout=1.0)
            except Exception:
                guard_logger.exception("failed to notify escalation webhook")

    audit_out = {"action": action, "escalation": escalation, "audit": audit}
    return action, escalation, audit_out
# -------------------- end guardrails helper ---------------------

def _safe_asdict(v):
    """
    Convert dataclasses to dicts, keep dicts unchanged, and fall back to vars().
    Helps tests avoid serializing custom objects manually.
    """
    if v is None:
        return None
    if dataclasses.is_dataclass(v):
        return dataclasses.asdict(v)
    if isinstance(v, dict):
        return v
    # Last resort fallback: try vars()
    try:
        return dict(vars(v))
    except Exception:
        return str(v)

def _cross_modal_coherence(vad_audio, vad_text) -> float:
    """
    Heuristic coherence index between audio and text channels.
    Uses absolute differences on Valence [-1..1] and Arousal [0..1].
    Returns a value in [0, 1], where 1 represents perfect coherence.
    """
    if not vad_audio or not vad_text:
        return 0.5  # Neutral default when data is missing

    try:
        va = float(vad_audio.get("valence", 0.0))
        aa = float(vad_audio.get("arousal", 0.5))
        vt = float(vad_text.get("valence", 0.0))
        at = float(vad_text.get("arousal", 0.5))

        # Normalized valence difference (max delta is 2 between -1 and +1)
        dv = abs(va - vt) / 2.0
        da = abs(aa - at) / 1.0
        coherence = max(0.0, min(1.0, 1.0 - (dv + da) / 2.0))
        return coherence
    except Exception:
        return 0.5


@lru_cache(maxsize=4)
def _get_fuzzy_engine(ruleset_path: str) -> FuzzyEngine:
    return FuzzyEngine(ruleset_path)

def run_pipeline(
    audio_path: str,
    metrics_port: int | None = None,
    run_id: str | None = None,
    *,
    blockchain_enabled: Optional[bool] = None,
    whisper_model_size: Optional[str] = None,
    fuzzy_ruleset_path: Optional[str] = None,
    use_mfcc: Optional[bool] = None,
) -> Dict[str, Any]:

    """
    Minimal orchestrator with Prometheus instrumentation.
    - If metrics_port is provided, exposes /metrics on that port.
    """

    effective_metrics_port = metrics_port if metrics_port is not None else SETTINGS.metrics.port
    if metrics_port is not None:
        init_metrics(metrics_port)

    model_size = SETTINGS.models.whisper_size
    if run_id is None:
        run_id = SETTINGS.run_id
    ruleset_path = os.path.abspath(fuzzy_ruleset_path) if fuzzy_ruleset_path else os.path.abspath(SETTINGS.fusion.fuzzy_ruleset)
    fuzzy_engine = _get_fuzzy_engine(ruleset_path)
    metrics_labels = {
        "model_size": model_size,
        "run_id": run_id,
    }

    # Ensure the logger is bound locally
    logger = get_logger("pipeline")
    logger.info("run:start", extra={"extra_fields": {"run_id": run_id}})

    original_model_size: Optional[str] = None
    restore_model_size = False
    if whisper_model_size:
        original_model_size = whisper_asr.get_model_size()
        if original_model_size != whisper_model_size:
            whisper_asr.set_model_size(whisper_model_size)
            restore_model_size = True
        metrics_labels["model_size"] = whisper_model_size

    use_mfcc_flag = SETTINGS.models.use_mfcc if use_mfcc is None else bool(use_mfcc)
    try:
        # --- 1) ASR ---
        with timeit(asr_latency, metrics_labels):
            asr_res = whisper_asr.transcribe(audio_path)  # ASRResult(text, words, confidence)
        raw_text = asr_res.text or ""
        asr_conf_raw = float(asr_res.confidence or 0.0)
        asr_conf_downstream = asr_conf_raw
        metrics_labels["model_size"] = whisper_asr.get_model_size()

        # --- 2) Emotion ---
        with timeit(audio_emotion_latency, metrics_labels):
            emo_audio = predict_audio_emotion(audio_path, use_mfcc=use_mfcc_flag)   # {"probs": {...}, "vad": {...}, "confidence": float}
            # structured logging for audio result
            logger.info("emo_audio.result", extra={"extra_fields": {
                "vad": _safe_asdict(getattr(emo_audio, "vad", None)),
                "probs": getattr(emo_audio, "probs", None),
                "confidence": float(getattr(emo_audio, "confidence", 0.0)),
                "metadata": getattr(emo_audio, "metadata", None)
            }})
            # 2.1) SNR
            snr_db = None
            try:
                snr_db = _safe_asdict(getattr(emo_audio, "metadata", {}).get("snr_db"))
            except Exception:
                snr_db = None

            # 2.2) Adjust ASR confidence (soft influence)

            # If snr_db is numeric, update the labeled gauge
            if snr_db is not None:
                try:
                    set_gauge(audio_snr_db_gauge, metrics_labels, float(snr_db))
                except Exception as exc:
                    logger.debug("failed to set audio_snr_db_gauge: %s", exc)

            if snr_db is not None:
                if snr_db < 5.0:
                    # Penalize textual confidence when SNR is very low
                    asr_conf_downstream *= 0.5
                elif snr_db < 12.0:
                    asr_conf_downstream *= 0.8
                # else leave asr_conf

            # 2.3) Optionally disable MFCC usage downstream if SNR is very low
            if snr_db is not None:
                if snr_db < 3.0:
                    logger.info("audio:low_snr_disable_mfcc", extra={"extra_fields": {"snr_db": snr_db, "run_id": run_id}})

        with timeit(text_emotion_latency, metrics_labels):
            emo_text  = predict_text_emotion(raw_text)      # {"probs": {...}, "vad": {...}, "confidence": float}

        text_metadata = getattr(emo_text, "metadata", {}) or {}
        audio_metadata = getattr(emo_audio, "metadata", {}) or {}
        text_conf = float(getattr(emo_text, "confidence", 0.0) or 0.0)
        coverage = float(text_metadata.get("coverage", 0.0) or 0.0)
        audio_conf = float(getattr(emo_audio, "confidence", 0.0) or 0.0)
        text_probs = getattr(emo_text, "probs", {}) or {}
        audio_probs = getattr(emo_audio, "probs", {}) or {}
        text_top = max(text_probs, key=text_probs.get) if text_probs else "neutral"

        # --- 3) Fusion (fuzzy decides w_text; linear fallback)
        with timeit(fusion_latency, metrics_labels):
            # 3.1 Obtain continuous arousal and valence signals
            # Prefer arousal from the audio (prosody) channel and averaged valence
            aro_audio = float(getattr(emo_audio.vad, "arousal", 0.5))
            val_audio = float(getattr(emo_audio.vad, "valence", 0.0))   # assume [-1,1] when already normalized
            val_text  = float(getattr(emo_text.vad, "valence", 0.0))

            # If models output valence in [0,1], map to [-1,1]: v2 = v*2 - 1
            def to_minus1_1(v: float) -> float:
                try:
                    v = float(v)
                except Exception:
                    return 0.0
                # Respect existing [-1,1]; otherwise map [0,1] to [-1,1]
                if -1.0 <= v <= 1.0:
                    return v
                return (max(0.0, min(1.0, v)) * 2.0) - 1.0

            arousal  = max(0.0, min(1.0, aro_audio))
            valence  = max(-1.0, min(1.0, (to_minus1_1(val_audio) + to_minus1_1(val_text)) / 2.0))

            # --- Cross-modal coherence (audio vs text) ---
            vad_audio_dict = {"valence": val_audio, "arousal": aro_audio}
            vad_text_dict = {"valence": val_text, "arousal": getattr(emo_text.vad, "arousal", 0.0)}
            coherence = _cross_modal_coherence(vad_audio_dict, vad_text_dict)
            logger.info("cross_modal_coherence_gauge", extra={"extra_fields": {"coherence": coherence, "run_id": run_id}})
            set_gauge(cross_modal_coherence_gauge, metrics_labels, coherence)


            asr_mod_cfg = SETTINGS.fusion.asr_modulation
            post_cfg = SETTINGS.fusion.post_fuzzy

            def _ratio(value: float, floor: float) -> float:
                if floor <= 0.0:
                    return 1.0
                return max(0.0, min(1.0, value / floor))

            coverage_factor = asr_mod_cfg.coverage_low_factor + (1.0 - asr_mod_cfg.coverage_low_factor) * _ratio(coverage, asr_mod_cfg.coverage_floor)
            text_conf_factor = asr_mod_cfg.text_conf_low_factor + (1.0 - asr_mod_cfg.text_conf_low_factor) * _ratio(text_conf, asr_mod_cfg.text_conf_floor)
            asr_conf_effective = asr_conf_downstream * coverage_factor * text_conf_factor
            asr_conf_effective = max(asr_mod_cfg.cap_min, min(asr_mod_cfg.cap_max, asr_conf_effective))
            set_gauge(asr_confidence_gauge, metrics_labels, asr_conf_effective)

            # 3.2 Fuzzy engine determines w_text (with linear fallback)
            try:
                fuzzy_out = fuzzy_engine.infer_w_text(asr_conf=asr_conf_effective, arousal=arousal, valence=valence)
                w_text = float(fuzzy_out["w_text"])
                w_text = max(post_cfg.min_weight, min(post_cfg.max_weight, w_text))
            except Exception:
                # Linear fallback keeps the pipeline running
                w_text = max(post_cfg.min_weight, min(post_cfg.max_weight, asr_conf_effective))
                fuzzy_out = {
                    "w_text": w_text,
                    "details": {
                        "inputs": {"asr_conf": asr_conf_effective, "arousal": arousal, "valence": valence},
                        "fired_rules": [{"if": ["<fallback>"], "then": "linear clamp", "strength": 1.0}],
                        "out_sets": {"low": 0.0, "mid": 1.0, "high": 0.0}
                    }
                }
            # --- Post-fuzzy heuristic adjustments ---
            valence_text = to_minus1_1(val_text)
            abs_valence_text = abs(valence_text)

            heur_arousal_audio = float(audio_metadata.get("heur_arousal", audio_metadata.get("arousal_smoothed", 0.5)) or 0.5)
            raw_scores_audio = audio_metadata.get("raw_scores") if isinstance(audio_metadata.get("raw_scores"), dict) else {}
            audio_neutral_score = float(raw_scores_audio.get("neutral", 0.0)) if raw_scores_audio else 0.0
            text_neutral_score = float(text_probs.get("neutral", 0.0))
            audio_top = max(audio_probs, key=audio_probs.get) if audio_probs else "neutral"
            audio_top_score = float(audio_probs.get(audio_top, 0.0)) if audio_probs else 0.0
            audio_non_neutral = audio_top != "neutral" and audio_top_score >= post_cfg.audio_non_neutral_threshold

            if asr_conf_raw >= 0.95 or asr_conf_effective >= 0.9:
                w_text = min(w_text, post_cfg.high_conf_cap)

            if coverage < asr_mod_cfg.coverage_floor or text_conf < asr_mod_cfg.text_conf_floor:
                w_text = min(w_text, post_cfg.low_evidence_cap)

            if abs_valence_text >= 0.55:
                w_text += 0.08
            elif abs_valence_text >= 0.35:
                w_text += 0.04

            if coverage >= 0.45 or text_conf >= 0.65:
                w_text += 0.06
            elif coverage >= 0.3 or text_conf >= 0.5:
                w_text += 0.03

            if coverage < 0.12 and text_conf < 0.4:
                w_text -= 0.10

            if audio_non_neutral:
                w_text = min(w_text, post_cfg.audio_override_cap)
                if text_neutral_score >= 0.5:
                    w_text -= post_cfg.neutral_audio_relief * 1.2
                elif text_neutral_score >= 0.4:
                    w_text -= post_cfg.neutral_audio_relief
                if audio_conf >= 0.6:
                    w_text -= post_cfg.audio_conf_boost
                if text_top == "disgust" and coverage < 0.35:
                    w_text -= 0.08
            else:
                if audio_neutral_score >= 0.6 and coverage >= 0.25:
                    w_text += 0.03

            if heur_arousal_audio >= 0.7 and audio_conf >= 0.6 and coverage < 0.3:
                w_text -= 0.04

            w_text = max(post_cfg.min_weight, min(post_cfg.max_weight, w_text))
            w_audio = 1.0 - w_text

            # 3.3 Fuse emotion probabilities into final scores
            probs_fused: Dict[str, float] = {}
            for emo in set(emo_audio.probs) | set(emo_text.probs):
                pa = emo_audio.probs.get(emo, 0.0)
                pt = emo_text.probs.get(emo, 0.0)
                probs_fused[emo] = (pa * w_audio) + (pt * w_text)
            s = sum(probs_fused.values()) or 1.0
            probs_fused = {k: v / s for k, v in probs_fused.items()}

            neg_labels = {"anger", "fear", "sadness"}
            text_top_label = max(text_probs, key=text_probs.get) if text_probs else "neutral"
            audio_top_label = audio_top

            if text_top_label in neg_labels and audio_top_label in neg_labels and text_top_label == audio_top_label:
                target = text_top_label
                boost = min(0.12, probs_fused.get("neutral", 0.0) * 0.4)
                if boost > 0:
                    probs_fused[target] = probs_fused.get(target, 0.0) + boost
                    probs_fused["neutral"] = max(0.0, probs_fused.get("neutral", 0.0) - boost)
                    s = sum(probs_fused.values()) or 1.0
                    probs_fused = {k: v / s for k, v in probs_fused.items()}
                    w_text = w_audio = 0.5
            elif audio_top_label in neg_labels and text_top_label == "neutral" and audio_conf >= 0.58 and heur_arousal_audio >= 0.55:
                shift = min(0.08, probs_fused.get("neutral", 0.0) * 0.35)
                if shift > 0:
                    probs_fused[audio_top_label] = probs_fused.get(audio_top_label, 0.0) + shift
                    probs_fused["neutral"] = max(0.0, probs_fused.get("neutral", 0.0) - shift)
                    s = sum(probs_fused.values()) or 1.0
                    probs_fused = {k: v / s for k, v in probs_fused.items()}

        # --- Guardrails: risk detection and escalation ---
        tmp_event = {
            "ts": now_iso(),
            "asr_conf": asr_conf_effective,
            "asr_conf_raw": asr_conf_raw,
            "asr_conf_snr": asr_conf_downstream,
            "weights": {"audio": 1.0 - w_text, "text": w_text},
        }
        # escalation_needed, escalation_info = check_guardrails(probs_fused, raw_text, tmp_event, run_id, txid_parent=None)
        action, escalation, guard_audit = check_guardrails(probs_fused=probs_fused,
                                                  text=raw_text,
                                                  stored_event=tmp_event,
                                                  run_id=run_id,
                                                  txid_parent=None)
        # attach audit info (so it gets persisted in stored_event later)
        if guard_audit:
            tmp_event.setdefault("audit_meta", {}).update(guard_audit)

        try:
            guard_logger.info("guardrails:action", extra={"extra_fields": {"run_id": run_id, "action": action, "audit": guard_audit}})
            level = None
            if escalation and isinstance(escalation, dict):
                level = escalation.get("level", "unknown")
            else:
                level = "notice" if action == "soften" else ("high" if action == "use_safe_template" else "unknown")
            inc_guardrails(level=level, labels=metrics_labels)
        except Exception:
            pass

        # Determine response_hint from configuration or guard audit outcome
        response_hint = None
        templates = SETTINGS.guardrails.templates
        if action == "suppress_and_escalate":
            response_hint = templates.safe
        elif action == "use_safe_template":
            response_hint = templates.safe
        elif action == "soften":
            response_hint = templates.soft

        # --- 4) Response + safety (PII) ---
        response = _plan_response(raw_text, probs_fused, response_hint=response_hint, action=action)

        redacted_text = redact(raw_text)
        if redacted_text != raw_text:
            # Increment labeled counter
            try:
                inc_counter(pii_redactions_total, metrics_labels, 1.0)
            except Exception:
                logger.exception("failed to inc pii_redactions_total")

        # --- 5) Audit trail (redacted event + blockchain-ready hash) ---
        event = {
            "ts": now_iso(),
            "asr_conf": asr_conf_effective,
            "asr_conf_raw": asr_conf_raw,
            "asr_conf_snr": asr_conf_downstream,
            "emotion_audio_conf": float(getattr(emo_audio, "confidence", 0.0)),
            "emotion_text_conf": float(getattr(emo_text, "confidence", 0.0)),
            "emotion_fused_top": max(probs_fused, key=probs_fused.get) if probs_fused else None,
            "text_redacted": redacted_text,
            "audio_metadata": audio_metadata,
            "weights": {"audio": 1.0 - w_text, "text": w_text},
            "cross_modal_coherence": coherence,
            "fusion_fuzzy": {
                "w_text": w_text,
                "w_audio": w_audio,
                "inputs": fuzzy_out["details"]["inputs"],
                "fired_rules": fuzzy_out["details"]["fired_rules"],
                "out_sets": fuzzy_out["details"]["out_sets"]
            },
            "escalation": tmp_event.get("escalation"),
            "audit_meta": tmp_event.get("audit_meta"),
            "ruleset_path": ruleset_path,
            "ruleset_version": getattr(fuzzy_engine, "version", "unknown"),
            "run_id": run_id,
            "notes": "MVP event; includes only redacted text and probabilities.",
        }
        txid, stored_event = record_event(event)
        anchor_info = blockchain_anchor_txid(txid, enabled=blockchain_enabled)

        try:
            fr = fuzzy_out["details"]["fired_rules"]
        except Exception:
            fr = []

        # Metadata for JSON export: inputs, sets, run_id, ruleset version
        meta = {
            "inputs": fuzzy_out["details"].get("inputs"),
            "out_sets": fuzzy_out["details"].get("out_sets"),
            "run_id": run_id,
            "ruleset_version": getattr(fuzzy_engine, "version", "unknown"),
            "ruleset_path": ruleset_path,
        }

        # Always export a JSON artifact
        json_dir = os.path.join(AUDIT_FIRED_DIR, "json")
        json_path = export_fired_rules_json(json_dir, txid, fr, meta=meta)

        # Export heatmap when matplotlib is available
        img_dir = os.path.join(AUDIT_FIRED_DIR, "heatmaps")
        matrix_repr = fired_rules_to_matrix(fr)
        img_path = save_heatmap_from_matrix(img_dir, txid, matrix_repr)

        # Log artifacts for observability
        logger.info("visual:exported_fired_rules", extra={"extra_fields": {"txid": txid, "json": json_path, "heatmap": img_path}})

        logger.info(
            "run:summary",
            extra={"extra_fields": {
                "run_id": run_id,
                "asr_conf_effective": asr_conf_effective,
                "asr_conf_raw": asr_conf_raw,
                "asr_conf_snr": asr_conf_downstream,
                "whisper_model_size": whisper_asr.get_model_size(),
                "weights": {"text": w_text, "audio": 1.0 - w_text},
                "top_emotion": max(probs_fused, key=probs_fused.get) if probs_fused else None,
                "ruleset_path": ruleset_path,
                "use_mfcc": use_mfcc_flag,
                "metrics_port": effective_metrics_port,
            }}
        )

        return {
            "text": raw_text,
            "text_redacted": redacted_text,
            "asr_confidence": asr_conf_effective,
            "asr_confidence_raw": asr_conf_raw,
            "asr_confidence_snr": asr_conf_downstream,
            "emotion_audio": _safe_asdict(emo_audio),
            "emotion_text": _safe_asdict(emo_text),
            "emotion_fused": probs_fused,
            "response": response,
            "audit": {"txid": txid, "stored_event": stored_event},
            "blockchain": anchor_info,
            "run_id": run_id,
            "ruleset_path": ruleset_path,
            "use_mfcc": use_mfcc_flag,
            "metrics_port": effective_metrics_port,
            "whisper_model_size": whisper_asr.get_model_size(),
        }
    except Exception as e:
        try:
            inc_counter(pipeline_errors_total, metrics_labels, 1.0)
        except Exception:
            logger.exception("failed to inc pipeline_errors_total")
        logger = get_logger("pipeline")
        logger.error("run:error", extra={"extra_fields": {"error": str(e)}})
        raise
    finally:
        if restore_model_size:
            target = original_model_size or SETTINGS.models.whisper_size
            whisper_asr.set_model_size(target)
