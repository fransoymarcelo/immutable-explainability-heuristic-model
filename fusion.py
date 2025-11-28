# fusion.py
from typing import Dict, Tuple

from config.settings import SETTINGS

_FUSION_SIMPLE_CFG = SETTINGS.fusion.fusion_simple


def fuse_simple(prob_text: Dict[str, float], prob_audio: Dict[str, float], asr_conf: float, ar_audio: float) -> Tuple[Dict[str, float], Dict]:
    """
    Heurística simple para obtener w_text y fired_rule.
    Devuelve (prob_final, fusion_details).
    """
    top_text_conf = max(prob_text.values()) if prob_text else 0.0
    top_audio_conf = max(prob_audio.values()) if prob_audio else 0.0

    # reglas simples (ajustables)
    if top_text_conf > _FUSION_SIMPLE_CFG.text_confidence_thresh and asr_conf > _FUSION_SIMPLE_CFG.asr_confidence_thresh:
        w_text = 0.85
        fired = "R_text_strong"
    elif top_audio_conf > _FUSION_SIMPLE_CFG.audio_confidence_thresh and ar_audio > _FUSION_SIMPLE_CFG.audio_arousal_thresh:
        w_text = 0.25
        fired = "R_audio_strong"
    else:
        w_text = 0.5
        fired = "R_balanced"

    # mezclar
    prob_final = {k: w_text * prob_text.get(k, 0.0) + (1 - w_text) * prob_audio.get(k, 0.0)
                  for k in prob_text.keys()}
    # normalizar por seguridad
    s = sum(prob_final.values()) or 1.0
    prob_final = {k: prob_final[k] / s for k in prob_final}
    details = {'w_text': w_text, 'fired_rule': fired, 'top_text_conf': top_text_conf, 'top_audio_conf': top_audio_conf}
    return prob_final, details
