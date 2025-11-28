# heuristic_audio.py
from typing import Tuple, Dict
import math
import numpy as np

# import your feature extractor
from scripts.evaluacion_cuantitativa import extract_audio_features

def _softmax(d):
    ex = {k: math.exp(v) for k, v in d.items()}
    s = sum(ex.values()) + 1e-9
    return {k: ex[k] / s for k in d}

def predict_audio_probs(audio_path: str) -> Tuple[Dict[str, float], Dict]:
    """
    Retorna (prob_dict, explanation).
    prob_dict: dict label->prob (suma 1)
    explanation: dict con rms/zcr,valence,arousal,raw_scores
    """
    rms_norm, zcr_norm = extract_audio_features(audio_path)
    # proxies (simple, calibrable)
    arousal = float(max(0.0, min(1.0, 0.7 * rms_norm + 0.3 * zcr_norm)))
    valence = float(max(0.0, min(1.0, 1.0 - zcr_norm)))  # sencilla
    # raw scoring toward emotions (tuneable)
    raw = {
        'joy': max(0.0, (valence - 0.5)) * arousal * 2.0,
        'anger': max(0.0, (0.5 - valence)) * arousal * 2.0,
        'sadness': max(0.0, (0.5 - valence)) * (1.0 - arousal),
        'fear': max(0.0, arousal * (0.5 - valence)),
        'disgust': max(0.0, (0.5 - valence)) * 0.5,
        'neutral': max(0.05, 1.0 - arousal)  # floor small
    }
    probs = _softmax(raw)
    explanation = {'rms': rms_norm, 'zcr': zcr_norm, 'valence': valence, 'arousal': arousal, 'raw': raw}
    return probs, explanation
