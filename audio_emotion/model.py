# audio_emotion/model.py
"""
Audio emotion model (deterministic, lightweight MVP).

- Includes ZCR, optional MFCC (when librosa is available), EMA smoothing,
  and returns `metadata` in the EmotionResult.
- Supports per-session smoothing (session_id) to avoid mixing conversation history.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import contextlib
import wave
import math
import os
import array

# numpy and librosa are optional; librosa is only used when `use_mfcc=True`
import numpy as np
try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

from common.types import EmotionResult, VAD
from config.settings import SETTINGS

# Emociones base (fijas en todo el proyecto)
EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "neutral"]


def _softmax(d: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if temperature <= 0:
        temperature = 1.0
    ex = {k: math.exp(v / temperature) for k, v in d.items()}
    s = sum(ex.values()) or 1.0
    return {k: ex[k] / s for k in d}


def _rms_zcr_any(path: str) -> Tuple[float, float]:
    """
    Compute normalized RMS and ZCR (0..1) using librosa for non-WAV formats.
    When librosa is unavailable or fails, return neutral values (0.0, 0.5).
    """
    if not HAS_LIBROSA:
        return 0.0, 0.5
    try:
        signal, sr = librosa.load(path, sr=22050)
        if signal.size == 0:
            return 0.0, 0.5
        rms = float(np.sqrt(np.mean(np.square(signal))))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=signal)))
        rms_norm = min(rms / 0.1, 1.0)
        zcr_norm = min(zcr / 0.5, 1.0)
        return rms_norm, zcr_norm
    except Exception:
        return 0.0, 0.5

def _rms_and_snr_wav(path: str, block: int = 1024, noise_percentile: float = 10.0) -> Tuple[float, Optional[float]]:
    """
    Calcula RMS (root mean square) global (normalizado ~[0..1]) y estima SNR en dB (float) para un WAV PCM mono/stereo.
    - block: tamaño de lectura en muestras.
    - noise_percentile: percentil bajo usado para estimar noise floor (ej. 10 -> 10th percentile).
    Retorna: (rms_norm, snr_db_or_None)

    """
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            samp_width = wf.getsampwidth()
            n_channels = wf.getnchannels()

            # Function that maps bytes to integers based on samp_width
            if samp_width == 1:
                fmt = "b"
                norm_factor = 128.0  # approx peak for 8-bit PCM unsigned/biased. Para 8-bit WAV el rango puede ser unsigned (0..255) con bias
            elif samp_width == 2:
                fmt = "h"
                norm_factor = 32768.0
            elif samp_width == 4:
                fmt = "i"
                norm_factor = 2 ** 31
            else:
                return 0.0, None

            energies = []  # energy por bloque (mean square)
            total_sq = 0.0
            total_count = 0

            while True:
                frames = wf.readframes(block)
                if not frames:
                    break
                arr = array.array(fmt)
                arr.frombytes(frames)

                # si stereo, promedio canales por pareja
                if n_channels > 1:
                    it = iter(arr)
                    mono = [(a + b) / 2.0 for a, b in zip(it, it)]
                else:
                    mono = arr

                if len(mono) == 0:
                    continue

                # Compute block energy (mean square)
                sum_sq = 0.0
                for x in mono:
                    v = float(x)
                    sum_sq += v * v
                block_energy = sum_sq / float(len(mono))  # mean square
                energies.append(block_energy)

                total_sq += sum_sq
                total_count += len(mono)

            if total_count == 0:
                return 0.0, None

            # RMS global (raw)
            rms = math.sqrt(total_sq / total_count)

            # Heuristic normalization to keep values within 0..1 (conservative)
            rms_norm = min(1.0, rms / (norm_factor * 0.92))  # 0.92 margen

            # Estimate noise floor: use a low percentile of 'energies'
            if len(energies) == 0:
                return rms_norm, None

            energies_sorted = sorted(energies)
            idx = max(0, int(len(energies_sorted) * (noise_percentile / 100.0)) - 1)
            noise_floor_msq = energies_sorted[idx] if idx < len(energies_sorted) else energies_sorted[0]

            # Avoid division by zero if noise_floor is extremely small (e.g., 0)
            eps = 1e-12
            signal_msq = (rms * rms)
            noise_msq = max(noise_floor_msq, eps)

            snr_linear = signal_msq / noise_msq
            # cap razonable
            if snr_linear <= 0:
                snr_db = None
            else:
                snr_db = 10.0 * math.log10(snr_linear)
                # limitar rango para evitar outliers extremos
                snr_db = max(-60.0, min(60.0, snr_db))

            return rms_norm, float(snr_db)
    except Exception:
        return 0.0, None

def _zero_crossing_rate_wav(path: str) -> float:
    """
    Estima ZCR como cruces por cero / total_muestras.
    Output típicamente en rango ~0..0.5; normalizamos luego externamente.
    """
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            samp_width = wf.getsampwidth()
            n_channels = wf.getnchannels()
            block = 1024
            total_zc = 0
            total_count = 0
            while True:
                frames = wf.readframes(block)
                if not frames:
                    break
                if samp_width == 1:
                    fmt = "b"
                elif samp_width == 2:
                    fmt = "h"
                elif samp_width == 4:
                    fmt = "i"
                else:
                    return 0.0
                arr = array.array(fmt)
                arr.frombytes(frames)

                if n_channels > 1:
                    it = iter(arr)
                    mono = [(a + b) / 2.0 for a, b in zip(it, it)]
                else:
                    mono = arr

                prev = None
                zc_block = 0
                for x in mono:
                    s = 1 if x >= 0 else -1
                    if prev is not None and s != prev:
                        zc_block += 1
                    prev = s
                total_zc += zc_block
                total_count += max(1, len(mono))

            if total_count == 0:
                return 0.0
            return float(total_zc) / float(total_count)
    except Exception:
        return 0.0


def _mfcc_features(path: str, n_mfcc: int = 13) -> Dict[str, Any]:
    """
    Extrae MFCC (media y deltas). Requiere librosa; si librosa no está instalado, lanza excepción.
    """
    if not HAS_LIBROSA:
        raise RuntimeError("librosa no está disponible. Instalarlo para MFCC.")
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2), axis=1)
    return {
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_delta": mfcc_delta.tolist(),
        "mfcc_delta2": mfcc_delta2.tolist(),
    }


def ema_smooth(prev: Optional[float], value: float, alpha: float = 0.2) -> float:
    """
    Media móvil exponencial simple.
    prev: valor anterior (None si no existe).
    alpha: factor (0,1], default 0.2.
    """
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev


# EMA state: global or per-session (dict)
_last_arousal: Optional[float] = None
_LAST_AROUSAL_BY_SESSION: Dict[str, float] = {}
_AUDIO_CFG = SETTINGS.audio


def predict(audio_path: str,
            use_mfcc: bool,
            ema_alpha: Optional[float] = None,
            session_id: Optional[str] = None) -> EmotionResult:
    """
    Predice emoción a partir de audio, integrando RMS, ZCR y MFCC (opcional),
    y aplicando EMA al arousal. Devuelve metadata con features y valores intermedios.
    - use_mfcc: True para extraer MFCC (librosa requerida).
    - ema_alpha: factor EMA.
    - session_id: si se provee, EMA se guarda por sesión.
    """
    global _last_arousal, _LAST_AROUSAL_BY_SESSION

    if ema_alpha is None:
        ema_alpha = float(_AUDIO_CFG.ema_alpha)

    metadata: Dict[str, Any] = {}

    ext = os.path.splitext(audio_path)[1].lower()

    # 1) RMS + SNR
    if ext == ".wav":
        arousal_rms, snr_db = _rms_and_snr_wav(audio_path)
        zcr_raw = _zero_crossing_rate_wav(audio_path)
        zcr_norm = min(1.0, zcr_raw * 10.0)
    else:
        arousal_rms, zcr_norm = _rms_zcr_any(audio_path)
        snr_db = None
        zcr_raw = None
    metadata["arousal_raw"] = arousal_rms
    metadata["snr_db"] = snr_db
    metadata["zcr_raw"] = zcr_raw
    metadata["zcr_norm"] = zcr_norm

    # 3) combinar RMS y ZCR
    arousal_combined = min(1.0, arousal_rms * (0.9 + 0.1 * zcr_norm))

    # 4) MFCC opcional
    mfcc_feats = None
    if use_mfcc and ext == ".wav" and HAS_LIBROSA:
        try:
            mfcc_feats = _mfcc_features(audio_path)
            metadata["mfcc_present"] = True
        except Exception:
            mfcc_feats = None
            metadata["mfcc_present"] = False
    else:
        metadata["mfcc_present"] = False

    # 5) Baselines VAD (antes de usar timbre)
    valence = 0.0
    dominance = 0.5

    # 6) timbre_score desde MFCC y ajustes
    timbre_score = None
    if mfcc_feats is not None:
        mfcc_mean = mfcc_feats.get("mfcc_mean")
        if mfcc_mean:
            high_band = mfcc_mean[6:] if len(mfcc_mean) > 6 else mfcc_mean
            timbre_score = sum(abs(v) for v in high_band) / (len(high_band) * 50.0)
            timbre_score = min(1.0, max(0.0, timbre_score))
        metadata["timbre_score"] = timbre_score

        if timbre_score is not None:
            # ajustar valence y arousal suavemente
            valence = max(-1.0, min(1.0, valence + (timbre_score - 0.5) * 0.2))
            arousal_combined = min(1.0, arousal_combined + timbre_score * 0.05)

    # 7) EMA (per session or global)
    if session_id is None:
        prev = _last_arousal
    else:
        prev = _LAST_AROUSAL_BY_SESSION.get(session_id)
    arousal_smoothed = ema_smooth(prev, arousal_combined, alpha=ema_alpha)
    if session_id is None:
        _last_arousal = arousal_smoothed
    else:
        _LAST_AROUSAL_BY_SESSION[session_id] = arousal_smoothed
    metadata["arousal_smoothed"] = arousal_smoothed

    # 8) Build emotion distribution (reuse heuristic_audio.py approach)
    heur_arousal = float(max(0.0, min(1.0, 0.7 * arousal_rms + 0.3 * zcr_norm)))
    heur_valence = float(max(0.0, min(1.0, 1.0 - zcr_norm)))

    pos_signal = max(0.0, heur_valence - 0.45)
    neg_signal = max(0.0, 0.55 - heur_valence)
    calm_signal = max(0.0, 0.6 - heur_arousal)

    joy_score = pos_signal * (0.6 + 0.40 * heur_arousal)
    anger_score = neg_signal * heur_arousal * 1.18
    sadness_score = neg_signal * calm_signal * 1.08
    fear_score = heur_arousal * (neg_signal * 0.72 + (1.0 - heur_valence) * 0.18)
    disgust_score = neg_signal * 0.75

    neutral_base = max(0.22, 0.70 - heur_arousal * 0.40 - pos_signal * 0.18 + calm_signal * 0.35)

    adjustments = []
    high_zcr = zcr_norm >= 0.55
    high_arousal = heur_arousal >= 0.65
    very_high_arousal = heur_arousal >= 0.8
    low_valence = heur_valence <= 0.45
    very_low_valence = heur_valence <= 0.35
    calm_low_arousal = heur_arousal <= 0.35

    if very_high_arousal and very_low_valence:
        anger_score *= 1.28
        fear_score *= 1.15
        neutral_base -= 0.10
        adjustments.append("very_high_arousal_low_valence")
    elif high_arousal and low_valence:
        anger_score *= 1.18
        fear_score *= 1.10
        neutral_base -= 0.06
        adjustments.append("high_arousal_low_valence")
    elif high_arousal and heur_valence > 0.55:
        joy_score *= 1.18
        neutral_base -= 0.04
        adjustments.append("high_arousal_positive")

    if calm_low_arousal and neg_signal >= 0.2:
        sadness_score *= 1.14
        neutral_base -= 0.04
        adjustments.append("calm_negative")

    if high_zcr:
        anger_score *= 1.08
        fear_score *= 1.05
        neutral_base -= 0.03
        adjustments.append("high_zcr")

    neutral_base = max(0.18, neutral_base)

    if neutral_base <= 0.2 and neg_signal < 0.18 and pos_signal < 0.18:
        neutral_base = max(neutral_base, 0.22)

    raw_scores = {
        "joy": joy_score,
        "anger": anger_score,
        "sadness": sadness_score,
        "fear": fear_score,
        "disgust": disgust_score,
        "neutral": neutral_base,
    }

    probs = _softmax(raw_scores, temperature=0.7)

    confidence = float(_AUDIO_CFG.wav_confidence if ext == ".wav" else _AUDIO_CFG.other_confidence)
    top = max(probs, key=probs.get)

    # 9) retornar EmotionResult (con metadata)
    return EmotionResult(
        probs=probs,
        vad=VAD(valence=valence, arousal=arousal_smoothed, dominance=dominance),
        confidence=confidence,
        top_label=top,
        metadata={
            **metadata,
            "raw_scores": raw_scores,
            "heur_arousal": heur_arousal,
            "heur_valence": heur_valence,
            "pos_signal": pos_signal,
            "neg_signal": neg_signal,
            "calm_signal": calm_signal,
            "adjustments": adjustments,
        },
    )
