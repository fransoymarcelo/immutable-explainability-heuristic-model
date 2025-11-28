# JSON Schema – “Blockchain-ready” record per dialog turn

```json
{
  "timestamp": "2025-10-01T12:34:56Z",
  "audio_path": "data/examples/mi_audio.wav",
  "transcription": "texto reconocido o vacío si no hay",
  "intent": "consulta|reclamo|agradecimiento|otro",
  "asr_confidence": 0.85,
  "text_emo_top": "joy",
  "audio_emo_top": "neutral",
  "final_emo_top": "joy",
  "conf_text": 0.62,
  "conf_audio": 0.48,
  "vad_text": {"valence": 0.6, "arousal": 0.5, "dominance": 0.5},
  "vad_audio": {"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
  "vad_final": {"valence": 0.4, "arousal": 0.42, "dominance": 0.5},
  "probs_text": {"joy": 0.5, "...": 0.0},
  "probs_audio": {"neutral": 0.6, "...": 0.0},
  "probs_final": {"joy": 0.51, "neutral": 0.3, "...": 0.0},
  "rules_triggered": ["Privilegiar texto por mayor confianza y desacuerdo"],
  "response_text": "¡Gracias por tu mensaje!...",
  "hash_audio": "sha256: ...",
  "hash_text": "sha256: ..."
}
```

- `hash_audio`: SHA‑256 del archivo de audio.
- `hash_text`: SHA‑256 de la transcripción (cadena).

> En Fase 3, estos campos se almacenan o anclan en blockchain (p.ej., hash on‑chain y datos cifrados off‑chain).
