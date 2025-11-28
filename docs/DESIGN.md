# Design document – Immutable Explainability vertical slice

This living document provides the technical overview of the affective voice pipeline delivered in this repository. It replaces the original Spanish template and should be used as reference when extending or swapping components (e.g. switching heuristics for neural models).

---

## 1. Executive summary
The pipeline processes Spanish speech, transcribes with Whisper, estimates emotions independently for text and audio, fuses both signals with a Mamdani fuzzy engine, applies empathetic guardrails, and emits an audit trail that is ready to be anchored on-chain.

Goals:
- Increase **accuracy** and **sensitivity** to subtle emotional cues.
- Preserve privacy via hashing/redaction.
- Provide a verifiable audit trail (“immutable explainability”) for later inspection or dispute resolution.

---

## 2. Requirements

### 2.1 Functional
- Speech-to-text (Whisper) with optional bypass (`use_gold_transcript`).
- Emotion detection from **text** and **audio** in parallel, including VAD (valence/arousal/dominance).
- Fuzzy fusion with post-processing heuristics to resolve conflicts.
- Empathetic response policy influenced by emotion and intent.
- JSON event per run (hashes, weights, guardrails) prepared for blockchain anchoring.

### 2.2 Non-functional
- Spanish (ES-AR / ES-LA) focus; resilience to moderate noise.
- Low latency and graceful degradation when confidence is low.
- Privacy by minimisation: hash and redact audio/text; no raw data persisted by default.
- Explainability: list of fired fuzzy rules, confidence scores, cross-modal coherence.

---

## 3. High-level architecture

```
Audio ──▶ Whisper ASR ──┐
                        │
                        ├─▶ Text emotion service
Audio ──▶ Audio emotion ─┘
         │
         └─▶ VAD & SNR metadata
            │
Fuzzy fusion ─▶ Guardrails & response ─▶ Audit log ─▶ (Blockchain anchor)
```

The orchestrator (`app/orchestrator.py`) coordinates these steps directly, invoking:
- Whisper transcription (`asr.whisper_asr`)
- Audio and text heuristic models
- Fuzzy fusion (with post-fusion adjustments)
- Guardrail checks and empathetic response selection
- Audit logging, redaction, and optional blockchain anchoring

---

## 4. Text channel
- **Input**: Whisper transcript.
- **Current implementation**: lexicon-based heuristic (baseline for reproducibility).
- **Future work**: fine-tuned BETO/BERT head for 7 emotions (joy, sadness, anger, fear, disgust, surprise, neutral) plus VAD regression.
- **Metrics**: macro-F1 per emotion; MAE for VAD dimensions.

---

## 5. Audio channel
- **Input**: mono WAV (16 kHz recommended).
- **Current implementation**: RMS/ZCR + handcrafted rules, optional MFCC, SNR estimation.
- **Future work**: wav2vec2.0/HuBERT embeddings + classifier; speaker normalisation.
- **Metrics**: macro-F1 per emotion; MAE for VAD.

---

## 6. Fuzzy fusion

### 6.1 Inputs
- `valence_text`, `arousal_text`, `coverage_text`, `confidence_text`
- `valence_audio`, `arousal_audio`, `confidence_audio`
- Cross-modal coherence (audio vs text VAD)

### 6.2 Membership functions (initial)
- Valence in [-1, 1]: negative / neutral / positive.
- Arousal in [0, 1]: low / medium / high.
- Confidence in [0, 1]: low / medium / high.

### 6.3 Rules (seed set > 20)
Examples:
- Positive valence & medium/high arousal → boost joy.
- Negative valence & high arousal → boost anger.
- Disagreement with high audio confidence → prioritise audio.
- Mutual low confidence → favour neutral response + ask clarification.
- Sustained coherence across runs → conservative updates (avoid sudden jumps).

### 6.4 Post-fuzzy heuristics
- Adjust weights when textual evidence is strong but audio is neutral (and vice versa).
- Penalise text when coverage is low or valence is near zero.
- Boost non-neutral audio when confidence ≥ threshold.

### 6.5 Output
- Final weights `w_text`, `w_audio`.
- Normalised fused probabilities, final VAD.
- `fuzzy_out.details.fired_rules` for explainability.

---

## 7. Guardrails & response policy
- Empathetic template matrix by emotion × intent (query, complaint, gratitude, other).
- Guardrails escalate when self-harm keywords or high-risk VAD combinations occur.
- Optional webhook for escalation (configurable in `config/defaults.yaml`).
- Low confidence fallback: ask the user to confirm understanding.

---

## 8. Immutable audit event
Each run generates a redacted JSON object containing:
- Timestamp, `run_id`, audio metadata, hashed text, hashed audio.
- Emotion probabilities per channel, fused result, VAD.
- Fired rules, guardrail action/escalation record.
- Response text and blockchain status when anchoring is enabled.

See `docs/JSON_SCHEMA.md` for the exact structure.

---

## 9. Experiment plan (baseline)
- Recommended datasets: **MEA Corpus 2023**, **InterFace**, **ELRA-S0329**.
- Keep a hold-out set, then perform k-fold evaluation when models mature.
- Metrics: macro-F1, precision, recall, accuracy, plus MAE for VAD.

---

## 10. Risks & mitigations
- Class imbalance → class weighting, targeted data collection.
- Ambiguity between channels → confidence-aware fuzzy rules + guardrails.
- Privacy → store hashes and redacted content only; secrets stay in `.env`.
- Latency → monitor Prometheus histograms; offload heavy models when necessary.

---

## 11. Roadmap
1. Finalise membership functions and rule set (>20 rules).
2. Integrate neural models while preserving the service interface.
3. Extend the test suite with regression data and batch smoke tests.
4. Document performance benchmarks (macro-F1, latency, blockchain overhead).
5. Prepare phase-3 deliverables (UI/demo integration).

---

## 12. References
- Tesis documentation (Fase 2).
- Whisper, wav2vec2.0, BETO/BERT Spanish references.
- Immutable explainability paper cited at the top of this README.

