# Roadmap – Phase 2 (Hybrid model design & development)

## Milestone A – Operational vertical slice
- CLI processes WAV (and optional sidecar TXT).
- Text and audio emotion estimators, fuzzy fusion, empathetic response.
- JSON audit entry persisted under `audit/`.

## Milestone B – Full fuzzy fusion
- Define membership functions and ≥20 rules.
- Benchmark against the heuristic fallback.

## Milestone C – Replace heuristics with learned models
- ASR: Whisper / Vosk / alternative production-grade engine.
- Text: Spanish BETO/BERT fine-tuned for multi-class emotions + VAD regression.
- Audio: pretrained embeddings (wav2vec2.0 / HuBERT / WavLM) + classifier.

## Milestone D – Formal evaluation
- Curated dataset and published metrics (macro-F1, MAE for VAD, latency).
- Short report with findings and qualitative analysis.

