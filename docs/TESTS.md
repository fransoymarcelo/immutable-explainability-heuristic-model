# Automated test guide

The project ships with a Pytest suite to validate the current heuristic pipeline and to serve as a safety net when swapping in neural / probabilistic models.

> Tip: create a virtual environment and install the optional extras (`pip install scikit-fuzzy librosa pandas spacy` + `python -m spacy download es_core_news_sm`) so that every test runs without skips.

## Layout

```
tests/
├── test_guardrails.py              # Guardrails, templates, escalation metadata
├── test_pipeline_integration.py    # End-to-end orchestrator (stubbed ASR/emotion)
├── test_text_model.py              # Lexical text emotion heuristics
└── test_heuristic_pipeline.py      # Audio heuristics, fuzzy fusion, PII redaction (legacy)
```

## Highlights

| Test file | Focus | What it asserts |
|-----------|-------|-----------------|
| `test_guardrails.py` | Safety policies | Regex detection, VAD thresholds, escalation payloads |
| `test_pipeline_integration.py` | Orchestrator flow | Guardrails trigger path, audit event shape, response text |
| `test_text_model.py` | Text heuristics | Probability normalisation, lemma metadata, top label selection |
| `test_heuristic_pipeline.py` | Heuristic components | Audio VAD range, fuzzy weights, PII redaction patterns |

Every test reads configuration via `config.settings.SETTINGS`, so changes in `defaults.yaml` automatically propagate to the suite.

## Running the suite

```bash
pytest
```

Targeted runs:

```bash
pytest tests/test_guardrails.py tests/test_pipeline_integration.py
pytest tests/test_text_model.py tests/test_heuristic_pipeline.py
```

If an optional dependency is missing, the test will be skipped with a clear message (e.g. `pytest.importorskip("spacy")`).

## Extending coverage

1. **Neural / external models** – keep the same assertions (normalised probabilities, guardrail decisions) when replacing heuristics; mock outbound HTTP calls if needed.
2. **Regression datasets** – create fixtures with annotated samples to monitor metrics such as macro-F1 or MAE after model upgrades.
3. **Batch pipelines** – add smoke tests for scripts (`scripts/run_meacorpus_full.py`, etc.) verifying that CSV outputs contain expected columns and value ranges.
