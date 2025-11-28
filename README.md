# Immutable Explainability Pipeline for Affective Voice (Spanish SER + Fuzzy Fusion + Blockchain)
![Status](https://img.shields.io/badge/Project-Thesis%20Research-blue) ![Python](https://img.shields.io/badge/Python-3.10%2B-green) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow)



This repository accompanies the thesis work on **Immutable Explainability**.  
It delivers an end-to-end slice that transcribes speech with Whisper, scores text and audio with lightweight heuristic models, fuses both channels through a Mamdani fuzzy engine, applies empathetic guardrails, and anchors redacted audit trails on the Sepolia testnet.

---

## 1. Prerequisites

| Requirement             | Notes                                                                  |
|------------------------|------------------------------------------------------------------------|
| Python 3.10+           | Local execution / development                                          |
| `ffmpeg`, `libsndfile` | Required for audio features (macOS: `brew install ffmpeg libsndfile`)  |
| Docker (optional)      | `docker` + `colima` recommended to reproduce the container workflow    |

### 1.1 Virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.2 Environment variables (blockchain)
Only required if you plan to anchor hashes on Sepolia. Create an `.env` file **without quotes**:
```ini
RPC_URL=https://sepolia.infura.io/v3/<your_project_id>
PRIVATE_KEY=<hex_private_key_with_gas_funds>
```

To run fully offline, omit the `.env` file and set `blockchain.enabled: false` in `config/local.yaml`.

### 1.3 Optional datasets
- `data/examples/` contains sample WAV files.
- The Spanish MEA Corpus should live under `data/spanish-mea-corpus-2023/` if you intend to run the batch scripts.

---

## 2. Project structure
```
immutable-explainability-heuristic-model/
├── app/               # Orchestrator entrypoints and CLI helpers
├── asr/               # Whisper wrapper and caching helpers
├── audio_emotion/     # Heuristic audio emotion model (RMS, ZCR, MFCC, EMA)
├── audit/             # Redacted audit trail and explainability artefacts
├── blockchain/        # Web3 client + contract artefacts for hash anchoring
├── common/            # Shared types and utilities
├── config/            # defaults.yaml, local overrides, typed settings loader
├── core/              # Mamdani fuzzy rulesets
├── data/              # Datasets (ignored by git)
├── docs/              # Additional documentation (CONFIG, DESIGN, TESTS, JSON schema)
├── scripts/           # Batch experiments and utilities
├── server/            # FastAPI application (`server.api`)
├── tests/             # Pytest suite (integration, guardrails, heuristic models)
├── Dockerfile / docker-compose.yml
├── README.md
└── requirements*.txt
```
> Runtime folders such as `runtime/` and `audit/` are created automatically (or mounted in Docker) to store audit evidence.

---

## 3. Configuration

1. Copy `config/defaults.yaml` to `config/local.yaml` for local overrides.
2. Adjust `models.whisper_size`, `fusion.fuzzy_ruleset`, `blockchain.enabled`, and other knobs as needed.

Common overrides exposed by the orchestrator (API + scripts):

| Flag / parameter      | Description                                                             |
|-----------------------|-------------------------------------------------------------------------|
| `whisper_model_size`  | Selects the Whisper model (`tiny`, `base`, `small`, …).                 |
| `blockchain_enabled`  | Enables/disables anchoring per request.                                |
| `use_mfcc`            | Toggles MFCC support in the heuristic audio model.                      |
| `use_gold_transcript` | Batch flag to bypass Whisper and rely on reference transcripts.         |

Environment variables listed in `docs/CONFIG.md` let you override defaults globally (e.g. `WHISPER_MODEL_SIZE`, `METRICS_PORT`, `RPC_URL`, `PRIVATE_KEY`).

---

## 4. Usage scenarios

### 4.1 CLI (single audio)
```bash
python -m app.cli run --audio data/examples/sample.wav --pretty
```
Optional: `--metrics-port 9000 --hold 30` keeps the Prometheus endpoint alive for 30 seconds.

### 4.2 REST API (local)
```bash
uvicorn server.api:app --host 0.0.0.0 --port 8080
```

Example request:
```bash
curl -X POST http://localhost:8080/analyze \
  -F "files=@data/examples/sample.wav" \
  -F "run_id=demo01" \
  -F "blockchain_enabled=true" \
  -F "expose_text=false" | jq
```

### 4.3 REST API via Docker
```bash
docker build -t affective-voice-pipeline:dev .
docker run --rm \
  --env-file .env \
  -p 8080:8080 -p 9000:9000 \
  -v "$(pwd)/runtime:/app/runtime" \
  -v "$(pwd)/audit:/app/audit" \
  affective-voice-pipeline:dev \
  uvicorn server.api:app --host 0.0.0.0 --port 8080
```

### 4.4 Batch processing scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/run_meacorpus_full.py` | Full MEA Corpus sweep with configurable limits | ```bash\npython scripts/run_meacorpus_full.py \\\n  --csv data/spanish-mea-corpus-2023/spanish-meacorpus-2023-dataset.csv \\\n  --audios_dir data/spanish-mea-corpus-2023/audios \\\n  --output_csv results_full.csv \\\n  --whisper_model_size small \\\n  --blockchain_enabled false\n``` |
| `scripts/run_pipeline_whisper_blockchain.py` | Whisper + blockchain on a subset | ```bash\npython scripts/run_pipeline_whisper_blockchain.py \\\n  --csv data/spanish-mea-corpus-2023/spanish-meacorpus-2023-dataset.csv \\\n  --audios_dir data/spanish-mea-corpus-2023/audios \\\n  --output_csv results_whisper_blockchain.csv \\\n  --metrics_json metrics_whisper_blockchain.json \\\n  --limit 10\n``` |
| `scripts/evaluacion_pipeline.py` | Quantitative evaluation with skip/limit and optional Whisper bypass | ```bash\npython scripts/evaluacion_pipeline.py \\\n  --csv data/spanish-mea-corpus-2023/spanish-mea-corpus-2023-dataset.csv \\\n  --audios_dir data/spanish-mea-corpus-2023/audios \\\n  --output_csv results_subset.csv \\\n  --skip 0 --limit 200 \\\n  --use_gold_transcript\n``` |

### 4.5 Quick scenarios

| Goal                         | How to achieve it                                                        |
|------------------------------|---------------------------------------------------------------------------|
| Skip Whisper                 | Use `--use_gold_transcript` or provide transcripts manually.             |
| Force Whisper processing     | Omit `--use_gold_transcript`; ensure `openai-whisper` is installed.      |
| Disable blockchain anchoring | `--blockchain_enabled false` per request or `blockchain.enabled: false`. |
| Process large datasets       | Use `--skip/--limit` to chunk processing (see scripts above).            |
| Inspect Prometheus metrics   | `http://localhost:<metrics_port>/metrics` (default `9000`).              |

### 4.6 Immutable explainability demo
1. Start the API with blockchain enabled (local or Docker).
2. Process an audio and capture the response:
   ```bash
   curl -X POST http://localhost:8080/analyze \
     -F "files=@data/examples/sample.wav" \
     -F "run_id=demo_explain" \
     -F "blockchain_enabled=true" \
     -o response.json
   jq '.results[0].blockchain' response.json
   ```
3. Confirm anchoring on Sepolia:
   ```bash
   python scripts/verify-txid-anchorage.py --txid "$(jq -r '.results[0].audit.txid' response.json)"
   ```

---

## 5. Observability & audit
- Redacted audit events live under `audit/events.log` (text is replaced by deterministic hashes).
- Prometheus metrics: `http://localhost:9000/metrics` (override with `METRICS_PORT`).
- Verify on-chain anchoring via `scripts/verify-txid-anchorage.py`.
- Each API response includes a `blockchain` block with status and transaction information.

---

## 6. Tests
```bash
PYTHONPATH=. pytest tests/test_guardrails.py \
                     tests/test_pipeline_integration.py \
                     tests/test_text_model.py \
                     tests/test_heuristic_pipeline.py
```

---

## 7. Additional documentation
- `docs/CONFIG.md` – configuration layers and override precedence.
- `docs/DESIGN.md` – architecture overview and fusion/guardrail details.
- `docs/TESTS.md` – explanation of the Pytest suite and fixtures.
- `docs/ROADMAP.md` – milestones for future development.
- `docs/JSON_SCHEMA.md` – audit-event and API-response schemas.

---

## 8. License & contribution

Licensed under the MIT License (see `LICENSE`).  
Contributions via pull requests are welcome—open an issue first for major changes and ensure tests pass.  
Never commit real credentials; keep `.env` local (use the snippet above or add your own `.env.example` locally for reference).

## 9. Acknowledgments

I would like to express my sincere gratitude to the research team behind Spanish MEACorpus 2023 for granting me access to the dataset and for their valuable work in developing such a crucial resource for advancing affective computing in Spanish.
Their contribution has been instrumental to the development, experimentation, and validation of this project.