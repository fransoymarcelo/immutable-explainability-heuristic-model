# server/api.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import tempfile, os, uuid, shutil

from app.orchestrator import run_pipeline
from utils.metrics import init_metrics
from config.settings import SETTINGS
from asr import whisper_asr

# ------------- Quick ENV-based configuration -------------
METRICS_PORT = SETTINGS.metrics.port
DEFAULT_RULESET = SETTINGS.fusion.fuzzy_ruleset
WHISPER_MODEL_SIZE = SETTINGS.models.whisper_size
USE_MFCC = SETTINGS.models.use_mfcc
ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")  # CORS can still be driven by ENV

# ------------- App -------------
app = FastAPI(title="Affective Voice API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize /metrics once
init_metrics(METRICS_PORT)

class AnalyzeItem(BaseModel):
    filename: str
    run_id: str
    asr_confidence: float = Field(..., description="Aggregated ASR confidence (0-1)")
    response: str
    emotion_fused: Dict[str, float]
    audit: Dict[str, Any]
    blockchain: Optional[Dict[str, Any]] = None
    fusion_fuzzy: Optional[Dict[str, Any]] = None
    # Optionally return raw and redacted text
    text: Optional[str] = None
    text_redacted: Optional[str] = None
    whisper_model_size: Optional[str] = None
    fuzzy_ruleset_path: Optional[str] = None
    use_mfcc: Optional[bool] = None
    metrics_port: Optional[int] = None

class AnalyzeResponse(BaseModel):
    results: List[AnalyzeItem]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "run_id": SETTINGS.run_id,
        "whisper_model_size_default": WHISPER_MODEL_SIZE,
        "whisper_model_size_active": whisper_asr.get_model_size(),
        "fuzzy_ruleset_default": DEFAULT_RULESET,
        "metrics_port": METRICS_PORT,
        "use_mfcc_default": USE_MFCC,
        "audio_backend": SETTINGS.models.audio_backend,
        "text_backend": SETTINGS.models.text_backend,
        "blockchain": {
            "enabled": SETTINGS.blockchain.enabled,
            "contract_info_path": SETTINGS.blockchain.contract_info_path,
        },
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    files: List[UploadFile] = File(..., description="One or more .wav/.mp3 audio files"),
    run_id: Optional[str] = Form(default=None),
    expose_text: bool = Form(default=False, description="If true, include text/text_redacted in the response"),
    whisper_model_size: Optional[str] = Form(default=None, description="Optional override for Whisper size"),
    blockchain_enabled: Optional[bool] = Form(
        default=None,
        description="Force blockchain anchoring on/off for this request",
    ),
    fuzzy_ruleset: Optional[str] = Form(default=None, description="Alternative fuzzy rules YAML path"),
    metrics_port: Optional[int] = Form(default=None, description="Prometheus /metrics port to use"),
    use_mfcc: Optional[bool] = Form(default=None, description="Override MFCC usage in the audio backend"),
):
    if not files:
        raise HTTPException(status_code=400, detail="Attach at least one audio file.")

    base_run_id = run_id or uuid.uuid4().hex[:8]
    results: List[AnalyzeItem] = []

    metrics_port_value = metrics_port if metrics_port is not None else METRICS_PORT

    for idx, file in enumerate(files, start=1):
        suffix = os.path.splitext(file.filename or "")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        rid = base_run_id if len(files) == 1 else f"{base_run_id}-{idx:02d}"

        try:
            out = run_pipeline(
                temp_path,
                metrics_port=metrics_port_value,
                run_id=rid,
                blockchain_enabled=blockchain_enabled,
                whisper_model_size=whisper_model_size,
                fuzzy_ruleset_path=fuzzy_ruleset,
                use_mfcc=use_mfcc,
            )
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        payload = AnalyzeItem(
            filename=file.filename or f"audio-{idx}",
            run_id=rid,
            asr_confidence=float(out.get("asr_confidence", 0.0)),
            response=out.get("response", ""),
            emotion_fused=out.get("emotion_fused", {}),
            audit=out.get("audit", {}),
            blockchain=out.get("blockchain"),
            fusion_fuzzy=out.get("fusion_fuzzy"),
            text=out.get("text", "") if expose_text else None,
            text_redacted=out.get("text_redacted", "") if expose_text else None,
            whisper_model_size=out.get("whisper_model_size"),
            fuzzy_ruleset_path=out.get("ruleset_path"),
            use_mfcc=out.get("use_mfcc"),
            metrics_port=out.get("metrics_port"),
        )
        results.append(payload)

    return AnalyzeResponse(results=results)
