# utils/metrics.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from contextlib import contextmanager
import time
import threading

import logging
from typing import Mapping

from config.settings import SETTINGS

_logger = logging.getLogger("metrics")

_started = False
_lock = threading.Lock()

_METRIC_LABELS = ("model_size", "run_id")
_DEFAULT_LABELS = {
    "model_size": SETTINGS.models.whisper_size,
    "run_id": SETTINGS.run_id or "bootstrap",
}

asr_latency = Histogram(
    "asr_latency_seconds",
    "ASR latency per file",
    _METRIC_LABELS,
)
audio_emotion_latency = Histogram(
    "audio_emotion_latency_seconds",
    "Audio emotion inference latency",
    _METRIC_LABELS,
)
text_emotion_latency = Histogram(
    "text_emotion_latency_seconds",
    "Text emotion inference latency",
    _METRIC_LABELS,
)
fusion_latency = Histogram(
    "fusion_latency_seconds",
    "Emotion fusion latency",
    _METRIC_LABELS,
)
pii_redactions_total = Counter(
    "pii_redactions_total",
    "Number of redactions applied to PII",
    _METRIC_LABELS,
)
pipeline_errors_total = Counter(
    "pipeline_errors_total",
    "Pipeline errors",
    _METRIC_LABELS,
)
asr_confidence_gauge = Gauge(
    "asr_confidence_gauge",
    "Estimated ASR confidence",
    _METRIC_LABELS,
)
audio_snr_db_gauge = Gauge(
    "audio_snr_db_gauge",
    "Estimated SNR (dB) for audio",
    _METRIC_LABELS,
)
cross_modal_coherence_gauge = Gauge(
    "cross_modal_coherence_gauge",
    "Heuristic audio-text coherence index",
    _METRIC_LABELS,
)

# Guardrails: how many guardrail events fired, grouped by level (notice/high/critical)
guardrails_triggered_total = Counter(
    "guardrails_triggered_total",
    "Guardrail activations by level",
    ("level", *_METRIC_LABELS),
)

def init_metrics(port: int | None = None):
    """Expose /metrics only once."""
    global _started
    if _started:
        return
    with _lock:
        if not _started:
            effective_port = port if port is not None else SETTINGS.metrics.port
            start_http_server(effective_port)
            _started = True
            # Pre-warm gauges/counters so they exist with default values
            try:
                guardrails_triggered_total.labels(level="unknown", **_DEFAULT_LABELS).inc(0)

                pii_redactions_total.labels(**_DEFAULT_LABELS).inc(0)
                pipeline_errors_total.labels(**_DEFAULT_LABELS).inc(0)

                asr_confidence_gauge.labels(**_DEFAULT_LABELS).set(0.0)
                audio_snr_db_gauge.labels(**_DEFAULT_LABELS).set(0.0)
                cross_modal_coherence_gauge.labels(**_DEFAULT_LABELS).set(0.0)
            except Exception:
                _logger.exception("init_metrics: failed to initialize default samples")

@contextmanager
def timeit(histogram, labels: Mapping[str, str]):
    """Record the duration of the block in the histogram with the provided labels."""
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.labels(**labels).observe(time.perf_counter() - start)


def inc_counter(counter, labels: Mapping[str, str], amount: float = 1.0):
    counter.labels(**labels).inc(amount)


def set_gauge(gauge, labels: Mapping[str, str], value: float):
    gauge.labels(**labels).set(value)

def inc_guardrails(level: str, labels: Mapping[str, str]):
    """
    Increment guardrails_triggered_total with the given level and labels (model_size, run_id).
    Labels should include 'model_size' and 'run_id'; defaults are applied otherwise.

     - Validate that the metric exists.
     - Log errors instead of silencing them completely.
     - Avoid breaking pipeline execution.
    """
    try:
        # Defaults when keys are missing
        model_size = labels.get("model_size", "unknown")
        run_id = labels.get("run_id", "unknown")

        # Verify the metric exists and supports .labels
        if "guardrails_triggered_total" not in globals():
            _logger.debug("inc_guardrails: metric guardrails_triggered_total not found in globals()")
        else:
            guardrails_triggered_total.labels(level=level, model_size=model_size, run_id=run_id).inc()
    except Exception as exc:
        # Log exceptions for debugging without re-raising
        _logger.exception("inc_guardrails failed to increment metric: %s", exc)