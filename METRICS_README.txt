MÉTRICAS (Prometheus) - Guía rápida

1) Instalar deps (además de ffmpeg):
   pip install -r requirements.txt

2) Ejecutar exponiendo /metrics (p.ej., puerto 9000) y mantiene /metrics vivo 60 segundos
   python -m app.cli --audio data/examples/mi_audio.wav --pretty --metrics-port 9000 --hold 60


3) Ver en el navegador:
   http://localhost:9000/metrics

4) Métricas:
   - asr_latency_seconds
   - audio_emotion_latency_seconds
   - text_emotion_latency_seconds
   - fusion_latency_seconds
   - pii_redactions_total
   - pipeline_errors_total
   - asr_confidence_gauge
   - audio_snr_db_gauge
   - cross_modal_coherence_gauge

Todas las métricas exponen las etiquetas `model_size` y `run_id`.
