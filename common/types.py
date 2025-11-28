"""
common/types.py

📘 Definición de estructuras de datos compartidas entre los módulos del sistema:
- Reconocimiento de voz (ASR)
- Análisis de emociones (audio y texto)
- Fusión afectiva y auditoría

Estas clases se implementan como dataclasses para facilitar
su serialización, comparación y trazabilidad en auditoría.
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any


# -------------------------------------------------------------------------
# 🎙️ ASR (Automatic Speech Recognition)
# -------------------------------------------------------------------------

@dataclass
class ASRWord:
    """
    Representa una palabra reconocida por el motor ASR con su tiempo y confianza.
    """
    word: str
    start: float = 0.0   # segundo en el que empieza la palabra
    end: float = 0.0     # segundo en el que termina
    confidence: float = 0.5  # [0,1] confianza estimada por el modelo


@dataclass
class ASRResult:
    """
    Resultado global de una transcripción ASR.
    """
    text: str                   # texto transcripto completo
    words: List[ASRWord]        # lista de palabras con tiempos
    confidence: float           # confianza global [0,1]


# -------------------------------------------------------------------------
# 💬 Emotion model (VAD and categorical distribution)
# -------------------------------------------------------------------------

@dataclass
class VAD:
    """
    Representa la emoción en un espacio continuo de tres dimensiones:
      - Valence: qué tan positiva o negativa es la emoción  [-1,1]
      - Arousal: nivel de activación o energía               [0,1]
      - Dominance: sensación de control/sumisión             [0,1]
    """
    valence: float
    arousal: float
    dominance: float


@dataclass
class EmotionResult:
    """
    Resultado de un clasificador de emoción (por audio o texto).

    - probs: diccionario con la probabilidad por cada emoción discreta
      Ejemplo: {"joy": 0.3, "anger": 0.1, "neutral": 0.4, ...}
    - vad: vector continuo de Valence–Arousal–Dominance
    - confidence: confianza global del modelo [0,1]
    - top_label: etiqueta dominante (opcional)
    - metadata: campo con features para emociones por voz (arousal_raw, zcr_raw, zcr_norm, mfcc_present, timbre_score, arousal_smoothed)
    """
    probs: Dict[str, float]
    vad: VAD
    confidence: float
    top_label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

# -------------------------------------------------------------------------
# 🧰 Serialization helper
# -------------------------------------------------------------------------

def to_dict(obj) -> Dict:
    """
    Convierte cualquier dataclass del módulo a diccionario.
    Equivalente a dataclasses.asdict(), pero más explícito.
    """
    return asdict(obj)
