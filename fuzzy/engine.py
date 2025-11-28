# fuzzy/engine.py
# Load YAML, perform fuzzification (triangles), Mamdani inference (min–max), aggregation,
# and centroid defuzzification to produce numeric w_text.
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import json, os
import math

try:
    import yaml  # PyYAML is common; otherwise fall back to json
except Exception:
    yaml = None

@dataclass
class Universe:
    lo: float
    hi: float

def _tri(x: float, a: float, b: float, c: float) -> float:
    """Triángulo [a,b,c]."""
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b:  return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)

def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoide [a,b,c,d]."""
    if x <= a or x >= d: return 0.0
    if b <= x <= c: return 1.0
    if a < x < b:  return (x - a) / (b - a + 1e-12)
    return (d - x) / (d - c + 1e-12)

def _mf_eval(x: float, params: List[float]) -> float:
    if len(params) == 3:
        return max(0.0, min(1.0, _tri(x, *params)))
    if len(params) == 4:
        return max(0.0, min(1.0, _trap(x, *params)))
    raise ValueError("Membership function must have 3 (tri) or 4 (trap) parameters")

class FuzzyEngine:
    def __init__(self, ruleset_path: str):
        self.ruleset_path = ruleset_path
        self.cfg = self._load_ruleset(ruleset_path)
        self.universe = {
            k: Universe(float(v[0]), float(v[1]))
            for k, v in self.cfg["universe"].items()
        }
        self.mf = self.cfg["membership"]
        self.rules = self.cfg["rules"]
        self.version = str(self.cfg.get("version", "unknown"))

    def get_version(self) -> str:
        return self.version

    def _load_ruleset(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ruleset no encontrado: {path}")
        text = open(path, "r", encoding="utf-8").read()
        if yaml:
            return yaml.safe_load(text)
        # Fallback when yaml is missing: support JSON when the file uses .json
        return json.loads(text)

    def _clip(self, x: float, var: str) -> float:
        u = self.universe[var]
        return max(u.lo, min(u.hi, x))

    def _fuzzify(self, var: str, x: float) -> Dict[str, float]:
        """Devuelve pertenencias {etiqueta: μ} para un var en x."""
        x = self._clip(x, var)
        out = {}
        for label, params in self.mf[var].items():
            mu = _mf_eval(x, list(map(float, params)))
            out[label] = mu
        return out

    def _parse_cond(self, token: str) -> Tuple[str, str]:
        # "asr_conf is high" -> ("asr_conf", "high")
        parts = token.split(" is ")
        if len(parts) != 2:
            raise ValueError(f"Condición inválida: {token}")
        return parts[0].strip(), parts[1].strip()

    def infer_w_text(self, asr_conf: float, arousal: float, valence: float) -> Dict[str, Any]:
        """Mamdani: devuelve w_text numérico y detalle para explicación."""
        # 1) Fuzzificar entradas
        asr_f = self._fuzzify("asr_conf", asr_conf)
        aro_f = self._fuzzify("arousal", arousal)
        val_f = self._fuzzify("valence", valence)

        # 2) Evaluar reglas (min sobre condiciones; max para OR si apareciera)
        # Agregamos en el dominio de salida 'w_text': conjuntos low/mid/high
        out_sets: Dict[str, float] = {"low": 0.0, "mid": 0.0, "high": 0.0}
        fired_rules: List[Tuple[List[str], str, float]] = []

        for rule in self.rules:
            conds: List[str] = rule["IF"]
            then: str = rule["THEN"]  # "w_text is X"
            out_var, out_label = self._parse_cond(then)

            # Implicit AND: minimum membership across all conditions
            mus: List[float] = []
            for c in conds:
                var, lab = self._parse_cond(c)
                if var == "asr_conf":
                    mus.append(asr_f.get(lab, 0.0))
                elif var == "arousal":
                    mus.append(aro_f.get(lab, 0.0))
                elif var == "valence":
                    mus.append(val_f.get(lab, 0.0))
                else:
                    raise ValueError(f"Variable desconocida en regla: {var}")
            firing = min(mus) if mus else 0.0
            out_sets[out_label] = max(out_sets[out_label], firing)
            fired_rules.append((conds, out_label, firing))

        # 3) Defuzzification (centroid) over w_text ∈ [0..1]
        # Build a simple discretization for the centroid
        lo, hi = self.universe["w_text"].lo, self.universe["w_text"].hi
        X = [lo + (hi - lo) * i / 200 for i in range(201)]
        # Para cada punto x, la pertenencia agregada es max de cada label recortado por firing
        def mu_agg(x: float) -> float:
            total = 0.0
            for label, firing in out_sets.items():
                if firing <= 0: 
                    continue
                params = self.mf["w_text"][label]
                mu = _mf_eval(x, list(map(float, params)))
                total = max(total, min(mu, firing))  # Mamdani: min, luego max
            return total

        num, den = 0.0, 0.0
        for x in X:
            mu = mu_agg(x)
            num += x * mu
            den += mu
        w_text = num / den if den > 0 else 0.5  # neutral si nada dispara

        return {
            "w_text": float(max(0.0, min(1.0, w_text))),
            "details": {
                "inputs": {"asr_conf": asr_conf, "arousal": arousal, "valence": valence},
                "fired_rules": [
                    {"if": conds, "then": f"w_text is {label}", "strength": firing}
                    for conds, label, firing in fired_rules
                ],
                "out_sets": out_sets
            }
        }
