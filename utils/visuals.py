# utils/visuals.py
import os
import json
from typing import List, Dict, Any, Optional
import math

# visual libs (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def fired_rules_to_matrix(fired_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convierte fired_rules (lista de {if:[...], then:..., strength:float})
    en una representación para JSON y matriz para heatmap.

    Devuelve dict:
    {
      "rule_ids": [...],           # ids ordenadas
      "conds": [...],              # list of unique conditions (tokens)
      "matrix": [[0..1,...], ...]  # rows = rules, cols = conditions; value = strength when rule contains the condition
      "then_labels": [...],        # etiqueta de salida por regla (ej. 'high','mid','low')
      "strengths": [...],          # strengths originales
    }
    """
    # Normalizar fired_rules structure
    # fired_rules puede ser [{'if': ['asr_conf is high'], 'then': 'w_text is high', 'strength': 0.9}, ...]
    cond_set = []
    for r in fired_rules:
        for c in r.get("if", []):
            if c not in cond_set:
                cond_set.append(c)
    rule_ids = [f"R{idx+1}" for idx in range(len(fired_rules))]
    matrix = []
    then_labels = []
    strengths = []
    for r in fired_rules:
        strength = float(r.get("strength", 0.0))
        strengths.append(strength)
        then = r.get("then", "")
        then_labels.append(then)
        row = []
        conds = r.get("if", [])
        for c in cond_set:
            row.append(strength if c in conds else 0.0)
        matrix.append(row)

    return {
        "rule_ids": rule_ids,
        "conds": cond_set,
        "matrix": matrix,
        "then_labels": then_labels,
        "strengths": strengths
    }

def export_fired_rules_json(outdir: str, txid: str, fired_rules: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Exporta JSON con fired_rules + meta. Retorna path del archivo.
    """
    ensure_dir(outdir)
    pdata = {
        "txid": txid,
        "fired_rules": fired_rules,
        "meta": meta or {}
    }
    path = os.path.join(outdir, f"{txid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pdata, f, ensure_ascii=False, indent=2)
    return path

def save_heatmap_from_matrix(outdir: str, txid: str, matrix_repr: Dict[str, Any], cmap: str = "viridis") -> Optional[str]:
    """
    Genera un heatmap PNG a partir de matrix_repr (salida de fired_rules_to_matrix).
    Requiere matplotlib; si no está, retorna None.
    """
    if not HAS_MPL:
        return None
    ensure_dir(outdir)
    m = matrix_repr["matrix"]
    conds = matrix_repr["conds"]
    rule_ids = matrix_repr["rule_ids"]
    strengths = matrix_repr["strengths"]

    # Matplotlib expects numeric matrix; guard contra empty
    if not m or not m[0]:
        # nothing to plot
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(conds)*0.6), max(2, len(rule_ids)*0.4)))
    im = ax.imshow(m, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)

    # ticks
    ax.set_yticks(range(len(rule_ids)))
    ax.set_yticklabels(rule_ids)
    ax.set_xticks(range(len(conds)))
    # rotar etiquetas si son largas
    ax.set_xticklabels(conds, rotation=45, ha='right', fontsize=8)
    ax.set_title(f"Fired rules strengths (txid={txid})")
    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("strength (0..1)")

    outpath = os.path.join(outdir, f"{txid}.png")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath
