from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULTS_PATH = CONFIG_DIR / "defaults.yaml"
LOCAL_OVERRIDE_PATH = CONFIG_DIR / "local.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(copy.deepcopy(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at top level")
    return data


_ENV_MAP: Dict[str, Iterable[str]] = {
    "WHISPER_MODEL_SIZE": ("models", "whisper_size"),
    "USE_MFCC": ("models", "use_mfcc"),
    "TEXT_EMO_BACKEND": ("models", "text_backend"),
    "AUDIO_EMO_BACKEND": ("models", "audio_backend"),
    "FUZZY_RULESET": ("fusion", "fuzzy_ruleset"),
    "METRICS_PORT": ("metrics", "port"),
    "AUDIT_DIR": ("paths", "audit_dir"),
    "AUDIT_FILE": ("paths", "audit_file"),
    "TEXT_EMO_DYNAMIC_LEXICON": ("paths", "dynamic_lexicon"),
    "ESCALATION_WEBHOOK": ("paths", "escalation_webhook"),
}


def _assign(config: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    target = config
    *parents, key = path
    for fragment in parents:
        target = target.setdefault(fragment, {})
        if not isinstance(target, dict):
            raise ValueError(f"Cannot assign into non-dict configuration path {'.'.join(path)}")
    target[key] = value


def _coerce_bool(raw: str) -> bool:
    return raw.lower() in {"1", "true", "yes", "on"}


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    for env_key, path in _ENV_MAP.items():
        raw = os.getenv(env_key)
        if raw is None:
            continue
        if env_key == "USE_MFCC":
            value = _coerce_bool(raw)
        elif env_key == "METRICS_PORT":
            value = int(raw)
        else:
            value = raw
        _assign(config, path, value)

    run_id = os.getenv("RUN_ID")
    if run_id:
        config.setdefault("runtime", {})["run_id"] = run_id


def load_settings_data() -> Dict[str, Any]:
    config = _load_yaml(DEFAULTS_PATH)
    if LOCAL_OVERRIDE_PATH.exists():
        overrides = _load_yaml(LOCAL_OVERRIDE_PATH)
        config = _deep_merge(config, overrides)

    _apply_env_overrides(config)

    runtime_cfg = config.setdefault("runtime", {})
    if not runtime_cfg.get("run_id"):
        prefix_len = int(runtime_cfg.get("run_id_prefix_length", 8))
        runtime_cfg["run_id"] = uuid.uuid4().hex[:prefix_len]

    return config

