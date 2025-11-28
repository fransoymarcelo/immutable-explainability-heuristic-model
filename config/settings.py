# config/settings.py
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .loader import load_settings_data


@dataclass(frozen=True)
class ModelsConfig:
    whisper_size: str
    use_mfcc: bool
    text_backend: str
    audio_backend: str


@dataclass(frozen=True)
class FusionSimpleConfig:
    text_confidence_thresh: float
    asr_confidence_thresh: float
    audio_confidence_thresh: float
    audio_arousal_thresh: float


@dataclass(frozen=True)
class FusionAsrModulationConfig:
    coverage_floor: float
    coverage_low_factor: float
    text_conf_floor: float
    text_conf_low_factor: float
    cap_min: float
    cap_max: float


@dataclass(frozen=True)
class FusionPostFuzzyConfig:
    high_conf_cap: float
    low_evidence_cap: float
    neutral_audio_relief: float
    audio_conf_boost: float
    audio_non_neutral_threshold: float
    audio_override_cap: float
    min_weight: float
    max_weight: float


@dataclass(frozen=True)
class FusionConfig:
    fuzzy_ruleset: str
    fusion_simple: FusionSimpleConfig
    asr_modulation: FusionAsrModulationConfig
    post_fuzzy: FusionPostFuzzyConfig


@dataclass(frozen=True)
class GuardrailThresholds:
    fear: float
    anger: float
    sadness: float
    arousal: float
    valence_neg: float


@dataclass(frozen=True)
class GuardrailTemplates:
    safe: str
    soft: str


@dataclass(frozen=True)
class GuardrailsConfig:
    thresholds: GuardrailThresholds
    risk_keywords: List[str]
    templates: GuardrailTemplates


@dataclass(frozen=True)
class PathsConfig:
    audit_dir: str
    audit_file: str
    dynamic_lexicon: Optional[str]
    escalation_webhook: Optional[str]


@dataclass(frozen=True)
class MetricsConfig:
    port: int


@dataclass(frozen=True)
class AudioHeuristics:
    ema_alpha: float
    base_neutral: float
    wav_confidence: float
    other_confidence: float


@dataclass(frozen=True)
class TextDynamicLexiconConfig:
    min_freq: int
    min_ratio: float


@dataclass(frozen=True)
class TextHeuristics:
    fallback_neutral: float
    dynamic_lexicon: TextDynamicLexiconConfig


@dataclass(frozen=True)
class BlockchainConfig:
    enabled: bool
    contract_info_path: str
    rpc_url_env: str
    private_key_env: str
    gas_limit: int
    wait_for_receipt: bool


@dataclass(frozen=True)
class Settings:
    models: ModelsConfig
    fusion: FusionConfig
    guardrails: GuardrailsConfig
    paths: PathsConfig
    metrics: MetricsConfig
    audio: AudioHeuristics
    text: TextHeuristics
    pii_patterns: Dict[str, str]
    blockchain: BlockchainConfig
    run_id: str
    raw: Dict[str, Any]

    # --- Compatibility helpers (legacy attribute names) ---
    @property
    def whisper_model_size(self) -> str:
        return self.models.whisper_size

    @property
    def use_mfcc(self) -> bool:
        return self.models.use_mfcc

    @property
    def fuzzy_ruleset(self) -> str:
        return self.fusion.fuzzy_ruleset

    @property
    def metrics_port(self) -> int:
        return self.metrics.port

    @property
    def audit_dir(self) -> str:
        return self.paths.audit_dir

    @property
    def audit_file(self) -> str:
        return self.paths.audit_file

    @property
    def RISK_KEYWORDS(self) -> List[str]:  # noqa: N802 (legacy camel case)
        return self.guardrails.risk_keywords

    @property
    def SAFE_TEMPLATE(self) -> str:  # noqa: N802
        return self.guardrails.templates.safe

    @property
    def SOFT_TEMPLATE(self) -> str:  # noqa: N802
        return self.guardrails.templates.soft

    @property
    def PII_PATTERNS(self) -> Dict[str, str]:  # noqa: N802
        return self.pii_patterns

    @property
    def THRESH_MIEDO(self) -> float:  # noqa: N802
        return self.guardrails.thresholds.fear

    @property
    def THRESH_IRA(self) -> float:  # noqa: N802
        return self.guardrails.thresholds.anger

    @property
    def THRESH_TRISTEZA(self) -> float:  # noqa: N802
        return self.guardrails.thresholds.sadness

    @property
    def THRESH_GUARD_AROUSAL(self) -> float:  # noqa: N802
        return self.guardrails.thresholds.arousal

    @property
    def THRESH_GUARD_VALENCE_NEG(self) -> float:  # noqa: N802
        return self.guardrails.thresholds.valence_neg

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)


def _build_models(config: Dict[str, Any]) -> ModelsConfig:
    data = config.get("models", {})
    return ModelsConfig(
        whisper_size=str(data.get("whisper_size", "small")),
        use_mfcc=bool(data.get("use_mfcc", False)),
        text_backend=str(data.get("text_backend", "heuristic")),
        audio_backend=str(data.get("audio_backend", "heuristic")),
    )


def _build_fusion(config: Dict[str, Any]) -> FusionConfig:
    data = config.get("fusion", {})
    simple = data.get("fusion_simple", {})
    fusion_simple = FusionSimpleConfig(
        text_confidence_thresh=float(simple.get("text_confidence_thresh", 0.6)),
        asr_confidence_thresh=float(simple.get("asr_confidence_thresh", 0.75)),
        audio_confidence_thresh=float(simple.get("audio_confidence_thresh", 0.6)),
        audio_arousal_thresh=float(simple.get("audio_arousal_thresh", 0.6)),
    )
    asr_mod = data.get("asr_modulation", {})
    asr_modulation = FusionAsrModulationConfig(
        coverage_floor=float(asr_mod.get("coverage_floor", 0.35)),
        coverage_low_factor=float(asr_mod.get("coverage_low_factor", 0.6)),
        text_conf_floor=float(asr_mod.get("text_conf_floor", 0.45)),
        text_conf_low_factor=float(asr_mod.get("text_conf_low_factor", 0.7)),
        cap_min=float(asr_mod.get("cap_min", 0.18)),
        cap_max=float(asr_mod.get("cap_max", 0.9)),
    )
    post = data.get("post_fuzzy", {})
    post_fuzzy = FusionPostFuzzyConfig(
        high_conf_cap=float(post.get("high_conf_cap", 0.75)),
        low_evidence_cap=float(post.get("low_evidence_cap", 0.65)),
        neutral_audio_relief=float(post.get("neutral_audio_relief", 0.1)),
        audio_conf_boost=float(post.get("audio_conf_boost", 0.08)),
        audio_non_neutral_threshold=float(post.get("audio_non_neutral_threshold", 0.45)),
        audio_override_cap=float(post.get("audio_override_cap", 0.7)),
        min_weight=float(post.get("min_weight", 0.2)),
        max_weight=float(post.get("max_weight", 0.82)),
    )
    return FusionConfig(
        fuzzy_ruleset=str(data.get("fuzzy_ruleset", "core/ruleset/default.yaml")),
        fusion_simple=fusion_simple,
        asr_modulation=asr_modulation,
        post_fuzzy=post_fuzzy,
    )


def _build_guardrails(config: Dict[str, Any]) -> GuardrailsConfig:
    data = config.get("guardrails", {})
    thresholds_data = data.get("thresholds", {})
    thresholds = GuardrailThresholds(
        fear=float(thresholds_data.get("fear", 0.70)),
        anger=float(thresholds_data.get("anger", 0.85)),
        sadness=float(thresholds_data.get("sadness", 0.85)),
        arousal=float(thresholds_data.get("arousal", 0.8)),
        valence_neg=float(thresholds_data.get("valence_neg", -0.4)),
    )
    templates_data = data.get("templates", {})
    templates = GuardrailTemplates(
        safe=str(templates_data.get("safe", "")),
        soft=str(templates_data.get("soft", "")),
    )
    risk_keywords = [str(item) for item in data.get("risk_keywords", [])]
    return GuardrailsConfig(
        thresholds=thresholds,
        risk_keywords=risk_keywords,
        templates=templates,
    )


def _build_paths(config: Dict[str, Any]) -> PathsConfig:
    data = config.get("paths", {})
    dynamic_path = data.get("dynamic_lexicon")
    if dynamic_path is not None:
        dynamic_path = str(dynamic_path)
    escalation = data.get("escalation_webhook")
    if escalation is not None:
        escalation = str(escalation)
    return PathsConfig(
        audit_dir=str(data.get("audit_dir", "audit")),
        audit_file=str(data.get("audit_file", "events.log")),
        dynamic_lexicon=dynamic_path,
        escalation_webhook=escalation,
    )


def _build_metrics(config: Dict[str, Any]) -> MetricsConfig:
    data = config.get("metrics", {})
    return MetricsConfig(port=int(data.get("port", 9000)))


def _build_audio(config: Dict[str, Any]) -> AudioHeuristics:
    data = config.get("audio", {})
    return AudioHeuristics(
        ema_alpha=float(data.get("ema_alpha", 0.2)),
        base_neutral=float(data.get("base_neutral", 0.60)),
        wav_confidence=float(data.get("wav_confidence", 0.65)),
        other_confidence=float(data.get("other_confidence", 0.50)),
    )


def _build_text(config: Dict[str, Any]) -> TextHeuristics:
    data = config.get("text", {})
    dynamic = data.get("dynamic_lexicon", {})
    dynamic_cfg = TextDynamicLexiconConfig(
        min_freq=int(dynamic.get("min_freq", 4)),
        min_ratio=float(dynamic.get("min_ratio", 1.8)),
    )
    return TextHeuristics(
        fallback_neutral=float(data.get("fallback_neutral", 0.5)),
        dynamic_lexicon=dynamic_cfg,
    )


def _build_blockchain(config: Dict[str, Any]) -> BlockchainConfig:
    data = config.get("blockchain", {})
    return BlockchainConfig(
        enabled=bool(data.get("enabled", False)),
        contract_info_path=str(data.get("contract_info_path", "contract_info.json")),
        rpc_url_env=str(data.get("rpc_url_env", "RPC_URL")),
        private_key_env=str(data.get("private_key_env", "PRIVATE_KEY")),
        gas_limit=int(data.get("gas_limit", 120000)),
        wait_for_receipt=bool(data.get("wait_for_receipt", True)),
    )


def load_settings() -> Settings:
    raw_config = load_settings_data()
    runtime = raw_config.get("runtime", {})
    return Settings(
        models=_build_models(raw_config),
        fusion=_build_fusion(raw_config),
        guardrails=_build_guardrails(raw_config),
        paths=_build_paths(raw_config),
        metrics=_build_metrics(raw_config),
        audio=_build_audio(raw_config),
        text=_build_text(raw_config),
        pii_patterns=dict(raw_config.get("pii", {}).get("patterns", {})),
        blockchain=_build_blockchain(raw_config),
        run_id=str(runtime.get("run_id")),
        raw=raw_config,
    )


SETTINGS = load_settings()
