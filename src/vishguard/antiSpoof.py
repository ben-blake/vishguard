"""Anti-spoof stage — clip -> SpoofVerdict.

Primary model: mo-thecreator/Deepfake-audio-detection (confirmed T1.1).
MelodyMachine/Deepfake-audio-detection-V2 always predicts real — do not use.
"""
from __future__ import annotations

from .types import AudioClip, SpoofConfig, SpoofVerdict

_PIPE_CACHE: dict = {}  # model_id -> pipeline


def _load_pipeline(cfg: SpoofConfig):
    if cfg.modelId not in _PIPE_CACHE:
        from transformers import pipeline

        _PIPE_CACHE[cfg.modelId] = pipeline(
            "audio-classification",
            model=cfg.modelId,
            device=0 if cfg.device == "cuda" else -1,
        )
    return _PIPE_CACHE[cfg.modelId]


def _buildRationale(p_synthetic: float) -> str:
    if p_synthetic >= 0.7:
        return (
            f"High-confidence synthetic voice (p_synth={p_synthetic:.2f}); "
            "prosody artifacts typical of TTS output."
        )
    if p_synthetic >= 0.4:
        return (
            f"Moderate synthetic likelihood (p_synth={p_synthetic:.2f}); "
            "inconclusive — recommend human review."
        )
    return (
        f"Low synthetic likelihood (p_synth={p_synthetic:.2f}); "
        "audio consistent with natural human speech."
    )


def detectSpoof(clip: AudioClip, cfg: SpoofConfig) -> SpoofVerdict:
    pipe = _load_pipeline(cfg)
    preds = pipe({"raw": clip.samples, "sampling_rate": clip.sampleRate})
    fake_score = next(p["score"] for p in preds if p["label"].lower() == "fake")
    return SpoofVerdict(
        pSynthetic=float(fake_score),
        modelId=cfg.modelId,
        rationale=_buildRationale(float(fake_score)),
    )
