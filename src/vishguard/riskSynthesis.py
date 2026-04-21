"""Risk-synthesis stage — combine spoof + tactics + transcript into RiskScore."""
from __future__ import annotations

from .promptLibrary import riskReasoningPrompt
from .types import LlmConfig, RiskScore, SpoofVerdict, Tactic, Transcript

_BENIGN_LABEL = "benign"
_SPOOF_WEIGHT = 40
_MALICIOUS_WEIGHT = 15
_TACTIC_CAP = 60
_BENIGN_PENALTY = 20


def _assignBand(score: int) -> str:
    if score >= 75:
        return "critical"
    if score >= 50:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def _computeScore(spoof: SpoofVerdict, tactics: tuple[Tactic, ...]) -> int:
    malicious_sum = sum(t.confidence for t in tactics if t.label != _BENIGN_LABEL)
    tactic_component = min(_MALICIOUS_WEIGHT * malicious_sum, _TACTIC_CAP)

    has_benign = any(t.label == _BENIGN_LABEL for t in tactics)
    benign_deduction = _BENIGN_PENALTY if has_benign else 0

    raw = _SPOOF_WEIGHT * spoof.pSynthetic + tactic_component - benign_deduction
    return max(0, min(100, round(raw)))


def _callLlmForReasoning(
    transcript: Transcript,
    spoof: SpoofVerdict,
    tactics: tuple[Tactic, ...],
    cfg: LlmConfig,
) -> str:
    from .tacticClassifier import _call_llm

    tactics_json = str(
        [
            {
                "label": t.label,
                "confidence": round(t.confidence, 2),
                "evidenceSpans": list(t.evidenceSpans),
            }
            for t in tactics
        ]
    )
    messages = riskReasoningPrompt(transcript.fullText, spoof.pSynthetic, tactics_json)
    return _call_llm(messages, cfg)


def synthesizeRisk(
    transcript: Transcript,
    spoof: SpoofVerdict,
    tactics: tuple[Tactic, ...],
    cfg: LlmConfig,
) -> RiskScore:
    score = _computeScore(spoof, tactics)
    band = _assignBand(score)
    reasoning = _callLlmForReasoning(transcript, spoof, tactics, cfg)
    return RiskScore(score=score, band=band, reasoning=reasoning)
