"""Risk-synthesis stage — combine spoof + tactics + transcript into RiskScore."""
from __future__ import annotations

from .types import LlmConfig, RiskScore, SpoofVerdict, Tactic, Transcript


def synthesizeRisk(
    transcript: Transcript,
    spoof: SpoofVerdict,
    tactics: tuple[Tactic, ...],
    cfg: LlmConfig,
) -> RiskScore:
    raise NotImplementedError("T2.5 riskSynthesis.synthesizeRisk — see docs/TASKS.md")
