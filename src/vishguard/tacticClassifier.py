"""Open-LLM tactic classification stage — Transcript -> tuple[Tactic, ...]."""
from __future__ import annotations

from .types import LlmConfig, Tactic, Transcript


def classifyTactics(transcript: Transcript, cfg: LlmConfig) -> tuple[Tactic, ...]:
    raise NotImplementedError("T2.4 tacticClassifier.classifyTactics — see docs/TASKS.md")
