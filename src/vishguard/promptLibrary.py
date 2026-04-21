"""Prompt variants for tactic classification and risk synthesis.

Phase 3 A/B tests v1 (bare) vs. v2 (taxonomy + few-shot + JSON schema).
Both variants read TACTIC_TAXONOMY from types.py so the label set cannot
drift between prompts and evaluation.
"""
from __future__ import annotations

from .types import TACTIC_TAXONOMY


def tacticPromptV1(transcript: str) -> str:
    raise NotImplementedError("T2.4 promptLibrary.tacticPromptV1 — see docs/TASKS.md")


def tacticPromptV2(transcript: str) -> str:
    raise NotImplementedError("T2.4 promptLibrary.tacticPromptV2 — see docs/TASKS.md")


def riskReasoningPrompt(transcript: str, pSynthetic: float, tacticsJson: str) -> str:
    raise NotImplementedError("T2.5 promptLibrary.riskReasoningPrompt — see docs/TASKS.md")
