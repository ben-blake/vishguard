"""SpeechT5 TTS briefing stage — RiskReport -> WAV path."""
from __future__ import annotations

from pathlib import Path

from .types import RiskReport, TtsConfig


def narrateBriefing(report: RiskReport, cfg: TtsConfig, outDir: Path) -> Path:
    raise NotImplementedError("T2.6 briefingTts.narrateBriefing — see docs/TASKS.md")
