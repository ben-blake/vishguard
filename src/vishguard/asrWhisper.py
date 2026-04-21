"""Whisper ASR stage — clip -> Transcript."""
from __future__ import annotations

from .types import AsrConfig, AudioClip, Transcript


def transcribe(clip: AudioClip, cfg: AsrConfig) -> Transcript:
    raise NotImplementedError("T2.2 asrWhisper.transcribe — see docs/TASKS.md")
