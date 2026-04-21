"""Whisper ASR stage — clip -> Transcript.

T1.3 spike finding: use WhisperProcessor + WhisperForConditionalGeneration
directly (not the pipeline) — transformers 5.x pipeline raises KeyError on
'num_frames' with dict inputs on some Colab runtimes.

WER evaluation must strip Whisper's added punctuation before comparing
against LibriSpeech-style refs (which have none). See _normalize in eval/.
"""
from __future__ import annotations

from .types import AsrConfig, AudioClip, Transcript


def transcribe(clip: AudioClip, cfg: AsrConfig) -> Transcript:
    raise NotImplementedError("T2.2 asrWhisper.transcribe — see docs/TASKS.md")
