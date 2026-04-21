"""Audio ingestion — decode WAV/MP3/M4A, resample to 16 kHz mono float32."""
from __future__ import annotations

from pathlib import Path

from .types import AudioClip


def ingest(path: Path) -> AudioClip:
    raise NotImplementedError("T2.1 loadAudio.ingest — see docs/TASKS.md")
