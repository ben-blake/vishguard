"""Audio ingestion — decode WAV/MP3/M4A, resample to 16 kHz mono float32."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .types import AudioClip

_TARGET_SR = 16_000


def ingest(path: Path) -> AudioClip:
    path = Path(path)
    samples, _ = librosa.load(str(path), sr=_TARGET_SR, mono=True)
    samples = samples.astype(np.float32)
    return AudioClip(
        samples=samples,
        sampleRate=_TARGET_SR,
        sourcePath=path,
        durationSec=len(samples) / _TARGET_SR,
    )
