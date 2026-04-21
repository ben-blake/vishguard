"""Anti-spoof stage — clip -> SpoofVerdict.

Primary model: MelodyMachine/Deepfake-audio-detection-V2. See
docs/ARCHITECTURE.md §2.2 for fallbacks and the Phase 1 smoke-test
acceptance criteria.
"""
from __future__ import annotations

from .types import AudioClip, SpoofConfig, SpoofVerdict


def detectSpoof(clip: AudioClip, cfg: SpoofConfig) -> SpoofVerdict:
    raise NotImplementedError("T2.3 antiSpoof.detectSpoof — see docs/TASKS.md")
