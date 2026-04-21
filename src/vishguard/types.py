"""Shared data types for the VishGuard pipeline.

All stages are pure functions over these frozen dataclasses. Do not add
mutable state. The tactic taxonomy is the single source of truth used by
prompts, evaluation manifests, and report schema validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


TACTIC_TAXONOMY: tuple[str, ...] = (
    "authority",
    "urgency",
    "pretexting",
    "credential_harvesting",
    "impersonation",
    "financial_manipulation",
    "fear_intimidation",
    "reward_prize",
    "tech_support",
    "benign",
)


RISK_BANDS: tuple[str, ...] = ("low", "medium", "high", "critical")


@dataclass(frozen=True)
class AudioClip:
    samples: Any  # numpy.ndarray at runtime; Any here avoids an import-time dep.
    sampleRate: int
    sourcePath: Path
    durationSec: float


@dataclass(frozen=True)
class AudioMeta:
    sourcePath: Path
    durationSec: float
    sampleRate: int


@dataclass(frozen=True)
class TranscriptSegment:
    startSec: float
    endSec: float
    text: str


@dataclass(frozen=True)
class Transcript:
    fullText: str
    segments: tuple[TranscriptSegment, ...]
    languageCode: str
    modelId: str


@dataclass(frozen=True)
class SpoofVerdict:
    pSynthetic: float
    modelId: str
    rationale: str


@dataclass(frozen=True)
class Tactic:
    label: str
    confidence: float
    evidenceSpans: tuple[str, ...]


@dataclass(frozen=True)
class RiskScore:
    score: int
    band: str
    reasoning: str


@dataclass(frozen=True)
class RiskReport:
    version: str
    audio: AudioMeta
    transcript: Transcript
    spoof: SpoofVerdict
    tactics: tuple[Tactic, ...]
    risk: RiskScore
    briefingAudioPath: Optional[Path]
    timings: dict[str, float]


@dataclass(frozen=True)
class AsrConfig:
    modelId: str = "openai/whisper-small"
    device: str = "cpu"
    chunkLengthSec: int = 30


@dataclass(frozen=True)
class SpoofConfig:
    modelId: str = "MelodyMachine/Deepfake-audio-detection-V2"
    device: str = "cpu"


@dataclass(frozen=True)
class LlmConfig:
    modelId: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda"
    loadIn4Bit: bool = True
    maxNewTokens: int = 512
    promptVariant: str = "v2"


@dataclass(frozen=True)
class TtsConfig:
    modelId: str = "microsoft/speecht5_tts"
    vocoderId: str = "microsoft/speecht5_hifigan"
    speakerEmbeddingId: str = "Matthijs/cmu-arctic-xvectors"
    device: str = "cpu"
