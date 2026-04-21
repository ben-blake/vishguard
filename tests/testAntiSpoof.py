"""T2.3: Tests for antiSpoof.detectSpoof — pipeline is mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vishguard.types import AudioClip, SpoofConfig, SpoofVerdict


@pytest.fixture
def clip() -> AudioClip:
    return AudioClip(
        samples=np.zeros(16_000, dtype=np.float32),
        sampleRate=16_000,
        sourcePath=Path("test.wav"),
        durationSec=1.0,
    )


@pytest.fixture
def cfg() -> SpoofConfig:
    return SpoofConfig(modelId="mo-thecreator/Deepfake-audio-detection", device="cpu")


def _fake_preds(fake_score: float) -> list[dict]:
    return [
        {"label": "fake", "score": fake_score},
        {"label": "real", "score": 1.0 - fake_score},
    ]


def _mock_pipe(fake_score: float) -> MagicMock:
    pipe = MagicMock()
    pipe.return_value = _fake_preds(fake_score)
    return pipe


class TestDetectSpoof:
    def test_returns_spoof_verdict(self, clip: AudioClip, cfg: SpoofConfig) -> None:
        from vishguard.antiSpoof import detectSpoof

        with patch("vishguard.antiSpoof._load_pipeline", return_value=_mock_pipe(0.8)):
            result = detectSpoof(clip, cfg)
        assert isinstance(result, SpoofVerdict)

    def test_p_synthetic_in_range(self, clip: AudioClip, cfg: SpoofConfig) -> None:
        from vishguard.antiSpoof import detectSpoof

        with patch("vishguard.antiSpoof._load_pipeline", return_value=_mock_pipe(0.8)):
            result = detectSpoof(clip, cfg)
        assert 0.0 <= result.pSynthetic <= 1.0

    def test_model_id_matches_config(self, clip: AudioClip, cfg: SpoofConfig) -> None:
        from vishguard.antiSpoof import detectSpoof

        with patch("vishguard.antiSpoof._load_pipeline", return_value=_mock_pipe(0.8)):
            result = detectSpoof(clip, cfg)
        assert result.modelId == cfg.modelId

    def test_rationale_is_nonempty_string(self, clip: AudioClip, cfg: SpoofConfig) -> None:
        from vishguard.antiSpoof import detectSpoof

        with patch("vishguard.antiSpoof._load_pipeline", return_value=_mock_pipe(0.8)):
            result = detectSpoof(clip, cfg)
        assert isinstance(result.rationale, str)
        assert len(result.rationale) > 0

    def test_high_fake_score_maps_to_high_p_synthetic(
        self, clip: AudioClip, cfg: SpoofConfig
    ) -> None:
        from vishguard.antiSpoof import detectSpoof

        with patch("vishguard.antiSpoof._load_pipeline", return_value=_mock_pipe(0.9)):
            result = detectSpoof(clip, cfg)
        assert result.pSynthetic >= 0.5

    def test_low_fake_score_maps_to_low_p_synthetic(
        self, clip: AudioClip, cfg: SpoofConfig
    ) -> None:
        from vishguard.antiSpoof import detectSpoof

        with patch("vishguard.antiSpoof._load_pipeline", return_value=_mock_pipe(0.05)):
            result = detectSpoof(clip, cfg)
        assert result.pSynthetic < 0.5
