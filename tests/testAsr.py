"""T2.2: Tests for asrWhisper.transcribe — model loading is mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vishguard.types import AsrConfig, AudioClip, Transcript


@pytest.fixture
def clip() -> AudioClip:
    return AudioClip(
        samples=np.zeros(16_000, dtype=np.float32),
        sampleRate=16_000,
        sourcePath=Path("test.wav"),
        durationSec=1.0,
    )


@pytest.fixture
def cfg() -> AsrConfig:
    return AsrConfig(modelId="openai/whisper-small", device="cpu")


def _make_mocks(decoded_text: str = "Hello world."):
    processor = MagicMock()
    proc_output = MagicMock()
    proc_output.input_features.to.return_value = MagicMock()
    processor.return_value = proc_output
    processor.batch_decode.return_value = [decoded_text]

    model = MagicMock()
    model.generate.return_value = [[1, 2, 3]]
    return processor, model


class TestTranscribe:
    def test_returns_transcript(self, clip: AudioClip, cfg: AsrConfig) -> None:
        from vishguard.asrWhisper import transcribe

        with patch(
            "vishguard.asrWhisper._load_processor_and_model",
            return_value=_make_mocks(),
        ):
            result = transcribe(clip, cfg)
        assert isinstance(result, Transcript)

    def test_full_text_is_stripped(self, clip: AudioClip, cfg: AsrConfig) -> None:
        from vishguard.asrWhisper import transcribe

        p, m = _make_mocks("  IRS calling.  ")
        with patch("vishguard.asrWhisper._load_processor_and_model", return_value=(p, m)):
            result = transcribe(clip, cfg)
        assert result.fullText == "IRS calling."

    def test_model_id_in_transcript(self, clip: AudioClip, cfg: AsrConfig) -> None:
        from vishguard.asrWhisper import transcribe

        with patch(
            "vishguard.asrWhisper._load_processor_and_model",
            return_value=_make_mocks(),
        ):
            result = transcribe(clip, cfg)
        assert result.modelId == cfg.modelId

    def test_segments_is_tuple(self, clip: AudioClip, cfg: AsrConfig) -> None:
        from vishguard.asrWhisper import transcribe

        with patch(
            "vishguard.asrWhisper._load_processor_and_model",
            return_value=_make_mocks(),
        ):
            result = transcribe(clip, cfg)
        assert isinstance(result.segments, tuple)
        assert len(result.segments) >= 1

    def test_segment_end_matches_clip_duration(self, clip: AudioClip, cfg: AsrConfig) -> None:
        from vishguard.asrWhisper import transcribe

        with patch(
            "vishguard.asrWhisper._load_processor_and_model",
            return_value=_make_mocks(),
        ):
            result = transcribe(clip, cfg)
        assert result.segments[-1].endSec == clip.durationSec

    def test_language_code_set(self, clip: AudioClip, cfg: AsrConfig) -> None:
        from vishguard.asrWhisper import transcribe

        with patch(
            "vishguard.asrWhisper._load_processor_and_model",
            return_value=_make_mocks(),
        ):
            result = transcribe(clip, cfg)
        assert isinstance(result.languageCode, str)
        assert len(result.languageCode) > 0
