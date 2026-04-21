"""T2.1: Tests for loadAudio.ingest — real WAV loading, no model mocks needed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vishguard.loadAudio import ingest
from vishguard.types import AudioClip

_TARGET_SR = 16_000


@pytest.fixture
def stereo_44k_wav(tmp_path: Path) -> Path:
    sr = 44_100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, retstep=False, dtype=np.float32)
    audio = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], axis=1)
    path = tmp_path / "stereo_44k.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def mono_16k_wav(tmp_path: Path) -> Path:
    sr = 16_000
    duration = 1.0
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    path = tmp_path / "mono_16k.wav"
    sf.write(str(path), samples, sr)
    return path


class TestIngest:
    def test_returns_audio_clip(self, mono_16k_wav: Path) -> None:
        clip = ingest(mono_16k_wav)
        assert isinstance(clip, AudioClip)

    def test_sample_rate_normalized_to_16k(self, stereo_44k_wav: Path) -> None:
        clip = ingest(stereo_44k_wav)
        assert clip.sampleRate == _TARGET_SR

    def test_audio_is_mono(self, stereo_44k_wav: Path) -> None:
        clip = ingest(stereo_44k_wav)
        assert clip.samples.ndim == 1

    def test_dtype_is_float32(self, stereo_44k_wav: Path) -> None:
        clip = ingest(stereo_44k_wav)
        assert clip.samples.dtype == np.float32

    def test_duration_close_to_source(self, stereo_44k_wav: Path) -> None:
        clip = ingest(stereo_44k_wav)
        assert abs(clip.durationSec - 2.0) < 0.1

    def test_source_path_stored(self, mono_16k_wav: Path) -> None:
        clip = ingest(mono_16k_wav)
        assert clip.sourcePath == mono_16k_wav

    def test_sample_count_matches_duration(self, mono_16k_wav: Path) -> None:
        clip = ingest(mono_16k_wav)
        expected = int(_TARGET_SR * clip.durationSec)
        assert abs(len(clip.samples) - expected) <= 1
