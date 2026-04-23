"""Whisper ASR stage — clip -> Transcript.

Uses WhisperProcessor + WhisperForConditionalGeneration directly to avoid
the transformers 5.x pipeline KeyError on 'num_frames' (see T1.3 notes in
CLAUDE.md).
"""
from __future__ import annotations

import torch

from .types import AsrConfig, AudioClip, Transcript, TranscriptSegment

_MODEL_CACHE: dict = {}  # model_id -> (processor, model)


def _load_processor_and_model(cfg: AsrConfig):
    if cfg.modelId not in _MODEL_CACHE:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(cfg.modelId)
        model = WhisperForConditionalGeneration.from_pretrained(cfg.modelId)
        model.to(cfg.device)  # type: ignore[arg-type]
        model.eval()
        _MODEL_CACHE[cfg.modelId] = (processor, model)
    return _MODEL_CACHE[cfg.modelId]


def transcribe(clip: AudioClip, cfg: AsrConfig) -> Transcript:
    processor, model = _load_processor_and_model(cfg)

    # WhisperProcessor silently truncates input to 30 s (480 000 samples).
    # Split longer audio into non-overlapping 30 s chunks and concatenate.
    chunk_samples = 30 * clip.sampleRate
    samples = clip.samples
    texts: list[str] = []
    segments: list[TranscriptSegment] = []
    offset = 0

    while offset < len(samples):
        chunk = samples[offset : offset + chunk_samples]
        inputs = processor(chunk, sampling_rate=clip.sampleRate, return_tensors="pt")
        input_features = inputs.input_features.to(cfg.device)

        with torch.no_grad():
            predicted_ids = model.generate(input_features)

        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        start_sec = offset / clip.sampleRate
        end_sec = min((offset + chunk_samples) / clip.sampleRate, clip.durationSec)

        if chunk_text:
            texts.append(chunk_text)
            segments.append(TranscriptSegment(startSec=start_sec, endSec=end_sec, text=chunk_text))

        offset += chunk_samples

    full_text = " ".join(texts)
    if not segments:
        segments = [TranscriptSegment(startSec=0.0, endSec=clip.durationSec, text="")]

    return Transcript(
        fullText=full_text,
        segments=tuple(segments),
        languageCode="en",
        modelId=cfg.modelId,
    )
