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

    inputs = processor(
        clip.samples,
        sampling_rate=clip.sampleRate,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(cfg.device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    segment = TranscriptSegment(startSec=0.0, endSec=clip.durationSec, text=text)
    return Transcript(
        fullText=text,
        segments=(segment,),
        languageCode="en",
        modelId=cfg.modelId,
    )
