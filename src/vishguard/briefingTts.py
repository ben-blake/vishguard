"""SpeechT5 TTS briefing stage — RiskReport -> WAV path."""
from __future__ import annotations

from pathlib import Path

import torch

from .types import RiskReport, TtsConfig

_SAMPLE_RATE = 16_000
_SPEAKER_IDX = 7306  # deterministic speaker from cmu-arctic-xvectors


def _buildBriefingText(report: RiskReport) -> str:
    tactic_labels = [t.label for t in report.tactics if t.label != "benign"]
    tactics_str = ", ".join(tactic_labels) if tactic_labels else "none detected"
    return (
        f"VishGuard alert. Risk score {report.risk.score} out of 100, "
        f"level {report.risk.band}. "
        f"Tactics identified: {tactics_str}. "
        f"{report.risk.reasoning}"
    )


def narrateBriefing(report: RiskReport, cfg: TtsConfig, outDir: Path) -> Path:
    import soundfile as sf
    from datasets import load_dataset
    from transformers import (
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5Processor,
    )

    text = _buildBriefingText(report)

    # Separate from_pretrained and .to() so Pylance can resolve the instance type.
    processor = SpeechT5Processor.from_pretrained(cfg.modelId)
    model = SpeechT5ForTextToSpeech.from_pretrained(cfg.modelId)
    model = model.to(cfg.device)  # type: ignore[arg-type]
    vocoder = SpeechT5HifiGan.from_pretrained(cfg.vocoderId)
    vocoder = vocoder.to(cfg.device)  # type: ignore[arg-type]

    embeddings_ds = load_dataset(cfg.speakerEmbeddingId, split="validation")
    # .float() converts Tensor -> FloatTensor to match generate_speech signature.
    xvector = embeddings_ds[_SPEAKER_IDX]["xvector"]  # type: ignore[index]
    speaker_embedding = torch.tensor(xvector).unsqueeze(0).float().to(cfg.device)

    # Processor __call__ is typed Optional in some stubs; split to allow .to().
    encoding = processor(text=text, return_tensors="pt")
    inputs = encoding.to(cfg.device)  # type: ignore[union-attr]

    with torch.no_grad():
        # generate_speech returns FloatTensor; stubs may narrow to tuple when
        # return_output_lengths is not set — annotate explicitly.
        speech: torch.Tensor = model.generate_speech(  # type: ignore[assignment]
            inputs["input_ids"], speaker_embedding, vocoder=vocoder  # type: ignore[arg-type]
        )

    outDir = Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    out_path = outDir / "briefing.wav"
    sf.write(str(out_path), speech.cpu().numpy(), samplerate=_SAMPLE_RATE)
    return out_path
