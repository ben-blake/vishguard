"""Generate a synthetic vishing call WAV for demo purposes.

Uses SpeechT5 (same model as the pipeline TTS stage) so the anti-spoof
detector has a good chance of flagging the output as synthetic.

Run on Colab T4 for speed (~30 s) or locally on CPU (~3-5 min):

    python scripts/makeDemoAudio.py

Output: artifacts/audio/demo_call.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Vishing script — rich tactic coverage for the demo:
#   authority, urgency, fear_intimidation, financial_manipulation,
#   credential_harvesting, pretexting
# SpeechT5 processor handles ~600 chars max; keep each segment under that.
# ---------------------------------------------------------------------------

SEGMENTS = [
    (
        "Hello. This is officer Daniel Reed from the Social Security "
        "Administration fraud division. We have detected suspicious activity "
        "linked to your social security number, which has now been suspended."
    ),
    (
        "You are facing a penalty of twenty four hundred dollars due to "
        "fraudulent use of your account. To avoid immediate arrest and a "
        "federal warrant being issued in your name, you must act now."
    ),
    (
        "Call our resolution hotline and provide your social security number "
        "and bank account details to verify your identity. "
        "You have twenty four hours before legal proceedings begin. "
        "This is your final warning."
    ),
]

OUT_PATH = ROOT / "artifacts" / "audio" / "demo_call.wav"
SPEAKER_IDX = 7306
SAMPLE_RATE = 16_000

MODEL_ID   = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
EMBED_ID   = "Matthijs/cmu-arctic-xvectors"


def main() -> None:
    import torch
    import soundfile as sf
    import numpy as np
    from huggingface_hub import snapshot_download
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading SpeechT5 models…")
    processor = SpeechT5Processor.from_pretrained(MODEL_ID)
    model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID).to(device)

    print("Loading speaker embedding…")
    import zipfile, tempfile
    snap = snapshot_download(repo_id=EMBED_ID, repo_type="dataset")
    zip_path = next(Path(snap).glob("*.zip"))
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        import csv
        rows = []
        for arrow in Path(tmp).rglob("*.arrow"):
            import pyarrow as pa
            tbl = pa.ipc.open_file(arrow).read_all()
            rows.extend(tbl.to_pydict()["xvector"])
        xvector = rows[SPEAKER_IDX]
    speaker_emb = torch.tensor(xvector).unsqueeze(0).float().to(device)

    print("Synthesising segments…")
    audio_parts: list[np.ndarray] = []
    silence = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)

    for i, text in enumerate(SEGMENTS, 1):
        print(f"  Segment {i}/{len(SEGMENTS)}: {text[:60]}…")
        inputs = processor(text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            speech: torch.Tensor = model.generate_speech(
                inputs["input_ids"], speaker_emb, vocoder=vocoder
            )
        audio_parts.append(speech.cpu().numpy())
        audio_parts.append(silence)

    combined = np.concatenate(audio_parts)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_PATH), combined, samplerate=SAMPLE_RATE)
    duration = len(combined) / SAMPLE_RATE
    print(f"\nSaved: {OUT_PATH}  ({duration:.1f}s)")


if __name__ == "__main__":
    main()
