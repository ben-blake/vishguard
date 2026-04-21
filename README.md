# VishGuard

Vishing-risk audio pipeline for CS 5542 Quiz Challenge 2 (UMKC, Spring 2026).
Analyzes a phone-call recording and returns a structured risk report:
transcript, synthetic-voice likelihood, social-engineering tactic labels,
aggregate risk score with reasoning, and a spoken briefing via SpeechT5.

See [`docs/`](./docs/) for PRD, architecture, and task plan.
Full reproduction instructions (eval scripts, sample outputs) are added
during Phase 4 — see [`docs/TASKS.md`](./docs/TASKS.md) T4.2.

## Local development (macOS, CPU-only)

Uses Python 3.12 to match the Colab 2026-04 runtime.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
pytest tests/
```

`bitsandbytes` is **not** installed locally — it has no macOS build.
The 4-bit LLM path (`tacticClassifier.py`, `riskSynthesis.py`) therefore
only runs on Colab. Whisper, librosa, SpeechT5, and the Streamlit UI all
work on CPU locally using `whisper-tiny` for demo speed.

## CLI usage

```bash
# Analyze a call recording (writes report.json + briefing.wav to out/)
vishguard run call.wav --out out/

# CPU-only (no GPU / no bitsandbytes) — uses whisper-tiny for speed
vishguard run call.wav --out out/ --device cpu --whisper openai/whisper-tiny --no-tts

# Prompt v1 vs v2 (for Phase 3 A/B eval)
vishguard run call.wav --out out/ --prompt v1
```

`--no-tts` skips the SpeechT5 briefing (required on macOS where the 4-bit
LLM path is also unavailable; use `--device cpu` together).

## Colab

Open [`notebooks/00_colabSetup.ipynb`](./notebooks/00_colabSetup.ipynb) in
a fresh Colab runtime and run the setup cell. Each notebook in this
project is self-contained — Colab runtimes are isolated, so no notebook
assumes another has run first.

The setup cell:

1. Mounts Google Drive (optional; used only for persisted outputs).
2. Clones or pulls this repo into `/content/vishguard`.
3. Runs `pip install -e .` + `pip install -r requirements.txt`.
4. Runs `pip install -r requirements-gpu.txt` for `bitsandbytes`.
5. Prints `vishguard.__version__` as a smoke test.
