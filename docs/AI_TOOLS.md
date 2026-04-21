# VishGuard — AI Tools Disclosure

Required for full credit under the CS 5542 Quiz Challenge 2 rubric's
transparency clause. One section per phase from [`TASKS.md`](./TASKS.md).
**Fill in at the end of every working session — do not reconstruct on
Day 7.**

Columns:

- **Tool** — product name + version if relevant
- **How used** — 1–2 sentences on the specific task
- **Origin** — `AI-generated`, `AI-assisted`, or `human-authored`
- **Where** — file path, commit SHA, or doc section

Keep an entry even when AI was *not* used ("no AI tools used this session")
so reviewers can confirm the log is real and not after-the-fact.

---

## Phase 0 — Planning (2026-04-21)

| Tool               | How used                                                                    | Origin         | Where                    |
|--------------------|-----------------------------------------------------------------------------|----------------|--------------------------|
| Claude Code (Opus) | Drafted PRD, ARCHITECTURE, TASKS, AI_TOOLS scaffold, and CLAUDE.md.         | AI-generated   | `docs/*.md`, `CLAUDE.md` |
| Ben (me)           | Wrote prompt, validated HF model IDs, reviewed scope, approved phase order. | Human-authored | prompt + review          |

Claude Code's draft covered the full planning doc set from a single prompt
that specified constraints, rubric weights, and the desired pipeline shape.
Ben's review added the tactic-taxonomy literature grounding and the
risk-first phase ordering.

---

## Phase 1 — Risk-down spikes + scaffolding (2026-04-21)

| Tool | How used | Origin | Where |
| --- | --- | --- | --- |
| Claude Code (Opus) | Authored repo scaffolding (T1.4): `pyproject.toml`, `requirements*.txt`, stubbed modules, frozen dataclasses in `types.py`, empty test files, eval stubs, `configs/default.yaml`. | AI-generated | `src/vishguard/*`, `tests/*`, `eval/*`, `configs/` |
| Claude Code (Opus) | Authored Colab setup notebook (T1.5) and 3 spike notebooks (T1.1 anti-spoof, T1.2 Qwen JSON, T1.3 Whisper WER). | AI-generated | `notebooks/00–03_*.ipynb` |
| Claude Code (Opus) | Aligned `requirements.txt` floors with Colab 2026-04 runtime and split `requirements-gpu.txt` for macOS portability. | AI-assisted | `requirements.txt`, `requirements-gpu.txt` |
| Claude Code (Sonnet) | Debugged T1.1 through 5 Colab errors (datasets v4, ZIP extraction, array shape, all-zero scores, wrong model ID). Confirmed `mo-thecreator` as working primary. | AI-assisted | `notebooks/01_spike_antiSpoof.ipynb` |
| Claude Code (Sonnet) | Ran T1.2, recorded per-script label analysis, implemented `tacticPromptV1` and `tacticPromptV2` with disambiguation notes and targeted few-shot. | AI-assisted | `notebooks/02_spike_llmJson.ipynb`, `src/vishguard/promptLibrary.py` |
| Claude Code (Sonnet) | Fixed T1.3 `KeyError: num_frames` (transformers 5.x pipeline regression); switched to direct `WhisperProcessor + WhisperForConditionalGeneration`; added punctuation normalization. | AI-assisted | `notebooks/03_spike_whisperWer.ipynb` |
| Claude Code (Sonnet) | Authored spike result JSON reports, created `artifacts/` folder, applied code-review fixes (SpoofConfig model ID, CLI entry point in pyproject.toml). | AI-generated | `artifacts/reports/*.json`, `src/vishguard/types.py`, `pyproject.toml` |
| Ben (me) | Supplied Colab pip freeze, identified correct model name `mo-thecreator/Deepfake-audio-detection`, ran all three spike notebooks on Colab T4, supplied all cell outputs, confirmed Python 3.12. | Human-authored | Colab runs + review |

Notes:

- Python 3.12 locally and on Colab 2026-04 runtime. `bitsandbytes` excluded from `requirements.txt` (no macOS build); Colab setup cell installs `requirements-gpu.txt` when CUDA is available.
- All Phase 1 spikes completed 2026-04-21. Results: T1.1 real<0.5=10/10 synth≥0.5=7/10; T1.2 valid_json=10/10; T1.3 clean_norm_WER=0.013 noisy_norm_WER=0.032.
- `MelodyMachine/Deepfake-audio-detection-V2` failed T1.1 — always predicts real. `mo-thecreator/Deepfake-audio-detection` is the confirmed working primary.
- transformers 5.x `pipeline('automatic-speech-recognition')` raises `KeyError: num_frames` with dict input on Colab; use `WhisperProcessor` + model directly instead.

---

## Phase 2 — Core pipeline (2026-04-21)

| Tool | How used | Origin | Where |
| --- | --- | --- | --- |
| Claude Code (Sonnet) | Authored TDD test suites (68 tests across 6 files) for all T2.1–T2.7 modules before implementing them; RED→GREEN cycle confirmed. | AI-generated | `tests/testIngestion.py`, `tests/testAsr.py`, `tests/testAntiSpoof.py`, `tests/testTacticClassifier.py`, `tests/testRiskSynthesis.py`, `tests/testReportSchema.py` |
| Claude Code (Sonnet) | Implemented `loadAudio.ingest` (librosa), `asrWhisper.transcribe` (WhisperProcessor direct), `antiSpoof.detectSpoof` (HF pipeline), `tacticClassifier.classifyTactics` (Qwen LLM + JSON retry), `riskSynthesis.synthesizeRisk` (weighted rule + LLM reasoning), `briefingTts.narrateBriefing` (SpeechT5), `reportSchema` (pydantic v2 round-trip), `orchestrator.runPipeline`, `cli.main`. | AI-generated | `src/vishguard/*.py` |
| Claude Code (Sonnet) | Implemented `promptLibrary.riskReasoningPrompt` (previously stubbed NotImplementedError). | AI-generated | `src/vishguard/promptLibrary.py` |
| Claude Code (Sonnet) | Fixed `configs/default.yaml` anti-spoof model ID from `MelodyMachine/Deepfake-audio-detection-V2` (always-real failure) to confirmed-working `mo-thecreator/Deepfake-audio-detection`. | AI-assisted | `configs/default.yaml` |
| Ben (me) | Reviewed all implementations for correctness, directed TDD workflow, approved scoring formula constants, confirmed 68/68 green. | Human-authored | code review |

---

## Phase 3 — Evaluation harness

| Tool                           | How used | Origin | Where |
|--------------------------------|----------|--------|-------|
| *TBD - fill at end of session* |          |        |       |

Special note for Phase 3: the **tactic-classification eval corpus** is
synthetically generated by a frontier LLM and then hand-verified. That
generation must be disclosed here with (a) model + version, (b) prompt
used, (c) number of examples generated vs. retained after hand review.

---

## Phase 4 — UI + polish

| Tool                           | How used | Origin | Where |
|--------------------------------|----------|--------|-------|
| *TBD - fill at end of session* |          |        |       |

---

## Phase 5 — Deliverables

| Tool                           | How used | Origin | Where |
|--------------------------------|----------|--------|-------|
| *TBD - fill at end of session* |          |        |       |

Expected entries this phase:

- Slide deck: Gamma / Canva / Google Slides — note any AI-generated
  slide copy or imagery.
- Demo video: QuickTime / OBS recording; note any AI-generated narration
  (if any). Original voice preferred for authenticity on a vishing-safety
  project.
- Final README polish may use Claude Code — disclose.

---

## Honest-log ground rules

1. If a session produces zero code/doc changes, log it as "no code
   changes; AI used only for brainstorming" rather than skipping.
2. Do not retroactively upgrade `AI-assisted` to `human-authored` after
   the fact to game the rubric. Graders can cross-check against commit
   diffs.
3. When Claude Code writes code and Ben reviews and edits, the origin is
   `AI-assisted` (not `AI-generated`). Reserve `AI-generated` for
   content shipped without meaningful human edits.
