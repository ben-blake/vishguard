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

## Phase 3 — Evaluation harness (2026-04-21)

| Tool | How used | Origin | Where |
| --- | --- | --- | --- |
| Claude Code (Sonnet) | Authored TDD tests (49 tests across 6 files) for all T3.1–T3.6 eval helper functions before implementing them; RED→GREEN cycle confirmed (117/117 total). | AI-generated | `tests/testEvalBuild*.py`, `tests/testEvalRun*.py` |
| Claude Code (Sonnet) | Implemented `buildSpoofSet.py` (LibriSpeech stream + SpeechT5 generation, manifest CSV builder and validator). | AI-generated | `eval/buildSpoofSet.py` |
| Claude Code (Sonnet) | Implemented `runSpoofEval.py` (accuracy, precision, recall, F1, EER, confusion matrix PNG via sklearn). | AI-generated | `eval/runSpoofEval.py` |
| Claude Code (Sonnet) | Generated all 50 tactic-classification corpus scripts in `eval/data/tactics.jsonl` — 5 per tactic covering all 10 taxonomy labels with multi-label overlap. | AI-generated | `eval/data/tactics.jsonl` |
| Claude Code (Sonnet) | Implemented `buildTacticSet.py` (corpus loader, per-entry and corpus-level validators, label coverage report). | AI-generated | `eval/buildTacticSet.py` |
| Claude Code (Sonnet) | Implemented `runTacticEval.py` (per-label binary F1, macro-F1 across prompt v1/v2/v3/v4, CSV output). | AI-generated | `eval/runTacticEval.py` |
| Claude Code (Sonnet) | Implemented `runAsrEval.py` (text normalization, Gaussian noise injection at target SNR dB, WER via jiwer, grouped bar chart). | AI-generated | `eval/runAsrEval.py` |
| Claude Code (Sonnet) | Implemented `runLatencyBench.py` (synthetic WAV generation, per-stage timing aggregation, CSV output). | AI-generated | `eval/runLatencyBench.py` |
| Claude Code (Sonnet) | Authored `notebooks/04_phase3_eval.ipynb` — Colab harness running T3.1–T3.6 in order, with setup cell, results cells, and zip-download cell. | AI-generated | `notebooks/04_phase3_eval.ipynb` |
| Claude Code (Sonnet) | Debugged 5 Colab runtime errors: `trust_remote_code` removed in datasets v4; `cmu-arctic-xvectors` loading script deprecated (switched to `snapshot_download`+ZIP); `loadAudio` import name wrong; `eval/data/` blocked by `.gitignore`; raw string passed to `classifyTactics` instead of `Transcript`. | AI-assisted | `eval/buildSpoofSet.py`, `eval/runSpoofEval.py`, `eval/runTacticEval.py`, `.gitignore` |
| Claude Code (Sonnet) | Fixed `tacticClassifier._extract_json` to use bracket-counting loop instead of greedy `\[.*\]` regex (over-matched closing bracket); added single-quote normalization fallback. Fixed `_parse_tactics` to handle flat `list[str]` format from model in addition to list-of-dicts. | AI-assisted | `src/vishguard/tacticClassifier.py` |
| Claude Code (Sonnet) | Iterated prompt v2→v3→v4 based on live Colab T4 eval results. v3: fixed `impersonation` definition (from "grandchild, colleague" to "named person, company, or agency"), added positive `pretexting` definition. v4: added examples 6–9 targeting co-occurrence regressions in `credential_harvesting` and `financial_manipulation`. | AI-assisted | `src/vishguard/promptLibrary.py` |
| Claude Code (Sonnet) | Organized Colab T3.1–T3.6 outputs into `artifacts/reports/` and `artifacts/plots/`; committed 7 artifact files (2 PNGs, 5 CSVs). | AI-assisted | `artifacts/reports/`, `artifacts/plots/` |
| Ben (me) | Directed TDD and eval workflow. Ran all T3.1–T3.6 cells on Colab T4, supplied all cell outputs and error traces, reviewed eval numbers, confirmed 117/117 green. Hand-verified all 50 tactic-corpus scripts; directed prompt iteration strategy. | Human-authored | Colab runs + review |

**Tactic corpus disclosure (T3.3):**

- **Model:** Claude claude-sonnet-4-6 (Anthropic, 2026-04) via Claude Code interactive session.
- **Prompt approach:** Claude Code generated all 50 scripts inline during the Phase 3 session, targeting balanced coverage of the 10-label taxonomy with realistic multi-label overlap (e.g., authority + fear_intimidation, impersonation + credential_harvesting).
- **Generated:** 50 scripts.
- **Hand-verified:** All 50 retained after review — labels match the intended tactics and the scripts are realistic enough for a classifier evaluation. No scripts were dropped.
- **Labeler:** Ben Blake (sole annotator).

---

## Phase 4 — UI + polish (2026-04-23)

| Tool | How used | Origin | Where |
| --- | --- | --- | --- |
| Claude Code (Sonnet) | Authored TDD tests (21 tests) for `pageReport.py` pure helper functions (`bandColor`, `spoofLabel`, `formatPct`, `tacticRows`, `timingsTable`) before implementing them; RED→GREEN cycle confirmed (138/138 total). | AI-generated | `tests/testUiPageReport.py` |
| Claude Code (Sonnet) | Implemented `pageReport.py` helper functions and `renderReport` (Streamlit columns, metrics, expanders, audio widget, timings table). | AI-generated | `src/vishguard/ui/pageReport.py` |
| Claude Code (Sonnet) | Implemented `app.py` Streamlit entry point: file uploader, sidebar config (ASR model, device, prompt variant, TTS toggle), spinner, `runPipeline` call, error display. | AI-generated | `app.py` |
| Claude Code (Sonnet) | Wrote full README (setup, Streamlit UI, CLI, model card table, Phase 3 eval results summary, eval reproduction steps, sample output, project structure, AI-tools link). | AI-generated | `README.md` |
| Claude Code (Sonnet) | Wrote `docs/FAILURES.md` — 3 failure cases from Phase 3: SpeechT5 anti-spoof recall=0.66, fear_intimidation/impersonation F1 weakness, Whisper hallucination on short LibriSpeech clips. | AI-generated | `docs/FAILURES.md` |
| Claude Code (Sonnet) | Ran `code-reviewer` agent on `src/vishguard/`; found 1 CRITICAL (`antiSpoof` unhandled `StopIteration`) and 1 HIGH (`tacticClassifier` unhandled `ValueError` on non-numeric confidence); applied both fixes. | AI-assisted | `src/vishguard/antiSpoof.py`, `src/vishguard/tacticClassifier.py` |
| Ben (me) | Directed TDD workflow, reviewed all implementations, confirmed 138/138 green. Reviewed README and FAILURES.md content. | Human-authored | code review |

---

## Phase 5 — Deliverables (2026-04-27)

| Tool | How used | Origin | Where |
| --- | --- | --- | --- |
| Claude Code (Sonnet) | Authored `notebooks/05_colabDemo.ipynb` — Colab T4 demo notebook: Cell 1 (repo setup), Cell 1b (SpeechT5 WAV generation + anti-spoof sanity check), Cell 2 (ngrok auth), Cell 2b (CLI pre-warm), Cell 3 (Streamlit launch), Cell 3b (GPU/log monitor), Cell 4 (shutdown). | AI-generated | `notebooks/05_colabDemo.ipynb` |
| Claude Code (Sonnet) | Debugged 5 Colab errors in demo notebook: `repo_type='dataset'` missing from `snapshot_download` (RepositoryNotFoundError); ZIP contains `.npy` not `.arrow` (IndexError); CLI positional arg order wrong (invalid choice); `--prompt` choices only v1/v2 (extended to v4); Whisper truncating at 30 s (chunking fix in `asrWhisper.py`). | AI-assisted | `notebooks/05_colabDemo.ipynb`, `src/vishguard/asrWhisper.py`, `src/vishguard/cli.py` |
| Claude Code (Sonnet) | Diagnosed p_synth=0.00 on demo audio (long multi-clause segments fool the model); rewrote Cell 1b with 6 short declarative sentences matching Phase 1 spike style (0.988–1.000 scores); added inline anti-spoof PASS/FAIL check before download. | AI-assisted | `notebooks/05_colabDemo.ipynb` |
| Claude Code (Sonnet) | Added `@st.cache_resource` pre-warm to `app.py` — loads all model weights into GPU on first page load so first analysis click takes ~30–60 s instead of 5+ min on Colab T4. | AI-generated | `app.py` |
| Claude Code (Sonnet) | Created `scripts/makeDemoAudio.py` — standalone local script to regenerate demo WAV outside of notebook context. | AI-generated | `scripts/makeDemoAudio.py` |
| Claude Code (Sonnet) | Fixed `app.py` `_prewarm` function: `Transcript` constructor called with nonexistent `durationSec` field and missing `segments`/`languageCode` fields. Added `TranscriptSegment` import and rebuilt dummy Transcript with all four required fields. | AI-assisted | `app.py` (commit 86b3ef3) |
| Claude Code (Sonnet) | Fixed `briefingTts.narrateBriefing`: replaced deprecated `load_dataset(cfg.speakerEmbeddingId)` (dataset scripts no longer supported) with `snapshot_download(repo_type="dataset")` + ZIP extraction pattern; added `.npy`/`.pt` fallback chain. | AI-assisted | `src/vishguard/briefingTts.py` (commit 984ca4c) |
| Claude Code (Sonnet) | Added demo video link (vimeo.com/1187092533) to README; added `artifacts/screenshots/streamlit.png` and `artifacts/screenshots/architecture.png` inline; fixed fenced code block language tag. | AI-assisted | `README.md` (commits 149caf6, 6349f01, 10fd50e) |
| Claude Code (Sonnet) | Generated `docs/vishguard_slides.pptx` (12 slides) via `scripts/makeSlides.py` (python-pptx): title, problem, business value, datasets, model card, pipeline architecture, prompt engineering, results, eval analysis, demo links, limitations, AI-tools disclosure. | AI-generated | `docs/vishguard_slides.pptx`, `scripts/makeSlides.py` |
| Ben (me) | Ran Cell 1b on Colab T4; verified new short-segment WAV scores p(synthetic) ≥ 0.5 on anti-spoof model; committed resulting `artifacts/audio/demo_call.wav`. Narrated and recorded demo video via QuickTime screen capture. Reviewed and approved all Phase 5 fixes and slide content. | Human-authored | Colab runs + video recording + review |

Notes:

- Demo audio (SpeechT5, 6 short declarative sentences) confirmed flagged as synthetic on Colab T4 after second generation pass.
- Streamlit pre-warm (`@st.cache_resource`) populates the Streamlit server's in-memory model cache; CLI pre-warm only warms the HF disk cache — both are needed for sub-60 s analysis on Colab.
- `briefingTts.py` dataset fix mirrors the same `snapshot_download` pattern used in `eval/buildSpoofSet.py` and the Colab notebook — consistent across all speaker-embedding loading paths.

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
