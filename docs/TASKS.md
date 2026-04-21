# VishGuard — Task Plan

5 phases across ~6 working days (2026-04-21 → 2026-04-27). Phase 1 front-loads
the three highest-risk unknowns; Phase 5 is reserved for deliverables
(slides + demo video) and cannot be compressed without losing the 10%
presentation rubric weight.

**Legend — status:** NOT_STARTED, IN_PROGRESS, DONE, BLOCKED, DROPPED.
**Legend — phase tags:** `[P1]` … `[P5]`.

---

## Phase 1 — Risk-down spikes + scaffolding (Day 1–2)

Goal: prove the three riskiest pieces work on representative input before
committing to the full build. If a spike fails, the plan adapts here
instead of on Day 6.

### T1.1 `[P1]` Anti-spoof model smoke test — **STATUS: DONE**

- ~~Load `MelodyMachine/Deepfake-audio-detection-V2`~~ — FAILED smoke test; always predicts real.
- **Working model: `mo-thecreator/Deepfake-audio-detection`** (promoted to primary in ARCHITECTURE.md §2.2).
- Ran on 10 LibriSpeech real clips + 10 SpeechT5 synthetic clips on Colab T4.
- **Results (2026-04-21):** real p(synth) all=0.0 (10/10 correct), synth p(synth) 7/10 ≥ 0.5.
- Separation visible on histogram. Acceptance criteria met.
- See `notebooks/01_spike_antiSpoof.ipynb`.

### T1.2 `[P1]` Open-LLM structured-output spike — **STATUS: DONE**

- Qwen/Qwen2.5-3B-Instruct loaded on Colab T4 in fp16 (`device: cuda:0`).
- **Results (2026-04-21):** valid_json=10/10, labels_all_in_taxonomy=10/10. Acceptance bar passed.
- Semantic patterns for Phase 3 prompt v2: `pretexting` overused as catch-all;
  `credential_harvesting` and `fear_intimidation` systematically underdetected;
  `benign` spuriously added to scam calls.
- See `notebooks/02_spike_llmJson.ipynb`.

### T1.3 `[P1]` Whisper WER floor check — **STATUS: DONE**

- Transcribed 5 LibriSpeech clean clips + 5 pink-noise copies (SNR 10 dB) via whisper-small.
- **Results (2026-04-21):** clean normalized WER=0.013 (raw=0.127), noisy normalized WER=0.032 (raw=0.146).
  Raw score inflated by Whisper punctuation insertions vs punct-free LibriSpeech refs.
  Both normalized scores well under acceptance bars. whisper-medium not needed.
- Worst-case: 'opened before them' → 'opened for them' (clean); 'concord' → 'concorde' (noisy).
- Use normalized WER (strip punctuation) in all Phase 3 eval scripts.
- See `notebooks/03_spike_whisperWer.ipynb`.

### T1.4 `[P1]` Repo scaffolding — **STATUS: DONE**

- All modules stubbed with `NotImplementedError`, `types.py` frozen dataclasses,
  `requirements.txt` / `requirements-gpu.txt`, `pyproject.toml` src-layout,
  `.gitignore`, `configs/default.yaml`, `artifacts/` folder.
- `python -c "import vishguard"` passes; `pytest tests/` collects 0 tests cleanly.
- Code review (2026-04-21): fixed `SpoofConfig` default model ID, added CLI entry point.

### T1.5 `[P1]` Colab setup notebook — **STATUS: DONE**

- `notebooks/00_colabSetup.ipynb` self-contained setup cell verified working on Colab T4.
- All three spike notebooks (01/02/03) share the same setup cell pattern and ran successfully.

---

## Phase 2 — Core pipeline (Day 2–4)

Goal: every stage in `ARCHITECTURE.md` is implemented as a pure function
and the orchestrator can run end-to-end on one seed audio file.

### T2.1 `[P2]` `loadAudio.py` — **STATUS: DONE**

- Decode WAV/MP3/M4A via `librosa.load`, resample to 16 kHz mono float32,
  return `AudioClip`.
- 7 unit tests with real in-memory WAV fixtures (stereo 44.1 kHz → mono 16 kHz,
  duration, dtype, sample count).
- **Acceptance:** `pytest tests/testIngestion.py` passes (7/7 2026-04-21).

### T2.2 `[P2]` `asrWhisper.py` — **STATUS: DONE**

- WhisperProcessor + WhisperForConditionalGeneration directly (not pipeline —
  T1.3 note). Single-segment output for now; `languageCode` defaults to "en".
- 6 unit tests with mocked `_load_processor_and_model`; verify strip, modelId,
  segments tuple, endSec == clip duration.
- **Acceptance:** `pytest tests/testAsr.py` passes (6/6 2026-04-21).

### T2.3 `[P2]` `antiSpoof.py` — **STATUS: DONE**

- `pipeline("audio-classification", model="mo-thecreator/Deepfake-audio-detection")`.
- Template rationale (3 bands: low / moderate / high).
- 6 unit tests with mocked pipeline; verify SpoofVerdict type, pSynthetic range,
  model ID, rationale, directional score.
- **Acceptance:** `pytest tests/testAntiSpoof.py` passes (6/6 2026-04-21).

### T2.4 `[P2]` `promptLibrary.py` + `tacticClassifier.py` — **STATUS: DONE**

- `promptLibrary.py`: `tacticPromptV1`, `tacticPromptV2`, `riskReasoningPrompt` all implemented.
- `tacticClassifier.py`: `_call_llm` with module-level cache, `_extract_json`,
  `_parse_tactics` (taxonomy filter), retry on parse failure, v1/v2 dispatch.
- 10 unit tests: JSON extraction, taxonomy filter, retry trigger, prompt variant routing.
- **Acceptance:** `pytest tests/testTacticClassifier.py` passes (10/10 2026-04-21).

### T2.5 `[P2]` `riskSynthesis.py` — **STATUS: DONE**

- `_computeScore`: `score = 40*pSynth + clamp(15*Σmalicious_conf, 0, 60) − 20*has_benign`.
- `_assignBand`: critical ≥75, high ≥50, medium ≥25, low <25.
- `synthesizeRisk` delegates reasoning string to `_callLlmForReasoning` (via tacticClassifier cache).
- 14 unit tests: low/medium/critical scenarios, clamping, band consistency.
- **Acceptance:** all three hand-crafted scenarios land in correct band (2026-04-21).

### T2.6 `[P2]` `briefingTts.py` — **STATUS: DONE**

- SpeechT5 + HiFi-GAN + cmu-arctic-xvectors speaker embedding (index 7306).
- `_buildBriefingText` produces "score / band / tactics / reasoning" speech text.
- Integration test runs on Colab (requires GPU); no additional unit tests beyond
  the mocked orchestrator path.
- **Acceptance:** WAV plays intelligibly on Colab T4 (to be verified in Phase 4 demo).

### T2.7 `[P2]` `orchestrator.py` + `reportSchema.py` + CLI — **STATUS: DONE**

- `orchestrator.runPipeline`: wires all 6 stages, captures per-stage latency,
  writes `report.json` via `reportSchema.saveReport`.
- `reportSchema`: `toDict`/`fromDict` (pydantic v2 validation), `saveReport`,
  `validateReport`.
- `cli.py`: `python -m vishguard.cli run path/to/call.wav --out out/ [--no-tts]`.
- 13 unit tests for reportSchema: round-trip, path coercion, save/validate file I/O.
- **Acceptance:** `pytest tests/testReportSchema.py` passes (13/13 2026-04-21);
  `pytest tests/` 68/68 green.

---

## Phase 3 — Evaluation harness (Day 4–5)

Goal: produce every number that will appear in the slide deck with a
reproducible script.

### T3.1 `[P3]` Anti-spoof eval corpus — **STATUS: IN_PROGRESS**

- Using LibriSpeech dev-clean (real) + SpeechT5-generated (fake) — 50 + 50 clips.
  ASVspoof 2019 LA subset skipped (registration-gated download; synthetic set
  sufficient for course rubric).
- `eval/buildSpoofSet.py` implemented with `buildManifest` + `validateManifestColumns`
  helpers; 7 unit tests green (2026-04-21).
- **Acceptance:** `eval/buildSpoofSet.py` outputs a manifest CSV with
  path + label + source columns.
- **Pending:** run on Colab T4 to generate `eval/out/spoof/manifest.csv`.

### T3.2 `[P3]` Anti-spoof metrics — **STATUS: IN_PROGRESS**

- `eval/runSpoofEval.py` implemented: `computeMetrics` (accuracy/precision/recall/F1),
  `computeEer` (sklearn roc_curve), confusion matrix PNG. 7 unit tests green (2026-04-21).
- **Acceptance:** CSV + PNG written to `eval/out/spoof/`.
- **Pending:** run on Colab T4 after T3.1 manifest is produced.

### T3.3 `[P3]` Tactic-classification corpus — **STATUS: DONE**

- 50 synthetic call scripts generated by Claude claude-sonnet-4-6 (2026-04-21),
  hand-verified by Ben Blake. All 50 retained.
- Covers all 10 taxonomy labels with realistic multi-label overlap.
- `eval/data/tactics.jsonl` committed; `eval/buildTacticSet.py` validates + reports.
- 8 unit tests green (2026-04-21). See `AI_TOOLS.md` Phase 3 for full disclosure.
- **Acceptance:** met — `eval/data/tactics.jsonl` exists with `text`, `labels`, `notes`.

### T3.4 `[P3]` Tactic-classification metrics — **STATUS: IN_PROGRESS**

- `eval/runTacticEval.py` implemented: `computePerLabelF1`, `computeMacroF1`,
  per-variant CSV output. 7 unit tests green (2026-04-21).
- **Acceptance:** CSV comparing v1 vs. v2 with ≥5-point macro-F1 gap expected.
- **Pending:** run on Colab T4 (requires Qwen GPU inference).

### T3.5 `[P3]` ASR WER eval — **STATUS: IN_PROGRESS**

- `eval/runAsrEval.py` implemented: `normalizeText`, `addNoise` (Gaussian at target
  SNR), `computeWer` (jiwer), bar chart output. 10 LibriSpeech clips × 3 SNR
  conditions × 2 models. 8 unit tests green (2026-04-21).
- **Acceptance:** CSV + bar chart written to `eval/out/asr/`.
- **Pending:** run on Colab T4.

### T3.6 `[P3]` End-to-end latency bench — **STATUS: IN_PROGRESS**

- `eval/runLatencyBench.py` implemented: synthetic WAV generation (silent clips),
  `summarizeTimings`, `makeTimingRow`, CSV output. 7 unit tests green (2026-04-21).
- Clips: 15 s / 45 s / 90 s silent WAVs through full orchestrator.
- **Acceptance:** table goes straight into the slide deck.
- **Pending:** run on Colab T4 (CPU run locally is also useful as baseline).

---

## Phase 4 — UI + polish (Day 5–6)

### T4.1 `[P4]` Streamlit app — **STATUS: NOT_STARTED**

- `app.py` → file uploader → orchestrator → render report.
- `pageReport.py`: header with risk score + band, expandable sections for
  transcript, spoof verdict, tactics (with evidence spans highlighted),
  and an audio widget for the TTS briefing.
- Show timings table.
- **Acceptance:** demo flow from upload → report runs cleanly in under
  3 minutes wall-clock on CPU for a 60-s clip (using `whisper-tiny` for
  local speed and `whisper-small` in the Colab demo recording).

### T4.2 `[P4]` README — **STATUS: NOT_STARTED**

- Setup (Python version, `pip install -r requirements.txt`,
  `.env.example`).
- Model card table linking to `ARCHITECTURE.md`.
- How to reproduce each evaluation.
- Sample outputs (screenshots + one report.json).
- AI-tools disclosure link.
- **Acceptance:** clean markdown render on GitHub, no broken links.

### T4.3 `[P4]` Failure-case write-up — **STATUS: NOT_STARTED**

- Pick 3 failures from Phase 3 (e.g., benign call mis-flagged,
  SpeechT5-generated deepfake passed as real, tactic taxonomy ambiguity).
- 1 paragraph + evidence each in `docs/FAILURES.md`.
- **Acceptance:** deck's "Limitations" slide pulls directly from here.

### T4.4 `[P4]` Code review pass — **STATUS: NOT_STARTED**

- Run the `code-reviewer` agent on `src/vishguard/` and address any
  CRITICAL / HIGH findings.
- **Acceptance:** no open CRITICAL items at end of Phase 4.

---

## Phase 5 — Deliverables (Day 6–7) — CANNOT BE COMPRESSED

### T5.1 `[P5]` PowerPoint deck (10+ slides) — **STATUS: NOT_STARTED**

Required slides (per Canvas rubric):

1. Title + author + course + date
2. Problem description
3. Why interesting / business value
4. Dataset & inputs
5. Models used (pipeline model card)
6. Pipeline architecture (screenshot of ARCHITECTURE.md ASCII diagram)
7. Prompt / input design
8. Results — ASR WER + tactic F1 + anti-spoof F1 + latency
9. Evaluation analysis (plots, failure cases)
10. Demo video link + GitHub link
11. Limitations (pull from FAILURES.md)
12. AI tools used (pull from AI_TOOLS.md)

- **Acceptance:** exported to `docs/vishguard_slides.pptx`, reviewed
  against the Canvas rubric checklist.

### T5.2 `[P5]` Demo video (1–2 min) — **STATUS: NOT_STARTED**

- Script: (0:00) problem intro → (0:15) Streamlit upload → (0:30)
  transcript + spoof + tactics populate → (0:50) risk score + spoken
  briefing plays → (1:10) one failure case → (1:25) GitHub URL.
- Record with QuickTime screen capture + mic. Upload to YouTube
  (unlisted) or Google Drive. Link in README + deck.
- **Acceptance:** video under 2:00, audible narration, shows end-to-end.

### T5.3 `[P5]` Final AI-tools pass — **STATUS: NOT_STARTED**

- Fill every remaining phase section in `AI_TOOLS.md` from commit
  history and session notes (not from memory).
- **Acceptance:** no blank "Tools used" entries.

### T5.4 `[P5]` Git tag + Canvas submission — **STATUS: NOT_STARTED**

- Tag `v1.0.0`, push, ensure GitHub repo is public.
- Upload deck to Canvas, paste GitHub + demo URLs in submission notes.
- **Acceptance:** Canvas shows submitted before 2026-04-27.

---

## Cross-cutting rules

- Commit before and after each AI-assisted change. Conventional commit
  prefixes: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`.
- Update `docs/AI_TOOLS.md` at the end of every session — never
  reconstruct on Day 7.
- Any stage that fails its Phase-3 eval below the "acceptance" bar gets
  an entry in `docs/FAILURES.md` instead of silent omission; the rubric
  rewards analysis over perfect numbers.
- No new scope admitted after Phase 3 is closed. Stretch goals in
  `PRD.md §4` are the only permitted additions.
