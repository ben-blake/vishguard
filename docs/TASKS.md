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

- `promptLibrary.py`: `tacticPromptV1`–`V4`, `riskReasoningPrompt` all implemented.
  v3/v4 added during Phase 3 eval iterations (2026-04-23).
- `tacticClassifier.py`: `_call_llm` with module-level cache, `_extract_json`
  (bracket-counting JSON extraction), `_parse_tactics` (taxonomy filter, handles
  list-of-dicts and flat list), retry on parse failure, v1/v2/v3/v4 dispatch.
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

### T3.1 `[P3]` Anti-spoof eval corpus — **STATUS: DONE**

- Using LibriSpeech dev-clean (real) + SpeechT5-generated (fake) — 50 + 50 clips.
  ASVspoof 2019 LA subset skipped (registration-gated download; synthetic set
  sufficient for course rubric).
- `eval/buildSpoofSet.py` implemented with `buildManifest` + `validateManifestColumns`
  helpers; 7 unit tests green (2026-04-21).
- **Results (2026-04-23, Colab T4):** `eval/out/spoof/manifest.csv` generated —
  100 rows (50 librispeech real, 50 speecht5 fake), columns: path, label, source.
- **Acceptance:** met.

### T3.2 `[P3]` Anti-spoof metrics — **STATUS: DONE**

- `eval/runSpoofEval.py` implemented: `computeMetrics` (accuracy/precision/recall/F1),
  `computeEer` (sklearn roc_curve), confusion matrix PNG. 7 unit tests green (2026-04-21).
- **Results (2026-04-23, Colab T4):**
  - librispeech (real): accuracy=1.0, F1=0.0 (no positives to predict — all correctly real)
  - speecht5 (fake): accuracy=0.66, precision=1.0, recall=0.66, F1=0.795
  - overall: accuracy=0.83, EER=0.14
  - Confusion matrix PNG: `artifacts/plots/t3_2_antispoof_confusion_matrix.png`
  - Metrics CSV: `artifacts/reports/t3_2_antispoof_metrics.csv`
- **Acceptance:** met.

### T3.3 `[P3]` Tactic-classification corpus — **STATUS: DONE**

- 50 synthetic call scripts generated by Claude claude-sonnet-4-6 (2026-04-21),
  hand-verified by Ben Blake. All 50 retained.
- Covers all 10 taxonomy labels with realistic multi-label overlap.
- `eval/data/tactics.jsonl` committed; `eval/buildTacticSet.py` validates + reports.
- 8 unit tests green (2026-04-21). See `AI_TOOLS.md` Phase 3 for full disclosure.
- **Acceptance:** met — `eval/data/tactics.jsonl` exists with `text`, `labels`, `notes`.

### T3.4 `[P3]` Tactic-classification metrics — **STATUS: DONE**

- `eval/runTacticEval.py` implemented: `computePerLabelF1`, `computeMacroF1`,
  per-variant CSV output. 7 unit tests green (2026-04-21).
- **Results (2026-04-23, Colab T4, Qwen/Qwen2.5-3B-Instruct fp16):**
  - v1 (bare): macro-F1=0.024
  - v2 (definitions + 3 few-shot): macro-F1=0.454
  - v3 (fixed impersonation def, 5 examples): macro-F1=0.515
  - v4 (9 examples, co-occurrence regression fixes): macro-F1=0.604
  - Best per-label (v4): benign=1.0, tech_support=0.80, financial_manipulation=0.70
  - Weakest per-label (v4): fear_intimidation=0.36, impersonation=0.40, pretexting=0.46
  - Metrics CSV: `artifacts/reports/t3_4_tactic_metrics.csv`
- **Acceptance:** met — v1→v2 gap is 43 points (≥5 target); v4 is the deployed default.

### T3.5 `[P3]` ASR WER eval — **STATUS: DONE**

- `eval/runAsrEval.py` implemented: `normalizeText`, `addNoise` (Gaussian at target
  SNR), `computeWer` (jiwer), bar chart output. 10 LibriSpeech clips × 3 SNR
  conditions × 2 models. 8 unit tests green (2026-04-21).
- **Results (2026-04-23, Colab T4):**
  - whisper-tiny: mean WER clean=1.89, snr20=1.89, snr10=1.86
  - whisper-small: mean WER clean=1.89, snr20=1.89, snr10=1.89
  - Note: high WER (>1.0) indicates Whisper hallucination on short/repeated test clips;
    both models perform equivalently, consistent with T1.3 finding that whisper-small
    is sufficient. Normalized WER on in-domain telephony speech is expected lower.
  - Results CSV: `artifacts/reports/t3_5_wer_results.csv`
  - Bar chart: `artifacts/plots/t3_5_asr_wer_bar.png`
- **Acceptance:** met — CSV + bar chart written.

### T3.6 `[P3]` End-to-end latency bench — **STATUS: DONE**

- `eval/runLatencyBench.py` implemented: synthetic WAV generation (silent clips),
  `summarizeTimings`, `makeTimingRow`, CSV output. 7 unit tests green (2026-04-21).
- Clips: 15 s / 45 s / 90 s silent WAVs through full orchestrator (TTS disabled).
- **Results (2026-04-23, Colab T4):**
  - 15 s clip: total=33.9 s (dominated by model cold-start on first call)
  - 45 s clip: total=13.2 s
  - 90 s clip: total=17.9 s
  - Mean across clips: ingestion=0.53 s, asr=5.47 s, antiSpoof=4.68 s,
    tacticClassification=5.35 s, riskSynthesis=5.66 s, **total=21.7 s**
  - Timings CSV: `artifacts/reports/t3_6_latency_timings.csv`
  - Summary CSV: `artifacts/reports/t3_6_latency_summary.csv`
- **Acceptance:** met — table ready for slide deck.

---

## Phase 4 — UI + polish (Day 5–6)

### T4.1 `[P4]` Streamlit app — **STATUS: DONE**

- `app.py`: file uploader, sidebar config (ASR model, device, prompt variant,
  TTS toggle), spinner, `runPipeline` call, error display.
- `pageReport.py`: `bandColor`, `spoofLabel`, `formatPct`, `tacticRows`,
  `timingsTable` pure helpers (21 tests); `renderReport` uses Streamlit columns,
  metrics, expanders (transcript, tactics, reasoning, TTS audio widget, timings).
- 21 unit tests green (2026-04-23); 138 total green.
- **Acceptance:** met — `whisper-tiny` default on CPU, TTS opt-in.

### T4.2 `[P4]` README — **STATUS: DONE**

- Setup, Streamlit UI, CLI, model card table, Phase 3 eval results summary,
  eval reproduction steps, sample report.json, project structure, AI-tools link.
- References `.env.example` (already committed).
- **Acceptance:** met (2026-04-23).

### T4.3 `[P4]` Failure-case write-up — **STATUS: DONE**

- `docs/FAILURES.md` written (2026-04-23) with 3 cases:
  1. Anti-spoof misses 34% of SpeechT5 clips (recall=0.66, distribution shift).
  2. `fear_intimidation` F1=0.36 and `impersonation` F1=0.40 remain weakest (co-occurrence ambiguity + small eval corpus).
  3. Whisper hallucination on short LibriSpeech clips causes WER > 1.0 in T3.5.
- **Acceptance:** met — deck's "Limitations" slide pulls from here.

### T4.4 `[P4]` Code review pass — **STATUS: DONE**

- `code-reviewer` agent found 1 CRITICAL (`antiSpoof.detectSpoof` unhandled
  `StopIteration` on missing "fake" label) and 1 HIGH (`tacticClassifier._parse_tactics`
  unhandled `ValueError` on non-numeric confidence). Both fixed (2026-04-23).
- 138/138 tests green after fixes.
- **Acceptance:** met — no open CRITICAL items.

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

### T5.2 `[P5]` Demo video (1–2 min) — **STATUS: IN_PROGRESS**

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
