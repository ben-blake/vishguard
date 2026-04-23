# VishGuard — Source of Truth

VishGuard is a CS 5542 Quiz Challenge 2 submission (UMKC, Spring 2026). It is
a solo, one-week academic project — **not a production system**. Grading
rubric weights drive every scope decision: completion 30%, output quality 20%,
evaluation 25%, tech design 15%, presentation 10%, plus bonus for multimodal.

## What it does

An audio pipeline that analyzes a phone-call recording and returns a
structured risk report:

1. Synthetic/deepfake voice likelihood
2. Full transcript (ASR)
3. Social-engineering tactic classification
4. Aggregate risk score with reasoning
5. Spoken risk briefing via TTS (bonus / multimodal extension)

## Planning artifacts

All planning lives in [`docs/`](./docs/) and is the authoritative spec:

- [`docs/PRD.md`](./docs/PRD.md) — problem, users, scope, non-goals, success
  criteria tied to rubric weights, business value.
- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — pipeline, HuggingFace
  model IDs with fallbacks, stage interfaces, shared report schema.
- [`docs/TASKS.md`](./docs/TASKS.md) — phased task list (5 phases), acceptance
  criteria, status. Phase 1 front-loads the highest-risk technical unknowns.
- [`docs/AI_TOOLS.md`](./docs/AI_TOOLS.md) — AI-assistance disclosure, filled
  in at the end of every session (required for full credit).

## Development workflow

- `/plan` before new modules, `/tdd` for classification logic,
  `/code-review` before each commit.
- Commit before and after each AI-assisted change.
- Update `docs/AI_TOOLS.md` at the end of every session — do not reconstruct
  it from memory on day 7.
- Short sessions, `/clear` between phases.

## Phase 1 key decisions (locked — do not revisit)

- **Anti-spoof model:** `mo-thecreator/Deepfake-audio-detection` — confirmed working.
  `MelodyMachine/Deepfake-audio-detection-V2` always predicts real (p(synth)≈0.0); do not use.
- **Tactic LLM:** `Qwen/Qwen2.5-3B-Instruct` in fp16 — 10/10 JSON reliability on T4.
  Prompt v4 (9 few-shot examples, co-occurrence fixes) is in `promptLibrary.py`; use it by default.
  v1–v3 retained for eval comparison only.
- **ASR:** `openai/whisper-small` via `WhisperProcessor` + `WhisperForConditionalGeneration`
  directly — **not** the pipeline (transformers 5.x pipeline raises `KeyError: num_frames`).
  Always strip punctuation before WER: `re.sub(r'[^\w\s]', '', text)`.
- **Python:** 3.12 locally and on Colab. `bitsandbytes` is GPU-only; excluded from `requirements.txt`.

## Phase 2 implementation notes (locked)

All modules implemented TDD (68 tests, 100% green as of 2026-04-21).
Phase 3 eval scripts added 49 more tests (117 total, all green as of 2026-04-23).

**Model caching:** every module uses a module-level dict (`_MODEL_CACHE` or
`_PIPE_CACHE`) keyed by `modelId`. This avoids reloading models on repeated
calls within a session. Do not refactor to instance state — the cache must
survive across multiple `transcribe`/`detectSpoof`/`classifyTactics` calls.

**Anti-spoof pipeline input:** `pipeline(...)` expects a dict
`{"raw": clip.samples.astype(float), "sampling_rate": clip.sampleRate}`.
Passing a bare numpy array causes a shape error. The label to extract is
`"fake"` (not `"LABEL_1"` — the model head uses string class names).

**Risk score formula (do not change constants without re-running Phase 3 eval):**

```text
score = 40 * pSynth + clamp(15 * sum(malicious_conf), 0, 60) − 20 * has_benign
```

Band thresholds: critical ≥ 75, high ≥ 50, medium ≥ 25, low < 25.
Score is clamped to [0, 100].

**TTS speaker embedding:** `Matthijs/cmu-arctic-xvectors`, index 7306 (verified
to produce intelligible output on Colab T4). Changing the index changes the
voice — keep it pinned for reproducible demo recordings.

**Quantization:** `tacticClassifier._load_tokenizer_and_model` checks
`cfg.loadIn4Bit` and imports `BitsAndBytesConfig` only when true. On macOS
this flag must be false (no `bitsandbytes` build). The `configs/default.yaml`
sets `llm.device: cuda` and `loadIn4Bit: true` for Colab; override to
`cpu`/`false` locally or via CLI `--device cpu`.

## Filename conventions

- Python modules: `camelCase.py` (e.g., `promptBuilder.py`, `riskSynthesis.py`)
- Package directories: lowercase (e.g., `src/vishguard/`)
- Docs: `UPPER_SNAKE.md` or `Title.md`
- Tests: `testXxx.py`
