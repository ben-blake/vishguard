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

## Filename conventions

- Python modules: `camelCase.py` (e.g., `promptBuilder.py`, `riskSynthesis.py`)
- Package directories: lowercase (e.g., `src/vishguard/`)
- Docs: `UPPER_SNAKE.md` or `Title.md`
- Tests: `testXxx.py`
