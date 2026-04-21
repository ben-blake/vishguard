# VishGuard — Architecture

This document locks in the pipeline, model choices, interfaces, and the
shared report schema. All module/file names use `camelCase.py` per the
repo convention.

---

## 1. Data-flow diagram

```text
                            ┌────────────────────────┐
 user uploads WAV/MP3 ───►  │   ingestion            │
                            │   (loadAudio.py)       │
                            │  - decode, resample    │
                            │  - mono 16kHz float32  │
                            │  - chunk if >30s       │
                            └────────────┬───────────┘
                                         │ AudioClip
                                         ▼
 ┌─────────────────────────┐   ┌────────────────────────┐
 │  ASR stage              │   │  Anti-spoof stage      │
 │  (asrWhisper.py)        │   │  (antiSpoof.py)        │
 │  - openai/whisper-small │   │  - wav2vec2 spoof head │
 │  - returns text + segs  │   │  - returns p(synth)    │
 └───────────┬─────────────┘   └────────────┬───────────┘
             │ Transcript                   │ SpoofVerdict
             │                              │
             ▼                              │
 ┌─────────────────────────┐                │
 │  Tactic classifier      │                │
 │  (tacticClassifier.py)  │                │
 │  - open LLM, zero-shot  │                │
 │  - returns Tactic[]     │                │
 └───────────┬─────────────┘                │
             │                              │
             ▼                              ▼
 ┌──────────────────────────────────────────────────────┐
 │             Risk synthesis                           │
 │             (riskSynthesis.py)                       │
 │   Combines SpoofVerdict + Tactic[] + transcript      │
 │   → RiskScore (0–100) with reasoning, evidence refs  │
 └─────────────────────────────┬────────────────────────┘
                               │ RiskReport (JSON)
              ┌────────────────┴───────────────┐
              ▼                                ▼
 ┌─────────────────────────┐      ┌────────────────────────┐
 │  TTS briefing (bonus)   │      │  Report artifact       │
 │  (briefingTts.py)       │      │  - JSON on disk        │
 │  - SpeechT5 + HiFi-GAN  │      │  - Streamlit render    │
 │  - returns WAV          │      │  - CLI print           │
 └─────────────────────────┘      └────────────────────────┘
```

---

## 2. Model choices (primary + fallbacks)

Every choice below must (a) be a pretrained HuggingFace checkpoint, (b) load
on Colab T4 (16 GB) under 4-bit quantization where relevant, and (c) be
usable for academic coursework without a gated-repo approval step.

### 2.1 ASR — speech to text

| Role     | HF model ID                                      | Notes                                                                           |
|----------|--------------------------------------------------|---------------------------------------------------------------------------------|
| Primary  | `openai/whisper-small`                           | 244M params, multilingual, ships clean; meets assignment's Whisper requirement. |
| Accuracy | `openai/whisper-medium`                          | 769M; use if Phase 3 eval shows WER ceiling issues.                             |
| Fallback | `distil-whisper/distil-large-v3` (English-only)  | 2x faster than whisper-large-v3, English-only, Apache 2.0.                      |
| Laptop   | `openai/whisper-tiny`                            | 39M; Streamlit demo on CPU.                                                     |

Libraries: `transformers` `AutoModelForSpeechSeq2Seq` + pipeline, or
`faster-whisper` (CTranslate2) if CPU speed becomes a Streamlit bottleneck.

### 2.2 Anti-spoof / synthetic-voice detection

| Role     | HF model ID                                  | Notes                                                                        |
|----------|----------------------------------------------|------------------------------------------------------------------------------|
| Primary  | `MelodyMachine/Deepfake-audio-detection-V2`  | wav2vec2-based V2 release, logits to real/fake. Public, non-gated.           |
| Alt A    | `motheecreator/Deepfake-audio-detection`     | Earlier wav2vec2 checkpoint; second opinion for ensemble in Phase 3.         |
| Alt B    | `mo-thecreator/Deepfake-audio-detection`     | Alt author version; use only if primary or Alt A repo disappears.            |
| Research | AASIST (ASVspoof reference) via raw PyTorch  | **Out of scope** - no HF wrapper; listed so reviewers see it was considered. |

**Phase-1 spike** must confirm that at least one of the above produces
stable logits on (a) ASVspoof 2019 LA samples and (b) SpeechT5-generated
deepfakes. If all three fail the smoke test, the fallback is to treat
"synthetic likelihood" as a prompted-LLM signal based on transcript
artifacts (TTS-style phrasing) — clearly documented as a degraded mode.

### 2.3 Tactic classifier — open LLM, zero-shot

Target: Colab T4 16 GB with 4-bit quantization.

| Role     | HF model ID                           | Size | Notes                                                                       |
|----------|---------------------------------------|------|-----------------------------------------------------------------------------|
| Primary  | `Qwen/Qwen2.5-3B-Instruct`            | 3B   | Apache 2.0, strong instruction following, fits T4 comfortably in fp16.      |
| Alt A    | `microsoft/Phi-3-mini-4k-instruct`    | 3.8B | MIT license, strong at structured output.                                   |
| Alt B    | `google/gemma-2-2b-it`                | 2B   | Gemma license (permissive for research); smallest, fastest fallback.        |
| Alt C    | `mistralai/Mistral-7B-Instruct-v0.3`  | 7B   | Needs 4-bit; use if 3B models underperform on tactic prompts.               |
| Blocked  | `meta-llama/Llama-3.1-8B-Instruct`    | 8B   | Gated - requires HF approval. **Not used**; flagged here for completeness.  |

Library: `transformers` + `bitsandbytes` (4-bit NF4) + `accelerate`. We will
*not* take a dependency on vLLM or llama.cpp to keep the repo portable.

Prompt strategy: structured JSON output with a fixed tactic taxonomy
(see §3.3). Phase 1 spike confirms the chosen LLM produces valid JSON
≥90% of the time with one-shot guardrails; if not, fall back to a
constrained-decoding wrapper (`outlines` or manual regex retry).

### 2.4 TTS — briefing + synthetic eval samples

| Role    | HF model ID                     | Notes                                                          |
|---------|---------------------------------|----------------------------------------------------------------|
| Primary | `microsoft/speecht5_tts`        | Required by assignment, MIT license.                           |
| Vocoder | `microsoft/speecht5_hifigan`    | Required companion.                                            |
| Speaker | `Matthijs/cmu-arctic-xvectors`  | HF dataset with x-vectors; pick a small set for variety.       |
| Alt     | `suno/bark-small`               | More natural prosody if SpeechT5 output feels robotic in demo. |

SpeechT5 serves double duty:

1. Generates the spoken risk briefing (bonus multimodal output).
2. Generates "known-synthetic" eval samples for the anti-spoof F1 metric.

This closes the loop cleanly in the slide deck: *"we used the same family
of model to both produce and detect synthetic speech."*

---

## 3. Stage interfaces

All stages are pure functions over dataclasses. No hidden global state,
no ambient config — orchestrator passes everything.

### 3.1 Data types (sketch, to be written in `src/vishguard/types.py`)

```python
@dataclass(frozen=True)
class AudioClip:
    samples: np.ndarray       # float32, mono, 16_000 Hz
    sampleRate: int           # always 16_000 after ingestion
    sourcePath: Path
    durationSec: float

@dataclass(frozen=True)
class TranscriptSegment:
    startSec: float
    endSec: float
    text: str

@dataclass(frozen=True)
class Transcript:
    fullText: str
    segments: tuple[TranscriptSegment, ...]
    languageCode: str
    modelId: str

@dataclass(frozen=True)
class SpoofVerdict:
    pSynthetic: float         # 0.0–1.0
    modelId: str
    rationale: str            # short English explanation for the report

@dataclass(frozen=True)
class Tactic:
    label: str                # one of TACTIC_TAXONOMY
    confidence: float         # 0.0–1.0
    evidenceSpans: tuple[str, ...]   # verbatim quotes from transcript

@dataclass(frozen=True)
class RiskScore:
    score: int                # 0–100
    band: str                 # "low" | "medium" | "high" | "critical"
    reasoning: str            # LLM-written 2–4 sentences

@dataclass(frozen=True)
class RiskReport:
    version: str              # semver of the schema
    audio: AudioMeta
    transcript: Transcript
    spoof: SpoofVerdict
    tactics: tuple[Tactic, ...]
    risk: RiskScore
    briefingAudioPath: Path | None
    timings: dict[str, float]  # perStage latency in seconds
```

### 3.2 Stage signatures

```python
def ingest(path: Path) -> AudioClip: ...
def transcribe(clip: AudioClip, cfg: AsrConfig) -> Transcript: ...
def detectSpoof(clip: AudioClip, cfg: SpoofConfig) -> SpoofVerdict: ...
def classifyTactics(transcript: Transcript, cfg: LlmConfig) -> tuple[Tactic, ...]: ...
def synthesizeRisk(
    transcript: Transcript,
    spoof: SpoofVerdict,
    tactics: tuple[Tactic, ...],
    cfg: LlmConfig,
) -> RiskScore: ...
def narrateBriefing(report: RiskReport, cfg: TtsConfig) -> Path: ...
def buildReport(*args) -> RiskReport: ...
```

The orchestrator (`orchestrator.py`) wires these stages, captures per-stage
latency, and writes `report.json` + `briefing.wav` to an output directory.

### 3.3 Tactic taxonomy

Frozen list, single source of truth. Every labeled example, prompt, and
eval row keys off this constant.

```python
TACTIC_TAXONOMY = (
    "authority",              # impersonating IRS, FBI, bank fraud dept
    "urgency",                # time pressure, "act now"
    "pretexting",             # fabricated backstory or scenario
    "credential_harvesting",  # asks for passwords, OTPs, card numbers, SSN
    "impersonation",          # claims to be a known contact / family member
    "financial_manipulation", # wire transfers, gift cards, crypto
    "fear_intimidation",      # threats of arrest, lawsuit, account closure
    "reward_prize",           # won a lottery, prize, refund
    "tech_support",           # your computer has a virus, need remote access
    "benign",                 # negative control — not a scam
)
```

Grounded in FBI IC3 / NIST SP 800-114 / MITRE ATT&CK T1566 references.

## 4. Report schema (shared JSON contract)

Single source of truth. All downstream rendering (Streamlit, CLI, demo
video overlay, slide screenshot) reads from this.

```json
{
  "version": "1.0.0",
  "audio": {
    "sourcePath": "samples/call_01.wav",
    "durationSec": 47.2,
    "sampleRate": 16000
  },
  "transcript": {
    "modelId": "openai/whisper-small",
    "languageCode": "en",
    "fullText": "Hello, this is calling from the IRS...",
    "segments": [
      {"startSec": 0.0, "endSec": 4.1, "text": "Hello, this is calling from the IRS"}
    ]
  },
  "spoof": {
    "modelId": "MelodyMachine/Deepfake-audio-detection-V2",
    "pSynthetic": 0.87,
    "rationale": "High-confidence synthetic; prosody artifacts typical of TTS output."
  },
  "tactics": [
    {
      "label": "authority",
      "confidence": 0.93,
      "evidenceSpans": ["this is calling from the IRS"]
    },
    {
      "label": "fear_intimidation",
      "confidence": 0.88,
      "evidenceSpans": ["there is a warrant for your arrest"]
    },
    {
      "label": "financial_manipulation",
      "confidence": 0.81,
      "evidenceSpans": ["pay the $2400 in gift cards"]
    }
  ],
  "risk": {
    "score": 92,
    "band": "critical",
    "reasoning": "High synthetic-voice likelihood combined with three classic vishing tactics (authority impersonation, fear, gift-card payment). Block and report."
  },
  "briefingAudioPath": "out/call_01_briefing.wav",
  "timings": {
    "ingestion": 0.12,
    "asr": 8.4,
    "antiSpoof": 1.9,
    "tacticClassification": 4.3,
    "riskSynthesis": 2.1,
    "ttsbriefing": 3.8,
    "total": 20.6
  }
}
```

Schema is versioned (`version` field). Evaluation scripts validate every
generated report against this schema to catch drift.

---

## 5. Repo layout

```text
vishguard/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── app.py                       # Streamlit entry point
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── TASKS.md
│   └── AI_TOOLS.md
├── src/vishguard/
│   ├── __init__.py
│   ├── types.py                 # dataclasses above
│   ├── loadAudio.py             # ingestion
│   ├── asrWhisper.py            # ASR stage
│   ├── antiSpoof.py             # anti-spoof stage
│   ├── promptLibrary.py         # tactic + risk prompts
│   ├── tacticClassifier.py      # LLM tactic stage
│   ├── riskSynthesis.py         # risk scoring
│   ├── briefingTts.py           # SpeechT5 bonus stage
│   ├── orchestrator.py          # wires stages, times them
│   ├── reportSchema.py          # JSON schema validation
│   ├── cli.py                   # batch CLI
│   └── ui/
│       ├── __init__.py
│       └── pageReport.py        # Streamlit render
├── tests/
│   ├── testIngestion.py
│   ├── testTacticClassifier.py
│   └── testReportSchema.py
├── eval/
│   ├── buildSpoofSet.py         # ASVspoof subset + SpeechT5 synthetics
│   ├── buildTacticSet.py        # LLM-gen + hand-verified
│   ├── runAsrEval.py
│   ├── runSpoofEval.py
│   ├── runTacticEval.py
│   └── runLatencyBench.py
├── data/                        # gitignored — datasets not committed
└── notebooks/
    └── 00_colabSetup.ipynb      # self-contained Colab runtime setup
```

Per feedback memory from prior project: every notebook must have a
self-contained setup cell (mount Drive, clone/pull repo, add `src/` to
`sys.path`) — Colab runtimes are isolated.

---

## 6. Configuration

Single `configs/default.yaml` (not committed as secret). Stage configs are
dataclasses that are hydrated from YAML so the Streamlit UI can let the
user flip, e.g., whisper-small vs. whisper-medium, or swap LLM.

No API keys required by default. If the user wants to plug in an
Anthropic/OpenAI LLM instead of the open HF LLM, that is a Streamlit
toggle reading from `.env` (gitignored). Academic-rubric-optimal path
stays fully open-source.

---

## 7. Evaluation approach (summary — details in `TASKS.md` Phase 3)

| Stage             | Metric                                   | Baseline vs. improved                                                         |
|-------------------|------------------------------------------|-------------------------------------------------------------------------------|
| ASR               | WER on ~20 clips, clean vs. SNR-reduced  | `whisper-tiny` vs. `whisper-small` (or `whisper-small` vs. `distil-whisper`). |
| Anti-spoof        | Accuracy / F1 on real + synthetic set    | Single model vs. 2-model ensemble (mean of logits).                           |
| Tactic classifier | Per-label F1 on ~40 hand-verified calls  | Prompt v1 (bare) vs. prompt v2 (taxonomy + few-shot + JSON schema).           |
| End-to-end        | Wall-clock seconds per 60s audio clip    | CPU vs. T4 GPU, and small vs. medium Whisper.                                 |

All numbers must land in the slide deck with real values and, where
applicable, a failure-case walkthrough.

---

## 8. Open risks carried into Phase 1

These are the 3 unknowns that can sink the project if unaddressed:

1. **Anti-spoof model out-of-distribution behavior.** HF deepfake-audio
   detectors are typically fine-tuned on clean TTS samples. Phone-compressed
   audio may degrade accuracy severely. *Mitigation:* Phase 1 spike
   benchmarks on 20 real + 20 SpeechT5 clips before committing.
2. **Open LLM structured-output reliability.** 3B-class models can hallucinate
   JSON or drift from the tactic taxonomy. *Mitigation:* Phase 1 spike
   measures valid-JSON rate over 30 samples; fall back to Qwen-7B in 4-bit
   if reliability < 90%.
3. **Evaluation corpus for tactics.** No permissively licensed vishing
   transcript corpus exists. *Mitigation:* generate 40–60 synthetic call
   scripts with a frontier LLM (disclosed in `AI_TOOLS.md`), hand-verify
   labels, and explicitly acknowledge the synthetic-eval limitation.
