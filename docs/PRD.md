# VishGuard — Product Requirements Document

**Course:** CS 5542 (UMKC) — Quiz Challenge 2
**Theme:** Foundation models for speech, music, and sound AI
**Deadline:** 2026-04-27
**Style:** Individual, ~1 week
**Submission:** Canvas (PowerPoint) + GitHub repo + demo video

---

## 1. Problem

Voice-based social engineering ("vishing") is the highest-growth fraud vector
aimed at consumers and enterprise help desks. Two forces amplify it:

1. **Loss scale.** The FBI IC3 2023 report attributes $1.3B+ in US losses to
   call-center and tech-support scams, with vishing driving a large share of
   credential and wire-fraud compromises. Enterprise help desks have been the
   entry point for high-profile breaches (e.g., MGM 2023, retail Scattered
   Spider campaigns in 2023–2024).
2. **Synthetic-voice availability.** Open-source TTS models (SpeechT5,
   VALL-E-X, XTTS, OpenVoice) and commercial APIs (ElevenLabs) lowered the
   cost of cloning a caller's voice to minutes of reference audio. FTC and
   CFPB have issued consumer advisories about deepfake family-emergency scams.

Listeners — both humans and automated call-center triage — currently have
**no structured, auditable signal** telling them whether a given call is
(a) synthetic, (b) using known social-engineering tactics, and (c) how those
two signals combine into an overall risk. VishGuard produces that signal.

## 2. Target user

Primary persona is a **SOC / fraud-ops analyst** or **call-center QA reviewer**
triaging a queue of suspect calls after the fact. VishGuard is an
*offline-batch audio forensics* tool, not a real-time caller-ID overlay.

Secondary persona is an **individual consumer** who received a suspicious
voicemail and wants a second opinion before calling back.

Both personas need the same output: one report, one glanceable risk score,
and the evidence chain that produced it.

### Why this framing matters for the rubric

- Offline triage is honest about what a 1-week academic build can deliver.
  Real-time streaming ASR + spoof detection with sub-second latency is a
  research problem; batch file analysis is achievable and demoable in 2
  minutes.
- Multiple personas give the evaluation section natural comparison pairs
  (terse analyst summary vs. consumer-friendly TTS briefing).

## 3. Scope

### In scope

- **Input:** pre-recorded call audio (WAV/MP3/M4A), up to ~3 minutes.
- **Stages:** ingestion → ASR → anti-spoof → tactic classification → risk
  synthesis → spoken briefing → report artifact.
- **Interface:** Streamlit app (upload → view report) for demo, plus a CLI
  entry point for batch evaluation.
- **Models:** pretrained HuggingFace checkpoints only. No fine-tuning from
  scratch. Prompt engineering and threshold tuning only.
- **Evaluation:** baseline vs. improved for each stage; small hand-labeled
  eval set; failure-case write-up.
- **Deliverables:** GitHub repo + README, 10+ slide PowerPoint, 1–2 min demo
  video, AI-tools disclosure.

### Non-goals (explicit)

- Real-time / streaming analysis during a live call.
- Speaker identification or voice-print matching.
- Training or fine-tuning any model from scratch.
- Adversarial robustness evaluation (laundered deepfakes, codec attacks).
- Multilingual coverage beyond English. If a non-English sample happens to
  work through Whisper's multilingual weights, it is a bonus, not a claim.
- Production PII handling, auth, storage, or retention policies.
- Integration with telephony stacks (SIP, Twilio, Genesys).
- A browser extension, mobile app, or Chrome overlay.

## 4. Success criteria — mapped to the rubric

| Rubric criterion               | Weight | What "meets" looks like for VishGuard                                                                                              |
|--------------------------------|--------|------------------------------------------------------------------------------------------------------------------------------------|
| Completion of working system   | 30%    | End-to-end pipeline runs on an uploaded audio file in Streamlit and produces the full report JSON + spoken briefing.               |
| Quality of outputs             | 20%    | Transcripts are readable; tactic labels are plausibly correct on a hand-verified set; risk score tracks intuition on failure demo. |
| Evaluation & insights          | 25%    | WER clean-vs-noisy, anti-spoof accuracy/F1 on mixed real+synthetic set, tactic classification prompt A/B, end-to-end latency.      |
| Technical design               | 15%    | Clear stage interfaces, swappable models, shared report schema, ASCII data-flow diagram, logged intermediate outputs.              |
| Presentation quality           | 10%    | 10+ slide PPTX covering every required section; 1–2 min demo with input → model running → output; clean README.                    |
| Bonus: multimodal innovation   | extra  | Spoken risk briefing via SpeechT5, plus SpeechT5-generated deepfakes used as held-out eval samples (speech→speech loop).           |

### Stretch goals (only if Phase 1–3 finish ahead)

- Diarization to separate caller vs. callee before tactic analysis.
- Noise-robustness sweep (SNR ladder) on anti-spoof accuracy.
- Per-tactic risk weights tuned against the hand-verified set.

## 5. Business value framing

For the Canvas slide deck, "business value" should name a concrete cost
avoidance story:

- Call-center QA currently samples <5% of calls manually. A pipeline that
  flags 100% of calls with a structured risk score lets analysts focus their
  human attention on the top-decile risk bucket.
- Consumers reach rate of US adults targeted by scam calls in 2023 exceeded
  60% per Truecaller / YouGov survey data. A downloadable self-service
  tool ("drop your voicemail in, get a report") has obvious product-market
  fit even as a free utility.
- Anti-phishing vendors (Proofpoint, Abnormal, KnowBe4) already sell
  text-channel versions. Voice-channel fraud tooling is a market adjacency
  with active M&A interest (Pindrop, Hiya, Nuance Gatekeeper).

VishGuard does not *solve* this market — it *demonstrates the plumbing*
using open foundation models in a one-week project.

## 6. Assumptions & constraints

- **Compute:** Google Colab free tier (T4 16GB) is the target execution
  environment. Any model that does not fit in 4-bit quantization on T4 is
  out. Streamlit is developed locally (CPU-acceptable for Whisper-small).
- **Licensing:** all chosen models must be usable for academic coursework
  without a gated approval step. Any gated model (Llama 3.1 instruct) is
  flagged in ARCHITECTURE.md with a non-gated fallback.
- **Data:** ASVspoof 2019 is used under its research license; no dataset
  requires redistribution. No real recorded phone calls with PII are
  committed to the repo.
- **Human bandwidth:** one student, ~6 working days. Scope is priced for
  solo execution, not team distribution.

## 7. Acceptance checklist

The project is "done" when all of the following are true:

- [ ] End-to-end Streamlit demo runs on a laptop in under 3 minutes per
      uploaded call.
- [ ] Report JSON schema matches [`ARCHITECTURE.md`](./ARCHITECTURE.md#report-schema).
- [ ] Every evaluation metric in §4 has a number in the deck with a
      baseline-vs-improved comparison.
- [ ] At least 3 documented failure cases in the deck with analysis.
- [ ] `AI_TOOLS.md` is filled per phase with no blank sections.
- [ ] Demo video is under 2 minutes and shows input → output.
- [ ] README covers setup, model choices, and how to reproduce the eval.
