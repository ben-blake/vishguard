# VishGuard — Demo Video Script (T5.2)

Target runtime: 1:45 – 2:00. Record with QuickTime screen capture + mic.

---

## 0:00 — Problem intro (15 s)

> "Vishing — voice phishing — is one of the fastest-growing social-engineering
> attacks. A caller impersonates a government agency or bank, creates urgency,
> and harvests credentials or money before the victim realizes what happened.
> VishGuard is an end-to-end audio pipeline that analyzes a recorded call and
> returns a structured risk report."

_Keep Streamlit tab open but in the background._

---

## 0:15 — Show the UI (15 s)

> "Here is the Streamlit interface. On the left sidebar I have the pipeline
> settings — ASR model, device, prompt variant, and an optional spoken briefing.
> I'm using whisper-tiny, CUDA, and the v4 tactic prompt, which achieves a
> macro-F1 of 0.604 across ten social-engineering tactic labels. I'll also
> enable the spoken briefing — VishGuard can narrate the risk
> summary as audio, which is the multimodal bonus component."

_Point to sidebar. Enable **Generate spoken briefing** checkbox. Do not click Analyze yet._

---

## 0:30 — Upload and analyze (20 s)

> "I'll upload the demo call. This is a synthetic vishing call generated with
> Microsoft SpeechT5."

_Drag `demo_call.wav` into the file uploader. Click **Analyze call**._

> "The pipeline is running — ingestion, Whisper ASR, deepfake detection, Qwen
> tactic classification, risk scoring, and finally the TTS briefing."

_Wait for spinner to resolve._

---

## 0:50 — Walk through results (30 s)

> "The anti-spoof stage flagged this call as synthetic with high confidence —
> that's the deepfake voice detector working correctly."

_Point to Synthetic probability metric._

> "The transcript shows the full text across all audio chunks — the chunking
> handles recordings longer than 30 seconds, which Whisper would otherwise
> silently truncate."

_Expand the Transcript section if collapsed._

> "The tactic classifier identified impersonation, urgency, fear and intimidation."

_Point to tactic chips or table._

> "The risk score comes out critical. The formula weights synthetic probability
> heavily, then adds malicious tactic confidence and subtracts a benign penalty.
> The reasoning text explains why in plain English."

_Point to risk score card and reasoning._

> "As a multimodal bonus, VishGuard also synthesizes a spoken briefing using
> SpeechT5. Let's play it."

_Click the audio player widget in the Spoken Briefing section and let it play (~10 s)._

---

## 1:20 — Limitations (15 s)

> "A known failure: the anti-spoof model misses about 34 percent of SpeechT5
> clips — recall is 0.66. That's a distribution-shift problem; the model was
> trained on different TTS systems. The fear-and-intimidation tactic label also
> has the lowest F1 at 0.36 due to co-occurrence ambiguity with impersonation.
> Both cases are documented in docs/FAILURES.md."

---

## 1:35 — Close (15 s)

> "The full source is on GitHub. The pipeline is under 400 lines of core Python,
> uses only open-source HuggingFace models, and runs on a free Colab T4. Thanks
> for watching."

_Show browser with GitHub repo URL visible._

---

## Recording checklist

- [ ] Colab T4 running, Streamlit tunnel open, pre-warm complete
- [ ] `demo_call.wav` downloaded locally and ready to drag-and-drop
- [ ] QuickTime → New Screen Recording → select tab + mic
- [ ] Sidebar set: `whisper-small`, `cuda`, `v4`, **TTS on**
- [ ] Do a dry run once before recording to confirm timing
- [ ] Upload to YouTube (unlisted) or Google Drive; paste link in README and slide deck

---

## Key numbers to mention (from Phase 3 eval)

| Metric | Value |
|---|---|
| Anti-spoof accuracy (overall) | 0.83 |
| Anti-spoof recall on synthetic | 0.66 |
| Tactic macro-F1 v4 | 0.604 |
| Tactic macro-F1 v1 (baseline) | 0.024 |
| Mean end-to-end latency (Colab T4) | 21.7 s |
