# VishGuard — Demo Video Script (T5.2)

Target runtime: ~1:45. Results are pre-loaded on screen before recording starts.

---

## Before recording

Have the Streamlit results page already showing for `demo_call.wav` — do not record the upload or wait. QuickTime → New Screen Recording → mic on.

---

## 0:00 — Problem + pipeline (20 s)

> "Vishing — voice phishing — uses synthetic voices and social-engineering
> tactics to steal credentials or money over the phone. VishGuard analyzes
> a recorded call and returns a structured risk report using four open-source
> models: Whisper for transcription, a deepfake detector, Qwen for tactic
> classification, and SpeechT5 for a spoken briefing."

---

## 0:20 — Results walkthrough (45 s)

_Screen shows the full results page._

> "This is a synthetic call I generated with SpeechT5. The anti-spoof model
> flagged it with high confidence — you can see the synthetic probability here."

_Point to Synthetic probability metric._

> "Whisper transcribed the full 40-second call in chunks — the complete text
> is in the transcript expander."

_Expand Transcript briefly._

> "The tactic classifier, running Qwen 2.5 3B with a 9-shot prompt, identified
> impersonation, urgency, fear and intimidation, financial manipulation, and
> credential harvesting."

_Point to tactic labels._

> "Those feed into the risk score — weighted by synthetic probability and
> tactic confidence. This call scores critical."

_Point to risk score card._

---

## 1:05 — Spoken briefing (15 s)

> "The multimodal bonus: VishGuard synthesizes a spoken briefing using SpeechT5.
> Here it is."

_Click play on the Spoken Briefing audio widget. Let it finish or fade after ~10 s._

---

## 1:20 — Limitations + close (25 s)

> "One known failure: the anti-spoof model misses about 34 percent of SpeechT5
> clips — recall is 0.66 due to distribution shift. The fear-and-intimidation
> tactic has the lowest F1 at 0.36. Both are documented in the repo.
> Full source on GitHub — all open-source models, runs on a free Colab T4.
> Thanks for watching."

_Show GitHub repo URL._

---

## Recording checklist

- [ ] Results page pre-loaded in browser before hitting record
- [ ] Sidebar shows: `whisper-small`, `cuda`, `v4`, TTS on
- [ ] Spoken Briefing section visible and audio player ready to click
- [ ] GitHub tab open and ready to switch to at the end
- [ ] QuickTime → New Screen Recording → mic on
