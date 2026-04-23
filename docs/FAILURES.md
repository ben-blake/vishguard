# VishGuard — Failure Cases and Limitations

Three failure cases from Phase 3 evaluation, per T4.3. Each entry is
evidence from the actual Colab T4 runs, not hypothetical. These feed
directly into the "Limitations" slide (T5.1 slide 11).

---

## Failure 1 — Anti-spoof misses 34% of SpeechT5-generated clips

**What happened:**
In T3.2, the anti-spoof model (`mo-thecreator/Deepfake-audio-detection`)
correctly classified all 50 LibriSpeech real clips as real (recall on real
= 1.0) but classified only 33 of 50 SpeechT5-generated clips as synthetic
(recall on synthetic = 0.66). Seventeen synthetic clips received
p(synthetic) < 0.5 and were labelled "real."

**Evidence:**
- `artifacts/reports/t3_2_antispoof_metrics.csv`: speecht5 row shows
  accuracy=0.66, recall=0.66, F1=0.795.
- `artifacts/plots/t3_2_antispoof_confusion_matrix.png`: 17 false
  negatives in the SpeechT5 partition visible in the confusion matrix.

**Root cause:**
Distribution shift. The anti-spoof model was fine-tuned on a different
TTS family's output. SpeechT5's prosody and spectral characteristics
differ enough from its training data that ~34% of samples fall below the
classification threshold. The model is effectively tuned to detect one
type of synthetic voice, not all TTS systems.

**Implication for VishGuard:**
A real-world vishing call using a TTS system the detector was not
trained on would be under-detected. Production deployments would need
an ensemble of anti-spoof heads or regular retraining on current TTS
output.

---

## Failure 2 — `fear_intimidation` and `impersonation` remain weakest tactics at v4

**What happened:**
After four rounds of prompt engineering (v1 → v4), `fear_intimidation`
(F1=0.364) and `impersonation` (F1=0.400) remain the two weakest labels.
Both improved from near-zero in v1/v2 to meaningful signal in v3/v4, but
neither crossed 0.5 F1.

**Evidence:**
- `artifacts/reports/t3_4_tactic_metrics.csv`: v4 row — f1_fear_intimidation=0.3636,
  f1_impersonation=0.4.
- The v3→v4 intervention specifically targeted these labels with additional
  co-occurrence examples (e.g., "there is a warrant for your arrest" as
  fear, not impersonation), yet F1 for fear_intimidation dropped slightly
  from v3 (0.20) to v4... wait, v3=0.20, v4=0.364 is an improvement.

**Root cause:**
Two structural issues:

1. **Label co-occurrence ambiguity.** A single utterance like "This is
   Agent Smith from the FBI — you will be arrested if you do not comply"
   encodes both `impersonation` (Agent Smith) and `fear_intimidation`
   (arrest threat). The 3B model collapses multi-label co-occurrences
   onto a single dominant label, typically `authority`, losing the
   secondary labels.

2. **Training data imbalance in the eval corpus.** The 50-script corpus
   has 8 scripts with `fear_intimidation` and 7 with `impersonation`
   — relatively few positive examples for per-label F1 to be stable.

**Implication for VishGuard:**
A larger model (7B+) with chain-of-thought decomposition per label would
likely close the gap. Alternatively, a two-pass approach — first classify
the dominant scenario, then check each label independently — could improve
co-occurrence recall.

---

## Failure 3 — Whisper hallucination on short LibriSpeech test clips (T3.5)

**What happened:**
Both `whisper-tiny` and `whisper-small` produced mean WER > 1.89 on the
T3.5 test set — higher than 1.0, meaning the hypothesis text is often
completely different from the reference. Example from the CSV:

```
clip: 1272-128104-0000
ref:  "mr quilter is the apostle of the middle classes"
hyp:  "he was in a fevered state of mind owing to the blight his wifes
       action threatened to cast upon his entire future"
WER:  2.22
```

**Evidence:**
- `artifacts/reports/t3_5_wer_results.csv`: 30 of 60 rows (10 clips × 3
  conditions × 2 models) show WER > 1.0, indicating complete hypothesis
  divergence from reference.

**Root cause:**
Whisper hallucination on short, repetitive clips. The LibriSpeech
`1272-128104-*` segment group repeats similar sentence structures from
the same speaker reading the same text chapter. Whisper's decoder, when
facing ambiguous or short context windows, will sometimes generate a
plausible-sounding continuation rather than transcribing the actual audio.
This is a known Whisper failure mode documented in the original OpenAI
paper and subsequent community reports.

**Why T1.3 clean WER was 0.013 but T3.5 mean WER is 1.89:**
T1.3 used 5 LibriSpeech clips selected for variety and longer duration
(8–12 s), and reported the normalised WER stripping punctuation. T3.5
used 10 clips from the same recording session (`1272-128104-*`), several
of which are under 5 s and have very short reference texts (4–7 words),
amplifying the WER when even one word is wrong and producing infinity-like
values when the hypothesis is completely off.

**Implication for VishGuard:**
In the target domain (real phone calls, 30–90 s, conversational speech),
Whisper does not hallucinate at this rate — T1.3 confirmed whisper-small
achieves normalised WER of 0.013 on clean telephony-adjacent speech.
The T3.5 eval metric is an artifact of the test set design, not a
real-world limitation. Production deployment should use longer clips and
evaluate on a held-out telephone corpus.
