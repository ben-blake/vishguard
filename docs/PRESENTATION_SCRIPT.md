# Presentation Script — CS 5542 Quiz Challenges

Total: 4:30 | Demo video: 3:44 | Talking: ~1:10

---

## Slide 1 — Title (~10 s)

"Today I'm presenting two AI systems I built for CS 5542 — Roomify, which generates interior renders from text descriptions, and VishGuard, which analyzes phone calls for vishing attacks. I'll do a quick overview of each, then play the demo."

---

## Slide 2 — Roomify (~30 s)

"Roomify generates photorealistic interior renders from structured room descriptions — no 3D software required. The dataset is a 200-image curated subset of SUN RGB-D, used as depth and Canny edge references for ControlNet conditioning. The approach is Stable Diffusion 1.5 with ControlNet, comparing three prompt strategies across a 90-image sweep on an A100. Results: mean CLIP alignment 0.27, LPIPS diversity 0.76. Key finding: styleAnchored and descriptive prompts outperform minimal, but depth conditioning trades CLIP score for spatial structure — 9 of the top 10 runs are uncontrolled. Limitation: automated metrics only, no human perceptual study. Next step: upgrade to SDXL and run a user study."

---

## Slide 3 — Roomify Demo

*(video plays)*

---

## Slide 4 — VishGuard (~30 s)

"VishGuard detects vishing attacks in phone recordings. The dataset is 100 audio clips — 50 real LibriSpeech, 50 synthetic — plus 50 hand-labeled call scripts for tactic evaluation. The pipeline chains Whisper ASR, a deepfake detector, and Qwen LLM for tactic classification. Results: anti-spoof accuracy of 0.83 with zero false positives, tactic macro-F1 of 0.604, mean latency 22 seconds on a T4 GPU. Key finding: four rounds of prompt engineering lifted F1 from 0.024 to 0.604 — co-occurrence negative examples were the unlock. Limitation: 34% miss rate on synthetic voice due to distribution shift. Future work: ensemble detectors and a larger LLM."

---

## Slide 5 — VishGuard Demo

*(video plays)*
