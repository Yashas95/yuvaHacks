# What TruthLens built vs. what it reuses

TruthLens uses some pretrained models and well-known forensic techniques. This page lists each of them with its
source and license, and says exactly what we wrote ourselves. Nothing here is copied from another forensics
project. Every pretrained model is loaded from its public release under its own license, unmodified.

## Pretrained models (used as-is, not retrained, not presented as ours)

Licenses are as listed on each model's Hugging Face page on 2026-11-19.

| Role in TruthLens | Model | Author | License | How we use it |
|---|---|---|---|---|
| AI-image score (images, video frames) | `haywoodsloan/ai-image-detector-deploy` | haywoodsloan | Apache-2.0 | one input among many, capped (see below) |
| Fallback AI-image score | `Organika/sdxl-detector` | Organika | CC-BY-NC-3.0 (non-commercial) | only if the first model fails to load |
| Face deepfake score | `prithivMLmods/Deep-Fake-Detector-v2-Model` (ViT, from `google/vit-base-patch16-224-in21k`) | prithivMLmods | Apache-2.0 | **scores shown, not counted**: failed our evaluation |
| Synthetic-voice score | `MelodyMachine/Deepfake-audio-detection-V2` | MelodyMachine | Apache-2.0 | **switched off**: it failed our evaluation (shown, not used) |
| Caption ↔ image match | `openai/clip-vit-base-patch32` | OpenAI | MIT (OpenAI CLIP repository; the Hugging Face card has no license tag) | cross-modal context check |
| Speech-to-text | `openai/whisper-base` | OpenAI | Apache-2.0 | transcript for caption ↔ speech check |
| Face detection | YuNet (`face_detection_yunet_2023mar.onnx`) | OpenCV Zoo | MIT | locating faces to crop |

**These models are not the product.** They are its weakest evidence. `truthlens/scoring.py` treats all of them as
one "ML" family. Unless independent non-ML evidence exists, they can never push a result past 44 % (Inconclusive).
We measured them on our own files (`tools/eval_detectors.py`, results in [METHODOLOGY.md](METHODOLOGY.md)) and
recorded where they fail. We swapped one out because it was saturated (AUC 0.25), withdrew another's "validated"
status after a false positive, and disabled the voice model completely.

## Published techniques we implemented ourselves

These are standard ideas from forensics literature. The code in `truthlens/` is our own implementation.

| Technique | Idea from | Our implementation |
|---|---|---|
| Error Level Analysis | N. Krawetz, *A Picture's Worth* (Black Hat, 2007) | `image_forensics.ela`: re-save, difference, **texture-normalized** with robust z-scores, then localized into boxed regions |
| Sensor-noise inconsistency | the camera-noise (PRNU) idea from Lukáš, Fridrich & Goljan (2006) | `image_forensics.noise_map`: block-wise noise residual, robust z-score, region boxes. This is a simplified local-consistency check, not full PRNU camera identification |
| Copy-move detection | keypoint-matching approaches in the literature | `image_forensics.copy_move`: OpenCV SIFT keypoints matched within the same image, clustered, boxed |
| Periodic-artifact spectrum | known traces of GAN upsampling and resampling | `image_forensics.spectrum`: FFT peak analysis |
| Temporal cut / duplicate detection | standard shot-boundary detection | `video_forensics`: full-frame difference scan, ignoring recorder start/stop, frozen-frame runs |
| Audio splice cues | standard audio forensics (silence gaps, noise floor) | `audio_forensics`: digital-silence gaps inside speech, noise-floor jumps, band limit |

## Designed and written by us

- **Evidence scoring** (`scoring.py`): detector families, strongest-in-family plus noisy-OR across families, the
  ML-only cap, authenticity damping scoped to each family, confidence and plausible range. Designed from the
  failures we measured (METHODOLOGY §1–4).
- **Cross-modal correlation** (`correlate.py`): a video cut matched to a *localized* audio edit; caption vs speech
  only for speech claims; claim date vs file date; same scene across files.
- **Non-ML provenance leads**: AI-generator signatures in EXIF/XMP/PNG chunks, screenshot vs camera origin, visible
  corner-watermark detector using the temporal median (`find_static_overlay`), TTS-native sample-rate lead.
- **Evaluation harness** (`tools/eval_detectors.py`): AUC, FPR and TPR, plus saturation detection on labelled files.
  Every model decision above came from it.
- **Explainable report** (`report.py`, `plain.py`, `assets/`): Simple and Expert views, a plain-English answer per
  file, and an offline case assistant (`chat.js`, rule-based, no LLM).
- **Chain of custody** (`evidence.py`): SHA-256 at intake, read-only copies, re-hash after analysis, analysis log.
- **Browser extension** (`extension/`): check a post, reel or photo on Instagram or any site in place (see below).
- **Web app** (`app.py`, `web/index.html`): upload, progress and the verdict all in one card, and hosted mode with
  per-visitor case isolation.

About 5,000 lines of Python, JavaScript and HTML in this repository were written for TruthLens. The pretrained
weights are downloaded at setup time and are not part of the repository.

## What makes TruthLens different from "run a deepfake classifier"

1. **It refuses to over-claim.** A classifier alone cannot produce "Strong". That needs provenance, pixel,
   temporal or audio-signal evidence, and the report explains why.
2. **It localizes and explains.** Every finding says where it is (region, frame, timestamp, audio segment), why it
   matters, how reliable the detector is, and what innocent explanations exist.
3. **It is cross-modal.** It checks picture against sound, caption against picture, caption against speech, and
   claim date against file date.
4. **It works where people actually see media.** The Chrome extension checks a post where it appears, including
   reels that the site streams and that can't be downloaded.
5. **It is measured.** Model choices are backed by our own dated evaluation numbers, including the ones that went
   against us.

## Other third-party software

FastAPI (MIT), Uvicorn (BSD-3), PyTorch (BSD-3), Hugging Face Transformers (Apache-2.0), OpenCV (Apache-2.0),
NumPy (BSD-3), SciPy (BSD-3), Pillow (MIT-CMU), Matplotlib (PSF-based), torchaudio (BSD-2), ffmpeg (LGPL/GPL, called as an external program). The landing-page
hand motif comes from Michelangelo's *The Creation of Adam* (public domain, via Wikimedia Commons).
