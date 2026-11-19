# How we found what actually detects manipulation

This is the dated record of what we tried, what we measured, and what we changed as a result.
Every number below was produced by this code on the files listed. With a handful of files per class
these are **anecdotes, not benchmarks** — the point is the method: measure, find the failure, fix the rule, re-measure.

Test set used throughout (truth known):

| File | Truth | Notes |
|---|---|---|
| `whatsapp_video.mp4` | real | selfie talking-head reel, burned-in captions |
| FaceTime screen capture (2.6 s, 2650×2160) | real | re-captured video call, Apple Core Media recorder |
| `Video-23481.mp4` | AI | Google Veo 3 (Gemini), re-uploaded via Instagram (572×1016, 465 kbps, metadata stripped) |
| `Wan_Avatar_PhotoAnimate.mp4` | AI | Wan avatar video |
| `tryonme-….png` | AI | output of an AI try-on app |
| `ReelAudio-34978.mp3` | AI | AI voice from a reel, 24 kHz |
| `tts_statement.wav` | AI | macOS `say` text-to-speech, 44.1 kHz |
| `test_assets/*` | edited | planted manipulations from `tools/make_test_assets.py` |

Real-person media is not committed to the repository (privacy); `test_assets/` is synthetic and is.

---

## 1. Correlated ML detectors were counted as independent evidence

**Observed.** The real WhatsApp selfie scored **94 % "Strong"**. The AI-image, face-swap and synthetic-voice
classifiers all fired, and the original aggregator combined them with noisy-OR as if they were independent
witnesses. They are not: compression, beauty filters and burned-in captions trip all three at once.

**Change** (`truthlens/scoring.py`).
- Detectors are grouped into **families** (provenance, pixel, temporal, audio-signal, ML, context, corroboration).
  Inside a family the strongest indicator dominates (`1 − (1−e₁)·Π(1 − 0.3·eₖ)`); families combine with noisy-OR.
- **ML-only cap:** if no non-ML family is strong enough, likelihood is capped at 44 % (Inconclusive) and confidence at Low.
- "Face and voice both flagged" became its own ML-family key (`xmodal_ml_agreement`) instead of counting as corroboration.

**Result.** WhatsApp selfie 94 % Strong → **44 % Inconclusive**. TryOnMe image stayed Strong (its evidence is a
generator signature in metadata, which is not ML).

## 2. The AI-image detector could not tell real from AI video

**Observed.** The Veo 3 video was missed. Per-frame scores showed why:

| Detector | Real WhatsApp | Veo 3 | Separates? |
|---|---|---|---|
| `Organika/sdxl-detector` (frames) | 1.000 on every frame | 1.000 on every frame | no — saturated |
| `prithivMLmods/Deep-Fake-Detector-v2-Model` (faces) | mean 0.655 | mean 0.662 | no |
| `MelodyMachine/Deepfake-audio-detection-V2` (voice) | 1.00 | 0.34 | inverted |

**Change.** Built `tools/eval_detectors.py` (offline unless `--allow-download`): runs candidate models on labelled
real/AI files and reports AUC, false-positive and true-positive rate at 0.5, and flags saturated models.

| Model (1 real / 4 AI) | AUC | FPR@0.5 | Note |
|---|---|---|---|
| `Organika/sdxl-detector` | 0.25 | 1.00 | saturated on real media |
| `Ateeqq/ai-vs-human-image-detector` | 0.25 | 1.00 | saturated on real media — deleted |
| `haywoodsloan/ai-image-detector-deploy` | 1.00 | 0.00 | real 0.065 vs AI 0.96–1.00 |

`haywoodsloan` became the primary AI-image model and was marked *validated*: a validated detector that is confident
across the whole file could lift an ML-only finding to Moderate (never Strong). Veo 3 went 44 % → 66 %.

**Re-test that reversed it.** A second real video (the FaceTime screen capture) scored **0.998** with `haywoodsloan`.
With 2 real / 4 AI files: **AUC 0.62, FPR 0.50**. The validation had rested on a single real file, so the
validated status was removed (`VALIDATED_MODELS` is empty, with the reason recorded in `config.py`). The model stays
primary because it is no worse than the alternatives, but it can no longer lift a finding on its own.

## 3. Scoring bugs found by the FaceTime false alarm (88 % → 44 %)

The real FaceTime capture scored **88 % Strong**. Three rule bugs explained most of it:

1. **Fake corroboration.** "Visual cut and audio anomaly at the same moment" matched cuts against the voice
   classifier's "synthetic" segment, which covered the entire 0–2.6 s clip — any cut matches that.
   *Fix* (`correlate.py`): only localized audio **edit** evidence (digital-silence gaps, noise-floor splices) counts,
   and segments longer than 40 % of the clip are ignored.
2. **Start-up "cut".** A discontinuity at 0.02 s was the first frame of the recording.
   *Fix* (`video_forensics.py`): ignore discontinuities in the first and last 0.25 s.
3. **Weak evidence unlocking ML.** One weak cut (0.45 × 0.40 = 0.18) counted as "independent evidence" and removed
   the ML-only cap. *Fix:* a non-ML family must reach 0.25 to count.

Consequence we report rather than hide: the Wan avatar video's earlier "Strong" came from the same fake
corroboration, so it is now Inconclusive too.

## 4. "Not AI-generated" was cancelling splice evidence

**Observed.** With the better AI-image model, the planted spliced photo dropped 69 % → 38 %: the model correctly
said "not AI-generated", and that authenticity signal damped the ELA / sensor-noise evidence.
A real photo with a pasted region is still "not AI-generated".

**Change.** Authenticity signals from ML detectors only damp the ML family; other authenticity signals
(e.g. caption matches the image) damp the whole score. Pinned by `tests/test_core.py`.

## 5. The synthetic-voice model is wrong on every file we have

| File | Truth | `MelodyMachine/Deepfake-audio-detection-V2` |
|---|---|---|
| ReelAudio | AI | 0 % synthetic |
| macOS TTS | AI | 0 % |
| WhatsApp selfie | real | 100 % |
| FaceTime capture | real | 100 % |

Flipping it would be tuning to four anecdotes, so it is **switched off** (`UNRELIABLE_AUDIO_MODELS` in
`config.py`): its scores are shown in the report, marked "failed evaluation, ignored", and never become findings.
Previously the report said the AI reel "leans towards a natural human voice"; now it says it cannot judge by ear.
An earlier test of `mo-thecreator/Deepfake-audio-detection` scored both TTS and real voices as fake.

**Non-ML clue added instead.** Audio stored at **22.05 / 24 kHz** — native output rates of many TTS engines
(e.g. OpenAI TTS and Coqui XTTS at 24 kHz, Piper at 22.05 kHz) — is reported as a weak provenance lead
(reliability 0.25). It fires on the 24 kHz ReelAudio, not on the 44.1 kHz macOS TTS file.

## 6. Other checks we added and how we tested them

- **Visible watermark / logo detector** (`find_static_overlay`). Uses the temporal median: a moving background blurs
  away while an overlay — even semi-transparent — stays sharp; the area around it must be moving. Tested on a copy of
  the Veo video with a planted "Veo" mark: found in the correct corner; no detection on four clean videos. It also
  flags burned-in subtitles, so it is a lead for a human to read, not proof (reliability 0.30).
  A first version (pixel-variance only) missed the planted mark and false-flagged a static object; it was rewritten.
- **SynthID recommendation.** Google embeds an invisible SynthID watermark in Veo / Gemini / Imagen output that
  survives re-compression; the report tells the reviewer to check it (Gemini app or SynthID Detector, where available).
- **Screenshot vs camera origin** (`plain.origin`). Apple writes `Screenshot` into EXIF `UserComment` / XMP;
  both screenshot samples are identified, with camera make/model used for camera photos.
- **Caption vs image (CLIP).** Tested on the Veo video: a matching caption is not flagged; "flood rescue in Chennai"
  is flagged as not matching the visual content.
- **Caption vs speech.** Only runs when the claim is about speech ("said", "announcing", quotes…); a caption that
  describes the scene was previously flagged as "not supported by what is said".

## Current results (2026-11-19, this code)

| File | Truth | TruthLens |
|---|---|---|
| WhatsApp selfie video | real | 40 % Inconclusive · Low |
| FaceTime screen capture | real | 44 % Inconclusive · Low |
| Veo 3 via Instagram | AI | 44 % Inconclusive · Low |
| Wan avatar video | AI | 44 % Inconclusive · Low |
| TryOnMe image | AI | **93 % Strong** · High (metadata signature) |
| ReelAudio (24 kHz) | AI | 48 % Moderate · Medium (sample-rate lead + noise-floor jumps) |
| macOS TTS voice | AI | 9 % No significant indicators — **missed** |
| Planted spliced photo | edited | 45 % Moderate (ELA + sensor-noise boxes at the paste) |
| Planted joined video | edited | **74 % Strong** · High (cut at 4.0 s + audio anomaly) |
| Planted voice note | edited | 32 % Inconclusive (silence gap, noise jump, band limit found) |
| iPhone screenshot | unknown | 5 % · identified as a screenshot |

**What this means.** TruthLens reliably reports *forensic* evidence — provenance tags, splices, cuts, planted
regions — and explains where. It does **not** yet reliably separate AI-generated video or voice from heavily
processed real media using ML alone, and it says so instead of guessing: no false "Strong" on either real video.

## Next steps that the data supports

1. Grow the labelled set to 10+ real and 10+ AI files per modality (ideally after WhatsApp/Instagram re-upload) and
   re-run `tools/eval_detectors.py` before trusting any classifier.
2. Evaluate newer voice-deepfake models (candidates found but not yet tested:
   `abhishtagatya/hubert-base-960h-itw-deepfake`, `garystafford/wav2vec2-deepfake-voice-detector`).
3. Video-native (temporal) generation detectors — research models exist but none are packaged for local use yet.
