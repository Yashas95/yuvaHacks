# TruthLens — Handoff for Claude Code

**Context:** 24-hour hackathon (InnovEdge Hackathon 2026), problem statement: *TruthLens — Multimodal Digital
Forensics & Deepfake Detection*. Build a working prototype that takes image/video/audio/text evidence and produces
an **explainable forensic report**: flagged regions/frames/segments, why each was flagged, calibrated confidence,
cross-modal correlation, source trace (hashes), limitations. Judges value explainability and honest uncertainty
over raw accuracy. The rubric says "a detection score is an indicator, not a verdict".

Team priorities: **output quality over UI polish**, and don't leave everything to the final hour.

Machine: **MacBook Air M4, 16 GB RAM, macOS, Apple Silicon (MPS available)**. Disk was full; ~14 GB free now.
Do not download large models without checking `df -h /System/Volumes/Data`.

---

## 1. How to run

```bash
cd ~/TruthLens
.venv/bin/python app.py                 # website -> http://localhost:8000
.venv/bin/python cli.py file1.jpg clip.mp4 --context "claimed caption"   # CLI, prints verdict + report path
.venv/bin/python download_models.py     # pre-download all HF models (offline demo)
HF_HUB_OFFLINE=1 .venv/bin/python app.py  # force offline (use at the venue)
tools/run_demo_batch.sh                 # runs every file in demo_samples/ and prints verdicts
.venv/bin/python tools/make_test_assets.py  # regenerates test_assets/ with PLANTED manipulations
```

- venv: `.venv` = **Python 3.15**, torch 2.15 (MPS ok), transformers 5.23, opencv-python-headless **5.1** (no
  Haar cascades in OpenCV 5 — we use YuNet ONNX at `weights/face_detection_yunet_2023mar.onnx`), scipy,
  fastapi, uvicorn, matplotlib. ffmpeg/ffprobe at `/opt/homebrew/bin`. No exiftool, no ollama.
- The server may still be running from the previous session (`pgrep -f app.py`; kill before restarting).
- Each analysis creates `cases/TL-<timestamp>-<id>/` with `evidence/` (read-only copies), `artifacts/`
  (heatmaps, timelines, spectrograms), `report.json`, `report.html` (self-contained, base64 images).

## 2. Architecture (all in `truthlens/`)

| File | Role |
|---|---|
| `config.py` | Model candidate lists (tried in order), `RELIABILITY` per detector family, limits |
| `models.py` | Lazy, fault-tolerant HF pipeline loading; `fake_probability()` maps labels to P(synthetic); `STATUS` feeds report |
| `evidence.py` | `Case` (intake → SHA-256 → read-only copy → re-verify at end; analysis log = chain of custody), `Indicator` dataclass |
| `image_forensics.py` | Metadata (EXIF/XMP/PNG chunks/C2PA, AI-generator signatures, editors), ELA (texture-normalized, localized boxes), sensor-noise inconsistency, SIFT copy-move, FFT periodic artifacts, ML AI-image classifier, YuNet faces + face deepfake classifier. Each detector is wrapped in `safe()` so one failure never kills the file |
| `video_forensics.py` | ffprobe container tags (AI tool / editor signatures — whole-word match only), A/V duration mismatch, full-frame temporal scan (discontinuities = cuts/splices, duplicated frames), ≤32 sampled frames → image AI + face models, timeline chart, flagged-frame strip, audio track → audio pipeline |
| `audio_forensics.py` | Digital-silence gaps, background-noise-floor jumps (splice, with `_refine_change`), band-limit cutoff (TTS vocoder hint), ML synthetic-speech per 4 s chunk, Whisper transcript with timestamps, waveform+spectrogram with shaded flags |
| `text_forensics.py` | Claim extraction (dates, years, recency words, entities, numbers), persuasion-language flags |
| `correlate.py` | Cross-modal: CLIP caption↔image/keyframes (vs distractor captions), CLIP scene labels, claim date vs EXIF/creation_time, claim vs transcript, **video cut ↔ audio anomaly at same timestamp**, synthetic voice over clean video (dub hypothesis), dHash same-scene links across files |
| `scoring.py` | Aggregation → likelihood %, plausible range, band, confidence (High/Med/Low), recommended human checks |
| `pipeline.py` | Orchestrates ingest → analyze → text → correlate → score → integrity re-verify → report |
| `report.py` | HTML report: overall assessment, key findings, evidence inventory + hashes, per-file visuals + indicator table (ID, location, observation, why, signal bar, reliability bar, alternative explanations), transcript (flag overlap highlighted), cross-modal table, claim analysis, recommended checks, limitations, analysis history, models used. Print CSS → "Download PDF" = window.print; JSON download embedded |
| `app.py` (root) | FastAPI: `/` upload page (drag-drop + claim box + progress polling), `/api/analyze`, `/api/status/{job}`, `/case/{id}`, `/case/{id}/report.json`, `/api/cases` |

Models (all downloaded to `~/.cache/huggingface`, verified loading):
- image_ai: `Organika/sdxl-detector` (labels artificial/human)
- face_deepfake: `prithivMLmods/Deep-Fake-Detector-v2-Model` (Realism/Deepfake)
- audio_deepfake: `MelodyMachine/Deepfake-audio-detection-V2` (fake/real) — **unreliable, see §4**
- clip: `openai/clip-vit-base-patch32` — downloaded at handoff time, **correlation with CLIP not yet tested end-to-end**
- asr: `openai/whisper-base` (works, CPU)

## 3. Test results so far

Synthetic planted tests (`test_assets/`) — all detected at the right location:
- spliced_photo.jpg: ELA + noise boxes at the exact paste (620,420) and clone (420,150). Copy-move didn't fire (too few SIFT keypoints on a synthetic texture — fine on real photos).
- voice_note.wav: digital silence 4.01–4.58 s ✓, noise-floor jump ✓ (reported 5.75 s; true 7.0 s but voice covers 5.9–7.1 so it's ambiguous), band-limit 8.7 kHz ✓.
- joined_clip.mp4: cut at exactly 4.00 s ✓, cross-modal "visual cut + audio anomaly at same moment" ✓. Fixed false positive: "veo" was matching random bytes.

Real samples (`demo_samples/`, copied from ~/Downloads):
| File | Truth | Result | Verdict |
|---|---|---|---|
| tryonme-1774514325128.png | AI (user's TryOnMe app) | 95% Strong — generator signature in metadata + C2PA + classifier | ✅ |
| Wan_Avatar_PhotoAnimate.mp4 | AI avatar video | 97% Strong — frames, face, voice, cut↔audio corroboration | ✅ |
| freevideo-abyss-bridge-greatsword-duel-max.mp4 | likely AI-generated | 93% Strong (ML-only evidence) | ✅ probably |
| IMG_2770.jpg | EXIF says "Screenshot"; very polished studio image next to TryOnMe outputs — truth unknown | 57% Moderate (classifier 99.99%) | ? |
| whatsapp_video.mp4 | **real** selfie talking-head reel, burned-in captions | 94% Strong | ❌ FALSE POSITIVE |
| tts_statement.wav | **synthetic** (macOS `say`) | 8% | ❌ MISSED |

## 4. Open problems — do these next, in order

### P1. Fix scoring so ML-only evidence can't produce "Strong" (in progress when handed off)
Cause of the WhatsApp false positive: image AI, face and audio classifiers all react to the same nuisance
(compression, beauty filter, burned-in captions) but `aggregate()` combines them with noisy-OR as if independent,
plus the `xmodal` "Both face and voice flagged as synthetic" adds again.
Planned fix in `scoring.py`:
1. Group detectors into **families**: provenance (`metadata_*`), pixel (`ela, noise, copy_move, fft`),
   temporal (`video_*`), audio_signal (`audio_bandwidth, audio_digital_silence, audio_splice`),
   ml (`ml_image_ai, ml_face_deepfake, audio_ml` + the ML-ML agreement finding), context (`text_language,
   xmodal_caption_*, xmodal_dates`), corroboration (`xmodal_corroboration` from forensic-forensic agreement).
2. Within a family: `e_f = 1 - (1-e1) * Π(1 - 0.3*e_k)` (strongest dominates, others add a little).
   Across families: noisy-OR.
3. **ML-only cap**: if no non-ML family has e ≥ 0.15, cap likelihood at ~45–50% and confidence at Low, and add
   a report line "Finding rests on ML classifiers only — no independent forensic/metadata evidence."
4. In `correlate.py`, give "Both face and voice flagged as synthetic" its own detector key (e.g.
   `xmodal_ml_agreement`, add to `RELIABILITY`) and put it in the ml family.
5. Lower `RELIABILITY`: ml_image_ai 0.40, ml_face_deepfake 0.35, audio_ml 0.30.
Expected after fix: WhatsApp → Inconclusive; Wan stays Strong (has temporal + cut↔audio corroboration);
TryOnMe stays Strong (metadata signature). Re-run `tools/run_demo_batch.sh` to confirm. Don't tune to individual files.

### P2. Replace / ensemble the audio deepfake model
`MelodyMachine/Deepfake-audio-detection-V2` saturates to 0/1 and is inverted on real-world audio (TTS→real,
real voice→fake). `tools/test_audio_models.py` was running at handoff (log: `.audio_test.log`), testing
`mo-thecreator/Deepfake-audio-detection`, `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`,
`garystafford/wav2vec2-deepfake-voice-detector` on TTS vs real vs Wan. Pick the one that separates them; update
`config.MODELS["audio_deepfake"]`. If none works, keep the model but drop `audio_ml` reliability to ~0.2 and say so
in limitations. Consider adding prosody features (pitch variance via autocorrelation) as a non-ML signal.

### P3. Test CLIP cross-modal end-to-end
Run e.g. `cli.py demo_samples/Wan_Avatar_PhotoAnimate.mp4 --context "Video of a news anchor in Delhi recorded today, 19 November 2026"`
and a matching vs mismatching caption on an image. Check thresholds in `correlate.py` (cap < 0.19 mismatch,
≥ 0.24 consistent — CLIP ViT-B/32 cosine). Verify `_clip_text_sim` works in transformers 5 (uses
`model.text_model` + `model.text_projection`).

### P4. Second image-AI model for an ensemble (optional)
`umm-maybe/AI-image-detector` is already a fallback candidate. Running both and averaging reduces single-model false
positives. Keep disk usage in mind (~350 MB).

### P5. Deliverables for teammates/judges
- `README.md` (what it does, architecture diagram, how to run, how scoring works, limitations).
- `setup.sh`: create venv (Python ≥3.12), `pip install -r requirements.txt`, download YuNet ONNX
  (`curl -L -o weights/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`),
  `python download_models.py`.
- `.gitignore`: `.venv/ cases/ demo_samples/ test_assets/ .*.log`.
- Prepare a **demo script**: (1) TryOnMe image → metadata signature, (2) planted spliced photo → ELA/noise boxes,
  (3) Wan avatar video + misleading caption → face+voice+cut↔audio+caption mismatch, (4) a real clip → low/inconclusive
  with honest explanation. Show the report's "why" and "alternative explanations" columns — that's the rubric.

## 5. Gotchas
- `device_commit`-style parallel edits bit us once: always verify file content (`md5`) after syncing.
- OpenCV 5: no `CascadeClassifier`; use YuNet (`cv2.FaceDetectorYN`). SIFT is available.
- transformers 5 pipelines: `pipe(img, top_k=10)`, audio `pipe({"raw": arr, "sampling_rate": 16000}, top_k=5)`.
- Whisper prints harmless deprecation warnings.
- Never search raw compressed bytes for short keywords (false matches); only whole-word in readable metadata, or ≥9-char strings in raw bytes.
- Only use the demo media files; never analyse or publish unrelated personal files.
- Scores are heuristic; the report already states this in Limitations. Keep that honesty — it's what judges grade.
