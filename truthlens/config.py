"""Central configuration for TruthLens.

Every model is OPTIONAL. If a model is missing or fails to load, the
corresponding indicator is skipped and the gap is written into the report's
"Limitations" section, so the pipeline never crashes during a demo.
"""
from pathlib import Path

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = ROOT / "cases"

# Hugging Face model candidates, tried in order. First one that loads wins.
MODELS = {
    # Whole-image "AI generated vs real photo" classifier
    "image_ai": [
        "haywoodsloan/ai-image-detector-deploy",   # SwinV2, retrained on recent generators (2026); see VALIDATED_MODELS
        "Organika/sdxl-detector",                  # fallback only: saturates (≈1.0) on compressed real video
        "umm-maybe/AI-image-detector",
    ],
    # Face-crop deepfake classifier (used on faces in images and video frames)
    "face_deepfake": [
        "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "dima806/deepfake_vs_real_image_detection",
    ],
    # Synthetic / cloned speech classifier
    "audio_deepfake": [
        "MelodyMachine/Deepfake-audio-detection-V2",
        "mo-thecreator/Deepfake-audio-detection",
    ],
    # Image <-> text consistency
    "clip": ["openai/clip-vit-base-patch32"],
    # Speech-to-text with timestamps
    "asr": ["openai/whisper-base"],
}

# How much each detector family is trusted when aggregating (0..1).
# These are deliberately conservative and are reported to the user.
RELIABILITY = {
    "metadata_ai_signature": 0.90,   # e.g. Stable Diffusion params in PNG chunk
    "metadata_editor": 0.55,
    "metadata_missing": 0.15,
    "metadata_dates": 0.40,
    "ela": 0.35,
    "noise": 0.35,
    "copy_move": 0.70,
    "fft": 0.25,
    "ml_image_ai": 0.40,        # lowered: fires on compression, filters, burned-in text
    "ml_face_deepfake": 0.35,
    "video_temporal": 0.40,
    "video_duplicate_frames": 0.45,
    "video_metadata": 0.35,
    "video_watermark": 0.30,   # visible corner overlay: a lead for a human to read, not proof on its own
    "audio_ml": 0.30,          # lowered: model is unreliable on real-world audio (see HANDOFF P2)
    "audio_bandwidth": 0.20,
    "audio_digital_silence": 0.45,
    "audio_splice": 0.40,
    "text_language": 0.15,
    "xmodal_caption_visual": 0.45,
    "xmodal_caption_transcript": 0.40,
    "xmodal_dates": 0.55,
    "xmodal_corroboration": 0.65,
    "xmodal_ml_agreement": 0.45,   # face and voice classifiers agree; still ML-vs-ML
}

# Models that separated real from AI media in tools/eval_detectors.py. Their consistent, confident output may lift
# an ML-only finding to "Moderate" (never "Strong": that still needs independent forensic evidence).
# Re-run the evaluation with more files (aim for 10+ real and 10+ AI) and update this list with the results.
VALIDATED_MODELS = {
    # 2026-11-19 re-test with 2 real / 4 AI files: haywoodsloan/ai-image-detector-deploy AUC 0.62, FPR 0.50
    # (scored a real screen-captured FaceTime call 0.998). Not reliable enough to lift ML-only findings, so none
    # is listed. Add a model back only after it separates 10+ real from 10+ AI files in tools/eval_detectors.py.
}
VALIDATED_RELIABILITY = 0.55   # replaces RELIABILITY["ml_image_ai"] when the loaded model is validated

VIDEO_MAX_SAMPLED_FRAMES = 32     # frames sent to ML models
VIDEO_TEMPORAL_MAX_FRAMES = 3000  # frames scanned for temporal checks
AUDIO_CHUNK_SECONDS = 4.0
