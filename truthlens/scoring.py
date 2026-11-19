"""Evidence aggregation into a calibrated-style likelihood with explicit uncertainty.

Method (stated in the report):
  * each manipulation indicator contributes  e = score x reliability
  * contributions combine as a noisy-OR:     P = 1 - prod(1 - e)
  * authenticity-supporting signals dampen:  P *= (1 - 0.5 * max(a))
  * confidence in the assessment depends on how many independent detector families
    actually ran, whether ML models were available, and whether modalities agree.
The weights are expert-set heuristics, NOT trained on a benchmark; the report says so.
"""
from __future__ import annotations

import numpy as np

BANDS = [(0.20, "No significant indicators", "green"),
         (0.45, "Inconclusive — weak indicators", "amber"),
         (0.70, "Moderate indicators of manipulation", "orange"),
         (1.01, "Strong indicators of manipulation", "red")]


def band(p):
    for th, label, color in BANDS:
        if p < th:
            return label, color
    return BANDS[-1][1], BANDS[-1][2]


# Detector families. Indicators inside one family are not independent (e.g. three
# ML classifiers reacting to the same compression artifacts), so they are combined
# with dominance rather than noisy-OR. Families themselves are combined with noisy-OR.
FAMILY = {
    "metadata_ai_signature": "provenance", "metadata_editor": "provenance",
    "metadata_missing": "provenance", "metadata_dates": "provenance", "video_metadata": "provenance",
    "video_watermark": "provenance", "audio_format": "provenance",
    "ela": "pixel", "noise": "pixel", "copy_move": "pixel", "fft": "pixel",
    "video_temporal": "temporal", "video_duplicate_frames": "temporal",
    "audio_bandwidth": "audio_signal", "audio_digital_silence": "audio_signal", "audio_splice": "audio_signal",
    "ml_image_ai": "ml", "ml_face_deepfake": "ml", "audio_ml": "ml", "xmodal_ml_agreement": "ml",
    "text_language": "context", "xmodal_caption_visual": "context",
    "xmodal_caption_transcript": "context", "xmodal_dates": "context",
    "xmodal_corroboration": "corroboration",
}
ML_FAMILIES = {"ml"}
ML_ONLY_CAP = 0.44          # likelihood ceiling when no non-ML family is present (keeps it in the Inconclusive band)
VALIDATED_CAP = 0.66        # ceiling when a validated detector is confident: "Moderate", never "Strong"
VALIDATED_MIN_SCORE = 0.85  # validated detector must be this confident (and consistent) to lift the ML-only cap
ML_ONLY_MIN_E = 0.25        # a non-ML family must reach this to count as independent evidence
WITHIN_FAMILY_WEIGHT = 0.3  # contribution of each additional indicator in the same family


UNVALIDATED_DISCOUNT = 0.35  # weight kept by unvalidated ML signals when a validated detector confidently disagrees
ML_DETECTORS = {"ml_image_ai", "ml_face_deepfake", "audio_ml", "xmodal_ml_agreement"}


def _family_scores(man, contradicted: bool = False) -> dict:
    """Combine indicators per family: strongest dominates, others add a little."""
    groups: dict[str, list[float]] = {}
    for i in man:
        w = i.score * i.reliability
        if contradicted and i.detector in ML_DETECTORS and not getattr(i, "validated", False):
            w *= UNVALIDATED_DISCOUNT
        groups.setdefault(FAMILY.get(i.detector, i.detector), []).append(min(0.95, w))
    out = {}
    for fam, es in groups.items():
        es = sorted(es, reverse=True)
        prod = 1.0
        for e in es[1:]:
            prod *= (1 - WITHIN_FAMILY_WEIGHT * e)
        out[fam] = 1 - (1 - es[0]) * prod
    return out


def aggregate(indicators, families_run: set, models_missing: int = 0) -> dict:
    man = [i for i in indicators if i.supports == "manipulation"]
    aut = [i for i in indicators if i.supports == "authenticity"]
    # A validated AI detector that confidently says "not AI-generated" outweighs unvalidated ML detectors
    # (face / voice models that our evaluation showed cannot separate real from AI).
    contradicted = any(getattr(i, "validated", False) and i.detector == "ml_image_ai" and i.score >= 0.8 for i in aut)
    fam_e = _family_scores(man, contradicted)
    # "Looks real" from an AI-generation detector only speaks against AI generation. It must not weaken
    # splice / edit / provenance evidence: a real photo with a pasted region is still "not AI-generated".
    a_ml = max([i.score * i.reliability for i in aut if i.detector in ML_DETECTORS], default=0.0)
    a_other = max([i.score * i.reliability for i in aut if i.detector not in ML_DETECTORS], default=0.0)
    if "ml" in fam_e:
        fam_e["ml"] *= (1 - 0.5 * a_ml)
    p = 1 - float(np.prod([1 - e for e in fam_e.values()])) if fam_e else 0.0
    p *= (1 - 0.5 * a_other)
    a = max(a_ml, a_other)

    # ML-only cap: classifiers alone must not produce a confident verdict.
    non_ml_strength = max([e for f, e in fam_e.items() if f not in ML_FAMILIES], default=0.0)
    ml_only = bool(fam_e) and non_ml_strength < ML_ONLY_MIN_E
    validated_ml = any(getattr(i, "validated", False) and i.detector == "ml_image_ai" and i.score >= VALIDATED_MIN_SCORE
                       for i in man)
    if ml_only:
        p = min(p, VALIDATED_CAP if validated_ml else ML_ONLY_CAP)
    p = float(np.clip(p, 0.01, 0.97))

    strong = [i for i in man if i.score * i.reliability >= 0.3]
    corroborated = any(i.detector == "xmodal_corroboration" and i.supports == "manipulation" for i in indicators)
    modalities_flagging = {i.modality for i in strong}
    conflict = bool(strong) and a >= 0.3
    conf = 0.25 + 0.07 * min(len(families_run), 8) - 0.06 * models_missing
    conf += 0.12 if corroborated or len(modalities_flagging) >= 2 else 0
    conf += 0.08 if (len(strong) >= 2 and not conflict and not ml_only) else 0
    conf -= 0.12 if conflict else 0
    if any(i.detector == "metadata_ai_signature" for i in man):
        conf += 0.15
    if ml_only:
        # ML-only: Low confidence, or Medium when a validated detector is confident and consistent
        conf = min(conf, 0.5) if validated_ml else min(conf, 0.44)
    conf = float(np.clip(conf, 0.15, 0.9))
    conf_label = "High" if conf >= 0.7 else "Medium" if conf >= 0.45 else "Low"
    width = round((1 - conf) * 30)
    label, color = band(round(p, 2))   # label matches the rounded percentage shown to the user
    top = sorted(man, key=lambda i: -i.score * i.reliability)[:6]
    return {"likelihood": round(p * 100), "range": [max(0, round(p * 100) - width), min(100, round(p * 100) + width)],
            "band": label, "color": color, "confidence": conf_label, "confidence_value": round(conf, 2),
            "n_manipulation_indicators": len(man), "n_authenticity_signals": len(aut),
            "conflicting_signals": conflict, "corroborated": corroborated,
            "ml_only": ml_only, "validated_ml": validated_ml,
            "family_scores": {f: round(e, 3) for f, e in fam_e.items()},
            "top": [i.id for i in top]}


def recommended_checks(indicators, modalities) -> list[str]:
    det = {i.detector for i in indicators if i.supports == "manipulation"}
    checks = []
    if "metadata_ai_signature" in det or "ml_image_ai" in det or "fft" in det:
        checks.append("Run a reverse image search (Google Lens, TinEye) to find the earliest copy and whether it was published as AI art.")
    if {"ml_image_ai", "ml_face_deepfake", "video_watermark"} & det or {"image", "video"} & set(modalities or []):
        checks.append("Check for Google's SynthID watermark (Gemini app: “Was this made with Google AI?”, or SynthID Detector). "
                      "It survives re-compression and identifies Veo / Gemini / Imagen output; a negative result does not rule out other generators.")
    if "video_watermark" in det:
        checks.append("Read the zoomed corner overlay: a generator name (Veo, Kling, Hailuo, Meta AI, Sora) is direct provenance evidence.")
    if {"ela", "noise", "copy_move"} & det:
        checks.append("Zoom into the boxed regions at 200–400% and check shadows, reflections, edges and lighting direction for consistency.")
    if "copy_move" in det:
        checks.append("Compare the 'source' and 'copy' boxes: identical texture/noise in both confirms cloning.")
    if "ml_face_deepfake" in det:
        checks.append("Inspect flagged faces frame-by-frame for blending seams at the jaw/hairline, teeth and eye reflections.")
    if {"video_temporal", "video_duplicate_frames", "audio_splice", "audio_digital_silence"} & det:
        checks.append("Play the video/audio around each flagged timestamp at 0.25x and check whether speech, motion and background noise are continuous.")
    if "audio_ml" in det or "audio_bandwidth" in det:
        checks.append("Compare the voice with a verified recording of the same speaker; if possible confirm the statement with the person directly.")
    if {"xmodal_dates", "xmodal_caption_visual", "xmodal_caption_transcript"} & det:
        checks.append("Verify the claim's date/location independently (news archives, weather records, landmarks visible in the media).")
    checks.append("Obtain the original file from the first uploader or device — re-shared copies lose metadata and compress away evidence.")
    checks.append("Treat this report as investigative leads; a human examiner must confirm before any conclusion is published.")
    return checks
