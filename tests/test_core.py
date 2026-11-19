"""Fast regression tests for the scoring and plain-language rules (no models, no media files).

Each test pins a decision we made after a real failure; the comment says which one.
Run:  .venv/bin/python tests/test_core.py      (also works with pytest)
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from truthlens.config import RELIABILITY as R  # noqa: E402
from truthlens.plain import origin  # noqa: E402
from truthlens.scoring import ML_ONLY_CAP, aggregate  # noqa: E402


def ind(det, score, supports="manipulation", validated=False):
    return NS(detector=det, score=score, reliability=R[det], supports=supports, modality="x", id=det, validated=validated)


def test_ml_only_is_capped():
    # Real WhatsApp selfie scored 94% "Strong" when three ML classifiers were noisy-OR'd as independent.
    r = aggregate([ind("ml_image_ai", .99), ind("ml_face_deepfake", .9), ind("audio_ml", 1.0),
                   ind("xmodal_ml_agreement", .75)], set())
    assert r["ml_only"] and r["likelihood"] <= ML_ONLY_CAP * 100 and r["confidence"] == "Low"


def test_weak_independent_evidence_does_not_unlock_ml():
    # Real FaceTime capture: one weak "cut" (0.45 x 0.40 = 0.18) must not lift ML-only evidence above the cap.
    r = aggregate([ind("video_temporal", .45), ind("ml_image_ai", 1.0), ind("ml_face_deepfake", .83)], set())
    assert r["ml_only"] and r["likelihood"] <= 44


def test_provenance_signature_is_strong():
    # TryOnMe image: generator signature in metadata is direct evidence, not a statistical guess.
    r = aggregate([ind("metadata_ai_signature", .95), ind("ml_image_ai", .9)], set())
    assert not r["ml_only"] and r["likelihood"] >= 70


def test_not_ai_generated_does_not_cancel_splice_evidence():
    # Planted spliced photo dropped 69% -> 38% when an AI detector's "looks real" damped ELA/noise evidence.
    splice = [ind("ela", 1.0), ind("noise", 1.0)]
    alone = aggregate(splice, set())["likelihood"]
    with_auth = aggregate(splice + [ind("ml_image_ai", .96, supports="authenticity")], set())["likelihood"]
    assert with_auth == alone


def test_correlated_detectors_do_not_stack():
    # Within one family the strongest indicator dominates; adding near-duplicates adds little.
    one = aggregate([ind("ela", .8)], set())["likelihood"]
    three = aggregate([ind("ela", .8), ind("noise", .8), ind("fft", .8)], set())["likelihood"]
    assert three - one < 15


def test_screenshot_origin_from_exif():
    # iPhone / macOS write "Screenshot" into EXIF UserComment.
    ev = {"modality": "image", "name": "IMG_2770.jpg", "indicators": [],
          "metadata": {"exif": {"UserComment": "ASCII\x00\x00\x00Screenshot"}, "png_text": {}}}
    assert origin(ev)["kind"] == "screenshot"


def test_camera_origin():
    ev = {"modality": "image", "name": "x.jpg", "indicators": [], "metadata": {"exif": {"Make": "Apple", "Model": "iPhone 15"}}}
    assert origin(ev)["kind"] == "camera"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
