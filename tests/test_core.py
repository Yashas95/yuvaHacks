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


def _cbor_text(t: bytes) -> bytes:
    return (bytes([0x60 + len(t)]) if len(t) < 24 else bytes([0x78, len(t)])) + t


def test_c2pa_ai_source_type_is_read_exactly(tmp=None):
    """A C2PA manifest declaring trainedAlgorithmicMedia is found and decoded by length, not by keyword.
    Mirrors the Gemini-generated MP3 that scored 40 % before the audio path read provenance (2026-11-20)."""
    import tempfile
    from truthlens.provenance import scan
    uri = b"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    blob = (b"ID3\x04GEOBapplication/c2pa\x00jumbjumdc2pa c2pa.signature " + _cbor_text(b"description")
            + _cbor_text(b"Created by Google Generative AI.") + _cbor_text(b"digitalSourceType") + _cbor_text(uri)
            + _cbor_text(b"parameters") + b"\x06\x03\x55\x04\x0a\x0c\x0aGoogle LLC" + b"\xff" * 64)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(blob)
    p = scan(Path(f.name))
    Path(f.name).unlink()
    assert p["c2pa_present"] and p["ai_source_types"] == ["trainedAlgorithmicMedia"]
    assert p["actions"] == ["Created by Google Generative AI."] and p["signers"] == ["Google LLC"]
    # the same words loose in compressed data, with no manifest, must not count
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"\xff" * 32 + b"Created by Google Generative AI" + b"\xff" * 32)
    q = scan(Path(f.name))
    Path(f.name).unlink()
    assert not q["c2pa_present"] and not q["ai_source_types"]


def test_face_model_that_failed_evaluation_is_not_counted():
    """The face model scored a real selfie like the AI videos (0.81 vs 0.80 / 0.88), so it must not create findings."""
    from truthlens.config import UNRELIABLE_FACE_MODELS
    from truthlens.evidence import Findings
    from truthlens.video_forensics import check_face_ml
    rows = [{"frame": k, "time": k / 25, "p_ai": 0.1, "faces": 1, "p_face_fake": 0.85,
             "face_boxes": [{"bbox": [0, 0, 10, 10], "p_fake": 0.85}]} for k in range(5)]
    counter = iter(range(100))
    out = Findings(NS(id="E01"), lambda: f"E01-{next(counter):02d}", "video")
    metrics = {"face_model": next(iter(UNRELIABLE_FACE_MODELS))}
    assert check_face_ml(out, metrics, rows) == [] and len(out) == 0 and metrics["face_model_unreliable"]
    metrics = {"face_model": "some/validated-model"}
    assert len(check_face_ml(out, metrics, rows)) == 5 and out[0].detector == "ml_face_deepfake"


def test_captured_media_skips_checks_that_would_describe_the_copy():
    """A reel the extension screenshots or records gets ELA / noise / frequency numbers about the capture, not the media."""
    from truthlens.acquisition import CAPTURED, FILE_LEVEL_IMAGE_CHECKS, METHODS, skipped_image_checks
    assert skipped_image_checks(NS(acquisition={})) == set()                      # a normal upload keeps every check
    assert skipped_image_checks(NS(acquisition={"method": "original"})) == set()  # the original file, fetched by the extension
    for m in CAPTURED - {"screen-capture"}:
        assert skipped_image_checks(NS(acquisition={"method": m})) == FILE_LEVEL_IMAGE_CHECKS
    shot = skipped_image_checks(NS(acquisition={"method": "screen-capture"}))     # the site's own icons repeat in a screenshot
    assert shot == FILE_LEVEL_IMAGE_CHECKS | {"copy_move"}
    assert set(METHODS) >= CAPTURED | {"original"}


def test_extension_acquisition_json_is_validated():
    from truthlens.acquisition import describe, parse
    assert parse("not json") == parse('{"method": "teleport"}') == parse("[1]") == parse(None) == {}
    a = parse('{"method": "recording", "seconds": 6.04, "width": 720, "height": 1280, "page_url": "https://www.instagram.com/reel/x/",'
              ' "source_url": "javascript:alert(1)", "page_title": "' + "t" * 500 + '", "evil": 1}')
    assert a["method"] == "recording" and a["seconds"] == 6.04 and (a["width"], a["height"]) == (720, 1280)
    assert "source_url" not in a and "evil" not in a and len(a["page_title"]) == 200
    assert parse('{"method": "recording", "seconds": 9999, "width": -4}') == {"method": "recording"}
    assert describe(a) == "short recording of the playing video (6.0 s), 720×1280 px on instagram.com"


def test_every_capture_method_gets_a_plain_label():
    """Adding a capture method without a label in plain.py raised KeyError in the report (found while adding image copies)."""
    from truthlens.acquisition import CAPTURED, limitation
    for m in sorted(CAPTURED):
        ev = {"modality": "image", "name": "x.png", "indicators": [], "metadata": {},
              "acquisition": {"method": m, "page_url": "https://example.com/p"}}
        o = origin(ev)
        assert o["kind"] == "capture" and "example.com" in o["label"], m
        assert limitation("x.png", ev["acquisition"]), m


def _docx(path, core, app="", body="word " * 400):
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("docProps/core.xml", f"<cp:coreProperties xmlns:cp='x' xmlns:dc='y'>{core}</cp:coreProperties>")
        z.writestr("docProps/app.xml", f"<Properties>{app}</Properties>")
        z.writestr("word/document.xml", f"<w:document><w:body><w:p><w:t>{body}</w:t></w:p></w:body></w:document>")


def test_document_made_by_a_program_is_flagged_but_one_typed_in_word_is_not():
    """A ChatGPT-made .docx scored 1 % "looks normal": .docx wasn't analysed at all (2026-11-20). Now the file's own
    fingerprint is read; a document typed in Word with an edit history raises nothing."""
    import tempfile
    from truthlens.document_forensics import analyze_document
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        case = NS(artifacts_dir=tmp)
        gen, word = tmp / "gen.docx", tmp / "word.docx"
        _docx(gen, "<dc:creator>Un-named</dc:creator><cp:lastModifiedBy>Un-named</cp:lastModifiedBy><cp:revision>1</cp:revision>"
                   "<dcterms:created>2026-09-03T06:16:26Z</dcterms:created><dcterms:modified>2026-09-03T06:16:26Z</dcterms:modified>")
        _docx(word, "<dc:creator>Asha</dc:creator><cp:revision>14</cp:revision>", "<Application>Microsoft Office Word</Application><TotalTime>95</TotalTime>")
        counter = iter(range(100))
        r = analyze_document(NS(id="E01", path=gen), case, lambda: f"E01-{next(counter):02d}")
        assert {i.detector for i in r["indicators"]} == {"document_generator", "document_history"} and r["metrics"]["library"] == "docx.js"
        assert aggregate(r["indicators"], set())["likelihood"] >= 45
        r = analyze_document(NS(id="E02", path=word), case, lambda: f"E02-{next(counter):02d}")
        assert r["indicators"] == [] and r["metrics"]["made_with"] == "Microsoft Word"


def test_chatbot_leftovers_count_but_ordinary_writing_does_not():
    from truthlens.ai_text import check_ai_writing
    from truthlens.evidence import Findings
    counter = iter(range(100))
    out = Findings(NS(id="E01"), lambda: f"E01-{next(counter):02d}", "text")
    check_ai_writing(out, "Certainly! Here is your report. " + "The team met on Monday and fixed the pump. " * 30 + "[Your Name]")
    assert [i.detector for i in out] == ["text_chatbot_leftover"] and out[0].score >= 0.9
    out = Findings(NS(id="E02"), lambda: f"E02-{next(counter):02d}", "text")
    m = check_ai_writing(out, "We drove to the market early, bought rice and onions, and came home before the rain. " * 20)
    assert len(out) == 0 and m["ai_vocab_per_1000"] == 0
    assert check_ai_writing(out, "too short")["skipped"]


def test_unsupported_file_is_never_reported_as_normal():
    """A file type TruthLens can't read used to get "No clear signs of editing or AI, 1 %" after zero checks."""
    import tempfile
    from truthlens.evidence import Case
    from truthlens.pipeline import run_case
    from truthlens.plain import build
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "archive.zip"
        f.write_bytes(b"PK\x03\x04 not analysable")
        _, r = run_case([(f, f.name)], "", lambda *a: None, case=Case(root=Path(tmp) / "cases"))
    p = build(r)
    assert r["overall"]["not_analysed"] and p["headline"] == "We couldn't check this file" and p["files"][0]["origin"]["kind"] == "unsupported"


if __name__ == "__main__":
    tests =[(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} passed")
