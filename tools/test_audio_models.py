"""Compare candidate synthetic-speech models on known samples (TTS vs real voice)."""
import sys

sys.path.insert(0, ".")
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import pipeline  # noqa: E402

from truthlens.media_io import load_audio  # noqa: E402

CANDS = sys.argv[1:] or ["mo-thecreator/Deepfake-audio-detection", "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
                         "garystafford/wav2vec2-deepfake-voice-detector"]
SAMPLES = {"TTS (synthetic)": "demo_samples/tts_statement.wav", "Real selfie voice": "demo_samples/whatsapp_video.mp4",
           "Wan avatar": "demo_samples/Wan_Avatar_PhotoAnimate.mp4"}
for mid in CANDS:
    try:
        snapshot_download(mid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.bin"] )
        p = pipeline("audio-classification", model=mid, device="cpu")
        print("MODEL", mid, p.model.config.id2label, flush=True)
        for name, f in SAMPLES.items():
            y, _ = load_audio(f, 16000)
            outs = []
            for s in range(0, min(len(y), 16000 * 16), 64000):
                ch = y[s:s + 64000]
                if len(ch) > 16000:
                    r = p({"raw": ch, "sampling_rate": 16000}, top_k=5)
                    outs.append({x["label"]: round(x["score"], 3) for x in r})
            print("   ", name, outs, flush=True)
    except Exception as e:
        print("FAIL", mid, type(e).__name__, str(e)[:150], flush=True)
print("TESTDONE")
