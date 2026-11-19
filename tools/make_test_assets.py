"""Create test media with KNOWN planted manipulations (for self-testing the pipeline)."""
import io
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

out = Path("test_assets"); out.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# --- image: textured scene, a region pasted from a differently-compressed source + a cloned patch
h, w = 768, 1024
yy, xx = np.mgrid[0:h, 0:w]
base = np.stack([120 + 60 * np.sin(xx / 90) + 30 * np.cos(yy / 50),
                 110 + 50 * np.cos(xx / 130 + yy / 170),
                 100 + 40 * np.sin((xx + yy) / 110)], -1)
base += rng.normal(0, 6, base.shape)
img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
buf = io.BytesIO(); img.save(buf, "JPEG", quality=92); img = Image.open(buf).convert("RGB")
patch = Image.fromarray(np.clip(rng.normal(150, 2, (180, 220, 3)) + np.linspace(0, 60, 220)[None, :, None], 0, 255).astype(np.uint8))
pb = io.BytesIO(); patch.save(pb, "JPEG", quality=40); patch = Image.open(pb)
img.paste(patch, (620, 420))                      # splice
src = img.crop((120, 120, 300, 300)); img.paste(src, (420, 150))   # copy-move
img.save(out / "spliced_photo.jpg", "JPEG", quality=90)

# --- audio: 44.1 kHz file whose content is band-limited at 8 kHz, with digital silence gap and noise-floor jump
sr = 44100
t = np.arange(int(sr * 12)) / sr
voice = 0.25 * np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.1 * np.sin(2 * np.pi * 360 * t)
noise = rng.normal(0, 0.003, t.size)
noise[t > 7] *= 12                                 # room tone jumps at 7s (splice)
y = voice * (np.sin(2 * np.pi * 0.5 * t) > -0.3) + noise
y[(t > 4.0) & (t < 4.6)] = 0.0                    # perfect digital silence
from scipy import signal
b, a = signal.butter(10, 8000 / (sr / 2)); y = signal.filtfilt(b, a, y)
y[(t > 4.0) & (t < 4.6)] = 0.0
with wave.open(str(out / "voice_note.wav"), "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())

# --- video: two different scenes joined (hard cut at 4s) + the spliced audio
ff = "ffmpeg"
subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=4",
                "-f", "lavfi", "-i", "mandelbrot=size=640x360:rate=25", "-i", str(out / "voice_note.wav"),
                "-filter_complex", "[1:v]trim=duration=4,setpts=PTS-STARTPTS[b];[0:v][b]concat=n=2:v=1:a=0[v]",
                "-map", "[v]", "-map", "2:a", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(out / "joined_clip.mp4")], check=True)
print("created:", [p.name for p in out.iterdir()])
