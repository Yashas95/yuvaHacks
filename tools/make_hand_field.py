"""Build the halftone 'two hands' density field used by the front-page animation.

Source image (public domain): Michelangelo, "The Creation of Adam" (c. 1512), Wikimedia Commons:
https://commons.wikimedia.org/wiki/File:Michelangelo_-_Creation_of_Adam_(cropped).jpg  (1920 px wide copy)

    .venv/bin/python tools/make_hand_field.py path/to/adam.jpg

Writes truthlens/hand_field.json: one density grid covering both arms (values 0-9, row-major),
plus where the gap between the fingertips is, so the page can open it up for the headline.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

X0, Y0, X1, Y1 = 455, 328, 975, 436   # forearms + hands only, in the 1920x871 image
SPLIT = 728                            # x between the two fingertips
STEP = 2                               # painting pixels per grid cell


def main(src: str):
    im = cv2.imread(src)
    if im is None or im.shape[1] != 1920:
        sys.exit("expected the 1920 px wide Creation of Adam image")
    i16 = im.astype(np.int16)
    b, g, r = i16[..., 0], i16[..., 1], i16[..., 2]
    strict = ((r - b) > 60) & (r > 105) & ((r - g) > 22)       # Adam: warm skin vs warm plaster under his hand
    loose = ((r - b) > 38) & (r > 105) & ((r - g) > 12)        # God: paler skin
    skin = loose.copy(); skin[:, :SPLIT] = strict[:, :SPLIT]
    skin = skin.astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))     # gentle: keep gaps between fingers
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
    crop = np.zeros_like(skin)
    crop[340:445, 285:SPLIT] = skin[340:445, 285:SPLIT]          # Adam: stop above the knee
    crop[318:440, SPLIT:1010] = skin[318:440, SPLIT:1010]        # God: stop before cherub / drape
    crop[400:445, 880:1010] = 0                                  # warm plaster under God's forearm
    m = crop[Y0:Y1, X0:X1]
    keep = np.zeros_like(m)
    for part in (slice(0, SPLIT - X0), slice(SPLIT - X0, X1 - X0)):   # largest blob per arm
        sub = m[:, part]
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub)
        if n > 1:
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            keep[:, part] = np.where(lab == j, 255, 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[Y0:Y1, X0:X1].astype(np.float32)
    shade = 1 - gray / 255.0                                      # darker paint -> bigger dots
    inside = keep > 0
    if inside.any():
        lo, hi = np.percentile(shade[inside], [5, 95])
        shade = np.clip((shade - lo) / max(1e-6, hi - lo), 0, 1)
    dens = (keep / 255.0) * (0.62 + 0.38 * shade)                # silhouette first, shading second
    h, w = dens.shape
    small = cv2.resize(dens, (w // STEP, h // STEP), interpolation=cv2.INTER_AREA)
    q = np.clip(np.round(small * 9), 0, 9).astype(int)
    out = {"w": int(q.shape[1]), "h": int(q.shape[0]), "split": round((SPLIT - X0) / (X1 - X0), 4),
           "aspect": round((X1 - X0) / (Y1 - Y0), 4), "data": "".join(map(str, q.ravel())),
           "source": "Michelangelo, The Creation of Adam (public domain), via Wikimedia Commons"}
    dst = Path(__file__).resolve().parent.parent / "truthlens" / "hand_field.json"
    dst.write_text(json.dumps(out))
    print(f"wrote {dst}  grid {q.shape[1]}x{q.shape[0]}  non-empty {int((q > 0).sum())} cells")


if __name__ == "__main__":
    main(sys.argv[1])
