"""Deterministic synthetic 'screenshot' so parity tests are reproducible without
any external files. Draws text, a code-ish block, and a labelled button (for a
future grounding check)."""
from PIL import Image, ImageDraw
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "test_image.png")


def make_image(path: str = OUT) -> str:
    W, H = 640, 400
    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)
    # title bar
    d.rectangle([0, 0, W, 36], fill=(40, 44, 52))
    d.text((12, 10), "build.log  —  CI status", fill=(220, 220, 220))
    # body text
    d.text((16, 56), "Running test suite...", fill=(30, 30, 30))
    d.text((16, 80), "PASS  tests/test_auth.py", fill=(20, 130, 40))
    d.text((16, 104), "PASS  tests/test_api.py", fill=(20, 130, 40))
    d.text((16, 128), "FAIL  tests/test_db.py  (timeout)", fill=(190, 30, 30))
    # a code box
    d.rectangle([16, 168, W - 16, 300], outline=(180, 180, 190), width=2)
    d.text((28, 180), "def connect(dsn):", fill=(60, 60, 70))
    d.text((28, 204), "    return pool.get(dsn)", fill=(60, 60, 70))
    d.text((28, 228), "# TODO: add retry/backoff", fill=(150, 120, 30))
    # a labelled button (grounding target)
    d.rectangle([460, 330, 600, 372], fill=(60, 120, 230))
    d.text((492, 344), "Re-run", fill=(255, 255, 255))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return os.path.abspath(path)


if __name__ == "__main__":
    print(make_image())
