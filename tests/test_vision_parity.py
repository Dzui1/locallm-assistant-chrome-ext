"""M1 gate: MLX SigLIP2-NaViT tower must match the PyTorch reference dump.

Reuses the EXACT preprocessed inputs from artifacts/ref_inputs.npz so preprocessing
is removed as a variable. Gate: vit_out cosine-sim > 0.999."""
import os, sys
import numpy as np
import mlx.core as mx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ovis_mlx.vision import Siglip2VisionModel
from ovis_mlx import weights as W

MODEL_DIR = os.path.join(ROOT, "Ovis2.5-2B")
ART = os.path.join(ROOT, "artifacts")


def cos(a, b):
    a, b = np.asarray(a).ravel().astype(np.float64), np.asarray(b).ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    inp = np.load(os.path.join(ART, "ref_inputs.npz"))
    ref = np.load(os.path.join(ART, "ref_dump.npz"))
    pixel_values = mx.array(inp["pixel_values"].astype(np.float32))
    grid_thws = inp["grid_thws"]

    cfg = W.load_vit_config(MODEL_DIR)
    patch_in = cfg["num_channels"] * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2
    model = Siglip2VisionModel(cfg)
    raw = W.load_raw(MODEL_DIR)
    model.load_weights(W.vision_weights(raw, patch_in))
    model.eval()

    dump = {}
    out = model(pixel_values, grid_thws, dump=dump)
    mx.eval(out)

    ok = True
    for name, got in [("patch_embeds", dump["patch_embeds"]), ("vit_out", dump["vit_out"])]:
        g = np.asarray(got)
        r = ref[name]
        c = cos(g, r)
        md = float(np.max(np.abs(g - r)))
        thresh = 0.999
        status = "PASS" if c > thresh else "FAIL"
        ok = ok and c > thresh
        print(f"[{status}] {name:14s} cos={c:.6f}  max|d|={md:.4e}  shape={g.shape}")
    print("M1", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
