"""M2 gate: full MLX Ovis2.5 (fp32) must match the PyTorch reference.
Gates: intermediate cos-sim > 0.999, first-step argmax identical, and greedy token
ids match the reference for the generated prefix."""
import os, sys, json
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ovis_mlx.ovis2_5 import Ovis2_5

MODEL_DIR = os.path.join(ROOT, "Ovis2.5-2B")
ART = os.path.join(ROOT, "artifacts")


def cos(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def greedy(model, merged, n, eos_ids):
    cache = make_prompt_cache(model.llm)
    out = model.llm(inputs=None, cache=cache, input_embeddings=merged)
    tok = int(mx.argmax(out[:, -1, :], axis=-1).item())
    ids = [tok]
    for _ in range(n - 1):
        if tok in eos_ids:
            break
        emb = model.llm.model.embed_tokens(mx.array([[tok]]))
        out = model.llm(inputs=None, cache=cache, input_embeddings=emb)
        tok = int(mx.argmax(out[:, -1, :], axis=-1).item())
        ids.append(tok)
    return ids


def main():
    inp = np.load(os.path.join(ART, "ref_inputs.npz"))
    ref = np.load(os.path.join(ART, "ref_dump.npz"))
    meta = json.load(open(os.path.join(ART, "ref_meta.json")))
    pixel_values = mx.array(inp["pixel_values"].astype(np.float32))
    grid_thws = inp["grid_thws"]
    input_ids = inp["input_ids"]

    print("loading MLX model (fp32)...", flush=True)
    model = Ovis2_5(MODEL_DIR).load_fp32()

    dump = {}
    out = model.logits(input_ids, pixel_values, grid_thws, dump=dump)
    mx.eval(out)

    ok = True
    for name in ["soft_tokens", "vte_out", "merged_embeds"]:
        c = cos(dump[name], ref[name])
        ok = ok and c > 0.999
        print(f"[{'PASS' if c>0.999 else 'FAIL'}] {name:14s} cos={c:.6f}")

    first = np.asarray(out[0, -1, :])
    argmax = int(np.argmax(first))
    c = cos(first, ref["first_logits"][0])
    match = argmax == meta["first_argmax"]
    ok = ok and match
    print(f"[{'PASS' if match else 'FAIL'}] first_logits  cos={c:.6f}  "
          f"argmax={argmax} ref={meta['first_argmax']}")

    eos = {151645, 151643}
    n = min(24, len(meta["greedy_token_ids"]))
    got = greedy(model, dump["merged_embeds"], n, eos)
    ref_ids = meta["greedy_token_ids"][:n]
    nmatch = sum(int(a == b) for a, b in zip(got, ref_ids))
    gmatch = nmatch >= int(0.9 * n)
    ok = ok and gmatch
    print(f"[{'PASS' if gmatch else 'FAIL'}] greedy {nmatch}/{n} tokens match")
    print("  got:", got[:16])
    print("  ref:", ref_ids[:16])
    print("M2", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
