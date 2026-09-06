"""M3 gate: the converted (LLM-4bit) MLX model should stay faithful to the reference.
Quantization may shift a few tokens; gate is first-argmax identical + greedy majority
match. Also reports peak GPU/unified memory."""
import os, sys, json
import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ovis_mlx.ovis2_5 import Ovis2_5

ART = os.path.join(ROOT, "artifacts")
OUT = os.path.join(ROOT, "ovis_mlx_model")


def greedy(model, merged, n, eos):
    cache = make_prompt_cache(model.llm)
    out = model.llm(inputs=None, cache=cache, input_embeddings=merged)
    tok = int(mx.argmax(out[:, -1, :], axis=-1).item()); ids = [tok]
    for _ in range(n - 1):
        if tok in eos: break
        emb = model.llm.model.embed_tokens(mx.array([[tok]]))
        out = model.llm(inputs=None, cache=cache, input_embeddings=emb)
        tok = int(mx.argmax(out[:, -1, :], axis=-1).item()); ids.append(tok)
    return ids


def main():
    inp = np.load(os.path.join(ART, "ref_inputs.npz"))
    meta = json.load(open(os.path.join(ART, "ref_meta.json")))
    pixel_values = mx.array(inp["pixel_values"].astype(np.float32))

    model = Ovis2_5.load_mlx(OUT)
    dump = {}
    out = model.logits(inp["input_ids"], pixel_values, inp["grid_thws"], dump=dump)
    mx.eval(out)

    argmax = int(np.argmax(np.asarray(out[0, -1, :])))
    match = argmax == meta["first_argmax"]
    print(f"[{'PASS' if match else 'FAIL'}] first argmax={argmax} ref={meta['first_argmax']}")

    n = min(24, len(meta["greedy_token_ids"]))
    got = greedy(model, dump["merged_embeds"], n, {151645, 151643})
    ref_ids = meta["greedy_token_ids"][:n]
    nmatch = sum(int(a == b) for a, b in zip(got, ref_ids))
    gmatch = nmatch >= int(0.9 * n)  # 8-bit default is near-lossless
    print(f"[{'PASS' if gmatch else 'FAIL'}] greedy {nmatch}/{n} match (quant vs fp32 ref)")
    print("  got:", got[:16])
    print("  ref:", ref_ids[:16])
    print(f"peak unified memory: {mx.get_peak_memory()/1e9:.2f} GB")
    ok = match and gmatch
    print("M3", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
