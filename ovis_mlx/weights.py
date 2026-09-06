"""Load Ovis2.5 checkpoint shards (HF safetensors) into MLX arrays and remap key
names onto our module trees. Used by parity tests (raw fp32) and by convert.py."""
from __future__ import annotations
import glob, json, os
import mlx.core as mx

VIT_PREFIX = "visual_tokenizer.vit.vision_model."
PATCH = "embeddings.patch_embedding."
POS = "embeddings.position_embedding."


def load_raw(model_dir: str) -> dict:
    """All tensors from every shard, as mx.float32 arrays keyed by original name."""
    out = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        for k, v in mx.load(shard).items():
            out[k] = v.astype(mx.float32)
    return out


def vision_weights(raw: dict, patch_in_dim: int) -> list[tuple[str, mx.array]]:
    """Map checkpoint keys -> Siglip2VisionModel parameter paths."""
    out = {}
    for k, v in raw.items():
        if not k.startswith(VIT_PREFIX):
            continue
        sub = k[len(VIT_PREFIX):]
        if sub.startswith(PATCH):
            # Conv2d (out,in,kh,kw) -> Linear (out, in*kh*kw)
            name = "patch_embedding." + sub[len(PATCH):]
            if name.endswith("weight"):
                v = v.reshape(v.shape[0], patch_in_dim)
            out[name] = v
        elif sub.startswith(POS):
            out["position_embedding." + sub[len(POS):]] = v
        elif sub.startswith("encoder.layers."):
            out["layers." + sub[len("encoder.layers."):]] = v
        elif sub.startswith("post_layernorm."):
            out[sub] = v
    return list(out.items())


def head_and_vte_weights(raw: dict):
    """visual_tokenizer.head.{0.weight,1.weight,1.bias} and vte.weight."""
    head = {
        "proj.weight": raw["visual_tokenizer.head.0.weight"],
        "norm.weight": raw["visual_tokenizer.head.1.weight"],
        "norm.bias": raw["visual_tokenizer.head.1.bias"],
    }
    return list(head.items()), raw["vte.weight"]


def load_vit_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)["vit_config"]


def load_llm_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)["llm_config"]
