"""Top-level Ovis2.5 model in MLX: vision tower + visual tokenizer head + VTE +
merge_multimodal + Qwen3 LLM (reused from mlx_lm)."""
from __future__ import annotations
import json, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.models import qwen3

from .vision import Siglip2VisionModel
from .visual_tokenizer import VisualHead, soft_tokens, visual_embeds, N_INDICATORS
from . import weights as W

IMAGE_PLACEHOLDER_ID = -200
VIDEO_PLACEHOLDER_ID = -201
VISUAL_ATOM_ID = -300
INDICATOR_IDS = [-301, -302, -303, -304]


def _qwen3_args(llm_cfg: dict) -> qwen3.ModelArgs:
    return qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=llm_cfg["hidden_size"],
        num_hidden_layers=llm_cfg["num_hidden_layers"],
        intermediate_size=llm_cfg["intermediate_size"],
        num_attention_heads=llm_cfg["num_attention_heads"],
        rms_norm_eps=llm_cfg["rms_norm_eps"],
        vocab_size=llm_cfg["vocab_size"],
        num_key_value_heads=llm_cfg["num_key_value_heads"],
        max_position_embeddings=llm_cfg["max_position_embeddings"],
        rope_theta=llm_cfg["rope_theta"],
        head_dim=llm_cfg["head_dim"],
        tie_word_embeddings=llm_cfg["tie_word_embeddings"],
        rope_scaling=llm_cfg.get("rope_scaling"),
    )


class Ovis2_5:
    """Container (not an nn.Module) holding the three sub-models so each can be
    quantized / dtyped independently."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.vit_cfg = W.load_vit_config(model_dir)
        self.llm_cfg = W.load_llm_config(model_dir)
        self.visual_vocab = 65536
        self.hidden_stride = self.vit_cfg["hidden_stride"]

        self.vision = Siglip2VisionModel(self.vit_cfg)
        patch_in = (self.vit_cfg["num_channels"] * self.vit_cfg["temporal_patch_size"]
                    * self.vit_cfg["patch_size"] ** 2)
        head_dim = self.visual_vocab - N_INDICATORS
        merged_in = self.vit_cfg["hidden_size"] * self.hidden_stride ** 2
        self.head = VisualHead(merged_in, head_dim, eps=self.vit_cfg["layer_norm_eps"])
        self.llm = qwen3.Model(_qwen3_args(self.llm_cfg))
        self.vte_weight = None
        self._patch_in = patch_in

    # ---- loading ----
    def load_fp32(self):
        raw = W.load_raw(self.model_dir)
        self.vision.load_weights(W.vision_weights(raw, self._patch_in))
        head_w, vte = W.head_and_vte_weights(raw)
        self.head.load_weights(head_w)
        self.vte_weight = vte
        llm_w = [(k[len("llm."):], v) for k, v in raw.items() if k.startswith("llm.")]
        self.llm.load_weights(llm_w)
        for m in (self.vision, self.head, self.llm):
            m.eval()
        return self

    def cast_vision(self, dtype):
        """Cast vision tower + head + vte to dtype (LLM is handled separately)."""
        self.vision.update(tree_map(lambda a: a.astype(dtype), self.vision.parameters()))
        self.head.update(tree_map(lambda a: a.astype(dtype), self.head.parameters()))
        self.vte_weight = self.vte_weight.astype(dtype)
        return self

    def quantize_llm(self, bits=4, group_size=64):
        nn.quantize(self.llm, group_size=group_size, bits=bits)
        return self

    def save_mlx(self, out_dir, quant=None):
        os.makedirs(out_dir, exist_ok=True)
        flat = {}
        for name, arr in tree_flatten(self.vision.parameters()):
            flat[f"vision.{name}"] = arr
        for name, arr in tree_flatten(self.head.parameters()):
            flat[f"head.{name}"] = arr
        flat["vte.weight"] = self.vte_weight
        for name, arr in tree_flatten(self.llm.parameters()):
            flat[f"llm.{name}"] = arr
        mx.save_safetensors(os.path.join(out_dir, "weights.safetensors"), flat)
        meta = {"vit_config": self.vit_cfg, "llm_config": self.llm_cfg,
                "visual_vocab": self.visual_vocab, "quant": quant}
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_mlx(cls, out_dir):
        """Load a converted (optionally LLM-quantized) MLX model."""
        with open(os.path.join(out_dir, "meta.json")) as f:
            meta = json.load(f)
        self = cls.__new__(cls)
        self.model_dir = out_dir
        self.vit_cfg = meta["vit_config"]
        self.llm_cfg = meta["llm_config"]
        self.visual_vocab = meta["visual_vocab"]
        self.hidden_stride = self.vit_cfg["hidden_stride"]
        self.vision = Siglip2VisionModel(self.vit_cfg)
        head_dim = self.visual_vocab - N_INDICATORS
        merged_in = self.vit_cfg["hidden_size"] * self.hidden_stride ** 2
        self.head = VisualHead(merged_in, head_dim, eps=self.vit_cfg["layer_norm_eps"])
        self.llm = qwen3.Model(_qwen3_args(self.llm_cfg))
        if meta.get("quant"):
            nn.quantize(self.llm, group_size=meta["quant"]["group_size"],
                        bits=meta["quant"]["bits"])
        flat = mx.load(os.path.join(out_dir, "weights.safetensors"))
        groups = {"vision": [], "head": [], "llm": []}
        for k, v in flat.items():
            if k == "vte.weight":
                self.vte_weight = v
            else:
                pre, rest = k.split(".", 1)
                groups[pre].append((rest, v))
        self.vision.load_weights(groups["vision"])
        self.head.load_weights(groups["head"])
        self.llm.load_weights(groups["llm"])
        for m in (self.vision, self.head, self.llm):
            m.eval()
        return self

    # ---- forward pieces ----
    def encode_image(self, pixel_values, grid_thws, dump=None):
        vit_out = self.vision(pixel_values, grid_thws, dump=dump)
        tokens = soft_tokens(vit_out, self.head, self.hidden_stride, self.visual_vocab)
        if dump is not None:
            dump["soft_tokens"] = tokens
        vemb = visual_embeds(tokens, self.vte_weight)
        if dump is not None:
            dump["vte_out"] = vemb
        return vemb

    def merge_multimodal(self, input_ids, pixel_values=None, grid_thws=None, dump=None):
        ids = np.asarray(input_ids).reshape(-1)
        L = ids.shape[0]
        clamped = mx.array(np.where(ids < 0, 0, ids).astype(np.int32))[None]
        text_embeds = self.llm.model.embed_tokens(clamped)[0]  # (L, H)
        if pixel_values is None:
            return text_embeds[None]

        vemb = self.encode_image(pixel_values, grid_thws, dump=dump)  # (A, H)
        ind_idx = mx.array(
            list(range(self.visual_vocab - N_INDICATORS, self.visual_vocab)))
        ind_embeds = self.vte_weight[ind_idx]                          # (4, H)

        source = mx.concatenate([text_embeds, vemb, ind_embeds], axis=0)
        gather = np.arange(L, dtype=np.int64)
        atom = 0
        for i, tok in enumerate(ids.tolist()):
            if tok == VISUAL_ATOM_ID:
                gather[i] = L + atom
                atom += 1
            elif tok in INDICATOR_IDS:
                gather[i] = L + vemb.shape[0] + INDICATOR_IDS.index(tok)
        merged = source[mx.array(gather)][None]                        # (1, L, H)
        if dump is not None:
            dump["merged_embeds"] = merged
        return merged

    def logits(self, input_ids, pixel_values=None, grid_thws=None, dump=None):
        merged = self.merge_multimodal(input_ids, pixel_values, grid_thws, dump=dump)
        out = self.llm(inputs=None, input_embeddings=merged)
        return out  # (1, L, vocab)
