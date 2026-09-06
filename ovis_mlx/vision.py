"""SigLIP2-NaViT vision tower in MLX (port of Siglip2VisionTransformer).

Simplifications valid for the Ovis2.5-2B checkpoint (vit_config.fullatt_block_indexes
is null, single image per request):
  * Every encoder layer uses FULL attention over the image's patches. The window
    reorder in the reference is a no-op after its final inverse permutation (full
    attention is permutation-invariant, rotary is permuted alongside), so we operate
    directly in natural, 2x2-block-grouped patch order and skip get_window_index.
  * One image == one attention sequence, so no cu_seqlens block-diagonal masking.

Bicubic positional-embedding interpolation depends only on (PE weights, grid) — not on
activations — so it is precomputed once per grid in `interpolate_pos_embed` (torch, to
match the reference's F.interpolate bit-for-bit) and fed in as a plain array.
"""
from __future__ import annotations
import math
import mlx.core as mx
import mlx.nn as nn


# ---- positional embedding (bicubic), computed with torch for parity ----
def interpolate_pos_embed(pe_weight, grid_thws, position_embedding_size: int,
                          hidden_stride: int):
    """Replicates Siglip2VisionEmbeddings position-embedding branch.

    pe_weight: (num_patches, embed_dim) array (np or mx-convertible)
    returns: mx.array (total_patches, embed_dim) in natural block-grouped order.
    """
    import torch
    import numpy as np
    if isinstance(pe_weight, mx.array):
        pe_weight = np.array(pe_weight.astype(mx.float32))
    pe = torch.tensor(np.asarray(pe_weight, dtype=np.float32), dtype=torch.float32)
    ori = position_embedding_size
    embed_dim = pe.shape[-1]
    base = pe.reshape(ori, ori, embed_dim).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    outs = []
    for t, h, w in np.asarray(grid_thws).tolist():
        p = torch.nn.functional.interpolate(base, size=(h, w), mode="bicubic",
                                            align_corners=False)
        p = p.permute(0, 2, 3, 1).reshape(1, h * w, -1)[0].repeat(t, 1)
        p = p.reshape(t, h // hidden_stride, hidden_stride, w // hidden_stride,
                      hidden_stride, -1)
        p = p.permute(0, 1, 3, 2, 4, 5).reshape(t * h * w, -1)
        outs.append(p)
    out = torch.cat(outs, dim=0).numpy()
    return mx.array(out)


# ---- 2D vision rotary embedding ----
def vision_rotary(grid_thws, head_dim: int, hidden_stride: int, theta: float = 10000.0):
    """Returns (cos, sin) each of shape (total_patches, head_dim) in natural
    block-grouped order. Mirrors Siglip2Encoder.rot_pos_emb + the emb=cat(rope,rope)."""
    import numpy as np
    dim = head_dim // 2  # VisionRotaryEmbedding dim = hidden//heads//2
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))  # (dim/2,)
    pos_ids = []
    for t, h, w in np.asarray(grid_thws).tolist():
        hpos = np.arange(h).reshape(-1, 1).repeat(w, 1)
        hpos = hpos.reshape(h // hidden_stride, hidden_stride,
                            w // hidden_stride, hidden_stride)
        hpos = hpos.transpose(0, 2, 1, 3).reshape(-1)
        wpos = np.arange(w).reshape(1, -1).repeat(h, 0)
        wpos = wpos.reshape(h // hidden_stride, hidden_stride,
                            w // hidden_stride, hidden_stride)
        wpos = wpos.transpose(0, 2, 1, 3).reshape(-1)
        ids = np.stack([hpos, wpos], axis=-1)
        ids = np.tile(ids, (t, 1))
        pos_ids.append(ids)
    pos_ids = np.concatenate(pos_ids, axis=0)              # (N, 2)
    max_grid = int(np.asarray(grid_thws)[:, 1:].max())
    seq = np.arange(max_grid, dtype=np.float32)
    freqs = np.outer(seq, inv_freq)                        # (max_grid, dim/2)
    rope = freqs[pos_ids].reshape(pos_ids.shape[0], -1)    # (N, dim) = (N, head_dim/2)
    emb = np.concatenate([rope, rope], axis=-1)            # (N, head_dim)
    return mx.array(np.cos(emb)), mx.array(np.sin(emb))


def _rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_vision_rope(q, k, cos, sin):
    # q,k: (N, heads, head_dim); cos,sin: (N, head_dim)
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    q2 = (q * cos) + (_rotate_half(q) * sin)
    k2 = (k * cos) + (_rotate_half(k) * sin)
    return q2, k2


class Siglip2Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x, cos, sin):
        # x: (N, dim).  Single full-attention sequence (one image).
        N, _ = x.shape
        q = self.q_proj(x).reshape(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(N, self.num_heads, self.head_dim)
        q, k = apply_vision_rope(q, k, cos, sin)
        # -> (1, heads, N, head_dim)
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o[0].transpose(1, 0, 2).reshape(N, -1)
        return self.out_proj(o)


class Siglip2MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def __call__(self, x):
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, dim, hidden, num_heads, eps):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=eps)
        self.self_attn = Siglip2Attention(dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Siglip2MLP(dim, hidden)

    def __call__(self, x, cos, sin):
        x = x + self.self_attn(self.layer_norm1(x), cos, sin)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Siglip2VisionModel(nn.Module):
    """Holds patch-embed projection + encoder + post_layernorm. Position embeddings
    are added by the caller (precomputed bicubic). Weight names mirror the checkpoint
    under visual_tokenizer.vit.vision_model.*"""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dim = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.head_dim = dim // self.num_heads
        self.hidden_stride = cfg["hidden_stride"]
        self.patch_size = cfg["patch_size"]
        self.position_embedding_size = cfg["image_size"] // cfg["patch_size"]
        eps = cfg["layer_norm_eps"]
        in_dim = cfg["num_channels"] * cfg["temporal_patch_size"] * cfg["patch_size"] ** 2
        # Conv2d(kernel=stride=patch) collapses to a Linear over flattened patches.
        self.patch_embedding = nn.Linear(in_dim, dim)
        # position_embedding stored as Embedding weight; only its .weight is used.
        self.position_embedding = nn.Embedding(self.position_embedding_size ** 2, dim)
        self.layers = [Siglip2EncoderLayer(dim, cfg["intermediate_size"],
                                           self.num_heads, eps)
                       for _ in range(cfg["num_hidden_layers"])]
        self.post_layernorm = nn.LayerNorm(dim, eps=eps)

    def __call__(self, flatten_patches, grid_thws, dump: dict | None = None):
        x = self.patch_embedding(flatten_patches)
        pe = interpolate_pos_embed(self.position_embedding.weight, grid_thws,
                                   self.position_embedding_size, self.hidden_stride)
        x = x + pe.astype(x.dtype)
        if dump is not None:
            dump["patch_embeds"] = x
        cos, sin = vision_rotary(grid_thws, self.head_dim, self.hidden_stride)
        cos, sin = cos.astype(x.dtype), sin.astype(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        # The visual tokenizer consumes the LAST encoder-layer output (hidden_states[-1]),
        # i.e. BEFORE post_layernorm. post_layernorm is applied only to last_hidden_state,
        # which the tokenizer ignores. So we return the pre-norm hidden state.
        if dump is not None:
            dump["vit_out"] = self.post_layernorm(x)
        return x
