"""Visual tokenizer head (Linear->LayerNorm->softmax soft tokens) and the VTE
matmul that maps soft visual tokens into the LLM embedding space."""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

N_INDICATORS = 4  # INDICATOR_IDS = [-301,-302,-303,-304]


class VisualHead(nn.Module):
    """head = Sequential(Linear(vit_hidden*stride^2 -> vocab-4, bias=False),
                         LayerNorm(vocab-4))"""

    def __init__(self, in_dim: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(in_dim, head_dim, bias=False)
        self.norm = nn.LayerNorm(head_dim, eps=eps)

    def __call__(self, features):
        return self.norm(self.proj(features))


def soft_tokens(vit_out, head: VisualHead, hidden_stride: int, visual_vocab: int):
    """vit_out (N_patch, vit_hidden) -> soft tokens (N_atom, visual_vocab)."""
    n, d = vit_out.shape
    merge = hidden_stride * hidden_stride
    features = vit_out.reshape(n // merge, d * merge)
    logits = head(features)
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    pad = mx.zeros((probs.shape[0], N_INDICATORS), dtype=probs.dtype)
    return mx.concatenate([probs, pad], axis=-1)  # (N_atom, visual_vocab)


def visual_embeds(tokens, vte_weight):
    """soft tokens (N_atom, vocab) @ vte (vocab, hidden) -> (N_atom, hidden)."""
    return tokens.astype(vte_weight.dtype) @ vte_weight
