"""Streaming generation for Ovis2.5-MLX: prefill from merged multimodal embeddings,
then decode token-by-token with a KV cache. Supports temperature/top-p/top-k sampling,
repetition penalty, and the two-pass 'thinking budget' from the reference."""
from __future__ import annotations
import mlx.core as mx
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache

IM_END = 151645
THINK_END = 151668
EOS_IDS = {151645, 151643}
EARLY_STOP = ("\n\nConsidering the limited time by the user, I have to give the "
              "solution based on the thinking directly now.\n</think>\n\n")


def _step(model, cache, embeds, tokens, sampler, processors):
    logits = model.llm(inputs=None, cache=cache, input_embeddings=embeds)[:, -1, :]
    if processors:
        for p in processors:
            logits = p(tokens, logits)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok = sampler(logprobs)
    return int(tok.item())


def stream_generate(model, processor, messages, *, max_tokens=512,
                    temperature=0.0, top_p=0.0, top_k=0, repetition_penalty=1.05,
                    min_pixels=None, max_pixels=None,
                    enable_thinking=False, enable_thinking_budget=False,
                    thinking_budget=1024):
    """Yields dict events: {'type':'delta','text':str} then {'type':'done'}."""
    from .preprocess import DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS
    min_pixels = min_pixels or DEFAULT_MIN_PIXELS
    max_pixels = max_pixels or DEFAULT_MAX_PIXELS

    input_ids, pixel_values, grid_thws = processor.build_inputs(
        messages, min_pixels=min_pixels, max_pixels=max_pixels,
        enable_thinking=enable_thinking)
    pv = mx.array(pixel_values) if pixel_values is not None else None
    merged = model.merge_multimodal(input_ids, pv, grid_thws)

    sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
    processors = make_logits_processors(repetition_penalty=repetition_penalty)
    tok_de = processor.tokenizer
    cache = make_prompt_cache(model.llm)

    budgeted = enable_thinking and enable_thinking_budget
    tokens = mx.array([], dtype=mx.int32)
    embeds = merged
    produced = 0
    think_closed = False

    def emit(tid):
        nonlocal tokens
        tokens = mx.concatenate([tokens, mx.array([tid], dtype=mx.int32)])
        return tok_de.decode([tid])

    # ---- phase 1: (budgeted) thinking up to thinking_budget tokens ----
    limit = thinking_budget if budgeted else max_tokens
    while produced < limit:
        tid = _step(model, cache, embeds, tokens, sampler, processors)
        if tid in EOS_IDS:
            yield {"type": "done", "reason": "stop"}
            return
        if tid == THINK_END:
            think_closed = True
        yield {"type": "delta", "text": emit(tid)}
        produced += 1
        embeds = model.llm.model.embed_tokens(mx.array([[tid]]))
        if budgeted and think_closed:
            break

    if not budgeted:
        yield {"type": "done", "reason": "length"}
        return

    # ---- phase 2: budget hit without </think> -> inject early-stop, finish answer ----
    if not think_closed:
        stop_ids = tok_de(EARLY_STOP, add_special_tokens=False).input_ids
        for tid in stop_ids:
            emit(tid)
        yield {"type": "delta", "text": EARLY_STOP}
        embeds = model.llm.model.embed_tokens(mx.array([stop_ids]))

    while produced < max_tokens:
        tid = _step(model, cache, embeds, tokens, sampler, processors)
        if tid in EOS_IDS:
            yield {"type": "done", "reason": "stop"}
            return
        yield {"type": "delta", "text": emit(tid)}
        produced += 1
        embeds = model.llm.model.embed_tokens(mx.array([[tid]]))
    yield {"type": "done", "reason": "length"}
