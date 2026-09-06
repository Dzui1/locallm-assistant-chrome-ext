# Ovis2.5-2B — local on-screen vision assistant (native MLX)

A locally-hosted vision SLM that looks at your screen and answers questions, running
**natively in Apple's MLX** on Apple Silicon. Ovis2.5-2B is a custom composite model
(Qwen3-1.7B LLM + SigLIP2-NaViT vision tower + a 65k-entry visual embedding table) that
no shipping MLX package supports, so this repo **ports the architecture to MLX** from
scratch and serves it behind an OpenAI-compatible API, with a Chrome side-panel client.

```
Chrome extension (side panel)            Python MLX daemon (localhost:8000)
  captureVisibleTab -> base64    --->    FastAPI /v1/chat/completions (SSE)
  chat UI, Deep Analysis toggle          ovis_mlx port + Qwen3 (mlx-lm)
  render markdown, fold <think>  <---    streamed tokens
  grounding overlay from <box>
```

## Layout
| Path | What |
|---|---|
| `ovis_mlx/vision.py` | SigLIP2-NaViT tower (Conv2d→Linear patch embed, bicubic PE, 2D RoPE, full attention) |
| `ovis_mlx/visual_tokenizer.py` | soft-token head + VTE matmul |
| `ovis_mlx/ovis2_5.py` | top model: `merge_multimodal` + Qwen3 (reused from `mlx_lm`), load/save/quantize |
| `ovis_mlx/preprocess.py` | smart_resize + patchify + placeholder input_ids (pure numpy/PIL) |
| `ovis_mlx/generate.py` | streaming decode, sampling, two-pass thinking budget |
| `convert.py` | HF checkpoint → MLX dir (LLM quantized, vision bf16) |
| `server/server.py` | OpenAI-compatible FastAPI daemon |
| `extension/` | Chrome MV3 side-panel client |
| `reference/` , `tests/` | PyTorch oracle + parity gates |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Build the MLX model DONE
```bash
python convert.py            # default: LLM 8-bit (near-lossless, ~4.9GB peak), vision bf16
# alternatives: --bits 4 (smallest, ~4.6GB), --vision-dtype fp16, --no-quant
```
Output → `ovis_mlx_model/` (`weights.safetensors` + `meta.json`).

## Run the daemon
```bash
./run_daemon.sh              # uvicorn on 127.0.0.1:8000
curl localhost:8000/health
```
`OVIS_MAX_PIXELS` caps incoming images (default 1280²) — the unified-memory OOM guardrail
for Retina captures. Thinking mode is per-request: `enable_thinking` /
`enable_thinking_budget` / `thinking_budget`.

## Load the Chrome extension
`chrome://extensions` → Developer mode → **Load unpacked** → select `extension/`.
Click the toolbar icon to open the side panel → **Capture tab** → ask a question.
Toggle **Deep Analysis** for reflective `<think>` reasoning. If the model returns
`<box>` coordinates, the captured image is shown with a highlight overlay.

**Session memory:** the side panel keeps the full conversation in memory and sends it
every turn, so follow-ups like "explain more" stay anchored on the screenshot(s) from
that session. Nothing is persisted — **＋ New** (or closing the panel) clears context.
Because the whole history is resent, the daemon is fully stateless; the cost is that any
image in history is re-encoded per turn (fine for one screenshot; a KV/vision cache is
the optimization if sessions get long).

## Correctness (parity gates)
The MLX port is validated stage-by-stage against a PyTorch CPU oracle on a fixed input:
```bash
python reference/run_reference.py     # M0: build oracle + intermediate dumps
python tests/test_vision_parity.py    # M1: vision tower   (cos-sim 1.000000)
python tests/test_full_parity.py      # M2: full fp32 model (greedy 24/24 vs ref)
python tests/test_quant_parity.py     # M3: 8-bit quant     (greedy 24/24, ~4.9GB)
```

## Notes / known follow-ups
- **Perf:** first request pays vision-encode + prefill + MLX kernel compilation. Decode
  steady-state is the next optimization (vision tower currently bf16; the big
  `softmax@vte` matmul dominates image cost).
- **Scope:** the browser client only sees the tab viewport. A native ScreenCaptureKit
  daemon can sit behind the same HTTP API to capture any app, without touching the model.
