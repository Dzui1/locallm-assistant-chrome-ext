"""Ovis2.5-MLX local inference daemon — OpenAI-compatible /v1/chat/completions.

Loads the converted MLX model once, serializes GPU work with a lock, decodes
base64 images, enforces a server-side max_pixels cap (the unified-memory OOM
guardrail), and streams SSE chunks. Custom (non-OpenAI) request fields:
  enable_thinking, enable_thinking_budget, thinking_budget, max_pixels

Run:  uvicorn server.server:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import base64, io, json, os, threading, time
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ovis_mlx.ovis2_5 import Ovis2_5
from ovis_mlx.preprocess import OvisProcessor
from ovis_mlx.generate import stream_generate

MODEL_DIR = os.environ.get("OVIS_MLX_DIR", os.path.join(ROOT, "ovis_mlx_model"))
HF_DIR = os.environ.get("OVIS_HF_DIR", os.path.join(ROOT, "Ovis2.5-2B"))
# Hard cap so a Retina capture cannot explode sequence length / unified memory.
MAX_PIXELS_CAP = int(os.environ.get("OVIS_MAX_PIXELS", str(1280 * 1280)))
MODEL_ID = "ovis2.5-2b-mlx"

app = FastAPI(title="Ovis2.5-MLX")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_lock = threading.Lock()
_state: dict[str, Any] = {"model": None, "proc": None}


@app.on_event("startup")
def _load():
    t = time.time()
    _state["model"] = Ovis2_5.load_mlx(MODEL_DIR)
    _state["proc"] = OvisProcessor(HF_DIR)
    print(f"[ovis] model ready in {time.time()-t:.1f}s  ({MODEL_DIR})", flush=True)


class ChatRequest(BaseModel):
    messages: List[dict]
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    repetition_penalty: float = 1.05
    stream: bool = True
    enable_thinking: bool = False
    enable_thinking_budget: bool = False
    thinking_budget: int = 1024
    max_pixels: Optional[int] = None
    model: Optional[str] = None


def _decode_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        url = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")


def _to_ovis_messages(messages: List[dict]) -> List[dict]:
    """OpenAI chat messages -> Ovis messages (image_url -> PIL image item)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        items = []
        for it in content or []:
            t = it.get("type")
            if t == "text":
                items.append({"type": "text", "text": it.get("text", "")})
            elif t == "image_url":
                u = it["image_url"]["url"] if isinstance(it["image_url"], dict) else it["image_url"]
                items.append({"type": "image", "image": _decode_image(u)})
        out.append({"role": m["role"], "content": items})
    return out


def _chunk(delta: dict, finish=None, cid="chatcmpl-ovis", created=0):
    return ("data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n")


def _run_stream(req: ChatRequest):
    model, proc = _state["model"], _state["proc"]
    msgs = _to_ovis_messages(req.messages)
    max_pixels = min(req.max_pixels or MAX_PIXELS_CAP, MAX_PIXELS_CAP)
    created = int(time.time())
    cid = f"chatcmpl-{created}"
    with _lock:
        yield _chunk({"role": "assistant", "content": ""}, created=created, cid=cid)
        finish = "stop"
        for ev in stream_generate(
            model, proc, msgs, max_tokens=req.max_tokens,
            temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
            repetition_penalty=req.repetition_penalty, max_pixels=max_pixels,
            enable_thinking=req.enable_thinking,
            enable_thinking_budget=req.enable_thinking_budget,
            thinking_budget=req.thinking_budget,
        ):
            if ev["type"] == "delta":
                yield _chunk({"content": ev["text"]}, created=created, cid=cid)
            else:
                finish = "stop" if ev.get("reason") == "stop" else "length"
        yield _chunk({}, finish=finish, created=created, cid=cid)
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if req.stream:
        return StreamingResponse(_run_stream(req), media_type="text/event-stream")
    # non-streaming: accumulate
    text, finish = "", "stop"
    with _lock:
        model, proc = _state["model"], _state["proc"]
        msgs = _to_ovis_messages(req.messages)
        max_pixels = min(req.max_pixels or MAX_PIXELS_CAP, MAX_PIXELS_CAP)
        for ev in stream_generate(
            model, proc, msgs, max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k, repetition_penalty=req.repetition_penalty,
            max_pixels=max_pixels, enable_thinking=req.enable_thinking,
            enable_thinking_budget=req.enable_thinking_budget,
            thinking_budget=req.thinking_budget):
            if ev["type"] == "delta":
                text += ev["text"]
            elif ev.get("reason") == "length":
                finish = "length"
    return JSONResponse({
        "id": "chatcmpl-ovis", "object": "chat.completion", "model": MODEL_ID,
        "choices": [{"index": 0, "finish_reason": finish,
                     "message": {"role": "assistant", "content": text}}],
    })


@app.get("/health")
def health():
    return {"status": "ok" if _state["model"] is not None else "loading",
            "model": MODEL_ID, "max_pixels_cap": MAX_PIXELS_CAP}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}
