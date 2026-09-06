"""M3 — Convert the HF Ovis2.5-2B checkpoint to an MLX model directory.

Quantization policy (tuned for 18-32 GB unified memory):
  * Qwen3 LLM -> 4-bit (group size 64) — this is where the memory win is.
  * Vision tower + visual head + VTE -> bf16 — small (~400M) and precision-sensitive
    for the soft-token / grounding path, so left near-lossless.

Usage:
  python convert.py [--out ovis_mlx_model] [--bits 4] [--group-size 64]
                    [--vision-dtype bf16] [--no-quant]
"""

import argparse
import os

import mlx.core as mx

# import Ovis2_5 class from ovis_mlx/ovis2_5.py
from ovis_mlx.ovis2_5 import Ovis2_5

ROOT = os.path.dirname(os.path.abspath(__file__))
DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}


def main():
    # Add arguments options for bash/ convert.py
    # Each args are saved to args.[x] where x = --[x]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "Ovis2.5-2B"))
    ap.add_argument("--out", default=os.path.join(ROOT, "ovis_mlx_model"))
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--vision-dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--no-quant", action="store_true")
    args = ap.parse_args()

    print("loading fp32 checkpoint...", flush=True)
    model = Ovis2_5(args.model_dir).load_fp32()
    model.cast_vision(DTYPES[args.vision_dtype])

    quant = None
    # args.no_quant updated here.
    if not args.no_quant:
        print(
            f"quantizing LLM to {args.bits}-bit (gs={args.group_size})...", flush=True
        )
        model.quantize_llm(bits=args.bits, group_size=args.group_size)
        quant = {"bits": args.bits, "group_size": args.group_size}

    model.save_mlx(args.out, quant=quant)
    size = (
        sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
        / 1e9
    )
    print(f"saved -> {args.out}  ({size:.2f} GB on disk)", flush=True)


if __name__ == "__main__":
    main()
