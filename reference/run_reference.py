"""M0 — Reference oracle.

Runs the official Ovis2.5 modeling file on CPU with eager attention (flash-attn is
absent on macOS, so the modeling code already falls back to SDPA; we force eager for
determinism). Dumps a fixed input + every interesting intermediate tensor so the MLX
port can be checked stage-by-stage.

Outputs (in artifacts/):
  test_image.png        the fixed input image
  ref_inputs.npz        input_ids, pixel_values, grid_thws  (exact MLX inputs)
  ref_dump.npz          patch_embeds, per-layer vit hidden, vit_out, soft_tokens,
                        visual_embeds, merged_embeds, first_logits
  ref_meta.json         prompt, greedy token ids + decoded text, shapes
Run:  python reference/run_reference.py
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODEL_PATH = os.path.join(ROOT, "Ovis2.5-2B")
ART = os.path.join(ROOT, "artifacts")
sys.path.insert(0, HERE)
from make_test_image import make_image

torch.manual_seed(0)
PROMPT = "Describe what is on this screen. Which test failed?"
MAX_PIXELS = 896 * 896          # cap to keep the reference cheap/deterministic
N_GREEDY = 48


def main():
    os.makedirs(ART, exist_ok=True)
    from PIL import Image
    img_path = make_image()
    image = Image.open(img_path).convert("RGB")

    print("loading model on CPU (float32, eager attn)...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages, max_pixels=MAX_PIXELS, enable_thinking=False
    )
    pixel_values = pixel_values.to(torch.float32)
    print("input_ids", tuple(input_ids.shape),
          "pixel_values", tuple(pixel_values.shape),
          "grid_thws", grid_thws.tolist(), flush=True)

    np.savez(os.path.join(ART, "ref_inputs.npz"),
             input_ids=input_ids.cpu().numpy(),
             pixel_values=pixel_values.cpu().numpy(),
             grid_thws=grid_thws.cpu().numpy())

    dump = {}
    vit = model.visual_tokenizer.vit.vision_model
    handles = []

    def save(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            dump[name] = t.detach().cpu().float().numpy()
        return hook

    handles.append(vit.embeddings.register_forward_hook(save("patch_embeds")))
    for i, layer in enumerate(vit.encoder.layers):
        handles.append(layer.register_forward_hook(save(f"vit_layer_{i:02d}")))
    handles.append(vit.post_layernorm.register_forward_hook(save("vit_out")))
    handles.append(model.visual_tokenizer.register_forward_hook(save("soft_tokens")))
    handles.append(model.vte.register_forward_hook(save("vte_out")))

    print("running merge_multimodal (vision tower + tokenizer + vte + merge)...", flush=True)
    with torch.no_grad():
        merged = model.merge_multimodal(
            input_ids=input_ids, pixel_values=pixel_values, grid_thws=grid_thws
        )
    dump["merged_embeds"] = merged.detach().cpu().float().numpy()
    for h in handles:
        h.remove()

    print("first-step LLM logits...", flush=True)
    with torch.no_grad():
        attn = torch.ne(input_ids, model.text_tokenizer.pad_token_id)
        out = model.llm(inputs_embeds=merged, attention_mask=attn)
        first_logits = out.logits[:, -1, :].detach().cpu().float().numpy()
    dump["first_logits"] = first_logits

    print(f"greedy generate {N_GREEDY} tokens...", flush=True)
    with torch.no_grad():
        gen = model.generate(
            input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
            max_new_tokens=N_GREEDY, do_sample=False, num_beams=1,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=model.text_tokenizer.pad_token_id,
        )
    gen_ids = gen[0].cpu().tolist()
    gen_text = model.text_tokenizer.decode(gen_ids, skip_special_tokens=True)
    print("=== GREEDY OUTPUT ===\n" + gen_text + "\n=====================", flush=True)

    np.savez(os.path.join(ART, "ref_dump.npz"), **dump)
    top10 = np.argsort(-first_logits[0])[:10].tolist()
    meta = {
        "prompt": PROMPT, "max_pixels": MAX_PIXELS,
        "input_ids_len": int(input_ids.shape[1]),
        "grid_thws": grid_thws.tolist(),
        "greedy_token_ids": gen_ids,
        "greedy_text": gen_text,
        "first_logits_top10": top10,
        "first_argmax": int(np.argmax(first_logits[0])),
        "dump_shapes": {k: list(v.shape) for k, v in dump.items()},
    }
    with open(os.path.join(ART, "ref_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved artifacts:", sorted(os.listdir(ART)), flush=True)


if __name__ == "__main__":
    main()
