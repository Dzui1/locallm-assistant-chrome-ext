"""Pure-Python preprocessing, ported verbatim from modeling_ovis2_5.py
(VisualTokenizer.preprocess / smart_resize / _tokenize_with_visual_placeholder /
_merge_inputs / preprocess_inputs). No torch model needed — only the HF tokenizer +
SiglipImageProcessor. Produces the exact (input_ids, pixel_values, grid_thws) the MLX
model consumes."""
from __future__ import annotations
import json, math, os
from typing import List, Union, Dict
import numpy as np
import PIL.Image

IMAGE_PLACEHOLDER = "<image>"
IMAGE_PLACEHOLDER_ID = -200
VIDEO_PLACEHOLDER = "<video>"
VIDEO_PLACEHOLDER_ID = -201
VISUAL_ATOM_ID = -300
INDICATOR_IDS = [-301, -302, -303, -304]

DEFAULT_MIN_PIXELS = 448 * 448
DEFAULT_MAX_PIXELS = 1344 * 1792


def smart_resize(height, width, factor, min_pixels, max_pixels):
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width); height = factor
        else:
            height = round(factor / width * height); width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class OvisProcessor:
    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer, AutoImageProcessor
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_dir, do_center_crop=False)
        with open(os.path.join(model_dir, "config.json")) as f:
            vit = json.load(f)["vit_config"]
        self.patch_size = vit["patch_size"]
        self.temporal_patch_size = vit["temporal_patch_size"]
        self.hidden_stride = vit["hidden_stride"]

    # --- image -> patches (port of VisualTokenizer.preprocess, image branch) ---
    def preprocess_image(self, image: PIL.Image.Image, min_pixels, max_pixels):
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        rh, rw = smart_resize(height, width,
                              factor=self.patch_size * self.hidden_stride,
                              min_pixels=min_pixels, max_pixels=max_pixels)
        arr = self.image_processor.preprocess(
            image, size=dict(height=rh, width=rw), return_tensors="np"
        )["pixel_values"][0]  # (C, rh, rw)
        patches = np.array([arr])
        if patches.shape[0] % self.temporal_patch_size != 0:
            rep = np.repeat(patches[-1][np.newaxis],
                            self.temporal_patch_size - 1, axis=0)
            patches = np.concatenate([patches, rep], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = rh // self.patch_size, rw // self.patch_size
        hs, ps = self.hidden_stride, self.patch_size
        patches = patches.reshape(grid_t, self.temporal_patch_size, channel,
                                  grid_h // hs, hs, ps, grid_w // hs, hs, ps)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(grid_t * grid_h * grid_w,
                               channel * self.temporal_patch_size * ps * ps)
        grid_thw = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
        return flat.astype(np.float32), grid_thw

    def _tokenize_with_visual_placeholder(self, text: str) -> List[int]:
        ph = VIDEO_PLACEHOLDER if VIDEO_PLACEHOLDER in text else IMAGE_PLACEHOLDER
        ph_id = VIDEO_PLACEHOLDER_ID if VIDEO_PLACEHOLDER in text else IMAGE_PLACEHOLDER_ID
        chunks = [self.tokenizer(c, add_special_tokens=False).input_ids
                  for c in text.split(ph)]
        ids = chunks[0]
        for c in chunks[1:]:
            ids.append(ph_id)
            ids.extend(c)
        return ids

    def _merge_inputs(self, raw_ids, placeholder_id, grid_thws, ind_begin, ind_end):
        ids = []
        prev = 0
        positions = [i for i, v in enumerate(raw_ids) if v == placeholder_id]
        for pos, grid in zip(positions, grid_thws):
            ids.extend(raw_ids[prev:pos])
            n = int(np.prod(grid)) // (self.hidden_stride ** 2) // self.temporal_patch_size
            ids.extend([ind_begin] + [VISUAL_ATOM_ID] * n + [ind_end])
            prev = pos + 1
        ids.extend(raw_ids[prev:])
        return ids

    def build_inputs(self, messages: List[Union[str, Dict]],
                     min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
                     add_generation_prompt=True, enable_thinking=False):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking)
        ids = self._tokenize_with_visual_placeholder(text)

        images = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                images.extend([it["image"] for it in c if it.get("image") is not None])

        pixel_values, grid_thws = None, None
        if images:
            pv, gt = zip(*(self.preprocess_image(im, min_pixels, max_pixels)
                           for im in images))
            ids = self._merge_inputs(ids, IMAGE_PLACEHOLDER_ID, gt,
                                     INDICATOR_IDS[0], INDICATOR_IDS[1])
            pixel_values = np.concatenate(pv, axis=0)
            grid_thws = np.concatenate(gt, axis=0)
        input_ids = np.array(ids, dtype=np.int64)[None]
        return input_ids, pixel_values, grid_thws
