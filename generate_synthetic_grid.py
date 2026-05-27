# -*- coding: utf-8 -*-
"""
generate_synthetic_grid.py

Full RadEdit generation pipeline for the synthetic lesion failure mapping project.
Reads the attribute grid spec CSV, loads each preprocessed .npy background X-ray,
inserts a synthetic nodule using RadEdit, and saves the output image + metadata JSON.

Usage:
    python generate_synthetic_grid.py

Outputs:
    /Users/felicianoserrano3/Desktop/synthetic_lesion_outputs/images/
    /Users/felicianoserrano3/Desktop/synthetic_lesion_outputs/metadata/
    /Users/felicianoserrano3/Desktop/synthetic_lesion_outputs/generation_log.csv
"""

# ── 0. Install & auth (run once) ──────────────────────────────────────────────
# pip install diffusers transformers torch torchvision huggingface-hub accelerate Pillow
# hf auth login

import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from pathlib import Path
from datetime import datetime

from transformers import AutoModel, AutoTokenizer
from diffusers import (
    AutoencoderKL, DDIMScheduler,
    StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline
)

# ── 1. Config ──────────────────────────────────────────────────────────────────

IMAGE_DIR = Path("/Users/felicianoserrano3/Desktop/AlgoverseMDLF/Node21_data/node21/cxr_images/proccessed_data")
GRID_CSV  = Path("/Users/felicianoserrano3/Desktop/synthetic_grid_spec.csv")
OUT_DIR   = Path("/Users/felicianoserrano3/Desktop/synthetic_lesion_outputs")

IMG_SIZE        = 512
NUM_STEPS       = 200
GUIDANCE_SCALE  = 7.5
SKIP_RATIO      = 0.3
MASK_PADDING    = 24

# Device: CUDA on Linux/Windows, MPS on Apple Silicon, CPU as fallback
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.float16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE  = torch.float32   # MPS does not support float16 for all ops
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32

print(f"Using device: {DEVICE}")

(OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)

# ── 2. Load RadEdit pipeline ───────────────────────────────────────────────────

print("Loading RadEdit pipeline...")

unet = UNet2DConditionModel.from_pretrained(
    "microsoft/radedit", subfolder="unet", torch_dtype=DTYPE
)
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
text_encoder = AutoModel.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T", model_max_length=128, trust_remote_code=True
)
scheduler = DDIMScheduler(
    beta_schedule="linear",
    clip_sample=False,
    prediction_type="epsilon",
    timestep_spacing="trailing",
    steps_offset=1,
)

gen_pipeline = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
).to(DEVICE)

radedit = DiffusionPipeline.from_pipe(
    gen_pipeline, custom_pipeline="microsoft/radedit", trust_remote_code=True
)

print("Pipeline ready.\n")

# ── 3. Helper functions ────────────────────────────────────────────────────────

def load_npy_as_pil(npy_path: Path) -> Image.Image:
    """
    Load a .npy preprocessed X-ray and convert to a 512x512 PIL RGB image.
    Normalises to [0, 255] and converts grayscale to RGB (RadEdit expects RGB).
    """
    arr = np.load(npy_path)

    # If shape is (H, W) or (1, H, W), squeeze to (H, W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)

    # Normalise to uint8
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img


def make_masks(gt_x, gt_y, gt_w, gt_h, src_w, src_h):
    """
    Build edit_mask and keep_mask from the ground-truth bounding box.
    The edit mask is a padded ellipse around the nodule location.
    """
    scale_x = IMG_SIZE / src_w
    scale_y = IMG_SIZE / src_h

    x0 = int((gt_x - MASK_PADDING) * scale_x)
    y0 = int((gt_y - MASK_PADDING) * scale_y)
    x1 = int((gt_x + gt_w + MASK_PADDING) * scale_x)
    y1 = int((gt_y + gt_h + MASK_PADDING) * scale_y)

    x0 = max(0, min(x0, IMG_SIZE - 1))
    y0 = max(0, min(y0, IMG_SIZE - 1))
    x1 = max(0, min(x1, IMG_SIZE))
    y1 = max(0, min(y1, IMG_SIZE))

    # Ensure x1 > x0 and y1 > y0
    if x1 <= x0:
        x1 = min(IMG_SIZE, x0 + 20)
    if y1 <= y0:
        y1 = min(IMG_SIZE, y0 + 20)

    edit_mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(edit_mask)
    draw.ellipse([x0, y0, x1, y1], fill=255)

    keep_mask = ImageOps.invert(edit_mask)
    return edit_mask, keep_mask


def strip_mha_extension(source_name: str) -> str:
    return source_name.replace(".mha", "_processed")

# ── 4. Main generation loop ────────────────────────────────────────────────────

df = pd.read_csv(GRID_CSV)
print(f"Loaded grid spec: {len(df)} images to generate.\n")

log_rows = []
failed   = []

for idx, row in df.iterrows():
    synth_id    = row["synth_image_id"]
    source_name = row["source_img_name"]
    prompt      = row["radedit_prompt"]
    seed        = int(row["generation_seed"])
    gt_x, gt_y  = int(row["gt_x"]), int(row["gt_y"])
    gt_w, gt_h  = int(row["gt_width"]), int(row["gt_height"])

    base_name = strip_mha_extension(source_name)
    npy_path  = IMAGE_DIR / f"{base_name}.npy"

    if not npy_path.exists():
        print(f"[SKIP] {synth_id}: .npy not found at {npy_path}")
        failed.append({"synth_image_id": synth_id, "reason": "npy_not_found"})
        continue

    try:
        input_image = load_npy_as_pil(npy_path)
        src_w, src_h = input_image.size

        edit_mask, keep_mask = make_masks(gt_x, gt_y, gt_w, gt_h, src_w, src_h)

        torch.manual_seed(seed)
        result = radedit(
            prompt,
            weights=[GUIDANCE_SCALE],
            image=input_image,
            edit_mask=edit_mask,
            keep_mask=keep_mask,
            num_inference_steps=NUM_STEPS,
            invert_prompt="",
            skip_ratio=SKIP_RATIO,
            output_type="pil",
        )
        edited_image = result[0]

        out_img_path = OUT_DIR / "images" / f"{synth_id}.png"
        edited_image.save(out_img_path)

        metadata = {
            "synth_image_id":  synth_id,
            "image_path":      str(out_img_path),
            "source_image":    source_name,
            "size":            row["size_bin"],
            "location":        row["location_zone"],
            "contrast":        row["contrast_bin"],
            "anatomy_overlap": row["anatomy_overlap"],
            "prompt":          prompt,
            "skip_ratio":      SKIP_RATIO,
            "guidance_scale":  GUIDANCE_SCALE,
            "num_steps":       NUM_STEPS,
            "seed":            seed,
            "gt_x":            gt_x,
            "gt_y":            gt_y,
            "gt_width":        gt_w,
            "gt_height":       gt_h,
            "qc_pass":         None,
            "generated_at":    datetime.utcnow().isoformat(),
        }

        meta_path = OUT_DIR / "metadata" / f"{synth_id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        log_rows.append({**metadata, "status": "success"})
        print(f"[OK] {synth_id} — {prompt[:60]}")

    except Exception as e:
        print(f"[ERROR] {synth_id}: {e}")
        failed.append({"synth_image_id": synth_id, "reason": str(e)})

# ── 5. Save generation log ─────────────────────────────────────────────────────

log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUT_DIR / "generation_log.csv", index=False)

print(f"\n{'='*50}")
print(f"Generation complete.")
print(f"  Successful:     {len(log_rows)}")
print(f"  Failed/skipped: {len(failed)}")
if failed:
    print(f"\nFailed images:")
    for f in failed:
        print(f"  {f['synth_image_id']}: {f['reason']}")
print(f"\nOutputs saved to: {OUT_DIR.resolve()}")
