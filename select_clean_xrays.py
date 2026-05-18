"""
select_clean_xrays.py
---------------------
Identifies 50 clean background X-rays from NODE21 (no annotated nodules)
to serve as canvases for synthetic lesion insertion.

Selection criteria:
  1. Image does not appear in metadata.csv (which lists only nodule-positive images)
  2. Corresponding preprocessed .npy file exists and is valid
  3. Pixel value range is [0, 1] with sufficient contrast (std > 0.05)
     to exclude blank/corrupt scans
  4. Final 50 selected via random sample with a fixed seed (42) for reproducibility
"""

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ORIGINAL_DIR   = Path("/Users/felicianoserrano3/Desktop/AlgoverseMDLF/Node21_data/node21/cxr_images/original_data/images")
METADATA_CSV   = Path("/Users/felicianoserrano3/Desktop/AlgoverseMDLF/Node21_data/node21/cxr_images/original_data/metadata.csv")
PROCESSED_DIR  = Path("/Users/felicianoserrano3/Desktop/AlgoverseMDLF/Node21_data/node21/cxr_images/proccessed_data")
OUTPUT_DIR     = Path("/Users/felicianoserrano3/Desktop/AlgoverseMDLF/Node21_data/node21/cxr_images/clean_backgrounds")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLEAN        = 50
RANDOM_SEED    = 42
MIN_STD        = 0.05   # reject blank/near-uniform images

# ── Step 1: Load metadata and split by label ──────────────────────────────
print("Loading metadata...")
meta = pd.read_csv(METADATA_CSV)
nodule_images = set(meta[meta["label"] == 1]["img_name"].unique())
clean_in_meta = set(meta[meta["label"] == 0]["img_name"].unique())
print(f"  Nodule-positive (label=1) : {len(nodule_images)}")
print(f"  Clean (label=0)           : {len(clean_in_meta)}")

# ── Step 2: All .mha files on disk ────────────────────────────────────────
all_images = {p.name for p in ORIGINAL_DIR.glob("*.mha")}
print(f"  Total .mha images on disk : {len(all_images)}")

# ── Step 3: Clean candidates = label=0 AND not also in label=1 ────────────
candidate_names = sorted(clean_in_meta - nodule_images)
print(f"  Nodule-free candidates    : {len(candidate_names)}")

# ── Step 4: Quality filter — check preprocessed .npy files ────────────────
print("\nRunning quality filter on candidates...")
passed, rejected = [], []

for name in candidate_names:
    stem = Path(name).stem
    npy_path = PROCESSED_DIR / f"{stem}_processed.npy"

    if not npy_path.exists():
        rejected.append({"img_name": name, "reason": "no preprocessed .npy found"})
        continue

    arr = np.load(npy_path)

    # Check value range
    if arr.min() < 0 or arr.max() > 1:
        rejected.append({"img_name": name, "reason": f"value range out of bounds [{arr.min():.3f}, {arr.max():.3f}]"})
        continue

    # Check contrast
    std = float(arr.std())
    if std < MIN_STD:
        rejected.append({"img_name": name, "reason": f"insufficient contrast (std={std:.4f} < {MIN_STD})"})
        continue

    passed.append({
        "img_name":  name,
        "npy_path":  str(npy_path),
        "std":       round(std, 4),
        "min":       round(float(arr.min()), 4),
        "max":       round(float(arr.max()), 4),
        "shape":     list(arr.shape),
    })

print(f"  Passed quality filter : {len(passed)}")
print(f"  Rejected              : {len(rejected)}")

if len(passed) < N_CLEAN:
    print(f"\nWARNING: Only {len(passed)} images passed — selecting all of them.")
    N_CLEAN = len(passed)

# ── Step 5: Random sample ──────────────────────────────────────────────────
random.seed(RANDOM_SEED)
selected = random.sample(passed, N_CLEAN)
selected_sorted = sorted(selected, key=lambda x: x["img_name"])

# ── Step 6: Save outputs ───────────────────────────────────────────────────

# Selected images list
selection_path = OUTPUT_DIR / "clean_background_selection.json"
with open(selection_path, "w") as f:
    json.dump(selected_sorted, f, indent=2)

# Rejected log
rejected_path = OUTPUT_DIR / "rejected_candidates.json"
with open(rejected_path, "w") as f:
    json.dump(rejected, f, indent=2)

# Human-readable summary + documented criteria
summary = {
    "selection_criteria": {
        "1_no_nodule_annotation": "Image name does not appear in metadata.csv (label=1 rows only)",
        "2_preprocessed_file_exists": "Corresponding _processed.npy file exists in proccessed_data/",
        "3_valid_value_range": "Pixel values in [0.0, 1.0] after preprocessing",
        "4_sufficient_contrast": f"Pixel std > {MIN_STD} to exclude blank or corrupt scans",
        "5_random_sampling": f"Final {N_CLEAN} selected via random.sample(seed={RANDOM_SEED}) for reproducibility",
    },
    "counts": {
        "total_images_on_disk":     len(all_images),
        "nodule_positive":          len(nodule_images),
        "nodule_free_candidates":   len(candidate_names),
        "passed_quality_filter":    len(passed),
        "rejected":                 len(rejected),
        "selected":                 N_CLEAN,
    },
    "selected_images": [s["img_name"] for s in selected_sorted],
}

summary_path = OUTPUT_DIR / "selection_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

# ── Step 7: Print results ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Selected {N_CLEAN} clean background X-rays")
print(f"{'='*50}")
print(f"\nSelection criteria:")
for k, v in summary["selection_criteria"].items():
    print(f"  {k}: {v}")

print(f"\nCounts:")
for k, v in summary["counts"].items():
    print(f"  {k:<30}: {v}")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  {selection_path.name}")
print(f"  {rejected_path.name}")
print(f"  {summary_path.name}")

print(f"\nFirst 5 selected images:")
for s in selected_sorted[:5]:
    print(f"  {s['img_name']}  std={s['std']}  range=[{s['min']}, {s['max']}]")
