# Clean Background X-Ray Selection — NODE21

This document describes the 50 clean chest X-rays selected from the NODE21 dataset to serve as canvases for synthetic lesion insertion.

---

## How to Access the Selected Images

The file `clean_background_selection.json` in this folder contains the exact list of 50 selected filenames. To load them:

```python
import json
import numpy as np
from pathlib import Path

with open("clean_background_selection.json") as f:
    selection = json.load(f)

PROCESSED_DIR = Path("/path/to/cxr_images/proccessed_data")

images = []
for entry in selection:
    stem = Path(entry["img_name"]).stem
    arr = np.load(PROCESSED_DIR / f"{stem}_processed.npy")
    images.append(arr)

print(f"Loaded {len(images)} clean background images, shape: {images[0].shape}")
```

Replace `/path/to/cxr_images/proccessed_data` with the path on your machine.

---

## Selection Criteria

Images were selected by running `select_clean_xrays.py` against the NODE21 dataset. The following criteria were applied in order:

| # | Criterion | Detail |
|---|---|---|
| 1 | **No nodule annotation** | `label == 0` in `metadata.csv` — excludes all images with annotated nodules |
| 2 | **Preprocessed file exists** | Corresponding `_processed.npy` file present in `proccessed_data/` |
| 3 | **Valid value range** | Pixel values within `[0.0, 1.0]` after preprocessing |
| 4 | **Sufficient contrast** | Pixel standard deviation `> 0.05` — excludes blank or corrupt scans |
| 5 | **Random sampling** | Final 50 drawn via `random.sample(seed=42)` for reproducibility |

---

## Dataset Counts

| | Count |
|---|---|
| Total images on disk | 4,882 |
| Nodule-positive (label=1) | 1,134 |
| Nodule-free candidates (label=0) | 3,748 |
| Passed quality filter | 3,748 |
| Rejected | 0 |
| **Selected** | **50** |

---

## Reproducing the Selection

To reproduce the exact same 50 images from scratch, run:

```bash
python3 select_clean_xrays.py
```

The random seed is fixed at `42` inside the script, so the output is identical every time as long as the NODE21 dataset is unchanged.

---

## Output Files

| File | Description |
|---|---|
| `clean_background_selection.json` | List of 50 selected images with filename, std, and value range |
| `rejected_candidates.json` | Log of any candidates that failed quality checks |
| `selection_summary.json` | Full counts and criteria in machine-readable format |
| `CLEAN_BACKGROUNDS_README.md` | This file |

---

## Preprocessing Details

All images were preprocessed by `node21_preprocessor.py` before selection:
- Resized to **512 × 512**
- Normalized to **[0.0, 1.0]** float32
- CLAHE contrast enhancement applied

Preprocessed arrays are saved as `.npy` files in `proccessed_data/`.
