import os
import json
import numpy as np
import cv2
from pathlib import Path

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk
    SIMPLEITK_AVAILABLE = True
except ImportError:
    SIMPLEITK_AVAILABLE = False
    print("Warning: SimpleITK not installed. MHA support disabled.")
    print("Install with: pip install SimpleITK")


class NODE21Preprocessor:
    def __init__(self, input_dir, output_dir, target_size=(512, 512), apply_clahe=True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CLAHE for chest X-ray contrast enhancement
        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Track processing results
        self.stats = {"success": 0, "failed": 0, "skipped": 0, "errors": []}

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_dicom(self, image_path):
        """Load a DICOM file and return a uint16 numpy array."""
        if not PYDICOM_AVAILABLE:
            raise RuntimeError("pydicom is required for .dcm files. pip install pydicom")

        ds = pydicom.dcmread(str(image_path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply DICOM rescale slope/intercept when present
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        pixel_array = pixel_array * slope + intercept

        metadata = self._extract_dicom_metadata(ds)
        return pixel_array, metadata

    def load_mha(self, image_path):
        """Load a .mha file and return a 2D float32 numpy array."""
        if not SIMPLEITK_AVAILABLE:
            raise RuntimeError("SimpleITK is required for .mha files. pip install SimpleITK")
        itk_image = sitk.ReadImage(str(image_path))
        array = sitk.GetArrayFromImage(itk_image).astype(np.float32)
        # .mha can be 3D (1, H, W) — squeeze to 2D
        if array.ndim == 3:
            array = array.squeeze()
        return array, {}

    def load_standard_image(self, image_path):
        """Load PNG/JPG/BMP with OpenCV."""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV could not read file: {image_path}")
        return image.astype(np.float32), {}

    def load_image(self, image_path):
        """Dispatch to the correct loader based on file extension."""
        suffix = image_path.suffix.lower()
        if suffix == ".dcm":
            return self.load_dicom(image_path)
        elif suffix == ".mha":
            return self.load_mha(image_path)
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
            return self.load_standard_image(image_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    # ------------------------------------------------------------------
    # Processing steps
    # ------------------------------------------------------------------

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]. Handles flat images safely."""
        image = image.astype(np.float32)
        min_val, max_val = image.min(), image.max()
        if max_val - min_val > 1e-6:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)  # Flat image → all zeros
        return image

    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to a [0, 1] float image.
        Converts to uint8, enhances, then returns float32 [0, 1].
        """
        img_uint8 = (image * 255).astype(np.uint8)
        enhanced = self.clahe.apply(img_uint8)
        return enhanced.astype(np.float32) / 255.0

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize using INTER_AREA (best for downscaling)."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Single-image pipeline
    # ------------------------------------------------------------------

    def preprocess(self, image_path: Path):
        """
        Full preprocessing pipeline for one image.

        Returns
        -------
        processed : np.ndarray, shape (H, W), float32 in [0, 1]
        metadata  : dict
        """
        image, metadata = self.load_image(image_path)

        # 1. Normalize first so CLAHE operates on a consistent range
        image = self.normalize_image(image)

        # 2. Optional CLAHE enhancement (improves nodule visibility)
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)

        # 3. Resize to target dimensions
        image = self.resize_image(image)

        # 4. Final normalize after resize (eliminates any interpolation drift)
        image = self.normalize_image(image)

        return image, metadata

    # ------------------------------------------------------------------
    # Dataset pipeline
    # ------------------------------------------------------------------

    def process_dataset(self, file_format=".dcm"):
        """
        Process every image in input_dir that matches file_format.

        Outputs
        -------
        - <stem>_processed.npy  : preprocessed float32 array
        - <stem>_metadata.json  : DICOM metadata (if available)
        - processing_summary.json: overall stats + per-file errors
        """
        suffix = file_format if file_format.startswith(".") else f".{file_format}"
        image_files = sorted(self.input_dir.glob(f"*{suffix}"))

        if not image_files:
            print(f"No {suffix} files found in {self.input_dir}")
            return

        print(f"Found {len(image_files)} {suffix} file(s) in {self.input_dir}")
        print(f"Target size : {self.target_size}")
        print(f"CLAHE       : {'enabled' if self.apply_clahe else 'disabled'}")
        print("-" * 50)

        for idx, image_path in enumerate(image_files, start=1):
            print(f"[{idx:>4}/{len(image_files)}] {image_path.name} ... ", end="", flush=True)
            try:
                processed, metadata = self.preprocess(image_path)

                # Save array
                out_npy = self.output_dir / f"{image_path.stem}_processed.npy"
                np.save(out_npy, processed)

                # Save metadata when available
                if metadata:
                    out_json = self.output_dir / f"{image_path.stem}_metadata.json"
                    with open(out_json, "w") as f:
                        json.dump(metadata, f, indent=2)

                self.stats["success"] += 1
                print(f"OK  shape={processed.shape}  range=[{processed.min():.3f}, {processed.max():.3f}]")

            except Exception as exc:
                self.stats["failed"] += 1
                error_msg = f"{image_path.name}: {exc}"
                self.stats["errors"].append(error_msg)
                print(f"FAILED — {exc}")

        self._print_summary(len(image_files))
        self._save_summary()

    # ------------------------------------------------------------------
    # Metadata + summary helpers
    # ------------------------------------------------------------------

    def _extract_dicom_metadata(self, ds) -> dict:
        """Pull commonly useful DICOM tags into a plain dict."""
        tags = {
            "PatientID":          "PatientID",
            "StudyDate":          "StudyDate",
            "Modality":           "Modality",
            "Manufacturer":       "Manufacturer",
            "Rows":               "Rows",
            "Columns":            "Columns",
            "PixelSpacing":       "PixelSpacing",
            "BitsStored":         "BitsStored",
            "PhotometricInterp":  "PhotometricInterpretation",
            "WindowCenter":       "WindowCenter",
            "WindowWidth":        "WindowWidth",
        }
        metadata = {}
        for key, attr in tags.items():
            value = getattr(ds, attr, None)
            if value is not None:
                # pydicom sequences / DSfloat → plain Python types
                try:
                    metadata[key] = value if isinstance(value, (str, int, float)) else str(value)
                except Exception:
                    pass
        return metadata

    def _print_summary(self, total: int):
        print("\n" + "=" * 50)
        print("Preprocessing complete")
        print(f"  Total   : {total}")
        print(f"  Success : {self.stats['success']}")
        print(f"  Failed  : {self.stats['failed']}")
        if self.stats["errors"]:
            print("\nFailed files:")
            for err in self.stats["errors"]:
                print(f"  • {err}")
        print("=" * 50)

    def _save_summary(self):
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Summary saved → {summary_path}")


# ---------------------------------------------------------------------------
# Quick verification helper
# ---------------------------------------------------------------------------

def verify_outputs(output_dir: str, n_samples: int = 3):
    """Print shape and value-range for the first n_samples .npy files."""
    out = Path(output_dir)
    npy_files = sorted(out.glob("*.npy"))

    if not npy_files:
        print("No .npy files found to verify.")
        return

    print(f"\nVerifying {min(n_samples, len(npy_files))} of {len(npy_files)} output(s):")
    for path in npy_files[:n_samples]:
        arr = np.load(path)
        print(f"  {path.name:40s}  shape={arr.shape}  dtype={arr.dtype}  "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT_DIR  = "/Users/felicianoserrano3/Desktop/AlgoverseMDLF/Node21_data/node21/cxr_images/original_data/images"
    OUTPUT_DIR = "/Users/felicianoserrano3/Desktop/AlgoverseMDLF/Node21_data/node21/cxr_images/proccessed_data"

    preprocessor = NODE21Preprocessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=(512, 512),
        apply_clahe=True,
    )

    preprocessor.process_dataset(file_format=".mha")

    verify_outputs(OUTPUT_DIR, n_samples=3)
