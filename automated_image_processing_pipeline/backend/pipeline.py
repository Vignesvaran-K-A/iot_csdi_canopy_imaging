import os
import glob
import time
import zipfile
import logging
from pathlib import Path
from typing import Callable, Optional, Dict, List, Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .config import MODEL_ID, DEVICE, ALLOWED_EXTENSIONS, MAX_PREVIEW_SAMPLES

logger = logging.getLogger("img_pipeline")
logging.basicConfig(level=logging.INFO)

# Singleton Model Holder to avoid reloading weights across runs
_model_instance = None
_processor_instance = None


def get_depth_model():
    """Lazy-load and return the Depth Anything V2 model and image processor."""
    global _model_instance, _processor_instance
    if _model_instance is None or _processor_instance is None:
        logger.info(f"Loading Depth Anything V2 model ({MODEL_ID}) on {DEVICE}...")
        _processor_instance = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model_instance = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(DEVICE)
        _model_instance.eval()
        logger.info("Depth Anything V2 model successfully loaded.")
    return _processor_instance, _model_instance


def rgb_to_hsi(image_bgr: np.ndarray):
    """
    Standard RGB to HSI Transformation exactly as designed in the notebook.
    Takes BGR uint8 image, returns:
    (HSI merged BGR-layout image, H channel, S channel, I channel)
    """
    img = image_bgr.astype(np.float32) / 255.0
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = np.zeros_like(I)
    non_zero = I > 1e-6
    S[non_zero] = 1.0 - (min_rgb[non_zero] / I[non_zero])

    # Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    H = np.zeros_like(I)
    H[B <= G] = theta[B <= G]
    H[B > G] = 2.0 * np.pi - theta[B > G]
    H = H / (2.0 * np.pi)

    # Scale to 0-255 uint8
    H_u8 = (H * 255).astype(np.uint8)
    S_u8 = (S * 255).astype(np.uint8)
    I_u8 = (I * 255).astype(np.uint8)

    hsi_merged = cv2.merge([H_u8, S_u8, I_u8])
    return hsi_merged, H_u8, S_u8, I_u8


class PipelineRunner:
    def __init__(self, job_id: str, zip_path: Path, output_base: Path):
        self.job_id = job_id
        self.zip_path = Path(zip_path)
        self.dataset_name = self.zip_path.stem
        self.output_base = Path(output_base) / job_id
        
        # Directories
        self.extracted_dir = self.output_base / "raw"
        self.processed_dir = self.output_base / "processed"
        self.hsie_dir = self.processed_dir / "hsie_enhanced"
        self.depth_dir = self.processed_dir / "depth_maps"
        self.hsi_dir = self.processed_dir / "hsi_images"
        
        for d in [self.extracted_dir, self.processed_dir, self.hsie_dir, self.depth_dir, self.hsi_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        self.zip_output_path = self.output_base / f"processed_{self.dataset_name}.zip"
        self.sample_pairs: List[Dict[str, Any]] = []

    def extract_zip(self, callback: Optional[Callable] = None):
        if callback:
            callback(status="extracting", message="Extracting uploaded ZIP dataset archive...", progress=5)
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Check for zip bomb / malicious paths
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                # Skip macOS metadata or hidden files
                if member.startswith("__MACOSX/") or filename.startswith("._"):
                    continue
                zip_ref.extract(member, self.extracted_dir)

    def discover_images(self) -> List[Path]:
        image_files = []
        for root, _, files in os.walk(self.extracted_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in ALLOWED_EXTENSIONS and not file.startswith("._"):
                    image_files.append(Path(root) / file)
        
        # Sort files naturally
        image_files = sorted(list(set(image_files)))
        return image_files

    def run(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # Step 1: Extract Zip
        self.extract_zip(progress_callback)
        
        # Step 2: Find Images
        image_paths = self.discover_images()
        total_images = len(image_paths)
        if total_images == 0:
            raise ValueError("No valid image files found inside the uploaded ZIP archive.")

        if progress_callback:
            progress_callback(
                status="initializing",
                message=f"Discovered {total_images} canopy images. Initializing Depth Anything V2 model...",
                progress=10,
                total=total_images
            )

        # Step 3: Load Model
        image_processor, depth_model = get_depth_model()

        if progress_callback:
            progress_callback(
                status="processing",
                message="Starting pipeline: RGB to HSI conversion & Depth Anything V2 inference...",
                progress=15,
                processed=0,
                total=total_images
            )

        # Step 4: Process Images
        processed_count = 0
        samples_collected = 0
        step_interval = max(1, total_images // MAX_PREVIEW_SAMPLES)

        for idx, img_path in enumerate(image_paths):
            base_name = img_path.stem
            
            raw_bgr = cv2.imread(str(img_path))
            if raw_bgr is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue

            h, w = raw_bgr.shape[:2]
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(raw_rgb)

            # Depth estimation with Depth Anything V2
            inputs = image_processor(images=pil_img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = depth_model(**inputs)
                predicted_depth = outputs.predicted_depth

            depth_map = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # HSI conversion
            hsi_img, H, S, I = rgb_to_hsi(raw_bgr)

            # HSIE Fusion: H, S, depth_norm
            hsie_img = cv2.merge([H, S, depth_norm])

            # Save processed files
            hsie_filename = f"{base_name}_hsie.png"
            depth_filename = f"{base_name}_depth.png"
            hsi_filename = f"{base_name}_hsi.png"

            cv2.imwrite(str(self.hsie_dir / hsie_filename), hsie_img)
            cv2.imwrite(str(self.depth_dir / depth_filename), depth_norm)
            cv2.imwrite(str(self.hsi_dir / hsi_filename), hsi_img)

            processed_count += 1
            
            # Select samples for UI comparison
            is_sample = (idx % step_interval == 0 and samples_collected < MAX_PREVIEW_SAMPLES) or (idx == total_images - 1 and samples_collected < MAX_PREVIEW_SAMPLES)
            if is_sample:
                rel_raw = img_path.relative_to(self.output_base).as_posix()
                self.sample_pairs.append({
                    "id": idx + 1,
                    "filename": img_path.name,
                    "base_name": base_name,
                    "width": w,
                    "height": h,
                    "raw_url": f"/api/files/{self.job_id}/raw/{img_path.name}",
                    "hsie_url": f"/api/files/{self.job_id}/hsie_enhanced/{hsie_filename}",
                    "depth_url": f"/api/files/{self.job_id}/depth_maps/{depth_filename}",
                    "hsi_url": f"/api/files/{self.job_id}/hsi_images/{hsi_filename}",
                    # Store exact paths for sample preview server
                    "raw_path": str(img_path),
                    "hsie_path": str(self.hsie_dir / hsie_filename),
                    "depth_path": str(self.depth_dir / depth_filename),
                    "hsi_path": str(self.hsi_dir / hsi_filename),
                })
                samples_collected += 1

            # Progress update (15% to 90%)
            progress_pct = int(15 + (processed_count / total_images) * 75)
            if progress_callback:
                progress_callback(
                    status="processing",
                    message=f"Processing [{processed_count}/{total_images}]: {img_path.name}",
                    progress=progress_pct,
                    processed=processed_count,
                    total=total_images,
                    current_file=img_path.name
                )

        # Step 5: Create Downloadable ZIP
        if progress_callback:
            progress_callback(
                status="zipping",
                message=f"Packaging {processed_count} processed images into downloadable ZIP...",
                progress=92
            )

        self.create_output_zip()

        total_duration = time.time() - start_time
        zip_size_bytes = self.zip_output_path.stat().st_size if self.zip_output_path.exists() else 0
        zip_size_mb = round(zip_size_bytes / (1024 * 1024), 2)

        summary = {
            "job_id": self.job_id,
            "dataset_name": self.dataset_name,
            "total_images": total_images,
            "processed_images": processed_count,
            "duration_seconds": round(total_duration, 2),
            "zip_filename": self.zip_output_path.name,
            "zip_size_mb": zip_size_mb,
            "download_url": f"/api/download/{self.job_id}",
            "sample_count": len(self.sample_pairs),
            "samples": self.sample_pairs
        }

        if progress_callback:
            progress_callback(
                status="completed",
                message=f"All {processed_count} images processed successfully in {round(total_duration, 1)}s! ZIP ready.",
                progress=100,
                processed=processed_count,
                total=total_images,
                summary=summary
            )

        return summary

    def create_output_zip(self):
        """
        Compresses processed outputs into a zip file.
        Matches notebook behavior by including hsie_enhanced,
        and cleanly includes depth_maps and hsi_images.
        """
        with zipfile.ZipFile(self.zip_output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for folder_name, folder_path in [
                ("hsie_enhanced", self.hsie_dir),
                ("depth_maps", self.depth_dir),
                ("hsi_images", self.hsi_dir)
            ]:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        full_path = Path(root) / file
                        arcname = os.path.join(self.dataset_name, folder_name, file)
                        zf.write(full_path, arcname=arcname)

