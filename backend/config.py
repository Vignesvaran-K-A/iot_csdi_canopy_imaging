import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Model configuration
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Limits & settings
MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB max for uploaded ZIP
CHUNK_SIZE_BYTES = 1024 * 1024             # 1 MB stream chunk
MAX_PREVIEW_SAMPLES = 8                    # Number of raw vs processed samples in UI

