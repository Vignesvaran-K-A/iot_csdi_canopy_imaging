# Canopy Vision - Automated Image Processing Portal

A high-performance web portal for automating the end-to-end canopy image processing pipeline previously executed manually in Colab/Jupyter notebooks.

---

## 🌟 Key Features

1. **High-Speed ZIP Upload (~100 MB+)**
   - Streamed chunk-based upload directly to disk (bypasses memory buffering).
   - Real-time upload speed (MB/s), elapsed time, transferred MBs, and progress bar.
2. **Automated Internal Pipeline Execution**
   - No separate notebook runs or manual variable adjustments (`day_09`, `day_03`, etc.).
   - Discovers all raw canopy images (`.jpg`, `.jpeg`, `.png`, etc.) recursively from the extracted archive.
   - Automatically executes:
     - **Depth Anything V2 Estimation**: Generates normalized metric depth maps.
     - **RGB to HSI Color Transformation**: Computes Hue, Saturation, and Intensity channels.
     - **HSIE Multi-Channel Fusion**: Fuses Hue ($H$), Saturation ($S$), and Normalized Depth ($D$) into composite 3-channel representations.
3. **Downloadable Processed ZIP**
   - Automatically packages all processed results (`hsie_enhanced`, `depth_maps`, `hsi_images`) into `processed_<dataset_name>.zip`.
   - 1-click persistent browser download.
4. **Interactive Sidebar Sample Comparison**
   - Automatically extracts sample pairs for instant visual verification.
   - Compares **Raw Canopy Image** with:
     - **HSIE Enhanced** (Fused)
     - **Depth Map** (Depth Anything V2)
     - **HSI Color**
   - Click any comparison card to inspect in full-resolution side-by-side zoom modal.
5. **Real-Time Live Logs & Execution Steps**
   - Server-Sent Events (SSE) live updates displaying active file name, processed count, and terminal log messages.

---

## 🚀 How to Run the Portal

### Method 1: Using the Batch Launcher (Easiest)
Simply double-click:
```bat
start.bat
```

### Method 2: Command Line (PowerShell / Command Prompt)
Navigate to the project directory and run:
```bash
python run_portal.py
```

The portal will start on **`http://127.0.0.1:8000`** and automatically open in your default browser!

---

## 📁 Output Directory Structure
When a dataset (e.g. `day_09.zip`) is processed, the system creates:
```
data/
├── uploads/
│   └── job_<id>/
│       └── day_09.zip
└── processed/
    └── job_<id>/
        ├── raw/                # Extracted original images
        ├── processed/
        │   ├── hsie_enhanced/  # Multi-channel fused images (*_hsie.png)
        │   ├── depth_maps/     # Metric depth estimations (*_depth.png)
        │   └── hsi_images/     # HSI color images (*_hsi.png)
        └── processed_day_09.zip # Ready-to-download archive
```

