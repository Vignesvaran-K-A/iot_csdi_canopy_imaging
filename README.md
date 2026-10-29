# IoT CSDI Canopy Imaging

Repository for the IoT Canopy Imaging research project and image processing pipelines.

## 📁 Repository Structure

* **[`automated_image_processing_pipeline/`](./automated_image_processing_pipeline)**: End-to-end automated web portal for canopy image processing.
  * **Backend**: FastAPI web server with streaming 100MB+ ZIP upload, live SSE progress tracking, and high-speed ZIP export.
  * **Core Pipeline**: Depth Anything V2 depth estimation, RGB-to-HSI color transformation, and HSIE multi-channel fusion.
  * **Frontend**: Interactive web dashboard with real-time progress monitor, execution log terminal, and sidebar with sample comparison viewer (Raw vs HSIE / Depth / HSI).
  * **Notebooks**: Original research Colab notebook (`Okra_Shoot_HSI_Depth_Annotation.ipynb`).

---

## 🚀 Quick Start (Automated Image Pipeline Portal)

Navigate into the pipeline folder:
```bash
cd automated_image_processing_pipeline
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the portal:
```bash
python run_portal.py
```
Or double-click `start.bat` on Windows. The portal opens at **`http://127.0.0.1:8000`**.

See the [Pipeline Documentation](./automated_image_processing_pipeline/README.md) for full details.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
