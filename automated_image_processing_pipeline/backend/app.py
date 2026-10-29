import os
import uuid
import time
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .config import BASE_DIR, UPLOAD_DIR, PROCESSED_DIR, CHUNK_SIZE_BYTES
from .pipeline import PipelineRunner

logger = logging.getLogger("portal_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Okra Canopy Image Processing Pipeline Portal", version="1.0.0")

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory jobs tracking
jobs_db: Dict[str, Dict[str, Any]] = {}
job_runners: Dict[str, PipelineRunner] = {}


def update_job_status(job_id: str, **kwargs):
    if job_id in jobs_db:
        jobs_db[job_id].update(kwargs)
        if "message" in kwargs and kwargs["message"]:
            log_entry = {
                "timestamp": time.strftime("%H:%M:%S"),
                "text": kwargs["message"]
            }
            jobs_db[job_id]["logs"].append(log_entry)
            # Limit logs to latest 200
            if len(jobs_db[job_id]["logs"]) > 200:
                jobs_db[job_id]["logs"].pop(0)


def run_pipeline_task(job_id: str, zip_path: Path):
    try:
        runner = PipelineRunner(job_id=job_id, zip_path=zip_path, output_base=PROCESSED_DIR)
        job_runners[job_id] = runner

        def callback(**kwargs):
            update_job_status(job_id, **kwargs)

        summary = runner.run(progress_callback=callback)
        update_job_status(job_id, status="completed", progress=100, summary=summary)
    except Exception as e:
        logger.exception(f"Pipeline error for job {job_id}: {e}")
        update_job_status(job_id, status="error", message=f"Pipeline failed: {str(e)}", error=str(e))


@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    High-speed streaming upload for zip datasets (supports 100MB+ quickly).
    Saves file in chunks to prevent memory bloat and initiates pipeline immediately.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip archive datasets are supported.")

    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    dataset_name = Path(file.filename).stem

    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    destination_zip = job_upload_dir / file.filename

    logger.info(f"Receiving upload: {file.filename} -> {destination_zip}")

    # Stream chunks of 1MB directly to disk
    bytes_written = 0
    start_time = time.time()
    
    with open(destination_zip, "wb") as buffer:
        while True:
            chunk = await file.read(CHUNK_SIZE_BYTES)
            if not chunk:
                break
            buffer.write(chunk)
            bytes_written += len(chunk)

    upload_duration = time.time() - start_time
    upload_mb = round(bytes_written / (1024 * 1024), 2)
    upload_speed = round(upload_mb / max(0.01, upload_duration), 2)
    logger.info(f"Uploaded {upload_mb} MB in {round(upload_duration, 2)}s ({upload_speed} MB/s).")

    # Initialize job in DB
    jobs_db[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "dataset_name": dataset_name,
        "zip_path": str(destination_zip),
        "size_mb": upload_mb,
        "status": "queued",
        "progress": 0,
        "processed": 0,
        "total": 0,
        "current_file": None,
        "message": f"Dataset uploaded ({upload_mb} MB). Queued for processing.",
        "logs": [
            {"timestamp": time.strftime("%H:%M:%S"), "text": f"Uploaded {file.filename} ({upload_mb} MB) at {upload_speed} MB/s."}
        ],
        "summary": None,
        "error": None,
        "created_at": time.time()
    }

    # Launch processing in background thread
    background_tasks.add_task(run_pipeline_task, job_id, destination_zip)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "dataset_name": dataset_name,
        "size_mb": upload_mb,
        "message": "Upload complete. Processing started in background."
    }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status, progress percentage, current file, and logs."""
    if job_id not in jobs_db:
        job_dir = PROCESSED_DIR / job_id
        if job_dir.exists():
            samples = get_all_processed_images(job_id)
            zip_files = list(job_dir.glob("processed_*.zip"))
            zip_name = zip_files[0].name if zip_files else None
            zip_size_mb = round(zip_files[0].stat().st_size / (1024 * 1024), 2) if zip_files else 0
            return {
                "job_id": job_id,
                "dataset_name": job_id,
                "status": "completed",
                "progress": 100,
                "processed": len(samples),
                "total": len(samples),
                "message": f"Dataset ready with {len(samples)} processed images.",
                "logs": [{"timestamp": "--:--", "text": "Dataset loaded from disk."}],
                "summary": {
                    "job_id": job_id,
                    "dataset_name": job_id,
                    "total_images": len(samples),
                    "processed_images": len(samples),
                    "duration_seconds": 0,
                    "zip_filename": zip_name,
                    "zip_size_mb": zip_size_mb,
                    "download_url": f"/api/download/{job_id}",
                    "sample_count": len(samples),
                    "samples": samples
                }
            }
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs_db[job_id]


@app.get("/api/progress/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """Server-Sent Events (SSE) stream for real-time progress updates without polling lag."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def event_generator():
        last_progress = -1
        last_message = ""
        while True:
            job = jobs_db.get(job_id)
            if not job:
                break
            
            # Send update if state changed
            cur_progress = job.get("progress", 0)
            cur_msg = job.get("message", "")
            cur_status = job.get("status", "")

            # Send JSON payload
            import json
            yield f"data: {json.dumps(job)}\n\n"

            if cur_status in ["completed", "error"]:
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


def get_all_processed_images(job_id: str) -> List[Dict[str, Any]]:
    """Scan and return all processed images for a job dynamically."""
    job_dir = PROCESSED_DIR / job_id
    hsie_dir = job_dir / "processed" / "hsie_enhanced"
    raw_dir = job_dir / "raw"
    
    if not hsie_dir.exists():
        return []
    
    # Map raw files by stem to their actual filenames
    raw_map = {}
    if raw_dir.exists():
        for root, _, files in os.walk(raw_dir):
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} and not f.startswith("._"):
                    raw_map[Path(f).stem] = f

    items = []
    try:
        hsie_files = sorted([f for f in os.listdir(hsie_dir) if f.endswith("_hsie.png")])
        for idx, hsie_file in enumerate(hsie_files):
            base_name = hsie_file[:-9]  # strip '_hsie.png'
            raw_filename = raw_map.get(base_name, f"{base_name}.jpg")
            depth_filename = f"{base_name}_depth.png"
            hsi_filename = f"{base_name}_hsi.png"
            
            items.append({
                "id": idx + 1,
                "filename": raw_filename,
                "base_name": base_name,
                "raw_url": f"/api/files/{job_id}/raw/{raw_filename}",
                "hsie_url": f"/api/files/{job_id}/hsie_enhanced/{hsie_file}",
                "depth_url": f"/api/files/{job_id}/depth_maps/{depth_filename}",
                "hsi_url": f"/api/files/{job_id}/hsi_images/{hsi_filename}",
            })
    except Exception as e:
        logger.error(f"Error listing samples for {job_id}: {e}")
    return items


@app.get("/api/samples/{job_id}")
async def get_job_samples(job_id: str):
    """Return all processed image samples (raw vs processed HSIE, Depth, HSI) for UI sidebar display."""
    if job_id not in jobs_db and not (PROCESSED_DIR / job_id).exists():
        raise HTTPException(status_code=404, detail="Job not found.")
    
    samples = get_all_processed_images(job_id)
    return {"samples": samples, "total": len(samples)}


@app.get("/api/latest_job")
async def get_latest_job():
    """Return the most recently processed job ID for persistent browser sessions."""
    if jobs_db:
        # Get latest from in-memory DB
        latest_id = list(jobs_db.keys())[-1]
        return jobs_db[latest_id]
    
    # Or check directory
    if PROCESSED_DIR.exists():
        job_folders = [d for d in PROCESSED_DIR.iterdir() if d.is_dir() and d.name.startswith("job_")]
        if job_folders:
            latest_folder = max(job_folders, key=lambda d: d.stat().st_mtime)
            job_id = latest_folder.name
            samples = get_all_processed_images(job_id)
            return {
                "job_id": job_id,
                "dataset_name": job_id,
                "status": "completed",
                "progress": 100,
                "message": f"Loaded existing dataset ({len(samples)} images).",
                "total": len(samples),
                "processed": len(samples),
                "logs": [{"timestamp": "--:--", "text": "Loaded previous job session."}],
                "summary": {
                    "job_id": job_id,
                    "processed_images": len(samples),
                    "sample_count": len(samples),
                    "samples": samples
                }
            }
    return {"job_id": None}


@app.get("/api/download/{job_id}")
async def download_processed_zip(job_id: str):
    """Direct high-speed browser download of the processed zip file."""
    job_dir = PROCESSED_DIR / job_id
    zip_path = None

    # 1. Search for any generated zip archive directly in the job directory
    if job_dir.exists():
        candidates = sorted(list(job_dir.glob("processed_*.zip")), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            zip_path = candidates[0]

    # 2. Check in jobs_db if runner created it
    if not zip_path and job_id in jobs_db:
        summary = jobs_db[job_id].get("summary")
        if summary and "zip_filename" in summary:
            candidate = job_dir / summary["zip_filename"]
            if candidate.exists():
                zip_path = candidate

    # 3. Check active runner
    if not zip_path:
        runner = job_runners.get(job_id)
        if runner and runner.zip_output_path.exists():
            zip_path = runner.zip_output_path

    if not zip_path or not zip_path.exists():
        raise HTTPException(status_code=404, detail=f"Processed ZIP archive for {job_id} not found on disk.")

    logger.info(f"Serving processed ZIP download: {zip_path} ({round(zip_path.stat().st_size / (1024*1024), 2)} MB)")

    return FileResponse(
        path=str(zip_path),
        filename=zip_path.name,
        media_type="application/zip"
    )


@app.get("/api/files/{job_id}/{folder}/{filename}")
async def serve_image_file(job_id: str, folder: str, filename: str):
    """Serve specific raw or processed image files for comparison thumbnails and modals."""
    job_dir = PROCESSED_DIR / job_id
    
    if folder == "raw":
        # Search in raw extracted folder recursively
        raw_dir = job_dir / "raw"
        for root, _, files in os.walk(raw_dir):
            if filename in files:
                return FileResponse(str(Path(root) / filename))
        raise HTTPException(status_code=404, detail=f"Raw image {filename} not found.")

    valid_folders = {"hsie_enhanced", "depth_maps", "hsi_images"}
    if folder not in valid_folders:
        raise HTTPException(status_code=400, detail="Invalid folder type.")

    target_path = job_dir / "processed" / folder / filename
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {filename} not found in {folder}.")

    return FileResponse(str(target_path))


# Mount Frontend static files
FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the single page portal application."""
    index_file = FRONTEND_DIR / "index.html"
    return FileResponse(str(index_file))

