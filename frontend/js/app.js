// Automated Image Processing Portal - Frontend Application
document.addEventListener("DOMContentLoaded", () => {
  // Initialize Lucide icons
  if (window.lucide) {
    lucide.createIcons();
  }

  // State
  let selectedFile = null;
  let activeJobId = null;
  let activeViewMode = "hsie"; // 'hsie' | 'depth' | 'hsi'
  let cachedSamples = [];
  let sseSource = null;
  let pollInterval = null;

  // DOM Elements
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const selectedFileBar = document.getElementById("selected-file-bar");
  const selectedFileName = document.getElementById("selected-file-name");
  const selectedFileSize = document.getElementById("selected-file-size");
  const removeFileBtn = document.getElementById("remove-file-btn");
  const startBtn = document.getElementById("start-btn");

  const uploadProgressContainer = document.getElementById("upload-progress-container");
  const uploadProgressBar = document.getElementById("upload-progress-bar");
  const uploadStatusText = document.getElementById("upload-status-text");
  const uploadPercentText = document.getElementById("upload-percent-text");
  const uploadSpeedText = document.getElementById("upload-speed-text");
  const uploadSizeText = document.getElementById("upload-size-text");

  const statusSpinner = document.getElementById("status-spinner");
  const badgeStatus = document.getElementById("badge-status");
  const pipelineProgressBar = document.getElementById("pipeline-progress-bar");
  const progressMessage = document.getElementById("progress-message");
  const progressPercentage = document.getElementById("progress-percentage");
  const currentFileLabel = document.getElementById("current-file-label");
  const progressCountLabel = document.getElementById("progress-count-label");
  const logTerminal = document.getElementById("log-terminal");

  const step1 = document.getElementById("step-1");
  const step2 = document.getElementById("step-2");
  const step3 = document.getElementById("step-3");
  const step4 = document.getElementById("step-4");

  const completedBanner = document.getElementById("completed-banner");
  const completedStats = document.getElementById("completed-stats");
  const downloadZipBtn = document.getElementById("download-zip-btn");

  const samplesContainer = document.getElementById("samples-container");
  const samplesEmpty = document.getElementById("samples-empty");
  const sampleCountBadge = document.getElementById("sample-count-badge");
  const viewModeButtons = document.querySelectorAll(".view-mode-btn");

  const modal = document.getElementById("comparison-modal");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalTitle = document.getElementById("modal-title");
  const modalRawImg = document.getElementById("modal-raw-img");
  const modalProcessedImg = document.getElementById("modal-processed-img");
  const modalProcessedLabel = document.getElementById("modal-processed-label");

  // File Selection Handlers
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("border-emerald-500", "bg-slate-900");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("border-emerald-500", "bg-slate-900");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-emerald-500", "bg-slate-900");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  removeFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetFileSelection();
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
      alert("Please select a .zip dataset archive!");
      return;
    }

    selectedFile = file;
    const mbSize = (file.size / (1024 * 1024)).toFixed(2);
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = `${mbSize} MB`;

    dropZone.classList.add("hidden");
    selectedFileBar.classList.remove("hidden");
    startBtn.disabled = false;
  }

  function resetFileSelection() {
    selectedFile = null;
    fileInput.value = "";
    dropZone.classList.remove("hidden");
    selectedFileBar.classList.add("hidden");
    startBtn.disabled = true;
    uploadProgressContainer.classList.add("hidden");
  }

  // Upload and Execution Trigger
  startBtn.addEventListener("click", () => {
    if (!selectedFile) return;
    uploadAndRunPipeline(selectedFile);
  });

  function uploadAndRunPipeline(file) {
    startBtn.disabled = true;
    uploadProgressContainer.classList.remove("hidden");
    uploadProgressBar.style.width = "0%";
    uploadPercentText.textContent = "0%";

    const totalBytes = file.size;
    const startTime = Date.now();
    let lastLoaded = 0;
    let lastTime = startTime;

    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        uploadProgressBar.style.width = `${percent}%`;
        uploadPercentText.textContent = `${percent}%`;

        const loadedMB = (e.loaded / (1024 * 1024)).toFixed(1);
        const totalMB = (e.total / (1024 * 1024)).toFixed(1);
        uploadSizeText.textContent = `${loadedMB} / ${totalMB} MB`;

        // Calculate upload speed
        const currentTime = Date.now();
        const timeDiff = (currentTime - lastTime) / 1000;
        if (timeDiff >= 0.5) {
          const bytesDiff = e.loaded - lastLoaded;
          const speedMBs = (bytesDiff / (1024 * 1024) / timeDiff).toFixed(1);
          uploadSpeedText.textContent = `${speedMBs} MB/s`;
          lastLoaded = e.loaded;
          lastTime = currentTime;
        }
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const res = JSON.parse(xhr.responseText);
          activeJobId = res.job_id;
          uploadStatusText.textContent = "Upload complete! Initializing pipeline...";
          uploadProgressBar.style.width = "100%";
          uploadPercentText.textContent = "100%";
          
          appendLog(`[Upload] Archive ${res.filename} (${res.size_mb} MB) uploaded successfully.`);
          setPipelineActive(true);
          startTrackingProgress(activeJobId);
        } catch (err) {
          handleUploadError("Failed to parse server response.");
        }
      } else {
        handleUploadError(`Upload failed with status: ${xhr.status}`);
      }
    };

    xhr.onerror = () => handleUploadError("Network error occurred during upload.");

    xhr.open("POST", "/api/upload", true);
    xhr.send(formData);
  }

  function handleUploadError(msg) {
    alert(msg);
    startBtn.disabled = false;
    uploadStatusText.textContent = "Upload failed.";
    appendLog(`[Error] ${msg}`);
  }

  function setPipelineActive(active) {
    if (active) {
      statusSpinner.className = "w-3 h-3 rounded-full bg-emerald-400 animate-ping";
      badgeStatus.className = "text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
      badgeStatus.textContent = "Processing";
    } else {
      statusSpinner.className = "w-3 h-3 rounded-full bg-slate-600";
    }
  }

  // Progress Tracking via SSE with fallback polling
  function startTrackingProgress(jobId) {
    if (window.EventSource) {
      if (sseSource) sseSource.close();
      sseSource = new EventSource(`/api/progress/${jobId}/stream`);

      sseSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          renderJobUpdate(data);

          if (data.status === "completed" || data.status === "error") {
            sseSource.close();
          }
        } catch (err) {
          console.error("SSE parse error:", err);
        }
      };

      sseSource.onerror = () => {
        console.warn("SSE connection closed or lost. Falling back to polling.");
        sseSource.close();
        startPolling(jobId);
      };
    } else {
      startPolling(jobId);
    }
  }

  function startPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
      try {
        const resp = await fetch(`/api/status/${jobId}`);
        if (!resp.ok) return;
        const data = await resp.json();
        renderJobUpdate(data);

        if (data.status === "completed" || data.status === "error") {
          clearInterval(pollInterval);
        }
      } catch (e) {
        console.error("Polling error:", e);
      }
    }, 1000);
  }

  function renderJobUpdate(job) {
    const status = job.status || "";
    const progress = job.progress || 0;
    const message = job.message || "";
    const processed = job.processed || 0;
    const total = job.total || 0;
    const currentFile = job.current_file || "-";

    // Progress bar and labels
    pipelineProgressBar.style.width = `${progress}%`;
    progressPercentage.textContent = `${progress}%`;
    progressMessage.textContent = message;
    currentFileLabel.textContent = `Current: ${currentFile}`;
    progressCountLabel.textContent = total > 0 ? `${processed} / ${total} images` : "-";

    // Update Step Badges
    updateStepBadges(status, progress);

    // Sync logs
    if (job.logs && Array.isArray(job.logs)) {
      renderLogs(job.logs);
    }

    // Periodically update samples during processing or upon completion
    if (job.summary && job.summary.samples) {
      cachedSamples = job.summary.samples;
      renderSamples(cachedSamples);
    } else if (processed > 0 && processed % 5 === 0) {
      fetchSamples(job.job_id);
    }

    // Completed State
    if (status === "completed") {
      setPipelineActive(false);
      badgeStatus.className = "text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
      badgeStatus.textContent = "Finished";
      statusSpinner.className = "w-3 h-3 rounded-full bg-emerald-400";

      completedBanner.classList.remove("hidden");
      downloadZipBtn.href = `/api/download/${job.job_id}`;
      downloadZipBtn.setAttribute("download", job.summary ? job.summary.zip_filename : "processed_images.zip");

      if (job.summary) {
        completedStats.textContent = `Completed ${job.summary.processed_images} images in ${job.summary.duration_seconds}s (Archive: ${job.summary.zip_size_mb} MB)`;
      }

      // Final samples render
      fetchSamples(job.job_id);
    }

    if (status === "error") {
      setPipelineActive(false);
      badgeStatus.className = "text-xs px-2.5 py-0.5 rounded-full bg-red-500/20 text-red-300 border border-red-500/30";
      badgeStatus.textContent = "Failed";
      statusSpinner.className = "w-3 h-3 rounded-full bg-red-500";
      alert(`Pipeline error: ${job.error || "Unknown error"}`);
    }
  }

  function updateStepBadges(status, progress) {
    // Step 1: Extraction
    if (progress >= 5) step1.className = "step-badge p-2.5 rounded-xl border text-xs " + (progress > 10 ? "done" : "active");
    // Step 2: Model Ready
    if (progress >= 10) step2.className = "step-badge p-2.5 rounded-xl border text-xs " + (progress > 15 ? "done" : "active");
    // Step 3: Inference & Fusion
    if (progress >= 15) step3.className = "step-badge p-2.5 rounded-xl border text-xs " + (progress >= 90 ? "done" : "active");
    // Step 4: ZIP Bundle
    if (progress >= 90) step4.className = "step-badge p-2.5 rounded-xl border text-xs " + (progress === 100 ? "done" : "active");
  }

  function appendLog(text) {
    const time = new Date().toLocaleTimeString();
    const line = document.createElement("div");
    line.innerHTML = `<span class="text-slate-500">[${time}]</span> ${text}`;
    logTerminal.appendChild(line);
    logTerminal.scrollTop = logTerminal.scrollHeight;
  }

  function renderLogs(logs) {
    logTerminal.innerHTML = "";
    logs.forEach((log) => {
      const line = document.createElement("div");
      line.innerHTML = `<span class="text-slate-500">[${log.timestamp}]</span> ${log.text}`;
      logTerminal.appendChild(line);
    });
    logTerminal.scrollTop = logTerminal.scrollHeight;
  }

  // Sample Navigation Elements
  const sampleNavBar = document.getElementById("sample-nav-bar");
  const sampleSelect = document.getElementById("sample-select");
  const prevSampleBtn = document.getElementById("prev-sample-btn");
  const nextSampleBtn = document.getElementById("next-sample-btn");
  const randomSampleBtn = document.getElementById("random-sample-btn");
  let currentSampleIndex = 0;

  // Check and load latest job on initial page load
  async function checkLatestJob() {
    try {
      const resp = await fetch("/api/latest_job");
      if (resp.ok) {
        const data = await resp.json();
        if (data && data.job_id) {
          activeJobId = data.job_id;
          renderJobUpdate(data);
          fetchSamples(data.job_id);
        }
      }
    } catch (e) {
      console.warn("Could not check latest job:", e);
    }
  }
  checkLatestJob();

  async function fetchSamples(jobId) {
    try {
      const resp = await fetch(`/api/samples/${jobId}`);
      if (resp.ok) {
        const data = await resp.json();
        if (data.samples && data.samples.length > 0) {
          cachedSamples = data.samples;
          renderSamples(cachedSamples);
        }
      }
    } catch (err) {
      console.warn("Could not fetch samples:", err);
    }
  }

  // Render Sample Comparisons in Sidebar with Navigation Options
  function renderSamples(samples) {
    if (!samples || samples.length === 0) return;

    samplesEmpty.classList.add("hidden");
    sampleNavBar.classList.remove("hidden");
    sampleCountBadge.textContent = `${samples.length} images`;

    // Populate Dropdown if not already populated or if count changed
    if (sampleSelect.options.length !== samples.length) {
      sampleSelect.innerHTML = "";
      samples.forEach((sample, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = `${idx + 1}. ${sample.filename}`;
        sampleSelect.appendChild(opt);
      });
    }

    if (currentSampleIndex >= samples.length) {
      currentSampleIndex = 0;
    }
    sampleSelect.value = currentSampleIndex;

    renderActiveSample();
  }

  function renderActiveSample() {
    if (!cachedSamples || cachedSamples.length === 0) return;

    const sample = cachedSamples[currentSampleIndex];
    if (!sample) return;

    sampleSelect.value = currentSampleIndex;
    prevSampleBtn.disabled = currentSampleIndex === 0;
    nextSampleBtn.disabled = currentSampleIndex === cachedSamples.length - 1;

    // Determine processed image URL and title based on view mode
    let processedUrl = sample.hsie_url;
    let labelText = "HSIE FUSED";
    let descText = "Hue + Saturation + Depth";

    if (activeViewMode === "depth") {
      processedUrl = sample.depth_url;
      labelText = "DEPTH MAP";
      descText = "Depth Anything V2 Estimation";
    } else if (activeViewMode === "hsi") {
      processedUrl = sample.hsi_url;
      labelText = "HSI COLOR";
      descText = "Hue, Saturation, Intensity";
    }

    samplesContainer.innerHTML = "";

    // Primary Active Comparison Card
    const mainCard = document.createElement("div");
    mainCard.className = "sample-card bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 shadow-lg";

    mainCard.innerHTML = `
      <div class="flex items-center justify-between text-xs border-b border-slate-800/80 pb-2">
        <div class="flex items-center gap-2 truncate">
          <span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-400 font-bold">#${currentSampleIndex + 1} of ${cachedSamples.length}</span>
          <span class="font-mono text-slate-200 font-medium truncate max-w-[180px]" title="${sample.filename}">${sample.filename}</span>
        </div>
        <span class="text-[11px] text-slate-400 font-mono">${sample.width ? sample.width + "x" + sample.height : ""}</span>
      </div>

      <!-- Side-by-Side Images -->
      <div class="grid grid-cols-2 gap-3 cursor-pointer comparison-view-trigger">
        <div class="flex flex-col gap-1.5">
          <div class="flex items-center justify-between">
            <span class="text-[11px] text-slate-400 font-semibold tracking-wider">RAW CANOPY</span>
          </div>
          <div class="img-thumb-container border border-slate-700/80 rounded-lg overflow-hidden bg-black aspect-[4/3]">
            <img src="${sample.raw_url}" alt="Raw ${sample.filename}" class="w-full h-full object-cover" loading="lazy">
          </div>
        </div>

        <div class="flex flex-col gap-1.5">
          <div class="flex items-center justify-between">
            <span class="text-[11px] text-emerald-400 font-semibold tracking-wider">${labelText}</span>
          </div>
          <div class="img-thumb-container border border-emerald-500/40 rounded-lg overflow-hidden bg-black aspect-[4/3]">
            <img src="${processedUrl}" alt="Processed ${sample.filename}" class="w-full h-full object-cover" loading="lazy">
          </div>
        </div>
      </div>

      <div class="flex items-center justify-between pt-1 text-[11px] text-slate-400 border-t border-slate-800/60">
        <span class="text-slate-500 italic">${descText}</span>
        <button class="flex items-center gap-1 text-emerald-400 hover:text-emerald-300 font-medium">
          <span>Click to Zoom & Compare</span>
          <i data-lucide="maximize-2" class="w-3.5 h-3.5"></i>
        </button>
      </div>
    `;

    mainCard.querySelector(".comparison-view-trigger").addEventListener("click", () => {
      openModal(sample);
    });
    mainCard.querySelector("button").addEventListener("click", () => {
      openModal(sample);
    });

    samplesContainer.appendChild(mainCard);

    // Thumbnail strip for quick selection of surrounding images
    if (cachedSamples.length > 1) {
      const thumbStrip = document.createElement("div");
      thumbStrip.className = "flex flex-col gap-1.5 pt-2";
      thumbStrip.innerHTML = `<span class="text-[11px] text-slate-400 font-medium">Quick Select Images:</span>`;

      const grid = document.createElement("div");
      grid.className = "grid grid-cols-4 gap-2 overflow-x-auto max-h-48 pr-1";

      cachedSamples.forEach((item, idx) => {
        const isCurrent = idx === currentSampleIndex;
        const thumb = document.createElement("div");
        thumb.className = `p-1 rounded-lg border cursor-pointer transition-all ${
          isCurrent
            ? "border-emerald-500 bg-emerald-500/10 shadow-md ring-1 ring-emerald-500/50"
            : "border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-850"
        }`;
        thumb.innerHTML = `
          <div class="aspect-square rounded overflow-hidden bg-black mb-1">
            <img src="${item.raw_url}" alt="${item.filename}" class="w-full h-full object-cover" loading="lazy">
          </div>
          <div class="text-[9px] font-mono text-center truncate ${isCurrent ? 'text-emerald-400 font-bold' : 'text-slate-400'}">
            #${idx + 1}
          </div>
        `;
        thumb.addEventListener("click", () => {
          currentSampleIndex = idx;
          renderActiveSample();
        });
        grid.appendChild(thumb);
      });

      thumbStrip.appendChild(grid);
      samplesContainer.appendChild(thumbStrip);
    }

    if (window.lucide) lucide.createIcons();
  }

  // Navigation Event Listeners
  if (sampleSelect) {
    sampleSelect.addEventListener("change", (e) => {
      currentSampleIndex = parseInt(e.target.value) || 0;
      renderActiveSample();
    });
  }

  if (prevSampleBtn) {
    prevSampleBtn.addEventListener("click", () => {
      if (currentSampleIndex > 0) {
        currentSampleIndex--;
        renderActiveSample();
      }
    });
  }

  if (nextSampleBtn) {
    nextSampleBtn.addEventListener("click", () => {
      if (currentSampleIndex < cachedSamples.length - 1) {
        currentSampleIndex++;
        renderActiveSample();
      }
    });
  }

  if (randomSampleBtn) {
    randomSampleBtn.addEventListener("click", () => {
      if (cachedSamples.length > 0) {
        currentSampleIndex = Math.floor(Math.random() * cachedSamples.length);
        renderActiveSample();
      }
    });
  }

  // View Mode Button Switcher (HSIE, Depth, HSI)
  viewModeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      viewModeButtons.forEach((b) => {
        b.className = "view-mode-btn flex-1 py-1.5 px-2 rounded-lg font-medium text-slate-400 hover:text-slate-200 transition-colors";
      });
      btn.className = "view-mode-btn flex-1 py-1.5 px-2 rounded-lg font-medium text-emerald-400 bg-slate-800 shadow transition-colors";
      activeViewMode = btn.dataset.view;
      renderActiveSample();
    });
  });

  // Modal Comparison View
  function openModal(sample) {
    modalTitle.textContent = `Comparison: ${sample.filename} (#${currentSampleIndex + 1} of ${cachedSamples.length})`;
    modalRawImg.src = sample.raw_url;

    if (activeViewMode === "depth") {
      modalProcessedImg.src = sample.depth_url;
      modalProcessedLabel.textContent = "Depth Anything V2 Estimation";
    } else if (activeViewMode === "hsi") {
      modalProcessedImg.src = sample.hsi_url;
      modalProcessedLabel.textContent = "RGB to HSI Conversion";
    } else {
      modalProcessedImg.src = sample.hsie_url;
      modalProcessedLabel.textContent = "HSIE Fusion Output [H, S, Depth]";
    }

    modal.classList.remove("hidden");
  }

  modalCloseBtn.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.classList.add("hidden");
  });
});

