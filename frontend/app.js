/**
 * AI Attendance System — Frontend Controller
 * Handles video upload, WebSocket progress streaming, attendance logs, and student directory.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ─── State ───────────────────────────────────────────────────
  let selectedVideoFile = null;
  let selectedRecordsFile = null;
  let selectedEnrollImage = null;
  let currentTaskId = null;
  let activeWebSocket = null;
  let pollingInterval = null;
  let allEnrolledStudents = [];
  let currentAttendanceRecords = [];

  // ─── DOM Elements ────────────────────────────────────────────
  // Navigation
  const navTabs = document.querySelectorAll('.nav-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const deviceBadge = document.getElementById('device-badge');
  const statusIndicator = document.getElementById('status-indicator');
  const statusText = document.getElementById('status-text');

  // Video Tab
  const videoDropzone = document.getElementById('video-dropzone');
  const videoFileInput = document.getElementById('video-file-input');
  const previewVideoPlayer = document.getElementById('preview-video-player');
  const videoEmptyState = document.getElementById('video-empty-state');
  const videoMeta = document.getElementById('video-meta');
  const videoFilename = document.getElementById('video-filename');
  const videoFilesize = document.getElementById('video-filesize');
  const previewStatusBadge = document.getElementById('preview-status-badge');
  const btnProcessVideo = document.getElementById('btn-process-video');
  const btnClearVideo = document.getElementById('btn-clear-video');
  const btnQuickSample = document.getElementById('btn-quick-sample');

  // Pipeline Monitor
  const pipelineMonitor = document.getElementById('pipeline-monitor');
  const pipelineMessage = document.getElementById('pipeline-message');
  const pipelinePercent = document.getElementById('pipeline-percent');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const stepCards = {
    extracting: document.getElementById('step-extracting'),
    detecting: document.getElementById('step-detecting'),
    tracking: document.getElementById('step-tracking'),
    embedding: document.getElementById('step-embedding'),
    liveness: document.getElementById('step-liveness'),
    matching: document.getElementById('step-matching'),
    compiling: document.getElementById('step-compiling'),
  };

  // Results
  const resultsContainer = document.getElementById('results-container');
  const resMatchedCount = document.getElementById('res-matched-count');
  const resUnmatchedCount = document.getElementById('res-unmatched-count');
  const resTotalFaces = document.getElementById('res-total-faces');
  const resProcessingTime = document.getElementById('res-processing-time');
  const recognizedStudentsGrid = document.getElementById('recognized-students-grid');
  const btnViewRecordsTab = document.getElementById('btn-view-records-tab');

  // Attendance Records Tab
  const recordsDatePicker = document.getElementById('records-date-picker');
  const recordsSearchInput = document.getElementById('records-search-input');
  const attendanceTableBody = document.getElementById('attendance-table-body');
  const statsTotalEnrolled = document.getElementById('stats-total-enrolled');
  const statsPresent = document.getElementById('stats-present');
  const statsAbsent = document.getElementById('stats-absent');
  const statsRate = document.getElementById('stats-rate');
  const btnDateToday = document.getElementById('btn-date-today');
  const btnDateYesterday = document.getElementById('btn-date-yesterday');
  const btnRefreshRecords = document.getElementById('btn-refresh-records');
  const btnClearRecords = document.getElementById('btn-clear-records');
  const btnExportCsv = document.getElementById('btn-export-csv');

  // Upload Records Modal
  const btnOpenUploadRecordsModal = document.getElementById('btn-open-upload-records-modal');
  const uploadRecordsModal = document.getElementById('upload-records-modal');
  const recordsDropzone = document.getElementById('records-dropzone');
  const recordsFileInput = document.getElementById('records-file-input');
  const recordsSelectedFilename = document.getElementById('records-selected-filename');
  const btnSubmitUploadRecords = document.getElementById('btn-submit-upload-records');
  const btnDownloadSampleCsv = document.getElementById('btn-download-sample-csv');

  // Students Tab
  const studentsDirectoryGrid = document.getElementById('students-directory-grid');
  const studentsSearchInput = document.getElementById('students-search-input');
  const studentsCountBadge = document.getElementById('students-count-badge');
  const btnOpenEnrollModal = document.getElementById('btn-open-enroll-modal');
  const enrollModal = document.getElementById('enroll-modal');
  const enrollForm = document.getElementById('enroll-form');
  const enrollName = document.getElementById('enroll-name');
  const enrollId = document.getElementById('enroll-id');
  const enrollImageInput = document.getElementById('enroll-image-input');
  const enrollImagePreviewBox = document.getElementById('enroll-image-preview-box');
  const enrollImagePlaceholder = document.getElementById('enroll-image-placeholder');
  const enrollImagePreview = document.getElementById('enroll-image-preview');
  const btnSubmitEnroll = document.getElementById('btn-submit-enroll');

  // Toast Container
  const toastContainer = document.getElementById('toast-container');

  // ─── Toast Notifications ─────────────────────────────────────
  function showToast(message, type = 'info', duration = 3500) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '⚠️';

    toast.innerHTML = `<span>${icon}</span><span style="flex: 1;">${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // ─── Modal Management ────────────────────────────────────────
  function openModal(modal) {
    modal.classList.add('active');
  }

  function closeModal(modal) {
    modal.classList.remove('active');
  }

  document.querySelectorAll('[data-close-modal]').forEach(btn => {
    btn.addEventListener('click', () => {
      const modalId = btn.getAttribute('data-close-modal');
      const targetModal = document.getElementById(modalId);
      if (targetModal) closeModal(targetModal);
    });
  });

  window.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) {
      closeModal(e.target);
    }
  });

  // ─── Navigation Tabs ─────────────────────────────────────────
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTabId = tab.getAttribute('data-tab');
      
      navTabs.forEach(t => t.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPane = document.getElementById(targetTabId);
      if (targetPane) targetPane.classList.add('active');

      // Refresh data on tab activation
      if (targetTabId === 'records-tab') {
        loadAttendanceRecords();
      } else if (targetTabId === 'students-tab') {
        loadStudents();
      }
    });
  });

  if (btnViewRecordsTab) {
    btnViewRecordsTab.addEventListener('click', () => {
      const recordsTabBtn = document.querySelector('[data-tab="records-tab"]');
      if (recordsTabBtn) recordsTabBtn.click();
    });
  }

  // ─── System Health Check ─────────────────────────────────────
  async function checkSystemHealth() {
    try {
      const res = await fetch('/health');
      if (!res.ok) throw new Error('Health check failed');
      const data = await res.json();
      
      const device = data.device ? data.device.toUpperCase() : 'UNKNOWN';
      deviceBadge.textContent = `DEVICE: ${device}`;
      if (device === 'CUDA') {
        deviceBadge.style.color = '#34d399';
        deviceBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
      }
      statusText.textContent = 'Backend Online';
      statusIndicator.style.color = 'var(--success)';
    } catch (err) {
      statusText.textContent = 'Offline / Error';
      statusIndicator.style.color = 'var(--danger)';
      deviceBadge.textContent = 'DISCONNECTED';
    }
  }

  // ─── Video File Selection & Preview ──────────────────────────
  function formatBytes(bytes, decimals = 1) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  function handleVideoFile(file) {
    if (!file) return;

    // Check extension
    const validExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExtensions.includes(fileExt)) {
      showToast(`Invalid file type "${fileExt}". Please upload a video file.`, 'error');
      return;
    }

    // Check file size (50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      showToast(`File size (${formatBytes(file.size)}) exceeds 50MB limit.`, 'error');
      return;
    }

    selectedVideoFile = file;

    // Display preview
    const videoUrl = URL.createObjectURL(file);
    previewVideoPlayer.src = videoUrl;
    previewVideoPlayer.style.display = 'block';
    videoEmptyState.style.display = 'none';

    videoFilename.textContent = file.name;
    videoFilesize.textContent = formatBytes(file.size);
    videoMeta.style.display = 'flex';

    previewStatusBadge.textContent = 'Ready to Process';
    previewStatusBadge.className = 'badge badge-match';

    btnProcessVideo.disabled = false;
    btnClearVideo.disabled = false;

    showToast(`Loaded "${file.name}"`, 'info');
  }

  videoDropzone.addEventListener('click', () => videoFileInput.click());
  videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleVideoFile(e.target.files[0]);
    }
  });

  videoDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    videoDropzone.classList.add('dragover');
  });

  videoDropzone.addEventListener('dragleave', () => {
    videoDropzone.classList.remove('dragover');
  });

  videoDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    videoDropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleVideoFile(e.dataTransfer.files[0]);
    }
  });

  btnClearVideo.addEventListener('click', () => {
    selectedVideoFile = null;
    previewVideoPlayer.pause();
    previewVideoPlayer.src = '';
    previewVideoPlayer.style.display = 'none';
    videoEmptyState.style.display = 'block';
    videoMeta.style.display = 'none';
    videoFileInput.value = '';
    previewStatusBadge.textContent = 'No video selected';
    previewStatusBadge.className = 'badge badge-live';
    btnProcessVideo.disabled = true;
    btnClearVideo.disabled = true;
  });

  // Load sample video button
  btnQuickSample.addEventListener('click', async () => {
    try {
      btnQuickSample.disabled = true;
      btnQuickSample.textContent = 'Loading classroom.mp4...';
      const res = await fetch('/sample-video');
      if (!res.ok) throw new Error('Sample video not found on server');
      const blob = await res.blob();
      const sampleFile = new File([blob], 'classroom.mp4', { type: 'video/mp4' });
      handleVideoFile(sampleFile);
    } catch (err) {
      showToast('Could not load sample classroom.mp4: ' + err.message, 'error');
    } finally {
      btnQuickSample.disabled = false;
      btnQuickSample.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6"/></svg>
        Load Sample classroom.mp4
      `;
    }
  });

  // ─── Pipeline Progress Visualizer ────────────────────────────
  const stepOrder = ['extracting', 'detecting', 'tracking', 'embedding', 'liveness', 'matching', 'compiling'];

  function resetPipelineSteps() {
    stepOrder.forEach(phase => {
      const card = stepCards[phase];
      if (card) {
        card.className = 'step-card waiting';
      }
    });
  }

  function updatePipelineSteps(currentPhase) {
    const currentIndex = stepOrder.indexOf(currentPhase);
    stepOrder.forEach((phase, idx) => {
      const card = stepCards[phase];
      if (!card) return;

      if (currentPhase === 'complete') {
        card.className = 'step-card complete';
      } else if (idx < currentIndex) {
        card.className = 'step-card complete';
      } else if (idx === currentIndex) {
        card.className = 'step-card active';
      } else {
        card.className = 'step-card waiting';
      }
    });
  }

  // ─── Process Attendance (Upload & Progress Streaming) ────────
  btnProcessVideo.addEventListener('click', async () => {
    if (!selectedVideoFile) return;

    btnProcessVideo.disabled = true;
    btnClearVideo.disabled = true;
    resultsContainer.classList.remove('active');

    // Show pipeline monitor
    pipelineMonitor.classList.add('active');
    pipelinePercent.textContent = '0%';
    progressBarFill.style.width = '0%';
    pipelineMessage.textContent = 'Uploading video file...';
    resetPipelineSteps();

    try {
      const formData = new FormData();
      formData.append('video', selectedVideoFile);

      const res = await fetch('/api/v1/attendance/upload-video', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await res.json();
      currentTaskId = data.task_id;
      showToast('Video uploaded. Initializing facial recognition pipeline...', 'info');

      // Connect WebSocket with polling fallback
      connectProgressWebSocket(currentTaskId);

    } catch (err) {
      showToast(err.message, 'error');
      pipelineMessage.textContent = 'Error: ' + err.message;
      pipelineMonitor.classList.remove('active');
      btnProcessVideo.disabled = false;
      btnClearVideo.disabled = false;
    }
  });

  function connectProgressWebSocket(taskId) {
    if (activeWebSocket) {
      activeWebSocket.close();
      activeWebSocket = null;
    }
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/attendance/ws/${taskId}`;

    let socketOpened = false;

    try {
      const ws = new WebSocket(wsUrl);
      activeWebSocket = ws;

      ws.onopen = () => {
        socketOpened = true;
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'heartbeat') return;
          handleProgressUpdate(msg);
        } catch (e) {
          console.error('WS parse error', e);
        }
      };

      ws.onerror = () => {
        // Fallback to polling if WebSocket has an issue
        if (!socketOpened) {
          startPollingFallback(taskId);
        }
      };

      ws.onclose = () => {
        activeWebSocket = null;
      };

      // Also set a safety timeout in case WebSocket doesn't fire
      setTimeout(() => {
        if (!socketOpened) {
          startPollingFallback(taskId);
        }
      }, 1500);

    } catch (e) {
      startPollingFallback(taskId);
    }
  }

  function startPollingFallback(taskId) {
    if (pollingInterval) return;

    pollingInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/attendance/status/${taskId}`);
        if (!res.ok) throw new Error('Status poll failed');
        const data = await res.json();

        handleProgressUpdate(data);

        if (data.status === 'completed' || data.status === 'error') {
          clearInterval(pollingInterval);
          pollingInterval = null;
        }
      } catch (err) {
        console.error('Polling error', err);
      }
    }, 700);
  }

  function handleProgressUpdate(update) {
    const progress = update.progress || 0.0;
    const phase = update.phase || update.status || 'processing';
    const message = update.message || '';

    const percentInt = Math.min(100, Math.round(progress * 100));
    pipelinePercent.textContent = `${percentInt}%`;
    progressBarFill.style.width = `${percentInt}%`;
    if (message) pipelineMessage.textContent = message;

    updatePipelineSteps(phase);

    if (phase === 'complete' || update.status === 'completed') {
      if (activeWebSocket) activeWebSocket.close();
      if (pollingInterval) clearInterval(pollingInterval);

      pipelinePercent.textContent = '100%';
      progressBarFill.style.width = '100%';
      updatePipelineSteps('complete');
      pipelineMessage.textContent = message || 'Attendance processing complete!';

      // Fetch complete task result
      fetchTaskFinalResult(currentTaskId);
    } else if (phase === 'error' || update.status === 'error') {
      if (activeWebSocket) activeWebSocket.close();
      if (pollingInterval) clearInterval(pollingInterval);

      pipelineMessage.textContent = 'Processing error: ' + (message || 'Unknown error');
      showToast('Processing error: ' + message, 'error');
      btnProcessVideo.disabled = false;
      btnClearVideo.disabled = false;
    }
  }

  async function fetchTaskFinalResult(taskId) {
    try {
      const res = await fetch(`/api/v1/attendance/status/${taskId}`);
      if (!res.ok) throw new Error('Failed to get final task result');
      const data = await res.json();
      const result = data.result || {};

      displayRecognitionResults(result);
      showToast('Attendance successfully processed and recorded!', 'success');

      // Refresh records and enrolled student lists
      loadAttendanceRecords();
      loadStudents();

    } catch (err) {
      showToast('Could not fetch final task summary: ' + err.message, 'warning');
    } finally {
      btnProcessVideo.disabled = false;
      btnClearVideo.disabled = false;
    }
  }

  // ─── Render Recognition Results ──────────────────────────────
  function displayRecognitionResults(result) {
    const matched = result.matched_students || [];
    const unmatched = result.unmatched_faces || 0;
    const totalFaces = result.total_unique_faces || (matched.length + unmatched);
    const procTime = result.processing_time_sec ? `${result.processing_time_sec.toFixed(1)}s` : '0.0s';

    resMatchedCount.textContent = matched.length;
    resUnmatchedCount.textContent = unmatched;
    resTotalFaces.textContent = totalFaces;
    resProcessingTime.textContent = procTime;

    recognizedStudentsGrid.innerHTML = '';

    if (matched.length === 0) {
      recognizedStudentsGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-icon">⚠️</div>
          <div>No enrolled students were recognized in this video clip.</div>
          <div style="font-size: 0.8rem; margin-top: 0.5rem; color: var(--text-muted);">
            Ensure enrolled photos have clear front-facing lighting, or enroll new students in the Students tab.
          </div>
        </div>
      `;
    } else {
      matched.forEach(rec => {
        const studentInfo = allEnrolledStudents.find(s => s.id === rec.student_id) || {};
        const enrollId = studentInfo.enrollment_id || rec.student_id;
        const confidencePct = Math.round(rec.confidence_score * 100);
        const livenessPct = Math.round(rec.liveness_score * 100);

        const card = document.createElement('div');
        card.className = 'student-card';
        card.innerHTML = `
          <img class="student-avatar" src="/api/v1/students/${rec.student_id}/photo" 
               onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" 
               alt="${rec.student_name}">
          <div class="student-avatar-placeholder" style="display: none;">
            ${(rec.student_name || 'U').charAt(0).toUpperCase()}
          </div>
          <div class="student-info">
            <div class="student-name">${rec.student_name}</div>
            <div class="student-id">${enrollId}</div>
            <div class="badge-row">
              <span class="badge ${confidencePct >= 60 ? 'badge-match' : 'badge-warning'}">
                🎯 ${confidencePct}% Match
              </span>
              <span class="badge badge-live">
                🛡️ Live ${livenessPct}%
              </span>
            </div>
          </div>
        `;
        recognizedStudentsGrid.appendChild(card);
      });
    }

    resultsContainer.classList.add('active');
  }

  // ─── Attendance Records Tab ──────────────────────────────────
  function getLocalDateString(date = new Date()) {
    const y = date.getFullYear();
    const m = String(date.getMonth() + 1).padStart(2, '0');
    const d = String(date.getDate()).padStart(2, '0');
    return `${y}-${m}-${d}`;
  }

  recordsDatePicker.value = getLocalDateString();

  btnDateToday.addEventListener('click', () => {
    recordsDatePicker.value = getLocalDateString(new Date());
    loadAttendanceRecords();
  });

  btnDateYesterday.addEventListener('click', () => {
    const d = new Date();
    d.setDate(d.getDate() - 1);
    recordsDatePicker.value = getLocalDateString(d);
    loadAttendanceRecords();
  });

  recordsDatePicker.addEventListener('change', () => loadAttendanceRecords());
  recordsSearchInput.addEventListener('input', () => filterAttendanceTable());
  btnRefreshRecords.addEventListener('click', () => loadAttendanceRecords());

  async function loadAttendanceRecords() {
    const selectedDate = recordsDatePicker.value;
    attendanceTableBody.innerHTML = `
      <tr>
        <td colspan="7" class="empty-state">
          <div class="empty-icon">⏳</div>
          <div>Loading attendance records for ${selectedDate}...</div>
        </td>
      </tr>
    `;

    try {
      // Fetch both history and total enrolled students for stats calculation
      const [historyRes, studentsRes] = await Promise.all([
        fetch(`/api/v1/attendance/history?date=${selectedDate}`),
        fetch('/api/v1/students'),
      ]);

      if (!historyRes.ok) throw new Error('Failed to fetch attendance history');
      const historyData = await historyRes.json();

      if (studentsRes.ok) {
        const studentsData = await studentsRes.json();
        allEnrolledStudents = studentsData.students || [];
      }

      currentAttendanceRecords = historyData.records || [];
      const totalEnrolled = allEnrolledStudents.length || historyData.total_students_enrolled || 0;
      const presentCount = currentAttendanceRecords.length;
      const absentCount = Math.max(0, totalEnrolled - presentCount);
      const ratePct = totalEnrolled > 0 ? Math.round((presentCount / totalEnrolled) * 100) : 0;

      // Update stat cards
      statsTotalEnrolled.textContent = totalEnrolled;
      statsPresent.textContent = presentCount;
      statsAbsent.textContent = absentCount;
      statsRate.textContent = `${ratePct}%`;

      renderAttendanceTable(currentAttendanceRecords);

    } catch (err) {
      showToast('Error loading attendance: ' + err.message, 'error');
      attendanceTableBody.innerHTML = `
        <tr>
          <td colspan="7" class="empty-state" style="color: var(--danger);">
            <div class="empty-icon">⚠️</div>
            <div>Failed to load records. ${err.message}</div>
          </td>
        </tr>
      `;
    }
  }

  function renderAttendanceTable(records) {
    if (!records || records.length === 0) {
      attendanceTableBody.innerHTML = `
        <tr>
          <td colspan="7" class="empty-state">
            <div class="empty-icon">📝</div>
            <div>No attendance records logged for this date.</div>
            <div style="font-size: 0.8rem; margin-top: 0.4rem; color: var(--text-muted);">
              Process a video or upload a CSV/JSON file to log attendance.
            </div>
          </td>
        </tr>
      `;
      return;
    }

    attendanceTableBody.innerHTML = '';
    records.forEach(r => {
      const student = allEnrolledStudents.find(s => s.id === r.student_id) || {};
      const enrollId = r.enrollment_id || student.enrollment_id || r.student_id;
      const studentName = r.student_name || student.name || 'Unknown';
      const confPct = Math.round((r.confidence_score || 0) * 100);
      const livePct = Math.round((r.liveness_score || 1) * 100);
      
      let formattedTime = '-';
      if (r.timestamp) {
        try {
          const d = new Date(r.timestamp);
          formattedTime = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch (_) {
          formattedTime = r.timestamp;
        }
      }

      const tr = document.createElement('tr');
      tr.setAttribute('data-search-text', `${studentName} ${enrollId}`.toLowerCase());
      tr.innerHTML = `
        <td>
          <img class="student-avatar" style="width: 38px; height: 38px; border-radius: var(--radius-sm);" 
               src="/api/v1/students/${r.student_id}/photo" 
               onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" 
               alt="${studentName}">
          <div class="student-avatar-placeholder" style="width: 38px; height: 38px; font-size: 0.85rem; display: none;">
            ${studentName.charAt(0).toUpperCase()}
          </div>
        </td>
        <td style="font-weight: 600; color: #fff;">${studentName}</td>
        <td style="font-family: var(--font-mono); color: var(--text-secondary); font-size: 0.82rem;">${enrollId}</td>
        <td style="color: var(--text-secondary); font-size: 0.82rem;">${formattedTime}</td>
        <td>
          <span class="badge ${confPct >= 60 ? 'badge-match' : 'badge-warning'}">
            ${confPct}%
          </span>
        </td>
        <td>
          <span class="badge badge-live">
            ${livePct}%
          </span>
        </td>
        <td>
          <span class="badge badge-match">
            ● Present
          </span>
        </td>
      `;
      attendanceTableBody.appendChild(tr);
    });
  }

  function filterAttendanceTable() {
    const query = recordsSearchInput.value.toLowerCase().trim();
    const rows = attendanceTableBody.querySelectorAll('tr[data-search-text]');
    rows.forEach(row => {
      const text = row.getAttribute('data-search-text') || '';
      row.style.display = text.includes(query) ? '' : 'none';
    });
  }

  // Clear Attendance Logs
  btnClearRecords.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear all attendance records? This action cannot be undone.')) {
      return;
    }

    try {
      const res = await fetch('/api/v1/attendance/clear', { method: 'POST' });
      if (!res.ok) throw new Error('Clear failed');
      showToast('All attendance logs cleared.', 'info');
      loadAttendanceRecords();
    } catch (err) {
      showToast('Error clearing logs: ' + err.message, 'error');
    }
  });

  // Export CSV
  btnExportCsv.addEventListener('click', () => {
    const date = recordsDatePicker.value;
    window.open(`/api/v1/attendance/export?date=${date}`, '_blank');
    showToast(`Exporting attendance CSV for ${date}...`, 'info');
  });

  // ─── Upload Attendance Records Modal (CSV / JSON) ────────────
  btnOpenUploadRecordsModal.addEventListener('click', () => {
    selectedRecordsFile = null;
    recordsFileInput.value = '';
    recordsSelectedFilename.textContent = 'Supported: .csv, .json';
    btnSubmitUploadRecords.disabled = true;
    openModal(uploadRecordsModal);
  });

  recordsDropzone.addEventListener('click', () => recordsFileInput.click());
  recordsFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleRecordsFile(e.target.files[0]);
    }
  });

  recordsDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    recordsDropzone.classList.add('dragover');
  });

  recordsDropzone.addEventListener('dragleave', () => {
    recordsDropzone.classList.remove('dragover');
  });

  recordsDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    recordsDropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleRecordsFile(e.dataTransfer.files[0]);
    }
  });

  function handleRecordsFile(file) {
    const validExts = ['.csv', '.json', '.txt'];
    const fileExt = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExts.includes(fileExt)) {
      showToast('Please select a .csv or .json file', 'error');
      return;
    }

    selectedRecordsFile = file;
    recordsSelectedFilename.textContent = `Selected: ${file.name} (${formatBytes(file.size)})`;
    btnSubmitUploadRecords.disabled = false;
  }

  btnSubmitUploadRecords.addEventListener('click', async () => {
    if (!selectedRecordsFile) return;

    btnSubmitUploadRecords.disabled = true;
    btnSubmitUploadRecords.textContent = 'Uploading & Syncing...';

    try {
      const formData = new FormData();
      formData.append('file', selectedRecordsFile);

      const res = await fetch('/api/v1/attendance/upload-records', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errorData.detail || 'Failed to upload attendance records');
      }

      const data = await res.json();
      showToast(data.message || 'Records imported successfully!', 'success');
      closeModal(uploadRecordsModal);
      loadAttendanceRecords();

    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      btnSubmitUploadRecords.disabled = false;
      btnSubmitUploadRecords.textContent = 'Upload & Sync Records';
    }
  });

  btnDownloadSampleCsv.addEventListener('click', () => {
    const today = getLocalDateString();
    const csvContent = 
`enrollment_id,student_name,timestamp,confidence_score,liveness_score
UG/04/BTCSE/2025/261,Aniket Gupta,${today}T09:30:00Z,0.92,1.0
UG/04/BTCSE/2025/262,Ronny Mondal,${today}T09:30:00Z,0.88,1.0
UG/04/BTCSE/2025/230,Deepak Kumar Roy,${today}T09:30:00Z,0.95,1.0
UG/04/BTCSE/2025/251,Jaydeep Dey,${today}T09:30:00Z,0.91,1.0`;

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'sample_attendance_records.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('Downloaded sample CSV template', 'info');
  });

  // ─── Student Directory & Enrollment ──────────────────────────
  async function loadStudents() {
    studentsDirectoryGrid.innerHTML = `
      <div class="empty-state" style="grid-column: 1 / -1;">
        <div class="empty-icon">⏳</div>
        <div>Loading enrolled students...</div>
      </div>
    `;

    try {
      const res = await fetch('/api/v1/students');
      if (!res.ok) throw new Error('Failed to load students');
      const data = await res.json();
      allEnrolledStudents = data.students || [];

      studentsCountBadge.textContent = `${allEnrolledStudents.length} Students Enrolled`;

      renderStudentsGrid(allEnrolledStudents);
    } catch (err) {
      showToast('Error loading students: ' + err.message, 'error');
      studentsDirectoryGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1; color: var(--danger);">
          <div class="empty-icon">⚠️</div>
          <div>Could not load students directory.</div>
        </div>
      `;
    }
  }

  function renderStudentsGrid(students) {
    if (!students || students.length === 0) {
      studentsDirectoryGrid.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
          <div class="empty-icon">👥</div>
          <div>No students enrolled yet.</div>
          <div style="font-size: 0.8rem; margin-top: 0.4rem; color: var(--text-muted);">
            Click "Enroll New Student" to add student face embeddings to the database.
          </div>
        </div>
      `;
      return;
    }

    studentsDirectoryGrid.innerHTML = '';
    students.forEach(s => {
      let createdStr = '';
      if (s.created_at) {
        try {
          createdStr = new Date(s.created_at).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
        } catch (_) {
          createdStr = s.created_at;
        }
      }

      const card = document.createElement('div');
      card.className = 'student-card';
      card.setAttribute('data-student-search', `${s.name} ${s.enrollment_id}`.toLowerCase());
      card.innerHTML = `
        <img class="student-avatar" src="/api/v1/students/${s.id}/photo" 
             onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" 
             alt="${s.name}">
        <div class="student-avatar-placeholder" style="display: none;">
          ${(s.name || 'U').charAt(0).toUpperCase()}
        </div>
        <div class="student-info">
          <div class="student-name">${s.name}</div>
          <div class="student-id">${s.enrollment_id}</div>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 0.2rem;">Enrolled: ${createdStr || 'Active'}</div>
        </div>
        <button class="btn btn-danger btn-sm" title="Delete Student" style="padding: 0.35rem 0.55rem;" data-delete-student="${s.id}" data-student-name="${s.name}">
          🗑️
        </button>
      `;
      studentsDirectoryGrid.appendChild(card);
    });

    // Attach delete handlers
    document.querySelectorAll('[data-delete-student]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const studentId = btn.getAttribute('data-delete-student');
        const sName = btn.getAttribute('data-student-name') || 'this student';

        if (!confirm(`Are you sure you want to delete ${sName}? Their face embedding and records will be deleted.`)) {
          return;
        }

        try {
          const res = await fetch(`/api/v1/students/${studentId}`, { method: 'DELETE' });
          if (!res.ok) throw new Error('Delete failed');
          showToast(`Student "${sName}" deleted.`, 'info');
          loadStudents();
          loadAttendanceRecords();
        } catch (err) {
          showToast('Failed to delete student: ' + err.message, 'error');
        }
      });
    });
  }

  studentsSearchInput.addEventListener('input', () => {
    const query = studentsSearchInput.value.toLowerCase().trim();
    const cards = studentsDirectoryGrid.querySelectorAll('[data-student-search]');
    cards.forEach(card => {
      const text = card.getAttribute('data-student-search') || '';
      card.style.display = text.includes(query) ? '' : 'none';
    });
  });

  // Enroll Student Modal
  btnOpenEnrollModal.addEventListener('click', () => {
    enrollForm.reset();
    selectedEnrollImage = null;
    enrollImagePreview.style.display = 'none';
    enrollImagePlaceholder.style.display = 'block';
    openModal(enrollModal);
  });

  enrollImagePreviewBox.addEventListener('click', () => enrollImageInput.click());
  enrollImageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      selectedEnrollImage = file;
      const url = URL.createObjectURL(file);
      enrollImagePreview.src = url;
      enrollImagePreview.style.display = 'block';
      enrollImagePlaceholder.style.display = 'none';
    }
  });

  enrollForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const name = enrollName.value.trim();
    const id = enrollId.value.trim();

    if (!name || !id) {
      showToast('Please enter both name and enrollment ID.', 'error');
      return;
    }

    if (!selectedEnrollImage) {
      showToast('Please select a front-facing face photo.', 'error');
      return;
    }

    btnSubmitEnroll.disabled = true;
    btnSubmitEnroll.textContent = 'Extracting Embedding...';

    try {
      const formData = new FormData();
      formData.append('name', name);
      formData.append('enrollment_id', id);
      formData.append('image', selectedEnrollImage);

      const res = await fetch('/api/v1/students/enroll', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Enrollment failed' }));
        throw new Error(errorData.detail || 'Enrollment failed');
      }

      const data = await res.json();
      showToast(`Student "${name}" successfully enrolled!`, 'success');
      closeModal(enrollModal);
      loadStudents();
      loadAttendanceRecords();

    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      btnSubmitEnroll.disabled = false;
      btnSubmitEnroll.textContent = 'Save & Extract Embedding';
    }
  });

  // ─── Initial Startup ─────────────────────────────────────────
  checkSystemHealth();
  loadStudents();
  loadAttendanceRecords();

  // Periodic health check every 30 seconds
  setInterval(checkSystemHealth, 30000);
});
