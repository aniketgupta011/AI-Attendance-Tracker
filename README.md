# AI-Powered Facial Recognition Attendance System

A local-first prototype that processes classroom video to automatically detect, track, and identify enrolled students using deep learning face recognition.

## Architecture

```
Video Upload → Frame Extraction → CLAHE → Face Detection (RetinaFace)
    → Face Tracking (IoU) → Best Crop Selection → Embedding (ArcFace)
    → Cosine Similarity Matching → Attendance Logging
```

**ML Stack:** InsightFace `buffalo_l` (RetinaFace + ArcFace) via ONNX Runtime with CUDA acceleration.

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

> **Note:** For GPU acceleration, ensure you have CUDA 11.8+ and install `onnxruntime-gpu`. The system will fall back to CPU if CUDA is unavailable.

### 2. Enroll Students (CLI)

```bash
# Enroll students with clear, front-facing photos
python cli.py enroll --name "John Doe" --id STU001 --image photos/john.jpg
python cli.py enroll --name "Jane Smith" --id STU002 --image photos/jane.jpg

# Verify enrollment
python cli.py list-students
```

### 3. Process Attendance Video (CLI)

```bash
# Process a classroom video (5-15 seconds, MP4/MOV/AVI)
python cli.py process classroom.mp4

# Output JSON for programmatic use
python cli.py process classroom.mp4 --json
```

### 4. Run the Web Dashboard & API Server

```bash
# Start server with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000

# Or run directly:
python main.py
```

- **Interactive Web Dashboard:** Open `http://localhost:8000/` in your browser.
  - **Video Attendance:** Drag-and-drop classroom video with real-time 7-phase WebSocket progress tracking and instant recognition results.
  - **Attendance Records:** Daily attendance stats, date filtering, search, CSV/JSON record imports, and 1-click CSV exports.
  - **Student Directory:** View enrolled students with photos and enroll new students with reference photos.
- **Interactive API Documentation:** `http://localhost:8000/docs`.

## API Endpoints

### Students

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/students/enroll` | Enroll a new student (multipart: name, enrollment_id, image) |
| `GET` | `/api/v1/students` | List all enrolled students |
| `GET` | `/api/v1/students/{id}` | Get student details |
| `GET` | `/api/v1/students/{id}/photo` | Fetch student enrollment photo |
| `DELETE` | `/api/v1/students/{id}` | Delete a student |

### Attendance

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/attendance/upload-video` | Upload video for processing (returns task_id) |
| `GET` | `/api/v1/attendance/status/{task_id}` | Poll task processing status |
| `WS` | `/api/v1/attendance/ws/{task_id}` | WebSocket for real-time progress |
| `GET` | `/api/v1/attendance/today` | Get today's attendance summary |
| `GET` | `/api/v1/attendance/history?date=YYYY-MM-DD` | Get attendance by date |
| `POST` | `/api/v1/attendance/upload-records` | Upload & import attendance records (CSV or JSON) |
| `GET` | `/api/v1/attendance/export?date=YYYY-MM-DD` | Export attendance records as CSV |

### Example: Upload Video with curl

```bash
# Upload a video
curl -X POST http://localhost:8000/api/v1/attendance/upload-video \
  -F "video=@classroom.mp4"

# Response: {"task_id": "abc-123", "status": "processing", ...}

# Poll status
curl http://localhost:8000/api/v1/attendance/status/abc-123

# Get today's attendance
curl http://localhost:8000/api/v1/attendance/today
```

## Project Structure

```
IDP/
├── config.py                # Centralized configuration
├── main.py                  # FastAPI entry point
├── cli.py                   # CLI entry point
├── requirements.txt         # Python dependencies
│
├── core/                    # ML pipeline modules
│   ├── pipeline.py          # Main orchestrator
│   ├── frame_extractor.py   # Video → frames + CLAHE
│   ├── face_detector.py     # InsightFace RetinaFace + ArcFace
│   ├── face_tracker.py      # IoU-based multi-face tracker
│   ├── face_embedder.py     # Best embedding selection
│   ├── face_matcher.py      # Cosine similarity matching
│   └── liveness.py          # Placeholder liveness module
│
├── api/                     # FastAPI routes
│   ├── attendance.py        # Video upload, WebSocket, results
│   └── students.py          # Student enrollment CRUD
│
├── storage/                 # Persistence layer
│   └── local_store.py       # JSON + .npy file storage
│
├── data/                    # Runtime data (gitignored)
│   ├── embeddings/          # Student face embeddings (.npy)
│   ├── enrollment_images/   # Original enrollment photos
│   ├── students.json        # Student metadata
│   ├── attendance_logs.json # Attendance records
│   └── temp/                # Temporary video uploads
│
└── models/                  # InsightFace models (gitignored, auto-downloaded)
```

## Configuration

All parameters are in [`config.py`](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAME_SAMPLE_FPS` | 4.0 | Frames extracted per second |
| `SIMILARITY_THRESHOLD` | 0.45 | Cosine similarity match cutoff |
| `DETECTION_THRESHOLD` | 0.5 | Minimum face detection confidence |
| `MAX_VIDEO_DURATION_SEC` | 15.0 | Maximum video length |
| `IOU_THRESHOLD` | 0.3 | Tracker bounding box overlap |
| `LIVENESS_THRESHOLD` | 0.85 | Liveness score cutoff |
| `DEVICE` | auto | `cuda` or `cpu` (auto-detected) |

## Hardware Requirements

- **GPU:** NVIDIA GPU with 4+ GB VRAM (tested on RTX 4050 Laptop, 6 GB)
- **VRAM Usage:** ~500 MB (RetinaFace ~300 MB + ArcFace ~200 MB)
- **CPU:** Works on CPU too, but significantly slower
- **RAM:** 8+ GB recommended

## Limitations (Prototype)

- **No liveness detection** — placeholder module always returns live=True
- **Simple IoU tracker** — may lose tracks during fast movement (ByteTrack upgrade planned)
- **File-based storage** — not suitable for concurrent multi-user access (PostgreSQL + pgvector planned)
- **Built-in Web Frontend** — Fully integrated interactive dashboard served at `http://localhost:8000/`
