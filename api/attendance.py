"""
Attendance API routes.

Handles:
- Video upload for attendance processing
- WebSocket progress streaming
- Attendance results retrieval
"""

import asyncio
import csv
import io
import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/attendance", tags=["attendance"])

# ─────────────────────────────────────────────────────────────────
# In-memory task tracking (replaced by Redis/Celery in production)
# ─────────────────────────────────────────────────────────────────

# Stores task status: {task_id: {"status": str, "progress": float, "result": dict}}
_tasks: dict[str, dict] = {}

# WebSocket connections waiting for task updates: {task_id: [websocket, ...]}
_ws_connections: dict[str, list[WebSocket]] = {}


# ─────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    phase: str
    progress: float
    message: str
    result: dict | None = None


class AttendanceRecordResponse(BaseModel):
    id: str
    student_id: str
    student_name: str
    enrollment_id: str = ""
    confidence_score: float
    liveness_score: float
    timestamp: str


class AttendanceSummaryResponse(BaseModel):
    date: str
    total_students_enrolled: int
    students_present: int
    records: list[AttendanceRecordResponse]


# ─────────────────────────────────────────────────────────────────
# Helper: Progress callback for WebSocket broadcasting
# ─────────────────────────────────────────────────────────────────

async def _broadcast_progress(task_id: str, phase: str, progress: float, message: str):
    """Send progress update to all WebSocket connections for a task."""
    data = {
        "task_id": task_id,
        "phase": phase,
        "progress": round(progress, 3),
        "message": message,
    }

    # Update in-memory task status
    if task_id in _tasks:
        _tasks[task_id].update({
            "phase": phase,
            "progress": progress,
            "message": message,
        })

    # Broadcast to connected WebSockets
    if task_id in _ws_connections:
        disconnected = []
        for ws in _ws_connections[task_id]:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.append(ws)

        # Clean up disconnected clients
        for ws in disconnected:
            _ws_connections[task_id].remove(ws)


def _make_sync_callback(task_id: str, loop: asyncio.AbstractEventLoop):
    """
    Create a synchronous progress callback that bridges to async WebSocket.

    The ML pipeline runs in a thread, so we need to schedule the async
    broadcast on the event loop from the sync context.
    """
    def callback(phase: str, progress: float, message: str):
        # Update task status synchronously (thread-safe for simple dict updates)
        if task_id in _tasks:
            _tasks[task_id].update({
                "status": "processing" if phase != "complete" and phase != "error" else phase,
                "phase": phase,
                "progress": progress,
                "message": message,
            })

        # Schedule async broadcast on the event loop
        try:
            future = asyncio.run_coroutine_threadsafe(
                _broadcast_progress(task_id, phase, progress, message),
                loop,
            )
            # Don't block on the future — fire and forget
            future.result(timeout=1.0)
        except Exception as e:
            logger.debug("WebSocket broadcast error (non-critical): %s", e)

    return callback


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@router.post("/upload-video", response_model=UploadResponse, status_code=202)
async def upload_video(video: UploadFile = File(...)):
    """
    Upload a classroom video for attendance processing.

    Accepts MP4, MOV, AVI, MKV, or WebM files up to 50 MB / 15 seconds.
    Returns immediately with a task_id for progress tracking via
    WebSocket or polling.
    """
    # Validate file extension
    ext = Path(video.filename or "").suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {config.ALLOWED_VIDEO_EXTENSIONS}",
        )

    # Read file content
    content = await video.read()

    if len(content) > config.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content)} bytes). "
                   f"Maximum: {config.MAX_UPLOAD_SIZE_BYTES} bytes",
        )

    # Get dependencies from app state
    from main import get_pipeline, get_store
    store = get_store()
    pipeline = get_pipeline()

    # Save temp file
    temp_path = store.save_temp_video(video.filename or "upload.mp4", content)

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Initialize task status
    _tasks[task_id] = {
        "status": "queued",
        "phase": "queued",
        "progress": 0.0,
        "message": "Video uploaded, processing queued",
        "result": None,
    }

    # Launch processing in background thread
    loop = asyncio.get_running_loop()
    callback = _make_sync_callback(task_id, loop)

    async def _process():
        try:
            result = await asyncio.to_thread(
                pipeline.process_video,
                temp_path,
                task_id,
                callback,
            )

            # Convert result to dict
            result_dict = asdict(result)
            _tasks[task_id]["status"] = result.status
            _tasks[task_id]["result"] = result_dict

            # Log attendance records if successful
            if result.status == "completed" and result.matched_students:
                records = [
                    {
                        "student_id": r.student_id,
                        "student_name": r.student_name,
                        "confidence_score": r.confidence_score,
                        "liveness_score": r.liveness_score,
                        "timestamp": r.timestamp,
                    }
                    for r in result.matched_students
                ]
                store.log_attendance(records)

        except Exception as e:
            logger.error("Processing error for task %s: %s", task_id, e, exc_info=True)
            _tasks[task_id]["status"] = "error"
            _tasks[task_id]["message"] = str(e)
        finally:
            store.cleanup_temp(temp_path)

    # Fire and forget the background task
    asyncio.create_task(_process())

    return UploadResponse(
        task_id=task_id,
        status="processing",
        message="Video uploaded successfully. Use WebSocket or polling to track progress.",
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the current processing status of a video upload task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task = _tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        phase=task.get("phase", "unknown"),
        progress=task.get("progress", 0.0),
        message=task.get("message", ""),
        result=task.get("result"),
    )


@router.websocket("/ws/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time progress updates.

    Connect after uploading a video to receive live updates as the
    ML pipeline processes frames, detects faces, and matches attendance.
    """
    await websocket.accept()

    # Register this connection
    if task_id not in _ws_connections:
        _ws_connections[task_id] = []
    _ws_connections[task_id].append(websocket)

    logger.info("WebSocket connected for task %s", task_id)

    try:
        # Send current status immediately
        if task_id in _tasks:
            task = _tasks[task_id]
            await websocket.send_json({
                "task_id": task_id,
                "phase": task.get("phase", "unknown"),
                "progress": task.get("progress", 0.0),
                "message": task.get("message", ""),
            })

        # Keep connection alive until client disconnects or task completes
        while True:
            try:
                # Wait for client messages (ping/pong or close)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
            except WebSocketDisconnect:
                break

    finally:
        # Unregister connection
        if task_id in _ws_connections:
            if websocket in _ws_connections[task_id]:
                _ws_connections[task_id].remove(websocket)
            if not _ws_connections[task_id]:
                del _ws_connections[task_id]

        logger.info("WebSocket disconnected for task %s", task_id)


@router.get("/today", response_model=AttendanceSummaryResponse)
async def get_today_attendance():
    """Get today's attendance summary with all records."""
    from main import get_store
    store = get_store()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records = store.get_attendance(today)
    total_enrolled = store.get_student_count()
    all_students = {s["id"]: s for s in store.list_students()}

    # Deduplicate by student_id (keep latest record per student)
    seen: dict[str, dict] = {}
    for r in records:
        sid = r.get("student_id", "")
        if sid not in seen:
            seen[sid] = r

    response_records = [
        AttendanceRecordResponse(
            id=r.get("id", ""),
            student_id=r.get("student_id", ""),
            student_name=r.get("student_name", "") or all_students.get(r.get("student_id", ""), {}).get("name", "Unknown"),
            enrollment_id=all_students.get(r.get("student_id", ""), {}).get("enrollment_id", ""),
            confidence_score=r.get("confidence_score", 0.0),
            liveness_score=r.get("liveness_score", 1.0),
            timestamp=r.get("timestamp", ""),
        )
        for r in seen.values()
    ]

    return AttendanceSummaryResponse(
        date=today,
        total_students_enrolled=total_enrolled,
        students_present=len(response_records),
        records=response_records,
    )


@router.get("/history")
async def get_attendance_history(date: str | None = None):
    """
    Get attendance for a specific date (YYYY-MM-DD).
    If no date is provided, returns today's attendance.
    """
    from main import get_store
    store = get_store()

    records = store.get_attendance(date)
    total_enrolled = store.get_student_count()
    all_students = {s["id"]: s for s in store.list_students()}

    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Deduplicate
    seen: dict[str, dict] = {}
    for r in records:
        sid = r.get("student_id", "")
        if sid not in seen:
            enriched = dict(r)
            student_info = all_students.get(sid, {})
            if not enriched.get("student_name"):
                enriched["student_name"] = student_info.get("name", "Unknown")
            enriched["enrollment_id"] = student_info.get("enrollment_id", "")
            seen[sid] = enriched

    return {
        "date": target_date,
        "total_students_enrolled": total_enrolled,
        "students_present": len(seen),
        "records": list(seen.values()),
    }


@router.post("/upload-records")
async def upload_attendance_records(file: UploadFile = File(...)):
    """
    Upload and import attendance records from a CSV or JSON file.

    Supported formats:
    - CSV: columns (student_id or enrollment_id or student_name, timestamp, confidence_score, liveness_score)
    - JSON: list of record objects
    """
    from main import get_store
    store = get_store()
    all_students = store.list_students()
    by_id = {s["id"]: s for s in all_students}
    by_enrollment = {s["enrollment_id"].strip().lower(): s for s in all_students}
    by_name = {s["name"].strip().lower(): s for s in all_students}

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    filename = file.filename or "records.csv"
    ext = Path(filename).suffix.lower()

    records_to_log = []
    now_iso = datetime.now(timezone.utc).isoformat()

    if ext == ".json":
        try:
            raw_data = json.loads(content.decode("utf-8"))
            if not isinstance(raw_data, list):
                raise ValueError("JSON file must contain an array of attendance records")
            for item in raw_data:
                if not isinstance(item, dict):
                    continue
                sid = str(item.get("student_id") or "").strip()
                enroll_id = str(item.get("enrollment_id") or "").strip().lower()
                sname = str(item.get("student_name") or item.get("name") or "").strip()

                # Resolve student
                student = by_id.get(sid) or by_enrollment.get(enroll_id) or by_name.get(sname.lower())
                final_sid = student["id"] if student else (sid or str(uuid.uuid4()))
                final_name = student["name"] if student else (sname or "Unknown")

                records_to_log.append({
                    "student_id": final_sid,
                    "student_name": final_name,
                    "confidence_score": float(item.get("confidence_score", 1.0)),
                    "liveness_score": float(item.get("liveness_score", 1.0)),
                    "timestamp": str(item.get("timestamp") or now_iso),
                })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON file: {e}")

    elif ext in [".csv", ".txt"]:
        try:
            text = content.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                # Normalize keys to lowercase stripped
                row_norm = {k.strip().lower(): v.strip() for k, v in row.items() if k}
                sid = row_norm.get("student_id", "")
                enroll_id = row_norm.get("enrollment_id", "") or row_norm.get("roll_no", "") or row_norm.get("id", "")
                sname = row_norm.get("student_name", "") or row_norm.get("name", "")

                student = by_id.get(sid) or by_enrollment.get(enroll_id.lower()) or by_name.get(sname.lower())
                final_sid = student["id"] if student else (sid or str(uuid.uuid4()))
                final_name = student["name"] if student else (sname or "Unknown")

                try:
                    conf = float(row_norm.get("confidence_score") or row_norm.get("confidence") or 1.0)
                except ValueError:
                    conf = 1.0

                try:
                    live = float(row_norm.get("liveness_score") or row_norm.get("liveness") or 1.0)
                except ValueError:
                    live = 1.0

                ts = row_norm.get("timestamp") or now_iso

                records_to_log.append({
                    "student_id": final_sid,
                    "student_name": final_name,
                    "confidence_score": conf,
                    "liveness_score": live,
                    "timestamp": ts,
                })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Please upload a .csv or .json file.",
        )

    if not records_to_log:
        raise HTTPException(status_code=400, detail="No valid records found in uploaded file")

    store.log_attendance(records_to_log)
    return {
        "status": "success",
        "message": f"Successfully imported {len(records_to_log)} attendance records",
        "count": len(records_to_log),
    }


@router.get("/export")
async def export_attendance_csv(date: str | None = None):
    """Export attendance records as a downloadable CSV."""
    from main import get_store
    store = get_store()

    records = store.get_attendance(date)
    all_students = {s["id"]: s for s in store.list_students()}
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Student ID", "Enrollment ID", "Student Name", "Confidence Score", "Liveness Score", "Timestamp"])

    for r in records:
        sid = r.get("student_id", "")
        student = all_students.get(sid, {})
        writer.writerow([
            r.get("id", ""),
            sid,
            student.get("enrollment_id", ""),
            r.get("student_name", "") or student.get("name", ""),
            r.get("confidence_score", 0.0),
            r.get("liveness_score", 1.0),
            r.get("timestamp", ""),
        ])

    csv_data = output.getvalue()
    filename = f"attendance_{target_date}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/clear")
async def clear_attendance_logs():
    """Clear all attendance logs."""
    from main import get_store
    store = get_store()
    store.clear_attendance()
    return {"status": "success", "message": "Attendance records cleared"}
