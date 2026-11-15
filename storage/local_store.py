"""
Local file-based storage layer.

Manages persistence for:
- Student metadata (students.json)
- Face embeddings (.npy files per student)
- Enrollment images (original photos)
- Attendance logs (attendance_logs.json)

Thread-safe via threading.Lock for concurrent FastAPI access.
Designed to be replaced by PostgreSQL + pgvector in a future phase.
"""

import json
import logging
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class LocalStore:
    """
    File-based storage for student data, embeddings, and attendance logs.

    File layout:
        data/
        ├── students.json               # [{id, name, enrollment_id, created_at}, ...]
        ├── attendance_logs.json         # [{id, student_id, timestamp, confidence, liveness}, ...]
        ├── embeddings/{student_id}.npy  # 512-D float32 vectors
        ├── enrollment_images/{student_id}.jpg
        └── temp/                        # Temporary video uploads
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Ensure data directories exist
        config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        config.ENROLLMENT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize JSON files if they don't exist
        if not config.STUDENTS_DB_PATH.exists():
            self._write_json(config.STUDENTS_DB_PATH, [])
        if not config.ATTENDANCE_LOGS_PATH.exists():
            self._write_json(config.ATTENDANCE_LOGS_PATH, [])

        logger.info("LocalStore initialized (data_dir=%s)", config.DATA_DIR)

    # ─────────────────────────────────────────────────────────────
    # JSON Helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _read_json(path: Path) -> list | dict:
        """Read a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: Path, data: list | dict):
        """Write data to a JSON file atomically."""
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp_path.replace(path)

    # ─────────────────────────────────────────────────────────────
    # Student Management
    # ─────────────────────────────────────────────────────────────

    def add_student(
        self,
        name: str,
        enrollment_id: str,
        embedding: np.ndarray,
        face_image: np.ndarray | None = None,
    ) -> dict:
        """
        Enroll a new student with their face embedding.

        Args:
            name: Student's full name.
            enrollment_id: Unique enrollment/roll number.
            embedding: 512-D face embedding vector.
            face_image: Optional BGR face image to save.

        Returns:
            Student dict with id, name, enrollment_id, created_at.

        Raises:
            ValueError: If enrollment_id already exists.
        """
        with self._lock:
            students = self._read_json(config.STUDENTS_DB_PATH)

            # Check for duplicate enrollment_id
            for s in students:
                if s["enrollment_id"] == enrollment_id:
                    raise ValueError(
                        f"Student with enrollment_id '{enrollment_id}' already exists"
                    )

            student_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            student = {
                "id": student_id,
                "name": name,
                "enrollment_id": enrollment_id,
                "created_at": now,
            }

            # Save embedding as .npy file
            emb_path = config.EMBEDDINGS_DIR / f"{student_id}.npy"
            np.save(str(emb_path), embedding.astype(np.float32))

            # Save enrollment image if provided
            if face_image is not None:
                img_path = config.ENROLLMENT_IMAGES_DIR / f"{student_id}.jpg"
                cv2.imwrite(str(img_path), face_image)

            # Append to students list
            students.append(student)
            self._write_json(config.STUDENTS_DB_PATH, students)

            logger.info(
                "Enrolled student: %s (id=%s, enrollment=%s)",
                name, student_id, enrollment_id,
            )

            return student

    def get_student(self, student_id: str) -> dict | None:
        """Get a student by their UUID."""
        with self._lock:
            students = self._read_json(config.STUDENTS_DB_PATH)
            for s in students:
                if s["id"] == student_id:
                    return s
            return None

    def get_student_by_enrollment_id(self, enrollment_id: str) -> dict | None:
        """Get a student by their enrollment/roll number."""
        with self._lock:
            students = self._read_json(config.STUDENTS_DB_PATH)
            for s in students:
                if s["enrollment_id"] == enrollment_id:
                    return s
            return None

    def list_students(self) -> list[dict]:
        """List all enrolled students."""
        with self._lock:
            return self._read_json(config.STUDENTS_DB_PATH)

    def delete_student(self, student_id: str) -> bool:
        """
        Delete a student and their embedding/image files.

        Returns:
            True if student was found and deleted, False otherwise.
        """
        with self._lock:
            students = self._read_json(config.STUDENTS_DB_PATH)
            original_count = len(students)
            students = [s for s in students if s["id"] != student_id]

            if len(students) == original_count:
                return False  # Student not found

            self._write_json(config.STUDENTS_DB_PATH, students)

            # Delete associated files
            emb_path = config.EMBEDDINGS_DIR / f"{student_id}.npy"
            if emb_path.exists():
                emb_path.unlink()

            img_path = config.ENROLLMENT_IMAGES_DIR / f"{student_id}.jpg"
            if img_path.exists():
                img_path.unlink()

            logger.info("Deleted student: %s", student_id)
            return True

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """
        Load all enrolled student embeddings into memory.

        Returns:
            Dict mapping student_id to 512-D numpy embedding vector.
        """
        with self._lock:
            students = self._read_json(config.STUDENTS_DB_PATH)

        embeddings: dict[str, np.ndarray] = {}
        for student in students:
            sid = student["id"]
            emb_path = config.EMBEDDINGS_DIR / f"{sid}.npy"
            if emb_path.exists():
                embeddings[sid] = np.load(str(emb_path)).astype(np.float32)
            else:
                logger.warning("Missing embedding file for student %s", sid)

        logger.info("Loaded %d/%d student embeddings", len(embeddings), len(students))
        return embeddings

    def get_student_count(self) -> int:
        """Get the total number of enrolled students."""
        with self._lock:
            students = self._read_json(config.STUDENTS_DB_PATH)
            return len(students)

    # ─────────────────────────────────────────────────────────────
    # Attendance Logs
    # ─────────────────────────────────────────────────────────────

    def log_attendance(self, records: list[dict]):
        """
        Append attendance records to the log file.

        Args:
            records: List of dicts with student_id, confidence_score,
                     liveness_score, timestamp.
        """
        if not records:
            return

        with self._lock:
            logs = self._read_json(config.ATTENDANCE_LOGS_PATH)

            for record in records:
                log_entry = {
                    "id": str(uuid.uuid4()),
                    "student_id": record.get("student_id"),
                    "student_name": record.get("student_name", ""),
                    "confidence_score": record.get("confidence_score", 0.0),
                    "liveness_score": record.get("liveness_score", 1.0),
                    "timestamp": record.get("timestamp",
                                            datetime.now(timezone.utc).isoformat()),
                }
                logs.append(log_entry)

            self._write_json(config.ATTENDANCE_LOGS_PATH, logs)

        logger.info("Logged %d attendance records", len(records))

    def get_attendance(self, date_str: str | None = None) -> list[dict]:
        """
        Get attendance records, optionally filtered by date.

        Args:
            date_str: Optional date string in YYYY-MM-DD format.
                      If None, returns today's records.

        Returns:
            List of attendance log dicts.
        """
        with self._lock:
            logs = self._read_json(config.ATTENDANCE_LOGS_PATH)

        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Filter by date prefix
        filtered = [
            log for log in logs
            if log.get("timestamp", "").startswith(date_str)
        ]

        return filtered

    def get_all_attendance(self) -> list[dict]:
        """Get all attendance records."""
        with self._lock:
            return self._read_json(config.ATTENDANCE_LOGS_PATH)

    def clear_attendance(self):
        """Clear all attendance logs (for testing)."""
        with self._lock:
            self._write_json(config.ATTENDANCE_LOGS_PATH, [])
        logger.info("Cleared all attendance logs")

    # ─────────────────────────────────────────────────────────────
    # Temporary File Management
    # ─────────────────────────────────────────────────────────────

    def save_temp_video(self, filename: str, content: bytes) -> Path:
        """
        Save an uploaded video file to the temp directory.

        Args:
            filename: Original filename (used for extension).
            content: Raw video bytes.

        Returns:
            Path to the saved temporary file.
        """
        ext = Path(filename).suffix or ".mp4"
        temp_name = f"{uuid.uuid4()}{ext}"
        temp_path = config.TEMP_DIR / temp_name

        with open(temp_path, "wb") as f:
            f.write(content)

        logger.debug("Saved temp video: %s (%d bytes)", temp_path, len(content))
        return temp_path

    def cleanup_temp(self, path: Path):
        """Delete a temporary file."""
        try:
            if path.exists():
                path.unlink()
                logger.debug("Cleaned up temp file: %s", path)
        except OSError as e:
            logger.warning("Failed to clean up temp file %s: %s", path, e)

    def cleanup_all_temp(self):
        """Delete all temporary files."""
        for f in config.TEMP_DIR.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        logger.info("Cleaned up all temp files")
