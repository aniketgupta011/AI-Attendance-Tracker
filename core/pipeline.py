"""
Main attendance pipeline orchestrator.

Combines all core modules into a single end-to-end pipeline:
Video → Frame Extraction → Face Detection → Tracking → Embedding Selection
→ Liveness Check → Database Matching → Attendance Results

Supports a progress_callback for real-time WebSocket updates.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

import config
from core.face_detector import FaceDetector
from core.face_embedder import FaceEmbedder
from core.face_matcher import FaceMatcher, MatchResult
from core.face_tracker import SimpleIoUTracker
from core.frame_extractor import FrameExtractor
from core.liveness import LivenessChecker, LivenessResult

logger = logging.getLogger(__name__)

# Type alias for the progress callback
# Signature: callback(phase: str, progress: float, message: str)
ProgressCallback = Callable[[str, float, str], None]


@dataclass
class AttendanceRecord:
    """A single attendance match result."""
    student_id: str
    student_name: str
    confidence_score: float
    liveness_score: float
    timestamp: str  # ISO format
    track_id: int


@dataclass
class AttendanceResult:
    """Complete result from processing a single video."""
    task_id: str
    status: str  # "completed", "error"
    matched_students: list[AttendanceRecord] = field(default_factory=list)
    unmatched_faces: int = 0
    total_unique_faces: int = 0
    total_frames_extracted: int = 0
    processing_time_sec: float = 0.0
    error_message: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _noop_callback(phase: str, progress: float, message: str):
    """Default no-op progress callback."""
    pass


class AttendancePipeline:
    """
    End-to-end attendance processing pipeline.

    Usage:
        pipeline = AttendancePipeline(store)
        result = pipeline.process_video("classroom.mp4", task_id="abc123")
    """

    def __init__(self, store):
        """
        Initialize the pipeline with all ML components.

        Args:
            store: A LocalStore instance for accessing enrolled embeddings
                   and student metadata.
        """
        self.store = store

        # Initialize components (models are lazy-loaded on first use)
        self.frame_extractor = FrameExtractor()
        self.face_detector = FaceDetector()
        self.face_embedder = FaceEmbedder()
        self.face_tracker = SimpleIoUTracker()
        self.face_matcher = FaceMatcher()
        self.liveness_checker = LivenessChecker()

        logger.info("AttendancePipeline initialized")

    def process_video(
        self,
        video_path: str | Path,
        task_id: str = "unknown",
        progress_callback: ProgressCallback | None = None,
    ) -> AttendanceResult:
        """
        Process a video file through the full attendance pipeline.

        Phases:
        1. Frame extraction (0.0 → 0.15)
        2. Face detection (0.15 → 0.45)
        3. Face tracking (0.45 → 0.55)
        4. Embedding selection (0.55 → 0.65)
        5. Liveness verification (0.65 → 0.75)
        6. Database matching (0.75 → 0.90)
        7. Results compilation (0.90 → 1.0)

        Args:
            video_path: Path to the video file to process.
            task_id: Unique identifier for progress tracking.
            progress_callback: Optional callback for real-time updates.

        Returns:
            AttendanceResult with all matched students and statistics.
        """
        callback = progress_callback or _noop_callback
        start_time = time.perf_counter()
        video_path = Path(video_path)

        logger.info("Processing video: %s (task_id=%s)", video_path, task_id)

        try:
            # ─── Phase 1: Frame Extraction ───────────────────────────
            callback("extracting", 0.05, "Extracting frames from video...")

            frames = self.frame_extractor.extract_frames(video_path)

            callback("extracting", 0.15, f"Extracted {len(frames)} frames")
            logger.info("Phase 1 complete: extracted %d frames", len(frames))

            if not frames:
                return AttendanceResult(
                    task_id=task_id,
                    status="completed",
                    processing_time_sec=time.perf_counter() - start_time,
                )

            # ─── Phase 2: Face Detection ─────────────────────────────
            callback("detecting", 0.15, "Detecting faces in frames...")

            all_detections: list[list] = []
            for i, frame in enumerate(frames):
                detections = self.face_detector.detect(frame.image)
                all_detections.append(detections)

                # Granular progress within detection phase
                det_progress = 0.15 + (0.30 * (i + 1) / len(frames))
                callback(
                    "detecting",
                    det_progress,
                    f"Detecting faces: frame {i + 1}/{len(frames)} "
                    f"({len(detections)} faces found)",
                )

            total_detections = sum(len(d) for d in all_detections)
            logger.info("Phase 2 complete: %d total detections across %d frames",
                        total_detections, len(frames))

            if total_detections == 0:
                return AttendanceResult(
                    task_id=task_id,
                    status="completed",
                    total_frames_extracted=len(frames),
                    processing_time_sec=time.perf_counter() - start_time,
                )

            # ─── Phase 3: Face Tracking ──────────────────────────────
            callback("tracking", 0.45, "Tracking faces across frames...")

            self.face_tracker.reset()
            for i, (frame, detections) in enumerate(zip(frames, all_detections)):
                self.face_tracker.update(frame.index, detections)

            tracks = self.face_tracker.get_all_tracks()

            callback("tracking", 0.55, f"Tracked {len(tracks)} unique faces")
            logger.info("Phase 3 complete: %d unique tracked faces", len(tracks))

            if not tracks:
                return AttendanceResult(
                    task_id=task_id,
                    status="completed",
                    total_frames_extracted=len(frames),
                    processing_time_sec=time.perf_counter() - start_time,
                )

            # ─── Phase 4: Embedding Selection ────────────────────────
            callback("embedding", 0.55, "Selecting best face embeddings...")

            track_embeddings: dict[int, np.ndarray] = {}
            for track in tracks:
                embedding = self.face_embedder.get_best_embedding(track)
                if embedding is not None:
                    track_embeddings[track.track_id] = embedding

            callback("embedding", 0.65, f"Generated {len(track_embeddings)} embeddings")
            logger.info("Phase 4 complete: %d valid embeddings from %d tracks",
                        len(track_embeddings), len(tracks))

            # ─── Phase 5: Liveness Verification ──────────────────────
            callback("liveness", 0.65, "Verifying face liveness...")

            tracks_crops: dict[int, list[np.ndarray]] = {}
            for track in tracks:
                crops = [
                    d.face_crop for d in track.detections
                    if d.face_crop is not None and d.face_crop.size > 0
                ]
                if crops:
                    tracks_crops[track.track_id] = crops

            liveness_results = self.liveness_checker.check_batch(tracks_crops)

            # Filter out spoofs
            live_embeddings: dict[int, np.ndarray] = {}
            for track_id, embedding in track_embeddings.items():
                liveness = liveness_results.get(track_id)
                if liveness and liveness.is_live:
                    live_embeddings[track_id] = embedding
                else:
                    logger.info("Track %d failed liveness check (score=%.3f)",
                                track_id,
                                liveness.score if liveness else 0.0)

            callback("liveness", 0.75,
                     f"{len(live_embeddings)}/{len(track_embeddings)} passed liveness")
            logger.info("Phase 5 complete: %d/%d passed liveness",
                        len(live_embeddings), len(track_embeddings))

            # ─── Phase 6: Database Matching ──────────────────────────
            callback("matching", 0.75, "Matching faces against enrolled students...")

            # Load enrolled embeddings
            enrolled = self.store.get_all_embeddings()
            self.face_matcher.load_embeddings(enrolled)

            match_results = self.face_matcher.match_batch(live_embeddings)

            callback("matching", 0.90, "Matching complete")
            logger.info("Phase 6 complete: matching done")

            # ─── Phase 7: Results Compilation ────────────────────────
            callback("compiling", 0.90, "Compiling attendance results...")

            matched_students: list[AttendanceRecord] = []
            unmatched_count = 0
            now = datetime.now(timezone.utc).isoformat()

            for track_id, match in match_results.items():
                if match and match.is_match:
                    # Get student name from store
                    student = self.store.get_student(match.student_id)
                    student_name = student["name"] if student else "Unknown"

                    liveness = liveness_results.get(track_id)
                    liveness_score = liveness.score if liveness else 0.0

                    matched_students.append(AttendanceRecord(
                        student_id=match.student_id,
                        student_name=student_name,
                        confidence_score=match.similarity,
                        liveness_score=liveness_score,
                        timestamp=now,
                        track_id=track_id,
                    ))
                else:
                    unmatched_count += 1

            processing_time = time.perf_counter() - start_time

            result = AttendanceResult(
                task_id=task_id,
                status="completed",
                matched_students=matched_students,
                unmatched_faces=unmatched_count,
                total_unique_faces=len(tracks),
                total_frames_extracted=len(frames),
                processing_time_sec=round(processing_time, 2),
            )

            callback("complete", 1.0,
                     f"Done! {len(matched_students)} students recognized "
                     f"in {processing_time:.1f}s")

            logger.info(
                "Pipeline complete: %d matched, %d unmatched, %d total tracks, "
                "%.2fs processing time",
                len(matched_students), unmatched_count, len(tracks),
                processing_time,
            )

            return result

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            logger.error("Pipeline error: %s", e, exc_info=True)
            callback("error", 0.0, f"Error: {str(e)}")

            return AttendanceResult(
                task_id=task_id,
                status="error",
                processing_time_sec=round(processing_time, 2),
                error_message=str(e),
            )
