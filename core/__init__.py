"""
Core ML pipeline package for the AI Attendance System.

This package contains all the machine learning components:
- frame_extractor: Video frame extraction with CLAHE + torchvision preprocessing
- face_detector: InsightFace RetinaFace face detection
- face_tracker: Simple IoU-based multi-face tracking
- face_embedder: ArcFace embedding extraction and quality selection
- face_matcher: Cosine similarity matching (GPU-accelerated via PyTorch)
- liveness: Heuristic + CNN anti-spoofing liveness verification
- pipeline: Main orchestrator combining all components
"""

from core.pipeline import AttendancePipeline

__all__ = ["AttendancePipeline"]
