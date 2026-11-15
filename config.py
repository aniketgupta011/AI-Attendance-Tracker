"""
Centralized configuration for the AI Attendance System.

All tunable parameters are defined here as module-level constants.
Import this module wherever configuration is needed.
"""

import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Project root is the directory containing this file
PROJECT_ROOT = Path(__file__).parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ENROLLMENT_IMAGES_DIR = DATA_DIR / "enrollment_images"
TEMP_DIR = DATA_DIR / "temp"
MODELS_DIR = PROJECT_ROOT / "models"

STUDENTS_DB_PATH = DATA_DIR / "students.json"
ATTENDANCE_LOGS_PATH = DATA_DIR / "attendance_logs.json"

# Ensure all data directories exist
for _dir in [DATA_DIR, EMBEDDINGS_DIR, ENROLLMENT_IMAGES_DIR, TEMP_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Device Configuration
# =============================================================================

def _detect_device() -> str:
    """
    Auto-detect whether CUDA (GPU) is available.

    Checks PyTorch first (most reliable), then falls back to ONNX Runtime.
    """
    # Primary: PyTorch CUDA check
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, OSError, Exception):
        pass

    # Fallback: ONNX Runtime CUDA check
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "cuda"
    except (ImportError, OSError, Exception):
        pass

    return "cpu"

DEVICE = _detect_device()

# PyTorch device object for torch-based modules (liveness, matcher)
def _get_torch_device():
    """Get a torch.device matching the detected DEVICE string."""
    try:
        import torch
        if DEVICE == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except (ImportError, OSError, Exception):
        return None

TORCH_DEVICE = _get_torch_device()

# ONNX Runtime execution providers ordered by preference
ONNX_PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if DEVICE == "cuda"
    else ["CPUExecutionProvider"]
)

# =============================================================================
# Frame Extraction Settings
# =============================================================================

# Target frames per second to extract from uploaded video.
# Lower = faster processing, higher = more faces captured.
# For a 30fps video with FRAME_SAMPLE_FPS=4, we process every ~8th frame.
FRAME_SAMPLE_FPS: float = 4.0

# Maximum allowed video duration in seconds
MAX_VIDEO_DURATION_SEC: float = 15.0

# Maximum upload file size in bytes (50 MB)
MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024

# Allowed video file extensions
ALLOWED_VIDEO_EXTENSIONS: set = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Allowed image file extensions (for enrollment)
ALLOWED_IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =============================================================================
# CLAHE (Contrast Limited Adaptive Histogram Equalization) Settings
# =============================================================================

CLAHE_CLIP_LIMIT: float = 2.0
CLAHE_TILE_GRID_SIZE: tuple = (8, 8)

# =============================================================================
# Face Detection Settings (InsightFace RetinaFace)
# =============================================================================

# InsightFace model pack name (downloaded automatically on first use)
INSIGHTFACE_MODEL_PACK: str = "buffalo_l"

# Minimum detection confidence to keep a face
DETECTION_THRESHOLD: float = 0.5

# Detection input size (width, height) — InsightFace default
DETECTION_INPUT_SIZE: tuple = (640, 640)

# =============================================================================
# Face Tracking Settings (IoU Tracker)
# =============================================================================

# Minimum IoU overlap to associate a detection with an existing track
IOU_THRESHOLD: float = 0.3

# Number of consecutive frames a track can be "missing" before expiry
MAX_FRAMES_LOST: int = 10

# Minimum number of detections in a track to consider it valid
MIN_TRACK_LENGTH: int = 2

# =============================================================================
# Face Quality Assessment
# =============================================================================

# Laplacian variance threshold — frames below this are considered too blurry
MIN_BLUR_VARIANCE: float = 50.0

# Maximum absolute yaw angle (degrees) to consider a face "frontal"
MAX_YAW_DEGREES: float = 45.0

# Maximum absolute pitch angle (degrees) to consider a face "frontal"
MAX_PITCH_DEGREES: float = 30.0

# =============================================================================
# Face Matching Settings
# =============================================================================

# Cosine similarity threshold for a positive match.
# InsightFace ArcFace embeddings are L2-normalized, so cosine similarity
# equals the dot product. Typical thresholds: 0.4 (lenient) to 0.6 (strict).
SIMILARITY_THRESHOLD: float = 0.45

# =============================================================================
# Liveness Detection Settings
# =============================================================================

# Liveness score threshold — faces below this are flagged as spoofs
LIVENESS_THRESHOLD: float = 0.85

# =============================================================================
# Server Settings
# =============================================================================

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS: list = ["*"]  # Allow all origins in development
