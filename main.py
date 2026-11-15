"""
AI Attendance System — FastAPI Application Entry Point.

Initializes the ML pipeline, storage layer, and mounts all API routes.
Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from api.attendance import router as attendance_router
from api.students import router as students_router
from core.pipeline import AttendancePipeline
from storage.local_store import LocalStore

# ─────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Global State (initialized during lifespan)
# ─────────────────────────────────────────────────────────────────

_store: LocalStore | None = None
_pipeline: AttendancePipeline | None = None


def get_store() -> LocalStore:
    """Get the global LocalStore instance."""
    global _store
    if _store is None:
        _store = LocalStore()
    return _store


def get_pipeline() -> AttendancePipeline:
    """Get the global AttendancePipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AttendancePipeline(get_store())
    return _pipeline


# ─────────────────────────────────────────────────────────────────
# Application Lifespan
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize ML pipeline and storage on startup.
    Clean up resources on shutdown.
    """
    global _store, _pipeline

    logger.info("=" * 60)
    logger.info("  AI Attendance System — Starting Up")
    logger.info("=" * 60)
    logger.info("Device: %s", config.DEVICE)
    logger.info("ONNX Providers: %s", config.ONNX_PROVIDERS)
    logger.info("Data directory: %s", config.DATA_DIR)
    logger.info("Models directory: %s", config.MODELS_DIR)

    # Initialize storage
    _store = LocalStore()
    student_count = _store.get_student_count()
    logger.info("Storage initialized: %d students enrolled", student_count)

    # Initialize ML pipeline
    _pipeline = AttendancePipeline(_store)
    logger.info("ML pipeline initialized")

    logger.info("=" * 60)
    logger.info("  Server ready at http://%s:%d", config.API_HOST, config.API_PORT)
    logger.info("  API docs at http://%s:%d/docs", config.API_HOST, config.API_PORT)
    logger.info("=" * 60)

    yield  # App is running

    # Shutdown
    logger.info("Shutting down — cleaning up temp files...")
    if _store:
        _store.cleanup_all_temp()
    logger.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Attendance System",
    description=(
        "Facial recognition attendance system using InsightFace (RetinaFace + ArcFace). "
        "Upload a classroom video to automatically detect, track, and identify enrolled "
        "students. Real-time progress via WebSocket."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(attendance_router)
app.include_router(students_router)

# Mount frontend directory for static assets
FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", tags=["frontend"])
async def root():
    """Serve the frontend application or system info if not found."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": "AI Attendance System",
        "version": "0.1.0",
        "status": "running",
        "device": config.DEVICE,
        "students_enrolled": _store.get_student_count() if _store else 0,
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "device": config.DEVICE,
        "onnx_providers": config.ONNX_PROVIDERS,
        "data_dir": str(config.DATA_DIR),
        "students_enrolled": _store.get_student_count() if _store else 0,
    }


@app.get("/sample-video", tags=["sample"])
async def get_sample_video():
    """Serve the bundled sample classroom video for quick testing."""
    from fastapi import HTTPException
    video_path = config.PROJECT_ROOT / "classroom.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Sample video not found")
    return FileResponse(str(video_path), media_type="video/mp4", filename="classroom.mp4")


# ─────────────────────────────────────────────────────────────────
# Direct Execution
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        log_level="info",
    )
