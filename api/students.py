"""
Student enrollment API routes.

Handles:
- Enrolling new students (face image → embedding → storage)
- Listing enrolled students
- Deleting students
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/students", tags=["students"])


# ─────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────

class StudentResponse(BaseModel):
    id: str
    name: str
    enrollment_id: str
    created_at: str


class StudentListResponse(BaseModel):
    total: int
    students: list[StudentResponse]


class EnrollResponse(BaseModel):
    message: str
    student: StudentResponse


class DeleteResponse(BaseModel):
    message: str
    student_id: str


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@router.post("/enroll", response_model=EnrollResponse, status_code=201)
async def enroll_student(
    name: str = Form(...),
    enrollment_id: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Enroll a new student with a face photo.

    The image is processed to detect a face, extract a 512-D embedding,
    and store both the embedding and metadata for future matching.

    Requires a clear, front-facing photo with exactly one visible face.
    """
    # Validate image extension
    ext = Path(image.filename or "").suffix.lower()
    if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type '{ext}'. Allowed: {config.ALLOWED_IMAGE_EXTENSIONS}",
        )

    # Read and decode image
    content = await image.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Decode image bytes to numpy array
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image file")

    # Detect face and extract embedding
    from main import get_pipeline, get_store
    pipeline = get_pipeline()
    store = get_store()

    face = pipeline.face_detector.detect_single(img)
    if face is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image. Please upload a clear, "
                   "front-facing photo with exactly one face visible.",
        )

    # Validate face quality
    if face.det_score < config.DETECTION_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"Face detection confidence too low ({face.det_score:.2f}). "
                   "Please upload a clearer photo.",
        )

    # Check if enrollment_id already exists
    existing = store.get_student_by_enrollment_id(enrollment_id)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Student with enrollment_id '{enrollment_id}' already exists "
                   f"(name: {existing['name']})",
        )

    # Enroll the student
    try:
        student = store.add_student(
            name=name,
            enrollment_id=enrollment_id,
            embedding=face.embedding,
            face_image=img,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    logger.info("Enrolled student: %s (%s)", name, enrollment_id)

    return EnrollResponse(
        message=f"Student '{name}' enrolled successfully",
        student=StudentResponse(
            id=student["id"],
            name=student["name"],
            enrollment_id=student["enrollment_id"],
            created_at=student["created_at"],
        ),
    )


@router.get("", response_model=StudentListResponse)
async def list_students():
    """List all enrolled students."""
    from main import get_store
    store = get_store()

    students = store.list_students()

    return StudentListResponse(
        total=len(students),
        students=[
            StudentResponse(
                id=s["id"],
                name=s["name"],
                enrollment_id=s["enrollment_id"],
                created_at=s["created_at"],
            )
            for s in students
        ],
    )


@router.get("/{student_id}", response_model=StudentResponse)
async def get_student(student_id: str):
    """Get a single student by their UUID."""
    from main import get_store
    store = get_store()

    student = store.get_student(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    return StudentResponse(
        id=student["id"],
        name=student["name"],
        enrollment_id=student["enrollment_id"],
        created_at=student["created_at"],
    )


@router.delete("/{student_id}", response_model=DeleteResponse)
async def delete_student(student_id: str):
    """Delete a student and their associated embedding/image."""
    from main import get_store
    store = get_store()

    deleted = store.delete_student(student_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    return DeleteResponse(
        message="Student deleted successfully",
        student_id=student_id,
    )


@router.get("/{student_id}/photo")
async def get_student_photo(student_id: str):
    """Serve the student's enrollment photo."""
    img_path = config.ENROLLMENT_IMAGES_DIR / f"{student_id}.jpg"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Photo not found")
    return FileResponse(str(img_path), media_type="image/jpeg")
