"""API routes package."""

from api.attendance import router as attendance_router
from api.students import router as students_router

__all__ = ["attendance_router", "students_router"]
