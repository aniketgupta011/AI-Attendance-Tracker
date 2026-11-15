"""
AI Attendance System — CLI Entry Point.

Standalone command-line interface for testing the ML pipeline
without running the FastAPI server.

Usage:
    python cli.py enroll --name "John Doe" --id STU001 --image face.jpg
    python cli.py process classroom.mp4
    python cli.py list-students
    python cli.py attendance [--date YYYY-MM-DD]
    python cli.py clear-attendance
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from core.face_detector import FaceDetector
from core.pipeline import AttendancePipeline
from storage.local_store import LocalStore

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


def _progress_callback(phase: str, progress: float, message: str):
    """Print progress updates to the console."""
    bar_width = 30
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r  [{bar}] {progress*100:5.1f}% │ {phase:<12s} │ {message}", end="", flush=True)
    if progress >= 1.0 or phase == "error":
        print()  # Newline at completion


# ─────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────

def cmd_enroll(args):
    """Enroll a new student."""
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read image: {image_path}")
        sys.exit(1)

    print(f"Enrolling student: {args.name} (ID: {args.id})")
    print(f"  Image: {image_path}")

    # Detect face
    detector = FaceDetector()
    face = detector.detect_single(img)

    if face is None:
        print("Error: No face detected in the image.")
        print("  Please use a clear, front-facing photo with exactly one face.")
        sys.exit(1)

    print(f"  Face detected (confidence: {face.det_score:.3f})")

    # Store
    store = LocalStore()

    try:
        student = store.add_student(
            name=args.name,
            enrollment_id=args.id,
            embedding=face.embedding,
            face_image=img,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"  ✓ Enrolled successfully!")
    print(f"  Student UUID: {student['id']}")


def cmd_process(args):
    """Process a video for attendance."""
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Processing video: {video_path}")
    print()

    store = LocalStore()
    pipeline = AttendancePipeline(store)

    result = pipeline.process_video(
        video_path=video_path,
        task_id="cli-task",
        progress_callback=_progress_callback,
    )

    print()
    print("=" * 60)
    print("  ATTENDANCE RESULTS")
    print("=" * 60)

    result_dict = asdict(result)

    print(f"  Status:           {result.status}")
    print(f"  Processing Time:  {result.processing_time_sec:.2f}s")
    print(f"  Frames Extracted: {result.total_frames_extracted}")
    print(f"  Unique Faces:     {result.total_unique_faces}")
    print(f"  Matched Students: {len(result.matched_students)}")
    print(f"  Unmatched Faces:  {result.unmatched_faces}")

    if result.matched_students:
        print()
        print("  Recognized Students:")
        print("  " + "-" * 56)
        print(f"  {'Name':<25s} {'ID':<12s} {'Confidence':<12s} {'Liveness':<10s}")
        print("  " + "-" * 56)

        for r in result.matched_students:
            student = store.get_student(r.student_id)
            enrollment_id = student["enrollment_id"] if student else "?"
            print(
                f"  {r.student_name:<25s} {enrollment_id:<12s} "
                f"{r.confidence_score:<12.4f} {r.liveness_score:<10.2f}"
            )

    if result.error_message:
        print(f"\n  Error: {result.error_message}")

    # Log attendance
    if result.matched_students:
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
        print(f"\n  ✓ Attendance logged ({len(records)} records)")

    # Also output raw JSON for programmatic use
    if args.json:
        print("\n--- JSON Output ---")
        print(json.dumps(result_dict, indent=2, default=str))


def cmd_list_students(args):
    """List all enrolled students."""
    store = LocalStore()
    students = store.list_students()

    if not students:
        print("No students enrolled.")
        return

    print(f"Enrolled Students ({len(students)} total):")
    print("-" * 70)
    print(f"{'Name':<25s} {'Enrollment ID':<15s} {'UUID':<38s}")
    print("-" * 70)

    for s in students:
        print(f"{s['name']:<25s} {s['enrollment_id']:<15s} {s['id']:<38s}")


def cmd_attendance(args):
    """Show attendance records."""
    store = LocalStore()
    records = store.get_attendance(args.date)

    date_label = args.date or "today"

    if not records:
        print(f"No attendance records for {date_label}.")
        return

    print(f"Attendance for {date_label} ({len(records)} records):")
    print("-" * 80)
    print(f"{'Name':<25s} {'Confidence':<12s} {'Liveness':<10s} {'Timestamp':<25s}")
    print("-" * 80)

    for r in records:
        print(
            f"{r.get('student_name', '?'):<25s} "
            f"{r.get('confidence_score', 0):<12.4f} "
            f"{r.get('liveness_score', 0):<10.2f} "
            f"{r.get('timestamp', '?'):<25s}"
        )


def cmd_clear_attendance(args):
    """Clear all attendance records."""
    store = LocalStore()
    store.clear_attendance()
    print("✓ All attendance records cleared.")


def cmd_delete_student(args):
    """Delete a student by enrollment ID."""
    store = LocalStore()

    student = store.get_student_by_enrollment_id(args.id)
    if student is None:
        print(f"Error: No student found with enrollment ID '{args.id}'")
        sys.exit(1)

    store.delete_student(student["id"])
    print(f"✓ Deleted student: {student['name']} ({args.id})")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Attendance System — Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py enroll --name "John Doe" --id STU001 --image face.jpg
  python cli.py process classroom.mp4
  python cli.py process classroom.mp4 --json
  python cli.py list-students
  python cli.py attendance --date 2025-01-15
  python cli.py delete-student --id STU001
  python cli.py clear-attendance
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # enroll
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new student")
    enroll_parser.add_argument("--name", required=True, help="Student's full name")
    enroll_parser.add_argument("--id", required=True, help="Enrollment/roll number")
    enroll_parser.add_argument("--image", required=True, help="Path to face photo")

    # process
    process_parser = subparsers.add_parser("process", help="Process a video for attendance")
    process_parser.add_argument("video", help="Path to the video file")
    process_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # list-students
    subparsers.add_parser("list-students", help="List all enrolled students")

    # attendance
    att_parser = subparsers.add_parser("attendance", help="Show attendance records")
    att_parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD), default: today")

    # delete-student
    del_parser = subparsers.add_parser("delete-student", help="Delete a student")
    del_parser.add_argument("--id", required=True, help="Enrollment ID to delete")

    # clear-attendance
    subparsers.add_parser("clear-attendance", help="Clear all attendance records")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "enroll": cmd_enroll,
        "process": cmd_process,
        "list-students": cmd_list_students,
        "attendance": cmd_attendance,
        "delete-student": cmd_delete_student,
        "clear-attendance": cmd_clear_attendance,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
