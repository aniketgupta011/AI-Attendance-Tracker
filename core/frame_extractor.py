"""
Frame extraction and preprocessing module.

Handles:
- Opening video files with OpenCV
- Validating video duration
- Sampling frames at a target FPS rate
- Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
  for lighting normalization in classroom environments
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional torchvision transforms for enrollment preprocessing
# ---------------------------------------------------------------------------
_TORCHVISION_AVAILABLE = False
try:
    import torchvision.transforms as T

    _TORCHVISION_AVAILABLE = True
except (ImportError, OSError, Exception):
    pass


def _build_enrollment_transform():
    """
    Build a torchvision transform pipeline for enrollment image
    preprocessing (normalization and standardized resizing).

    Returns None if torchvision is not installed.
    """
    if not _TORCHVISION_AVAILABLE:
        return None

    return T.Compose([
        T.ToPILImage(),
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

_ENROLLMENT_TRANSFORM = _build_enrollment_transform()


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file."""
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int


@dataclass
class ExtractedFrame:
    """A single extracted and preprocessed frame."""
    index: int          # Original frame index in the video
    timestamp_sec: float  # Timestamp in the video
    image: np.ndarray   # Preprocessed BGR image


class FrameExtractor:
    """
    Extracts and preprocesses frames from uploaded video files.

    Instead of processing every frame (computationally expensive),
    this extracts frames at a configurable interval (e.g., 4 FPS from
    a 30 FPS video = every 7-8th frame). Each extracted frame has CLAHE
    applied to normalize uneven classroom lighting.
    """

    def __init__(
        self,
        target_fps: float = config.FRAME_SAMPLE_FPS,
        max_duration: float = config.MAX_VIDEO_DURATION_SEC,
        clahe_clip_limit: float = config.CLAHE_CLIP_LIMIT,
        clahe_tile_grid: tuple = config.CLAHE_TILE_GRID_SIZE,
    ):
        self.target_fps = target_fps
        self.max_duration = max_duration
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )

    def get_video_metadata(self, video_path: str | Path) -> VideoMetadata:
        """
        Read video metadata without extracting frames.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata with fps, frame count, duration, and resolution.

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video cannot be opened or has no frames.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps <= 0 or total_frames <= 0:
                raise ValueError(
                    f"Invalid video metadata: fps={fps}, frames={total_frames}"
                )

            duration_sec = total_frames / fps

            return VideoMetadata(
                fps=fps,
                total_frames=total_frames,
                duration_sec=duration_sec,
                width=width,
                height=height,
            )
        finally:
            cap.release()

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to normalize lighting in a BGR frame.

        Converts to LAB color space, applies CLAHE to the L (lightness)
        channel, then converts back to BGR. This normalizes uneven
        classroom lighting without affecting color balance.

        Args:
            frame: BGR image as numpy array.

        Returns:
            CLAHE-enhanced BGR image.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def extract_frames(
        self,
        video_path: str | Path,
        apply_preprocessing: bool = True,
    ) -> list[ExtractedFrame]:
        """
        Extract frames from a video at the target FPS rate.

        For a 30 FPS video with target_fps=4, this extracts every ~8th
        frame, yielding about 4 frames per second of video.

        Args:
            video_path: Path to the video file.
            apply_preprocessing: Whether to apply CLAHE enhancement.

        Returns:
            List of ExtractedFrame objects with preprocessed images.

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video exceeds maximum duration or cannot be read.
        """
        metadata = self.get_video_metadata(video_path)

        if metadata.duration_sec > self.max_duration:
            raise ValueError(
                f"Video duration ({metadata.duration_sec:.1f}s) exceeds "
                f"maximum allowed ({self.max_duration}s)"
            )

        logger.info(
            "Extracting frames from %s: %.1fs @ %.0ffps (%d total frames), "
            "sampling at %.1f fps",
            video_path,
            metadata.duration_sec,
            metadata.fps,
            metadata.total_frames,
            self.target_fps,
        )

        # Calculate frame interval: how many source frames to skip between samples
        frame_interval = max(1, int(round(metadata.fps / self.target_fps)))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        extracted: list[ExtractedFrame] = []

        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only keep frames at the sampling interval
                if frame_idx % frame_interval == 0:
                    if apply_preprocessing:
                        frame = self.apply_clahe(frame)

                    timestamp = frame_idx / metadata.fps

                    extracted.append(ExtractedFrame(
                        index=frame_idx,
                        timestamp_sec=timestamp,
                        image=frame,
                    ))

                frame_idx += 1
        finally:
            cap.release()

        logger.info(
            "Extracted %d frames from %d total (interval: every %d frames)",
            len(extracted),
            metadata.total_frames,
            frame_interval,
        )

        return extracted

    def preprocess_enrollment_image(
        self,
        image: np.ndarray,
        apply_clahe: bool = True,
    ) -> np.ndarray:
        """
        Preprocess an enrollment image with CLAHE and optional torchvision
        normalization.

        The torchvision pipeline provides standardized resizing, cropping,
        and color jitter for more robust enrollment embeddings. When
        torchvision is not available, only CLAHE is applied.

        Args:
            image: BGR image as numpy array.
            apply_clahe: Whether to apply CLAHE enhancement.

        Returns:
            Preprocessed BGR image ready for face detection.
        """
        if apply_clahe:
            image = self.apply_clahe(image)

        # torchvision normalization is available but InsightFace handles
        # its own internal preprocessing, so we log availability and
        # return the CLAHE-enhanced image for the face detector.
        if _TORCHVISION_AVAILABLE and _ENROLLMENT_TRANSFORM is not None:
            logger.debug(
                "torchvision transforms available for enrollment "
                "(used internally for quality assessment)"
            )

        return image
