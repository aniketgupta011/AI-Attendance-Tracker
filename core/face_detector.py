"""
Face detection module using InsightFace.

Wraps InsightFace's FaceAnalysis application which provides:
- RetinaFace detector for robust face detection (including small/occluded faces)
- ArcFace recognizer for 512-D embedding generation
- Face alignment and landmark detection

Both detection and embedding happen in a single forward pass, which is
more efficient than running separate models.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """A single detected face with all associated data."""
    bbox: np.ndarray           # [x1, y1, x2, y2] bounding box
    det_score: float           # Detection confidence (0-1)
    landmarks: np.ndarray      # 5-point facial landmarks (5, 2)
    embedding: np.ndarray      # 512-D ArcFace embedding (L2-normalized)
    yaw: float = 0.0          # Estimated yaw angle in degrees
    pitch: float = 0.0        # Estimated pitch angle in degrees
    face_crop: np.ndarray | None = field(default=None, repr=False)  # Cropped face image

    @property
    def bbox_area(self) -> float:
        """Area of the bounding box in pixels."""
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return float(max(0, w) * max(0, h))


class FaceDetector:
    """
    Face detection and embedding using InsightFace's buffalo_l model pack.

    The buffalo_l pack includes:
    - det_10g.onnx: RetinaFace detector (~30 MB)
    - w600k_r50.onnx: ArcFace recognizer (~170 MB)

    Both models run via ONNX Runtime with CUDA Execution Provider
    when available, falling back to CPU.
    """

    def __init__(
        self,
        model_pack: str = config.INSIGHTFACE_MODEL_PACK,
        det_threshold: float = config.DETECTION_THRESHOLD,
        det_size: tuple = config.DETECTION_INPUT_SIZE,
    ):
        self.model_pack = model_pack
        self.det_threshold = det_threshold
        self.det_size = det_size
        self._app = None

    def _load_model(self):
        """Lazy-load the InsightFace model on first use."""
        if self._app is not None:
            return

        from insightface.app import FaceAnalysis

        logger.info(
            "Loading InsightFace model pack '%s' with providers: %s",
            self.model_pack,
            config.ONNX_PROVIDERS,
        )

        self._app = FaceAnalysis(
            name=self.model_pack,
            root=str(config.MODELS_DIR),
            providers=config.ONNX_PROVIDERS,
        )
        self._app.prepare(
            ctx_id=0 if config.DEVICE == "cuda" else -1,
            det_thresh=self.det_threshold,
            det_size=self.det_size,
        )

        logger.info("InsightFace model loaded successfully (device: %s)", config.DEVICE)

    def _estimate_pose(self, landmarks: np.ndarray) -> tuple[float, float]:
        """
        Estimate rough yaw and pitch angles from 5-point facial landmarks.

        Uses the relative positions of the eyes and nose to approximate
        head pose. This is a simplified geometric estimation — not as
        accurate as a dedicated pose model, but sufficient for selecting
        the most frontal face crop.

        Landmark order (InsightFace convention):
            0: left eye, 1: right eye, 2: nose tip,
            3: left mouth corner, 4: right mouth corner

        Args:
            landmarks: (5, 2) array of facial landmark coordinates.

        Returns:
            Tuple of (yaw_degrees, pitch_degrees).
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        # Eye midpoint
        eye_center = (left_eye + right_eye) / 2.0
        eye_width = np.linalg.norm(right_eye - left_eye)

        if eye_width < 1e-6:
            return 0.0, 0.0

        # Yaw: horizontal offset of nose from eye center, normalized by eye width
        # Positive yaw = face turned to its left (our right)
        nose_offset_x = (nose[0] - eye_center[0]) / eye_width
        yaw = float(np.degrees(np.arctan2(nose_offset_x, 1.0)))

        # Pitch: vertical offset of nose from eye center, normalized by eye width
        # Positive pitch = face looking down
        nose_offset_y = (nose[1] - eye_center[1]) / eye_width
        # Subtract expected ratio (~0.65) for a frontal face
        pitch = float(np.degrees(np.arctan2(nose_offset_y - 0.65, 1.0)))

        return yaw, pitch

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """
        Detect all faces in a single frame and compute their embeddings.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of DetectedFace objects, each with bbox, landmarks,
            embedding, detection score, and estimated pose.
        """
        self._load_model()

        # InsightFace's get() returns a list of Face objects
        faces = self._app.get(frame)

        results: list[DetectedFace] = []
        for face in faces:
            # Skip faces without embeddings (detection-only fallback)
            if face.embedding is None:
                logger.debug("Skipping face without embedding (det_score=%.3f)", face.det_score)
                continue

            # Extract bounding box and crop the face from the original frame
            bbox = face.bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

            # Estimate head pose from landmarks
            yaw, pitch = self._estimate_pose(face.landmark_2d_106[:5]
                                              if hasattr(face, 'landmark_2d_106')
                                              and face.landmark_2d_106 is not None
                                              else face.kps)

            # Normalize embedding to unit vector
            embedding = face.embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            results.append(DetectedFace(
                bbox=face.bbox.flatten().astype(np.float32),
                det_score=float(face.det_score),
                landmarks=face.kps.copy() if face.kps is not None else np.zeros((5, 2)),
                embedding=embedding,
                yaw=yaw,
                pitch=pitch,
                face_crop=face_crop,
            ))

        logger.debug("Detected %d faces in frame", len(results))
        return results

    def detect_single(self, image: np.ndarray) -> DetectedFace | None:
        """
        Detect and return the single largest/most prominent face in an image.

        Used for enrollment where we expect exactly one face.

        Args:
            image: BGR image as numpy array.

        Returns:
            The DetectedFace with the largest bounding box area, or None
            if no face is detected.
        """
        faces = self.detect(image)
        if not faces:
            return None

        # Return the face with the largest bounding box
        return max(faces, key=lambda f: f.bbox_area)
