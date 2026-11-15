"""
Simple IoU-based multi-face tracker.

Links face detections across frames by computing Intersection-over-Union
(IoU) between existing tracks and new detections, using the Hungarian
algorithm for optimal assignment.

This is a lightweight alternative to ByteTrack, sufficient for short
(5-15 second) uploaded videos where faces move relatively slowly between
sampled frames (3-5 FPS).
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from core.face_detector import DetectedFace

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """
    A tracked face identity across multiple frames.

    Each track accumulates face crops, embeddings, and quality scores
    across the frames where the face was detected. After tracking is
    complete, the best crop and embedding can be selected.
    """
    track_id: int
    detections: list[DetectedFace] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)
    frames_since_seen: int = 0

    @property
    def last_bbox(self) -> np.ndarray:
        """Most recent bounding box for this track."""
        return self.detections[-1].bbox

    @property
    def length(self) -> int:
        """Number of frames this track spans."""
        return len(self.detections)

    def add_detection(self, detection: DetectedFace, frame_idx: int):
        """Add a new detection to this track."""
        self.detections.append(detection)
        self.frame_indices.append(frame_idx)
        self.frames_since_seen = 0

    def compute_quality_score(self, detection: DetectedFace) -> float:
        """
        Compute a quality score for a single detection.

        Combines blur variance (sharpness) and frontality (low yaw/pitch)
        into a single score. Higher is better.

        Args:
            detection: The face detection to evaluate.

        Returns:
            Float quality score (higher = better quality face crop).
        """
        # Blur variance: Laplacian variance of the face crop
        blur_score = 0.0
        if detection.face_crop is not None and detection.face_crop.size > 0:
            import cv2
            gray = cv2.cvtColor(detection.face_crop, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Frontality: penalize high yaw and pitch angles
        # Score ranges from 0 (extreme side) to 1 (perfectly frontal)
        yaw_penalty = max(0.0, 1.0 - abs(detection.yaw) / 90.0)
        pitch_penalty = max(0.0, 1.0 - abs(detection.pitch) / 90.0)
        frontality = yaw_penalty * pitch_penalty

        # Detection confidence also contributes
        det_conf = detection.det_score

        # Weighted combination:
        # - blur_score is typically 50-2000+, normalize to ~0-1 range
        normalized_blur = min(blur_score / 500.0, 1.0)

        return normalized_blur * 0.4 + frontality * 0.4 + det_conf * 0.2

    def get_best_detection(self) -> DetectedFace | None:
        """
        Select the highest-quality detection from this track.

        Evaluates all detections based on blur variance, frontality,
        and detection confidence to find the best crop.

        Returns:
            The DetectedFace with the highest quality score, or None
            if the track has no detections.
        """
        if not self.detections:
            return None

        best_det = None
        best_score = -1.0

        for det in self.detections:
            score = self.compute_quality_score(det)
            if score > best_score:
                best_score = score
                best_det = det

        return best_det


def _compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """
    Compute Intersection-over-Union between two bounding boxes.

    Args:
        bbox_a: [x1, y1, x2, y2] array
        bbox_b: [x1, y1, x2, y2] array

    Returns:
        IoU value in [0, 1].
    """
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def _compute_iou_matrix(
    tracks: list[Track],
    detections: list[DetectedFace],
) -> np.ndarray:
    """
    Compute the IoU matrix between all active tracks and new detections.

    Args:
        tracks: List of active Track objects.
        detections: List of new DetectedFace objects.

    Returns:
        (num_tracks, num_detections) IoU matrix.
    """
    n_tracks = len(tracks)
    n_dets = len(detections)
    iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou_matrix[i, j] = _compute_iou(track.last_bbox, det.bbox)

    return iou_matrix


class SimpleIoUTracker:
    """
    Frame-by-frame IoU-based multi-object tracker.

    For each new frame's detections:
    1. Compute IoU between all active tracks and new detections
    2. Use Hungarian algorithm for optimal bipartite matching
    3. Update matched tracks with new detections
    4. Create new tracks for unmatched detections
    5. Expire tracks that haven't been seen for too long
    """

    def __init__(
        self,
        iou_threshold: float = config.IOU_THRESHOLD,
        max_frames_lost: int = config.MAX_FRAMES_LOST,
        min_track_length: int = config.MIN_TRACK_LENGTH,
    ):
        self.iou_threshold = iou_threshold
        self.max_frames_lost = max_frames_lost
        self.min_track_length = min_track_length

        self._tracks: list[Track] = []
        self._next_id: int = 0
        self._finished_tracks: list[Track] = []

    def reset(self):
        """Reset the tracker state for a new video."""
        self._tracks = []
        self._next_id = 0
        self._finished_tracks = []

    def _create_track(self, detection: DetectedFace, frame_idx: int) -> Track:
        """Create a new track from an unmatched detection."""
        track = Track(track_id=self._next_id)
        self._next_id += 1
        track.add_detection(detection, frame_idx)
        return track

    def update(
        self,
        frame_idx: int,
        detections: list[DetectedFace],
    ) -> list[Track]:
        """
        Update tracks with new detections from a single frame.

        Args:
            frame_idx: Index of the current frame.
            detections: List of DetectedFace objects from the detector.

        Returns:
            List of currently active tracks.
        """
        if not self._tracks:
            # No existing tracks — create one per detection
            for det in detections:
                self._tracks.append(self._create_track(det, frame_idx))
            return self._tracks

        if not detections:
            # No detections — increment all tracks' lost counter
            for track in self._tracks:
                track.frames_since_seen += 1
            self._expire_tracks()
            return self._tracks

        # Compute IoU matrix
        iou_matrix = _compute_iou_matrix(self._tracks, detections)

        # Hungarian algorithm: minimize cost = maximize IoU
        # Convert to cost matrix (1 - IoU)
        cost_matrix = 1.0 - iou_matrix
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for t_idx, d_idx in zip(track_indices, det_indices):
            if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                # Valid match — update track
                self._tracks[t_idx].add_detection(detections[d_idx], frame_idx)
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

        # Increment lost counter for unmatched tracks
        for t_idx in range(len(self._tracks)):
            if t_idx not in matched_tracks:
                self._tracks[t_idx].frames_since_seen += 1

        # Create new tracks for unmatched detections
        for d_idx in range(len(detections)):
            if d_idx not in matched_dets:
                self._tracks.append(self._create_track(detections[d_idx], frame_idx))

        self._expire_tracks()

        return self._tracks

    def _expire_tracks(self):
        """Remove tracks that have been lost for too long."""
        still_active = []
        for track in self._tracks:
            if track.frames_since_seen > self.max_frames_lost:
                # Track expired — save to finished tracks
                if track.length >= self.min_track_length:
                    self._finished_tracks.append(track)
                logger.debug(
                    "Track %d expired (length=%d, lost=%d frames)",
                    track.track_id, track.length, track.frames_since_seen,
                )
            else:
                still_active.append(track)
        self._tracks = still_active

    def get_all_tracks(self) -> list[Track]:
        """
        Get all valid tracks (active + finished) that meet the minimum length.

        Call this after processing all frames to get the final set of
        tracked face identities.

        Returns:
            List of Track objects representing unique face identities.
        """
        all_tracks = self._finished_tracks + self._tracks
        valid = [t for t in all_tracks if t.length >= self.min_track_length]
        logger.info(
            "Tracker finished: %d total tracks, %d valid (min_length=%d)",
            len(all_tracks), len(valid), self.min_track_length,
        )
        return valid
