"""
Face embedding extraction module.

Provides a thin abstraction over the embedding data already computed
by InsightFace during detection. Its main role is selecting the best
embedding from a tracked face's history based on quality criteria.
"""

import logging

import numpy as np

from core.face_tracker import Track

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """
    Selects the best embedding from a tracked face's detection history.

    Since InsightFace's FaceAnalysis computes both detection and embedding
    in a single forward pass, we don't need a separate embedding model.
    This class focuses on selecting the highest-quality embedding from
    the sequence of per-frame embeddings collected during tracking.
    """

    @staticmethod
    def get_best_embedding(track: Track) -> np.ndarray | None:
        """
        Select the best embedding for a tracked face.

        Uses the track's quality scoring to find the highest-quality
        detection, then returns its embedding.

        Args:
            track: A Track object with at least one detection.

        Returns:
            512-D L2-normalized embedding vector, or None if track is empty.
        """
        best_det = track.get_best_detection()
        if best_det is None:
            return None

        embedding = best_det.embedding.copy()

        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def get_average_embedding(track: Track) -> np.ndarray | None:
        """
        Compute the average embedding across all detections in a track.

        Averaging across multiple frames can improve robustness by
        smoothing out per-frame noise, but may blur the embedding
        if some frames have poor quality.

        Args:
            track: A Track object with at least one detection.

        Returns:
            512-D L2-normalized average embedding vector, or None.
        """
        if not track.detections:
            return None

        embeddings = np.array([
            det.embedding for det in track.detections
            if det.embedding is not None
        ])

        if len(embeddings) == 0:
            return None

        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

        # Re-normalize to unit vector
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding
