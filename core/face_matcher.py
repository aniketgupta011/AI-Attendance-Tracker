"""
Face matching module using cosine similarity.

Matches query face embeddings against a database of enrolled student
embeddings. Uses GPU-accelerated PyTorch tensors when CUDA is available
and the database is large enough to benefit, falling back to vectorized
NumPy operations otherwise.
"""

import logging
from dataclasses import dataclass

import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch import for GPU-accelerated matching
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    pass

# Minimum enrolled embeddings to use GPU (below this, NumPy is faster)
_GPU_MIN_EMBEDDINGS = 50


@dataclass
class MatchResult:
    """Result of matching a single face embedding against the database."""
    student_id: str
    similarity: float  # Cosine similarity score (0 to 1)
    is_match: bool     # Whether similarity exceeds threshold


class FaceMatcher:
    """
    Cosine similarity matcher against enrolled student embeddings.

    Preloads all enrolled embeddings into a matrix for vectorized
    matching. When PyTorch CUDA is available and the database exceeds
    a minimum size, matching runs on the GPU for faster throughput.
    """

    def __init__(self, threshold: float = config.SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self._student_ids: list[str] = []
        self._embedding_matrix_np: np.ndarray | None = None   # Always kept (CPU fallback)
        self._embedding_matrix_gpu: object | None = None       # torch.Tensor on CUDA
        self._use_gpu = False

    def load_embeddings(self, embeddings: dict[str, np.ndarray]):
        """
        Load enrolled student embeddings into the matcher.

        If PyTorch CUDA is available and there are enough embeddings,
        the matrix is also uploaded to the GPU.

        Args:
            embeddings: Dict mapping student_id to 512-D embedding vector.
        """
        if not embeddings:
            self._student_ids = []
            self._embedding_matrix_np = None
            self._embedding_matrix_gpu = None
            self._use_gpu = False
            logger.warning("No enrolled embeddings to load")
            return

        self._student_ids = list(embeddings.keys())
        self._embedding_matrix_np = np.stack(
            [embeddings[sid] for sid in self._student_ids]
        ).astype(np.float32)

        # Ensure all embeddings are L2-normalized (row-wise)
        norms = np.linalg.norm(self._embedding_matrix_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        self._embedding_matrix_np = self._embedding_matrix_np / norms

        # Attempt GPU upload for large databases
        self._use_gpu = False
        self._embedding_matrix_gpu = None

        if (
            _TORCH_AVAILABLE
            and config.TORCH_DEVICE is not None
            and config.TORCH_DEVICE.type == "cuda"
            and len(self._student_ids) >= _GPU_MIN_EMBEDDINGS
        ):
            try:
                self._embedding_matrix_gpu = torch.from_numpy(
                    self._embedding_matrix_np
                ).to(config.TORCH_DEVICE)
                self._use_gpu = True
                logger.info(
                    "Face matcher using GPU (%d embeddings on %s)",
                    len(self._student_ids),
                    config.TORCH_DEVICE,
                )
            except Exception as e:
                logger.warning("Failed to upload embeddings to GPU: %s", e)
                self._use_gpu = False

        if not self._use_gpu:
            logger.info(
                "Face matcher using CPU NumPy (%d embeddings)",
                len(self._student_ids),
            )

        logger.info(
            "Loaded %d enrolled embeddings (matrix shape: %s)",
            len(self._student_ids),
            self._embedding_matrix_np.shape,
        )

    def _match_gpu(self, query: np.ndarray) -> tuple[int, float]:
        """
        GPU-accelerated cosine similarity via PyTorch.

        Args:
            query: L2-normalized 512-D query vector (numpy).

        Returns:
            Tuple of (best_index, best_similarity).
        """
        query_tensor = torch.from_numpy(query).to(config.TORCH_DEVICE)
        similarities = self._embedding_matrix_gpu @ query_tensor
        best_idx = int(torch.argmax(similarities).cpu().item())
        best_sim = float(similarities[best_idx].cpu().item())
        return best_idx, best_sim

    def _match_cpu(self, query: np.ndarray) -> tuple[int, float]:
        """
        CPU NumPy cosine similarity (dot product of normalized vectors).

        Args:
            query: L2-normalized 512-D query vector (numpy).

        Returns:
            Tuple of (best_index, best_similarity).
        """
        similarities = self._embedding_matrix_np @ query
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        return best_idx, best_sim

    def match(self, query_embedding: np.ndarray) -> MatchResult | None:
        """
        Match a single query embedding against all enrolled embeddings.

        Uses GPU acceleration when available, falling back to NumPy.

        Args:
            query_embedding: 512-D L2-normalized face embedding.

        Returns:
            MatchResult for the best match if above threshold, else None.
        """
        if self._embedding_matrix_np is None or len(self._student_ids) == 0:
            return None

        # Normalize query
        query = query_embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Route to GPU or CPU matching
        if self._use_gpu and self._embedding_matrix_gpu is not None:
            best_idx, best_sim = self._match_gpu(query)
        else:
            best_idx, best_sim = self._match_cpu(query)

        is_match = best_sim >= self.threshold

        result = MatchResult(
            student_id=self._student_ids[best_idx],
            similarity=best_sim,
            is_match=is_match,
        )

        if is_match:
            logger.debug(
                "Match found: student=%s, similarity=%.4f",
                result.student_id, result.similarity,
            )
        else:
            logger.debug(
                "No match above threshold (best=%.4f, threshold=%.4f)",
                best_sim, self.threshold,
            )

        return result

    def match_batch(
        self,
        embeddings: dict[int, np.ndarray],
    ) -> dict[int, MatchResult | None]:
        """
        Match multiple embeddings (keyed by track_id) against the database.

        Deduplicates matches: if two tracks match the same student,
        keeps the one with the higher similarity score.

        Args:
            embeddings: Dict mapping track_id to 512-D embedding vector.

        Returns:
            Dict mapping track_id to MatchResult (or None for no match).
        """
        results: dict[int, MatchResult | None] = {}
        # Track best match per student to avoid duplicates
        student_best: dict[str, tuple[int, float]] = {}

        for track_id, emb in embeddings.items():
            result = self.match(emb)
            results[track_id] = result

            if result and result.is_match:
                sid = result.student_id
                if sid not in student_best or result.similarity > student_best[sid][1]:
                    student_best[sid] = (track_id, result.similarity)

        # Deduplicate: only keep the best track per student
        valid_tracks = set(tid for tid, _ in student_best.values())
        for track_id in results:
            if results[track_id] and results[track_id].is_match:
                if track_id not in valid_tracks:
                    # This track matched a student but another track was better
                    logger.debug(
                        "Deduplicating: track %d matched %s but a better match exists",
                        track_id, results[track_id].student_id,
                    )
                    results[track_id] = MatchResult(
                        student_id=results[track_id].student_id,
                        similarity=results[track_id].similarity,
                        is_match=False,  # Suppress duplicate
                    )

        matched_count = sum(1 for r in results.values() if r and r.is_match)
        logger.info(
            "Batch matching: %d tracks → %d matches (threshold=%.4f, gpu=%s)",
            len(embeddings), matched_count, self.threshold, self._use_gpu,
        )

        return results
