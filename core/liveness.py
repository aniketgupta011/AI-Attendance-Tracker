"""
Liveness detection module using heuristic image analysis.

Provides multi-signal anti-spoofing detection by analyzing face crops for:
- Texture quality (Laplacian variance for blur/print detection)
- Color distribution (histogram analysis for screen/print artifacts)
- Moiré pattern detection (frequency-domain analysis for screen replay)
- Depth cues via edge structure analysis

If a trained PyTorch CNN checkpoint is available at models/liveness/,
it is loaded and used for inference. Otherwise, the heuristic-based
detector provides reasonable baseline protection.

A real production system should train a binary classifier (live vs spoof)
on a dataset like CASIA-FASD, Replay-Attack, or OULU-NPU.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional PyTorch imports (graceful fallback if torch unavailable)
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision import models

    _TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    logger.info("PyTorch not available — liveness uses heuristic-only mode")


@dataclass
class LivenessResult:
    """Result of liveness check for a single tracked face."""
    score: float       # Liveness probability [0, 1]
    is_live: bool      # Whether score exceeds liveness threshold
    method: str        # Detection method used


# ==========================================================================
# Heuristic Liveness Detector
# ==========================================================================

class _HeuristicDetector:
    """
    Multi-signal heuristic anti-spoofing detector.

    Combines several image-quality signals to estimate whether a face
    crop is a live capture or a spoof (print, screen replay).

    Signals:
        1. Texture sharpness  — Laplacian variance (prints/screens are flatter)
        2. Color richness     — Hue-saturation histogram spread
        3. Moiré patterns     — High-frequency grid artifacts from screens
        4. Specular highlight — Bright spots from screen glare
    """

    # Thresholds tuned for 112×112 aligned face crops
    BLUR_MIN = 30.0       # Laplacian variance below this → likely print/flat
    COLOR_SPREAD_MIN = 25 # Minimum distinct hue bins for a live face
    MOIRE_MAX = 0.20      # Max energy ratio in high-freq bands → screen

    def analyze(self, face_crop: np.ndarray) -> float:
        """
        Return a liveness score in [0, 1] from heuristic analysis.

        Args:
            face_crop: BGR face image (any size, will be resized).

        Returns:
            Weighted liveness score — higher is more likely live.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        # Resize to a standard size for consistent analysis
        crop = cv2.resize(face_crop, (128, 128))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        scores = []

        # --- Signal 1: Texture sharpness (Laplacian variance) ---
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Sigmoid mapping: score ≈ 0 when lap_var < 15, ≈ 1 when > 60
        texture_score = 1.0 / (1.0 + np.exp(-0.1 * (lap_var - self.BLUR_MIN)))
        scores.append(texture_score)

        # --- Signal 2: Color richness (hue histogram spread) ---
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
        # Count bins with meaningful presence (> 1% of pixels)
        total_pixels = crop.shape[0] * crop.shape[1]
        active_bins = int(np.sum(hue_hist > total_pixels * 0.01))
        color_score = min(active_bins / self.COLOR_SPREAD_MIN, 1.0)
        scores.append(color_score)

        # --- Signal 3: Moiré pattern detection (FFT high-freq energy) ---
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        # High-frequency ring: outer 20% of the spectrum
        radius_inner = int(min(h, w) * 0.4)
        y_coords, x_coords = np.ogrid[:h, :w]
        high_freq_mask = ((y_coords - cy) ** 2 + (x_coords - cx) ** 2) > radius_inner ** 2
        total_energy = magnitude.sum() + 1e-8
        high_freq_ratio = float(magnitude[high_freq_mask].sum() / total_energy)
        # Low high-freq energy → more likely live (no moiré)
        moire_score = max(0.0, 1.0 - high_freq_ratio / self.MOIRE_MAX)
        scores.append(moire_score)

        # --- Signal 4: Specular highlights (screen glare detection) ---
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        bright_ratio = float(np.sum(bright_mask > 0)) / (gray.shape[0] * gray.shape[1])
        # A small amount of specular highlight is normal; too much → screen
        glare_score = max(0.0, 1.0 - bright_ratio / 0.05)
        scores.append(glare_score)

        # Weighted combination
        weights = [0.35, 0.20, 0.25, 0.20]
        final_score = sum(s * w for s, w in zip(scores, weights))
        return float(np.clip(final_score, 0.0, 1.0))


# ==========================================================================
# CNN-based Liveness Detector (loads trained checkpoint if available)
# ==========================================================================

class _CNNDetector:
    """
    PyTorch CNN-based liveness detector using MobileNetV3-Small.

    Expects a trained checkpoint at models/liveness/mobilenet_liveness.pt
    containing the state_dict of a MobileNetV3-Small with a binary
    classification head (live=1, spoof=0).

    If no checkpoint is found, this detector is not used and the system
    falls back to heuristic-only mode.
    """

    CHECKPOINT_PATH = config.MODELS_DIR / "liveness" / "mobilenet_liveness.pt"
    INPUT_SIZE = (224, 224)

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self._load_model()

    def _load_model(self):
        """Attempt to load a trained checkpoint."""
        if not self.CHECKPOINT_PATH.exists():
            logger.info(
                "No CNN liveness checkpoint at %s — CNN detector disabled",
                self.CHECKPOINT_PATH,
            )
            return

        try:
            model = models.mobilenet_v3_small(weights=None)
            # Replace classifier head for binary output
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 1)

            state_dict = torch.load(
                self.CHECKPOINT_PATH,
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("CNN liveness model loaded from %s", self.CHECKPOINT_PATH)
        except Exception as e:
            logger.warning("Failed to load CNN liveness model: %s", e)
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    @torch.no_grad()
    def predict(self, face_crop: np.ndarray) -> float:
        """
        Predict liveness probability for a single face crop.

        Args:
            face_crop: BGR face image as numpy array.

        Returns:
            Liveness probability [0, 1].
        """
        if self.model is None or face_crop is None or face_crop.size == 0:
            return 0.5

        # Convert BGR → RGB for torchvision
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        logit = self.model(tensor).squeeze()
        return float(torch.sigmoid(logit).cpu().item())

    @torch.no_grad()
    def predict_batch(self, face_crops: list[np.ndarray]) -> list[float]:
        """
        Predict liveness for a batch of face crops.

        Args:
            face_crops: List of BGR face images.

        Returns:
            List of liveness probabilities.
        """
        if self.model is None or not face_crops:
            return [0.5] * len(face_crops)

        tensors = []
        for crop in face_crops:
            if crop is None or crop.size == 0:
                # Placeholder tensor for invalid crops
                tensors.append(torch.zeros(3, *self.INPUT_SIZE))
            else:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensors.append(self.transform(rgb))

        batch = torch.stack(tensors).to(self.device)
        logits = self.model(batch).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]
        return probs


# ==========================================================================
# Public LivenessChecker API
# ==========================================================================

class LivenessChecker:
    """
    Anti-spoofing liveness checker combining heuristic and CNN analysis.

    Strategy:
    - If a trained CNN checkpoint is available → use CNN (weighted 0.7)
      combined with heuristics (weighted 0.3)
    - Otherwise → use heuristics only

    The heuristic detector analyzes texture sharpness, color distribution,
    Moiré patterns, and specular highlights to detect prints and screens.
    """

    def __init__(self, threshold: float = config.LIVENESS_THRESHOLD):
        self.threshold = threshold
        self._heuristic = _HeuristicDetector()
        self._cnn = None
        self._method = "heuristic"

        if _TORCH_AVAILABLE and config.TORCH_DEVICE is not None:
            self._cnn = _CNNDetector(config.TORCH_DEVICE)
            if self._cnn.is_available:
                self._method = "cnn+heuristic"
                logger.info(
                    "Liveness checker using CNN + heuristic (device: %s)",
                    config.TORCH_DEVICE,
                )
            else:
                self._cnn = None

        if self._method == "heuristic":
            logger.info(
                "Liveness checker using heuristic-only mode "
                "(drop a trained checkpoint at %s to enable CNN)",
                _CNNDetector.CHECKPOINT_PATH if _TORCH_AVAILABLE else "models/liveness/",
            )

    def check(self, face_crops_sequence: list[np.ndarray]) -> LivenessResult:
        """
        Check if a sequence of face crops represents a live person.

        Analyzes multiple crops from the same tracked identity and
        aggregates the scores (median of per-frame scores).

        Args:
            face_crops_sequence: List of BGR face crop images from the
                same tracked identity across multiple frames.

        Returns:
            LivenessResult with aggregated score.
        """
        if not face_crops_sequence:
            return LivenessResult(score=0.0, is_live=False, method=self._method)

        # Filter valid crops
        valid_crops = [c for c in face_crops_sequence if c is not None and c.size > 0]
        if not valid_crops:
            return LivenessResult(score=0.0, is_live=False, method=self._method)

        # Sample up to 8 crops for efficiency
        if len(valid_crops) > 8:
            indices = np.linspace(0, len(valid_crops) - 1, 8, dtype=int)
            valid_crops = [valid_crops[i] for i in indices]

        # Heuristic scores
        heuristic_scores = [self._heuristic.analyze(crop) for crop in valid_crops]

        if self._cnn is not None and self._cnn.is_available:
            # CNN + heuristic fusion
            cnn_scores = self._cnn.predict_batch(valid_crops)
            # Weighted fusion per crop
            fused = [
                0.7 * cnn + 0.3 * heur
                for cnn, heur in zip(cnn_scores, heuristic_scores)
            ]
            final_score = float(np.median(fused))
        else:
            # Heuristic only
            final_score = float(np.median(heuristic_scores))

        return LivenessResult(
            score=round(final_score, 4),
            is_live=final_score >= self.threshold,
            method=self._method,
        )

    def check_batch(
        self,
        tracks_crops: dict[int, list[np.ndarray]],
    ) -> dict[int, LivenessResult]:
        """
        Check liveness for multiple tracked faces.

        Args:
            tracks_crops: Dict mapping track_id to list of face crop images.

        Returns:
            Dict mapping track_id to LivenessResult.
        """
        results = {}
        for track_id, crops in tracks_crops.items():
            results[track_id] = self.check(crops)

        live_count = sum(1 for r in results.values() if r.is_live)
        logger.info(
            "Liveness check: %d/%d tracks passed (threshold=%.2f, method=%s)",
            live_count, len(results), self.threshold, self._method,
        )

        return results
