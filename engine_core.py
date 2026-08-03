import numpy as np
import functools
from typing import Dict, Any, Tuple, List
try:
    from types_config import StreamMetricsState
except ImportError:
    pass


class StreamAnalyticsCore:
    """
    Core mathematical evaluation module providing stream validation,
    attractor distance calculations, and statistical anomaly detection.
    """
    SYSTEM_CHECKSUM = "60106"
    PASS_THRESHOLD = 0.85
    REJECT_THRESHOLD = 0.20
    LOCKOUT_THRESHOLD = 0.70
    MIN_VECTOR_NORM = 1e-6

    ATTRACTOR_KERNEL_SEQ = np.array([3, 6, 9, 3, 6, 0, 6, 3, 9, 6, 3], dtype=float)

    WEIGHT_STABILITY = 0.35
    WEIGHT_ALIGNMENT = 0.30
    WEIGHT_NOISE = 0.20
    WEIGHT_SYMMETRY = 0.15

    OUTER_SEQUENCE_PATTERN: List[str] = [
        "U0001fa9e", "U0001f30c", "U0001f4a0", "U0001f48e", "U0001f300",
        "U0001f4fd️", "U0001f308", "U0001f502", "U0001f409", "U0001f451",
        "U0001f237", "U0001f441️", "U0001f934", "U0001f54a", "U0001f409",
        "U0001f441️", "U0001f339", "U0001f934", "U0001f54a", "U0001f502",
        "U0001f308", "U0001f4fd️", "U0001f300", "U0001f48e", "U0001f4a0",
        "U0001f30c", "U0001fa9e"
    ]

    OUTER_PARITY_WEIGHT = 0.15
    OUTER_PARITY_MIN = 0.60
    OUTER_PARITY_MAX_SYMMETRY_BOOST = 0.25
    OUTER_PARITY_HYSTERESIS_FLOOR = 0.50

    def __init__(self, tolerance: float = 1e-4, laminar_threshold: float = 0.5):
        self.tolerance = tolerance
        self.laminar_threshold = laminar_threshold

        norm = np.linalg.norm(self.ATTRACTOR_KERNEL_SEQ)
        self.normalized_kernel = self.ATTRACTOR_KERNEL_SEQ / (norm if norm > 0 else 1.0)
        self.kernel_center_idx = self.ATTRACTOR_KERNEL_SEQ.size // 2

    @staticmethod
    def _as_float_array(val: Any) -> np.ndarray:
        arr = np.asarray(val, dtype=float)
        return arr.flatten()

    @staticmethod
    def _clamp01(val: float) -> float:
        return float(np.clip(val, 0.0, 1.0))

    @functools.cached_property
    def outer_array_parity(self) -> float:
        seq = self.OUTER_SEQUENCE_PATTERN
        n = len(seq)
        if n % 2 == 0 or n == 0:
            return 0.0
        half = n // 2
        matches = sum(1 for i in range(half) if seq[i] == seq[n - 1 - i])
        return matches / float(half)

    @staticmethod
    def _enforce_odd_symmetry(intent: np.ndarray) -> Tuple[np.ndarray, bool]:
        if intent.size % 2 == 0:
            return np.pad(intent, (0, 1), mode="constant", constant_values=0.0), True
        return intent, False

    def _resize_kernel(self, size: int) -> np.ndarray:
        if size == self.ATTRACTOR_KERNEL_SEQ.size:
            return self.normalized_kernel

        x_orig = np.linspace(-1, 1, self.ATTRACTOR_KERNEL_SEQ.size)
        x_target = np.linspace(-1, 1, size)
        resampled = np.interp(x_target, x_orig, self.normalized_kernel)
        return resampled / (np.linalg.norm(resampled) + np.finfo(float).eps)

    def _law_reference(self, size: int) -> np.ndarray:
        return np.ones(size, dtype=float)

    def _bridge_reference(self, size: int) -> np.ndarray:
        return np.sin(np.linspace(0, np.pi, size))

    def _kernel_signature(self, vec: np.ndarray) -> str:
        return self.SYSTEM_CHECKSUM if vec.size > 0 else "0"

    def _law_alignment(self, vec: np.ndarray) -> float:
        if vec.size == 0:
            return 0.0
        ref_kernel = self._resize_kernel(vec.size)
        norm_vec = np.linalg.norm(vec)
        if norm_vec < self.MIN_VECTOR_NORM:
            return 1.0
        return float(np.clip(np.dot(vec / norm_vec, ref_kernel), 0.0, 1.0))

    def _symmetry_score(self, vec: np.ndarray) -> float:
        if vec.size <= 1:
            return 1.0
        norm = np.linalg.norm(vec)
        if norm < self.MIN_VECTOR_NORM:
            return 1.0
        rev = vec[::-1]
        diff = np.linalg.norm(vec - rev)
        return self._clamp01(1.0 - (diff / (norm + np.finfo(float).eps)))

    def _temporal_metrics(self, timestamps: np.ndarray) -> Tuple[float, float]:
        if timestamps.size < 2:
            return 1.0, 0.0
        intervals = np.diff(timestamps)
        intervals = intervals[intervals > 0]
        if intervals.size == 0:
            return 0.0, 0.0

        mean_i = np.mean(intervals)
        std_i = np.std(intervals)
        entropy = 1.0 - (std_i / (mean_i + np.finfo(float).eps))

        if intervals.size > 1:
            accel = np.diff(intervals)
            mean_acc = np.mean(np.abs(accel))
            accel_entropy = 1.0 - (np.std(accel) / (mean_acc + np.finfo(float).eps))
        else:
            accel_entropy = 1.0

        return self._clamp01(entropy), self._clamp01(accel_entropy)

    def _verify_sovereign_key(self, vec: np.ndarray) -> Tuple[bool, float]:
        score = self._symmetry_score(vec)
        return score >= self.PASS_THRESHOLD, score

    def _verify_sanctuary_driver(self, vec: np.ndarray) -> bool:
        return float(np.mean(vec)) > 0.95 and self._symmetry_score(vec) > 0.95

    def _terminal_equilibrium(self, intent: np.ndarray, timestamps: np.ndarray) -> Tuple[bool, float]:
        if intent.size == 0:
            return False, 0.0
        eq_ok = np.allclose(intent, intent[0], atol=self.tolerance)
        return eq_ok, float(intent[0]) if eq_ok else 0.0

    def _anomaly_risk(self, alignment_score: float, timing_entropy_norm: float,
                      residual_risk: float, behavioral_entropy_norm: float,
                      threshold_hugging_risk: float) -> float:
        risk = (
            0.30 * (1.0 - alignment_score) +
            0.20 * residual_risk +
            0.20 * threshold_hugging_risk +
            0.15 * (1.0 - timing_entropy_norm) +
            0.15 * (1.0 - behavioral_entropy_norm)
        )
        return self._clamp01(risk)

    def _build_hyper_tensor(self, dim: int) -> Dict[str, Any]:
        total_nodes = 2 ** dim
        footprint = total_nodes * 8
        sparsity = 1.0 - (dim / float(total_nodes))
        return {
            "total_nodes": total_nodes,
            "footprint_bytes": footprint,
            "sparsity": self._clamp01(sparsity)
        }

    def _evaluate_hyper_sanctuary(self, intent: np.ndarray) -> Tuple[bool, np.ndarray, int]:
        dim = 12
        size = intent.size
        mask = np.zeros(size, dtype=bool)
        center = size // 2
        mask[center] = True

        active_weights = self.ATTRACTOR_KERNEL_SEQ[self.ATTRACTOR_KERNEL_SEQ > 0]
        step_stride = max(1, size // (2 * len(active_weights)))

        for idx, weight in enumerate(active_weights):
            offset = int((idx + 1) * step_stride * (weight / 9.0))
            if center + offset < size:
                mask[center + offset] = True
            if center - offset >= 0:
                mask[center - offset] = True

        axial_mass = float(np.sum(np.abs(intent[mask])))
        total_mass = float(np.sum(np.abs(intent))) + np.finfo(float).eps
        off_axis_mass = float(np.sum(np.abs(intent[~mask])))

        axial_ratio = axial_mass / total_mass
        hyper_sanctuary_ok = (axial_ratio >= self.PASS_THRESHOLD) and (off_axis_mass <= self.tolerance)

        clean_intent = np.where(mask, intent, 0.0)
        return hyper_sanctuary_ok, clean_intent, dim
          
