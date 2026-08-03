import numpy as np
from typing import Dict, Any
try:
    from types_config import StreamMetricsState, TelemetryAcousticVector
except ImportError:
    pass


class DynamicTelemetryMapper:
    """
    State-Space Telemetry Mapper:
    Projects multi-variable system metrics into dynamic DSP audio parameters
    using an 8x8 linear transformation matrix.
    """

    # 8x8 Transformation Matrix: Maps S(t) -> A(t)
    # Input Vector S(t): [stability, law_alignment, symmetry, temporal_entropy,
    #                     sovereign_alignment, anomaly_risk, outer_parity, boundary_strength]
    PROJECTION_MATRIX_8X8 = np.array([
        [ 432.0,   12.0,    0.0,   -20.0,    10.0,  -100.0,    5.0,    0.0],  # f_0
        [   0.1,    0.4,    0.3,    -0.2,     0.2,    -0.5,    0.1,    0.1],  # harmonic_density
        [   0.3,    0.2,    0.3,     0.1,     0.1,    -0.8,    0.0,    0.0],  # phase_coherence
        [   1.0,    2.0,    3.0,    -1.0,     2.0,   -10.0,    1.0,    1.0],  # filter_q
        [  -0.05,  -0.05,  -0.05,    0.2,    -0.05,    0.8,   -0.02,  -0.02], # noise_floor
        [   1.0,    1.0,    2.0,    -1.0,     1.0,    -5.0,    0.5,    0.5],  # reverb_decay
        [   0.0,    0.1,    0.7,     0.0,     0.1,    -0.4,    0.2,    0.0],  # palindromic_weight
        [ 2000.0, 3000.0, 4000.0, -1000.0,  2000.0, -8000.0, 1000.0, 1000.0]   # cutoff_frequency
    ], dtype=float)

    # Base bias offset vector for parameter scaling
    BIAS_VECTOR_8 = np.array([
        432.0,  # Base f_0
        0.1,    # Base harmonic density
        0.1,    # Base phase coherence
        1.0,    # Base filter Q
        0.001,  # Base noise floor
        0.5,    # Base reverb decay (seconds)
        0.0,    # Base palindromic weight
        400.0   # Base cutoff frequency (Hz)
    ], dtype=float)

    @staticmethod
    def _clamp(val: float, min_val: float, max_val: float) -> float:
        return float(np.clip(val, min_val, max_val))

    def map_state(self, state: Dict[str, Any], lockout_override: bool = False) -> Dict[str, float]:
        """
        Transforms internal StreamMetricsState into TelemetryAcousticVector parameters.
        """
        if lockout_override:
            return {
                "f_0": 108.0,
                "harmonic_density": 0.0,
                "phase_coherence": 0.0,
                "filter_q": 0.5,
                "noise_floor": 0.95,
                "reverb_decay": 0.1,
                "palindromic_weight": 0.0,
                "cutoff_frequency": 200.0
            }

        # Extract normalized state vector S(t) [8 elements]
        s_vector = np.array([
            state.get("stability_score", 0.0),
            state.get("law_alignment", 0.0),
            state.get("symmetry_score", 0.0),
            state.get("temporal_entropy", 0.0),
            state.get("sovereign_alignment", 0.0),
            state.get("anomaly_risk", 0.0),
            state.get("outer_array_parity", 0.0),
            state.get("boundary_strength", 0.0)
        ], dtype=float)

        # Linear matrix transformation: A(t) = M * S(t) + Bias
        raw_acoustic = np.dot(self.PROJECTION_MATRIX_8X8, s_vector) + self.BIAS_VECTOR_8

        # Clamped acoustic dynamic ranges
        return {
            "f_0": self._clamp(raw_acoustic[0], 216.0, 864.0),
            "harmonic_density": self._clamp(raw_acoustic[1], 0.0, 1.0),
            "phase_coherence": self._clamp(raw_acoustic[2], 0.0, 1.0),
            "filter_q": self._clamp(raw_acoustic[3], 0.5, 20.0),
            "noise_floor": self._clamp(raw_acoustic[4], 0.0001, 1.0),
            "reverb_decay": self._clamp(raw_acoustic[5], 0.1, 8.0),
            "palindromic_weight": self._clamp(raw_acoustic[6], 0.0, 1.0),
            "cutoff_frequency": self._clamp(raw_acoustic[7], 200.0, 16000.0)
      }
      
