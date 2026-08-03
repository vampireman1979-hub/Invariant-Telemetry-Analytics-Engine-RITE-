import numpy as np
from typing import Dict, Any, Tuple, TypedDict, List


class StreamMetricsState(TypedDict):
    """Data contract representing internal telemetry and integrity metrics of a data stream."""
    structure_ok: bool
    timing_ok: bool
    parity: float
    center_value: float
    temporal_entropy: float
    accel_entropy: float
    symmetry_score: float
    stability_score: float
    maturation_level: float
    absolute_law_ok: bool
    sanctuary_driver_ok: bool
    law_alignment: float
    sovereign_key_verified: bool
    sovereign_alignment: float
    gate_status: str
    anomaly_risk: float
    terminal_equilibrium_ok: bool
    terminal_scalar: float
    boundary_strength: float
    hyper_dimension: int
    tensor_nodes: int
    hyper_sparsity: float
    hyper_sanctuary_ok: bool
    outer_array_parity: float


class TelemetryAcousticVector(TypedDict):
    """Schema for dynamic Digital Signal Processing (DSP) telemetry outputs."""
    f_0: float
    harmonic_density: float
    phase_coherence: float
    filter_q: float
    noise_floor: float
    reverb_decay: float
    palindromic_weight: float
    cutoff_frequency: float


class StreamProcessingResult(TypedDict):
    """Standard return object for the data processing pipeline."""
    Status: str
    Phase_State: str
    Yield: np.ndarray
    Feedback_Vector: np.ndarray
    Coherence: float
    Maturation_Index: float
    Symmetry_Score: float
    Invariant_Check: str
    Hyper_Volume_Footprint_Bytes: int
    Hyper_Sparsity_Ratio: float
    Acoustic_Target: TelemetryAcousticVector
  
