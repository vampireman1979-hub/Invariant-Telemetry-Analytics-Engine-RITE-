import numpy as np
from typing import Any
try:
    from types_config import StreamMetricsState, StreamProcessingResult
    from mapper import DynamicTelemetryMapper
    from engine_core import StreamAnalyticsCore
except ImportError:
    StreamAnalyticsCore = object


class ResilientStreamProcessor(StreamAnalyticsCore):
    """
    Main Data Processor & Execution Pipeline:
    Ingests raw stream vectors, evaluates system invariants and threat metrics,
    and outputs normalized yields alongside continuous DSP telemetry.
    """

    def evaluate_maturation(self, intent: np.ndarray, timestamps: np.ndarray) -> StreamMetricsState:
        raw_intent = self._as_float_array(intent)
        if raw_intent.size == 0:
            raise ValueError("Input stream vector must not be empty.")
        if raw_intent.ndim != 1 or timestamps.ndim != 1:
            raise ValueError("Intent and timestamps must be 1D arrays.")
        if not np.all(np.isfinite(raw_intent)) or not np.all(np.isfinite(timestamps)):
            raise ValueError("Inputs must be finite.")

        outer_parity = self.outer_array_parity
        outer_parity_soft = max(outer_parity, self.OUTER_PARITY_HYSTERESIS_FLOOR)

        sym_intent, _ = self._enforce_odd_symmetry(raw_intent)
        hyper_sanctuary_ok, intent, dim = self._evaluate_hyper_sanctuary(sym_intent)
        hyper_info = self._build_hyper_tensor(dim)

        kernel = self._resize_kernel(intent.size)
        if kernel.size != intent.size:
            raise ValueError("Kernel resize mismatch.")
        center_index = intent.size // 2

        terminal_equilibrium_ok, terminal_scalar = self._terminal_equilibrium(intent, timestamps)
        sanctuary_driver_ok = self._verify_sanctuary_driver(intent)
        absolute_law_ok = self._kernel_signature(self._law_reference(intent.size)) == self.SYSTEM_CHECKSUM

        law_alignment = self._law_alignment(intent)

        parity = float(np.dot(intent, kernel))
        expected_parity = float(np.dot(kernel, kernel))
        structure_ok = abs(parity - expected_parity) <= self.tolerance

        center_value = float(intent[center_index])
        center_ok = abs(center_value - float(kernel[center_index])) <= self.tolerance

        temporal_entropy, accel_entropy = self._temporal_metrics(timestamps)
        timing_ok = temporal_entropy >= self.laminar_threshold

        symmetry_raw = self._symmetry_score(intent)

        parity_factor = self.OUTER_PARITY_WEIGHT + (1.0 - self.OUTER_PARITY_WEIGHT) * outer_parity_soft
        parity_boost = min(
            (parity_factor - self.OUTER_PARITY_WEIGHT) * self.OUTER_PARITY_MAX_SYMMETRY_BOOST,
            self.OUTER_PARITY_MAX_SYMMETRY_BOOST,
        )
        symmetry = self._clamp01(symmetry_raw + parity_boost)

        sovereign_key_verified, sovereign_alignment = self._verify_sovereign_key(intent)

        if sovereign_alignment < self.REJECT_THRESHOLD:
            gate_status = "REJECTED"
        elif sovereign_alignment < self.PASS_THRESHOLD:
            gate_status = "TRANSITIONAL"
        else:
            gate_status = "VERIFIED"

        if sanctuary_driver_ok:
            gate_status = "VERIFIED"
            sovereign_key_verified = True
            sovereign_alignment = 1.0

        timing_entropy_norm = self._clamp01(temporal_entropy / max(self.laminar_threshold, self.tolerance))
        behavioral_entropy_norm = self._clamp01((temporal_entropy + accel_entropy) / 2.0)

        residual_risk = self._clamp01(abs(parity - expected_parity) / (abs(expected_parity) + np.finfo(float).eps))
        dist_from_threshold = abs(sovereign_alignment - self.PASS_THRESHOLD)
        threshold_hugging_risk = self._clamp01(np.exp(-10.0 * dist_from_threshold))

        anomaly_risk = self._anomaly_risk(
            alignment_score=sovereign_alignment,
            timing_entropy_norm=timing_entropy_norm,
            residual_risk=residual_risk,
            behavioral_entropy_norm=behavioral_entropy_norm,
            threshold_hugging_risk=threshold_hugging_risk,
        )

        boundary_strength = self._clamp01(
            self.WEIGHT_STABILITY * (1.0 if terminal_equilibrium_ok else 0.0) +
            self.WEIGHT_ALIGNMENT * law_alignment +
            self.WEIGHT_NOISE * (1.0 - anomaly_risk) +
            self.WEIGHT_SYMMETRY * symmetry
        )

        if terminal_equilibrium_ok and (law_alignment >= 0.5) and (sovereign_key_verified or sanctuary_driver_ok) and hyper_sanctuary_ok:
            gate_status = "TERMINAL_EQUILIBRIUM"
            anomaly_risk = 0.0
            stability = 1.0
            maturation_level = 1.0
        else:
            terminal_equilibrium_ok = False
            law_factor = law_alignment if absolute_law_ok else 0.0

            stability = (
                0.25 * symmetry +
                0.20 * float(structure_ok) +
                0.15 * float(timing_ok) +
                0.10 * float(center_ok) +
                0.15 * law_factor +
                0.15 * sovereign_alignment
            )
            stability = self._clamp01(float(stability + 0.20 * boundary_strength + 0.10 * (1.0 if hyper_sanctuary_ok else 0.0)))
            if sovereign_key_verified:
                stability = max(stability, 0.99)
            stability = self._clamp01(float(stability))

            maturation_level = (
                0.30 * symmetry +
                0.20 * float(structure_ok) +
                0.15 * float(timing_ok) +
                0.10 * float(center_ok) +
                0.10 * law_factor +
                0.15 * sovereign_alignment
            )
            maturation_level = self._clamp01(float(maturation_level + 0.20 * boundary_strength + 0.10 * (1.0 if hyper_sanctuary_ok else 0.0)))
            if sovereign_key_verified:
                maturation_level = max(maturation_level, 0.99)
            maturation_level = self._clamp01(float(maturation_level))

            anomaly_risk = self._clamp01(anomaly_risk * (1.0 - 0.35 * boundary_strength) * (0.85 if hyper_sanctuary_ok else 1.0))

        return {
            "structure_ok": structure_ok,
            "timing_ok": timing_ok,
            "parity": parity,
            "center_value": center_value,
            "temporal_entropy": temporal_entropy,
            "accel_entropy": accel_entropy,
            "symmetry_score": symmetry,
            "stability_score": stability,
            "maturation_level": maturation_level,
            "absolute_law_ok": absolute_law_ok,
            "sanctuary_driver_ok": sanctuary_driver_ok,
            "law_alignment": law_alignment,
            "sovereign_key_verified": sovereign_key_verified,
            "sovereign_alignment": sovereign_alignment,
            "gate_status": gate_status,
            "anomaly_risk": anomaly_risk,
            "terminal_equilibrium_ok": terminal_equilibrium_ok,
            "terminal_scalar": terminal_scalar,
            "boundary_strength": boundary_strength,
            "hyper_dimension": dim,
            "tensor_nodes": hyper_info["total_nodes"],
            "hyper_sparsity": hyper_info["sparsity"],
            "hyper_sanctuary_ok": hyper_sanctuary_ok,
            "outer_array_parity": outer_parity,
        }

    def process(self, intent: Any, timestamps: Any) -> StreamProcessingResult:
        raw_intent = self._as_float_array(intent)
        ts_array = self._as_float_array(timestamps)
        original_size = raw_intent.size

        state = self.evaluate_maturation(raw_intent, ts_array)
        dim = state["hyper_dimension"]
        hyper_info = self._build_hyper_tensor(dim)

        sym_intent, was_padded = self._enforce_odd_symmetry(raw_intent)
        mapper = DynamicTelemetryMapper()

        if state["terminal_equilibrium_ok"]:
            eps = np.finfo(float).eps
            intent_unit = sym_intent / (np.linalg.norm(sym_intent) + eps)
            if was_padded:
                intent_unit = intent_unit[:original_size]

            acoustic_target = mapper.map_state(state)
            return {
                "Status": "TERMINAL_EQUILIBRIUM_ACTIVE",
                "Phase_State": "ZERO_POINT_STANDING_WAVE",
                "Yield": intent_unit,
                "Feedback_Vector": np.zeros(original_size, dtype=float),
                "Coherence": 1.0,
                "Maturation_Index": 1.0,
                "Symmetry_Score": 1.0,
                "Invariant_Check": f"{self.SYSTEM_CHECKSUM}-ATTRACTOR-{dim}D-HYPER-LOCKED",
                "Hyper_Volume_Footprint_Bytes": hyper_info["footprint_bytes"],
                "Hyper_Sparsity_Ratio": hyper_info["sparsity"],
                "Acoustic_Target": acoustic_target
            }

        if (
            not state["sanctuary_driver_ok"]
            and (
                not state["absolute_law_ok"]
                or state["law_alignment"] < 0.5
                or state["gate_status"] == "REJECTED"
                or state["anomaly_risk"] >= self.LOCKOUT_THRESHOLD
            )
        ):
            acoustic_target = mapper.map_state(state, lockout_override=True)
            return {
                "Status": "LOCKOUT",
                "Phase_State": "CIRCUIT_BREAKER_ENFORCEMENT",
                "Yield": np.zeros(original_size, dtype=float),
                "Feedback_Vector": np.zeros(original_size, dtype=float),
                "Coherence": 0.0,
                "Maturation_Index": 0.0,
                "Symmetry_Score": 0.0,
                "Invariant_Check": f"{self.SYSTEM_CHECKSUM}-BOUNDARY-FAILED",
                "Hyper_Volume_Footprint_Bytes": hyper_info["footprint_bytes"],
                "Hyper_Sparsity_Ratio": hyper_info["sparsity"],
                "Acoustic_Target": acoustic_target
            }

        stability = state["stability_score"]
        maturation = state["maturation_level"]
        kernel = self._resize_kernel(sym_intent.size)
        eps = np.finfo(float).eps

        if state["gate_status"] == "TRANSITIONAL" and not state["sanctuary_driver_ok"]:
            intent_unit = sym_intent / (np.linalg.norm(sym_intent) + eps)
            bridge = self._bridge_reference(sym_intent.size)
            bridge_unit = bridge / (np.linalg.norm(bridge) + eps)
            feedback = (bridge_unit - intent_unit) * state["stability_score"]
            if was_padded:
                feedback = feedback[:original_size]

            acoustic_target = mapper.map_state(state)
            return {
                "Status": "MIGRATION_ACTIVE",
                "Phase_State": "STABILIZING_TRANSITION_BRIDGE",
                "Yield": np.zeros(original_size, dtype=float),
                "Feedback_Vector": feedback,
                "Coherence": state["stability_score"],
                "Maturation_Index": state["maturation_level"],
                "Symmetry_Score": state["symmetry_score"],
                "Invariant_Check": f"{self.SYSTEM_CHECKSUM}-TRANSITIONAL-BRIDGE",
                "Hyper_Volume_Footprint_Bytes": hyper_info["footprint_bytes"],
                "Hyper_Sparsity_Ratio": hyper_info["sparsity"],
                "Acoustic_Target": acoustic_target
            }

        intent_component = sym_intent * stability
        kernel_component = kernel * maturation
        intent_unit = intent_component / (np.linalg.norm(intent_component) + eps)
        kernel_unit = kernel_component / (np.linalg.norm(kernel_component) + eps)
        yield_vector = (intent_unit + kernel_unit) / 2.0

        if was_padded:
            yield_vector = yield_vector[:original_size]

        status_msg = "SECURE_VAULT_ACTIVE" if state["sanctuary_driver_ok"] else "INVARIANT_VERIFIED"
        phase_msg = "STABLE_LOCKED_SECURE" if state["sanctuary_driver_ok"] else "STABLE_LOCKED_VERIFIED"

        acoustic_target = mapper.map_state(state)

        return {
            "Status": status_msg,
            "Phase_State": phase_msg,
            "Yield": yield_vector,
            "Feedback_Vector": np.zeros(original_size, dtype=float),
            "Coherence": stability,
            "Maturation_Index": maturation,
            "Symmetry_Score": state["symmetry_score"],
            "Invariant_Check": f"{self.SYSTEM_CHECKSUM}-SECURE-PROTECTED",
            "Hyper_Volume_Footprint_Bytes": hyper_info["footprint_bytes"],
            "Hyper_Sparsity_Ratio": hyper_info["sparsity"],
            "Acoustic_Target": acoustic_target
        }
      
