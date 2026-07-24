"""Minimal linearized replay for screening ancillary-feedback parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model import LPVModel, MODEL_DT, discrete_model
from .observer import PreparedPairs


@dataclass(frozen=True)
class AncillaryConfig:
    velocity_error_gain: float
    command_error_gain: float
    velocity_error_reanchor_threshold: float
    velocity_command_margin: float
    velocity_command_rate_margin: float

    def serializable(self) -> dict[str, float]:
        return asdict(self)

    def authority_valid(self) -> bool:
        values = np.asarray(list(asdict(self).values()))
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            return False
        if self.velocity_error_gain == 0.0 and self.command_error_gain == 0.0:
            return self.velocity_command_margin == 0.0 and self.velocity_command_rate_margin == 0.0
        if min(
            self.velocity_error_gain,
            self.command_error_gain,
            self.velocity_error_reanchor_threshold,
            self.velocity_command_margin,
            self.velocity_command_rate_margin,
        ) <= 0.0:
            return False
        required = self.velocity_error_gain * self.velocity_error_reanchor_threshold
        return (
            self.velocity_command_rate_margin >= required
            and self.command_error_gain * self.velocity_command_margin > required
        )


def feedback_correction(config: AncillaryConfig, velocity_error: float, command_error: float) -> float:
    correction = float(np.clip(
        -config.velocity_error_gain * velocity_error - config.command_error_gain * command_error,
        -config.velocity_command_rate_margin,
        config.velocity_command_rate_margin,
    ))
    lower = (-config.velocity_command_margin - command_error) / MODEL_DT
    upper = (config.velocity_command_margin - command_error) / MODEL_DT
    return float(np.clip(correction, lower, upper))


def simulate_field_errors(
    prepared: PreparedPairs,
    disturbance: np.ndarray,
    valid: np.ndarray,
    config: AncillaryConfig,
) -> np.ndarray:
    if not config.authority_valid():
        raise ValueError(f"ancillary configuration lacks tube authority: {config}")
    size = prepared.dataset.size
    velocity_error = np.full(size, np.nan)
    state_error = np.zeros(2)
    command_error = 0.0
    segment = prepared.dataset.column("segment_id").astype(np.int64)
    previous_segment = -1

    for index in range(size):
        if segment[index] != previous_segment or not valid[index] or not np.isfinite(disturbance[index]):
            state_error[:] = 0.0
            command_error = 0.0
            previous_segment = int(segment[index])
            if not valid[index] or not np.isfinite(disturbance[index]):
                continue
        if abs(state_error[1]) > config.velocity_error_reanchor_threshold:
            state_error[:] = 0.0
            command_error = 0.0

        correction = feedback_correction(config, float(state_error[1]), command_error)
        next_state = np.asarray([
            prepared.ad00[index] * state_error[0]
            + prepared.ad01[index] * state_error[1]
            + prepared.bd0[index] * command_error,
            prepared.ad10[index] * state_error[0]
            + prepared.ad11[index] * state_error[1]
            + prepared.bd1[index] * command_error
            + disturbance[index],
        ])
        command_error += MODEL_DT * correction
        state_error = next_state
        velocity_error[index] = state_error[1]
    return velocity_error


def velocity_rmse(errors: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool) & np.isfinite(errors)
    return float(np.sqrt(np.mean(errors[selected] ** 2))) if np.any(selected) else float("nan")


def persistent_bias(model: LPVModel, config: AncillaryConfig) -> float:
    transition = discrete_model(model, -1.0)
    disturbance = np.zeros(240)
    disturbance[20:180] = 0.010
    velocity = np.zeros(240)
    state_error = np.zeros(2)
    command_error = 0.0
    for index, noise in enumerate(disturbance):
        if abs(state_error[1]) > config.velocity_error_reanchor_threshold:
            state_error[:] = 0.0
            command_error = 0.0
        correction = feedback_correction(config, float(state_error[1]), command_error)
        state_error = np.asarray([
            transition.ad00 * state_error[0] + transition.ad01 * state_error[1] + transition.bd0 * command_error,
            transition.ad10 * state_error[0] + transition.ad11 * state_error[1]
            + transition.bd1 * command_error + noise,
        ])
        command_error += MODEL_DT * correction
        velocity[index] = state_error[1]
    return float(np.mean(np.abs(velocity[150:180])))
