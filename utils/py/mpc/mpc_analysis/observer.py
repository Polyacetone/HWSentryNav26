"""Replay the production longitudinal observer on extracted transition pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import PairDataset, dilate_within_segments
from .model import LPVModel, discrete_model, nonlinear_term, schedule_rho, smooth_sign


MIN_INITIALIZATION_COUPLING = 1e-3


@dataclass(frozen=True)
class ObserverConfig:
    velocity_gain: float
    correction_clip: float
    reset_threshold: float

    def validate(self) -> None:
        values = np.asarray([self.velocity_gain, self.correction_clip, self.reset_threshold])
        if not np.all(np.isfinite(values)):
            raise ValueError("observer parameters must be finite")
        if self.correction_clip <= 0.0 or self.reset_threshold <= self.correction_clip:
            raise ValueError("correction clip must be positive and below reset threshold")


@dataclass(frozen=True)
class PreparedPairs:
    dataset: PairDataset
    model: LPVModel
    eligible: np.ndarray
    init_ad10: np.ndarray
    init_ad11: np.ndarray
    init_bd1: np.ndarray
    init_gd1: np.ndarray
    response_init_ad10: np.ndarray
    response_init_ad11: np.ndarray
    response_init_bd1: np.ndarray
    response_init_gd1: np.ndarray
    ad00: np.ndarray
    ad01: np.ndarray
    ad10: np.ndarray
    ad11: np.ndarray
    bd0: np.ndarray
    bd1: np.ndarray
    gd0: np.ndarray
    gd1: np.ndarray
    alpha_w: np.ndarray
    beta_w: np.ndarray
    gamma_w: np.ndarray
    nonlinear_start: np.ndarray
    nonlinear_response: np.ndarray


@dataclass(frozen=True)
class ReplayResult:
    prediction_available: np.ndarray
    validated: np.ndarray
    hard_reset: np.ndarray
    velocity_innovation: np.ndarray
    angular_velocity_innovation: np.ndarray


def model_observer_config(model: LPVModel) -> ObserverConfig:
    return ObserverConfig(
        model.observer.velocity_gain,
        model.observer.correction_clip,
        model.observer.reset_threshold,
    )


def prepare_pairs(dataset: PairDataset, model: LPVModel) -> PreparedPairs:
    size = dataset.size
    eligible = dataset.lpv_eligible()
    rho_start = schedule_rho(model, dataset.column("leg_h_start"), dataset.column("leg_psi_start"))
    rho_response = schedule_rho(model, dataset.column("leg_h_response"), dataset.column("leg_psi_response"))
    rho_transition = 0.5 * (rho_start + rho_response)

    names = (
        "init_ad10", "init_ad11", "init_bd1", "init_gd1",
        "response_init_ad10", "response_init_ad11", "response_init_bd1", "response_init_gd1",
        "ad00", "ad01", "ad10", "ad11", "bd0", "bd1", "gd0", "gd1",
        "alpha_w", "beta_w", "gamma_w", "nonlinear_start", "nonlinear_response",
    )
    values = {name: np.full(size, np.nan) for name in names}
    v_start = dataset.column("v_start")
    w_start = dataset.column("w_start")
    v_response = dataset.column("v_response")
    w_response = dataset.column("w_response")

    for index in np.flatnonzero(eligible):
        initial = discrete_model(model, float(rho_start[index]))
        response_initial = discrete_model(model, float(rho_response[index]))
        transition = discrete_model(model, float(rho_transition[index]))
        for prefix, discrete in (("init_", initial), ("response_init_", response_initial)):
            values[f"{prefix}ad10"][index] = discrete.ad10
            values[f"{prefix}ad11"][index] = discrete.ad11
            values[f"{prefix}bd1"][index] = discrete.bd1
            values[f"{prefix}gd1"][index] = discrete.gd1
        for name in ("ad00", "ad01", "ad10", "ad11", "bd0", "bd1", "gd0", "gd1", "alpha_w", "beta_w", "gamma_w"):
            values[name][index] = getattr(transition, name)
        values["nonlinear_start"][index] = nonlinear_term(model, float(v_start[index]), float(w_start[index]))
        values["nonlinear_response"][index] = nonlinear_term(model, float(v_response[index]), float(w_response[index]))

    return PreparedPairs(dataset, model, eligible, **values)


def replay_observer(prepared: PreparedPairs, config: ObserverConfig, *, hard_reset_enabled: bool) -> ReplayResult:
    config.validate()
    dataset = prepared.dataset
    size = dataset.size
    prediction_available = np.zeros(size, dtype=bool)
    validated = np.zeros(size, dtype=bool)
    hard_reset = np.zeros(size, dtype=bool)
    velocity_innovation = np.full(size, np.nan)
    angular_velocity_innovation = np.full(size, np.nan)

    segment = dataset.column("segment_id").astype(np.int64)
    v_start = dataset.column("v_start")
    w_start = dataset.column("w_start")
    v_response = dataset.column("v_response")
    w_response = dataset.column("w_response")
    v_command = dataset.column("v_cmd")
    w_command = dataset.column("w_cmd")
    v_command_response = dataset.column("v_cmd_response")

    initialized = False
    xh = 0.0
    previous_segment = -1
    for index in range(size):
        new_segment = segment[index] != previous_segment
        if new_segment:
            initialized = False
            previous_segment = int(segment[index])
        if not prepared.eligible[index]:
            initialized = False
            continue

        if new_segment:
            denominator = prepared.init_ad10[index]
            if abs(denominator) < MIN_INITIALIZATION_COUPLING:
                continue
            xh = (
                v_start[index]
                - prepared.init_ad11[index] * v_start[index]
                - prepared.init_bd1[index] * v_command[index]
                - prepared.init_gd1[index] * prepared.nonlinear_start[index]
            ) / denominator
            initialized = bool(np.isfinite(xh))
            if not initialized:
                continue
        elif not initialized:
            denominator = prepared.response_init_ad10[index]
            if abs(denominator) < MIN_INITIALIZATION_COUPLING:
                continue
            xh = (
                v_response[index]
                - prepared.response_init_ad11[index] * v_response[index]
                - prepared.response_init_bd1[index] * v_command_response[index]
                - prepared.response_init_gd1[index] * prepared.nonlinear_response[index]
            ) / denominator
            initialized = bool(np.isfinite(xh))
            continue

        xh_predicted = (
            prepared.ad00[index] * xh
            + prepared.ad01[index] * v_start[index]
            + prepared.bd0[index] * v_command[index]
            + prepared.gd0[index] * prepared.nonlinear_start[index]
        )
        velocity_predicted = (
            prepared.ad10[index] * xh
            + prepared.ad11[index] * v_start[index]
            + prepared.bd1[index] * v_command[index]
            + prepared.gd1[index] * prepared.nonlinear_start[index]
        )
        innovation = v_response[index] - velocity_predicted
        if not np.all(np.isfinite([xh_predicted, velocity_predicted, innovation])):
            initialized = False
            continue

        prediction_available[index] = True
        velocity_innovation[index] = innovation
        yaw_prediction = (
            prepared.alpha_w[index] * w_start[index]
            + prepared.beta_w[index] * w_command[index]
            - prepared.gamma_w[index] * smooth_sign(w_start[index], prepared.model.schedule.sgn_eps)
        )
        if np.isfinite(yaw_prediction):
            angular_velocity_innovation[index] = w_response[index] - yaw_prediction

        if hard_reset_enabled and abs(innovation) > config.reset_threshold:
            hard_reset[index] = True
            initialized = False
            continue

        xh = xh_predicted + config.velocity_gain * np.clip(
            innovation,
            -config.correction_clip,
            config.correction_clip,
        )
        if not np.isfinite(xh):
            initialized = False
            continue
        validated[index] = True

    return ReplayResult(
        prediction_available,
        validated,
        hard_reset,
        velocity_innovation,
        angular_velocity_innovation,
    )


def observer_duty(prepared: PreparedPairs, replay: ReplayResult, split: np.ndarray) -> float:
    selected = np.asarray(split, dtype=bool) & prepared.eligible
    transition_count = int(np.sum(selected))
    if not transition_count:
        return float("nan")
    segment_count = len(np.unique(prepared.dataset.column("segment_id")[selected]))
    return float(np.sum(replay.validated & selected) / (transition_count + segment_count))


def residual_masks(
    replay: ReplayResult,
    segment_id: np.ndarray,
    *,
    threshold: float = 0.12,
    dilation_radius: int = 5,
) -> dict[str, np.ndarray]:
    eligible = replay.prediction_available & np.isfinite(replay.velocity_innovation)
    direct_stress = eligible & (np.abs(replay.velocity_innovation) > threshold)
    stress = eligible & dilate_within_segments(direct_stress, segment_id, dilation_radius)
    return {
        "eligible": eligible,
        "stress": stress,
        "clean": eligible & ~stress,
    }
