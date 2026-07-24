"""LPV model loading and numerics kept in parity with the C++ implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


MODEL_DT = 0.05


@dataclass(frozen=True)
class ScheduleParameters:
    z_ref: float
    z_scale: float
    rho_clip: float
    sgn_eps: float


@dataclass(frozen=True)
class PlantParameters:
    ca00: float
    ca01: float
    ca10: float
    ca11: float
    cb0: float
    cb1: float
    dca00: float
    dca01: float
    dca10: float
    dca11: float
    dcb0: float
    dcb1: float
    gxh: float
    gv: float
    cf1: float
    cf2: float
    w_lam0: float
    w_k0: float
    w_cf0: float
    w_lam1: float
    w_k1: float
    w_cf1: float


@dataclass(frozen=True)
class ObserverParameters:
    velocity_gain: float
    correction_clip: float
    reset_threshold: float


@dataclass(frozen=True)
class LPVModel:
    schedule: ScheduleParameters
    plant: PlantParameters
    observer: ObserverParameters


@dataclass(frozen=True)
class DiscreteModel:
    rho: float
    ad00: float
    ad01: float
    ad10: float
    ad11: float
    bd0: float
    bd1: float
    gd0: float
    gd1: float
    alpha_w: float
    beta_w: float
    gamma_w: float


def _finite_float(mapping: dict[str, object], name: str) -> float:
    if name not in mapping:
        raise ValueError(f"Missing kinematic_model parameter: {name}")
    value = float(mapping[name])
    if not math.isfinite(value):
        raise ValueError(f"kinematic_model.{name} must be finite")
    return value


def load_model(path: Path) -> LPVModel:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = raw.get("/**", {}).get("ros__parameters", {}).get("kinematic_model", {})
    if not parameters:
        raise ValueError(f"Missing /**.ros__parameters.kinematic_model in {path}")

    schedule = ScheduleParameters(**{
        name: _finite_float(parameters, name)
        for name in ("z_ref", "z_scale", "rho_clip", "sgn_eps")
    })
    if schedule.z_scale <= 0.0 or schedule.rho_clip <= 0.0 or schedule.sgn_eps <= 0.0:
        raise ValueError("z_scale, rho_clip, and sgn_eps must be positive")

    plant_names = tuple(PlantParameters.__dataclass_fields__)
    plant = PlantParameters(**{
        name: _finite_float(parameters, name)
        for name in plant_names
    })
    observer = ObserverParameters(
        velocity_gain=_finite_float(parameters, "obs_lv"),
        correction_clip=_finite_float(parameters, "obs_v_correction_clip"),
        reset_threshold=_finite_float(parameters, "obs_v_reset_threshold"),
    )
    if not 0.0 < observer.correction_clip < observer.reset_threshold:
        raise ValueError("observer correction clip must be positive and below reset threshold")
    return LPVModel(schedule, plant, observer)


def schedule_rho(
    model: LPVModel,
    leg_h: np.ndarray | float,
    leg_psi: np.ndarray | float,
) -> np.ndarray:
    height = np.asarray(leg_h, dtype=np.float64)
    angle = np.asarray(leg_psi, dtype=np.float64)
    rho = (height * np.cos(angle) - model.schedule.z_ref) / model.schedule.z_scale
    return np.clip(rho, -model.schedule.rho_clip, model.schedule.rho_clip)


def smooth_sign(value: float, epsilon: float) -> float:
    return math.tanh(value / max(epsilon, 1e-6))


def zoh_v_matrices(
    a00: float,
    a01: float,
    a10: float,
    a11: float,
    b0: float,
    b1: float,
    g0: float,
    g1: float,
    dt: float = MODEL_DT,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Match nav_executor::zoh_v_matrices, including its regularization branches."""
    m00 = a00 * dt
    m01 = a01 * dt
    m10 = a10 * dt
    m11 = a11 * dt
    trace = m00 + m11
    determinant = m00 * m11 - m01 * m10
    discriminant = trace * trace - 4.0 * determinant

    if discriminant > 1e-8:
        root = math.sqrt(discriminant)
        lambda1 = 0.5 * (trace + root)
        lambda2 = 0.5 * (trace - root)
        delta = lambda1 - lambda2
        exp_lambda2 = math.exp(lambda2)
        beta = exp_lambda2 * math.expm1(delta) / delta
        alpha = math.exp(lambda1) - beta * lambda1
    elif discriminant < -1e-8:
        p = 0.5 * trace
        q = 0.5 * math.sqrt(-discriminant)
        exp_p = math.exp(p)
        beta = exp_p * math.sin(q) / q
        alpha = exp_p * (math.cos(q) - p * math.sin(q) / q)
    else:
        eigenvalue = 0.5 * trace
        exponential = math.exp(eigenvalue)
        beta = exponential
        alpha = exponential * (1.0 - eigenvalue)

    ad00 = alpha + beta * m00
    ad01 = beta * m01
    ad10 = beta * m10
    ad11 = alpha + beta * m11

    determinant_a = a00 * a11 - a01 * a10
    c = alpha - 1.0
    if abs(determinant_a) > 1e-6:
        inverse = 1.0 / determinant_a
        integral00 = c * a11 * inverse + beta * dt
        integral01 = c * (-a01) * inverse
        integral10 = c * (-a10) * inverse
        integral11 = c * a00 * inverse + beta * dt
    else:
        integral00 = dt + 0.5 * dt * dt * a00
        integral01 = 0.5 * dt * dt * a01
        integral10 = 0.5 * dt * dt * a10
        integral11 = dt + 0.5 * dt * dt * a11

    bd0 = integral00 * b0 + integral01 * b1
    bd1 = integral10 * b0 + integral11 * b1
    gd0 = integral00 * g0 + integral01 * g1
    gd1 = integral10 * g0 + integral11 * g1
    return ad00, ad01, ad10, ad11, bd0, bd1, gd0, gd1


def discrete_model(model: LPVModel, rho: float, dt: float = MODEL_DT) -> DiscreteModel:
    bounded_rho = min(max(float(rho), -model.schedule.rho_clip), model.schedule.rho_clip)
    plant = model.plant
    a00 = plant.ca00 + bounded_rho * plant.dca00
    a01 = plant.ca01 + bounded_rho * plant.dca01
    a10 = plant.ca10 + bounded_rho * plant.dca10
    a11 = plant.ca11 + bounded_rho * plant.dca11
    b0 = plant.cb0 + bounded_rho * plant.dcb0
    b1 = plant.cb1 + bounded_rho * plant.dcb1
    matrices = zoh_v_matrices(a00, a01, a10, a11, b0, b1, plant.gxh, plant.gv, dt)

    yaw_lambda = max(plant.w_lam0 + bounded_rho * plant.w_lam1, 1e-5)
    alpha_w = math.exp(-yaw_lambda * dt)
    integral_w = (1.0 - alpha_w) / yaw_lambda
    beta_w = integral_w * (plant.w_k0 + bounded_rho * plant.w_k1)
    gamma_w = integral_w * (plant.w_cf0 + bounded_rho * plant.w_cf1)
    return DiscreteModel(bounded_rho, *matrices, alpha_w, beta_w, gamma_w)


def nonlinear_term(model: LPVModel, velocity: float, angular_velocity: float) -> float:
    plant = model.plant
    return (
        plant.cf1 * smooth_sign(velocity, model.schedule.sgn_eps)
        + plant.cf2 * velocity * abs(angular_velocity)
    )
