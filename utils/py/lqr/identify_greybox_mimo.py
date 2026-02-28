#!/usr/bin/env python3
"""Grey-box system identification for wheel-leg balanced robot dynamics.

This script identifies a physics-informed model mapping (v_cmd, w_cmd) → (v_act, w_act)
for a wheel-leg balanced robot with LQR-controlled pitch dynamics.

Model structure (continuous-time, Euler-discretized at 10Hz):

  Longitudinal velocity (2nd order with hidden pitch state + nonlinear compensation):
    ẋ_h   = a11·x_h + a12·v_act + b1·v_cmd
    v̇_act = a21·x_h + a22·v_act + b2·v_cmd + cf1·sgn(v_act) + cf2·v_act·|w_act|

    x_h represents the hidden pitch dynamics during acceleration/deceleration,
    capturing the non-minimum phase behavior (robot tilts before moving).

  Angular velocity (1st order lag + nonlinear friction):
    ẇ_act = (1/τ)·(w_cmd - w_act) − cf3·sgn(w_act)

Input delay: 1 step (0.1s at 10Hz MPC rate).

Optimization:
  - Three-phase: w-only → v-only (with measured w) → joint refinement
  - Velocity RMSE + displacement integral penalty (configurable weights)
  - Global search (differential_evolution) + multi-pass local refinement

Outputs (saved to identify_result/<run_stamp>/):
  - model_greybox_10hz.npz     (A, B, C, D + nonlinear params + metadata)
  - model_greybox_10hz.txt     (human-readable)
  - per_file_metrics.csv
  - plots/*.png                (velocity, residuals, displacement, trajectory)

Usage:
  python identify_greybox_mimo.py --data-dir identify_data --w-vel 1.0 --w-pos 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import signal
    from scipy.optimize import differential_evolution, minimize
except ImportError as exc:
    raise RuntimeError("scipy is required: pip install scipy") from exc

# Numba is a required dependency for this script (performance-critical).
from numba import njit, prange


# ════════════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════════════

DT = 0.1            # MPC timestep (10 Hz)
INPUT_DELAY = 1      # command delay in MPC steps (0.1 s)
SGN_EPS = 0.05       # smooth-sign softness: tanh(x / eps)

PARAM_NAMES = ["a11", "a12", "a21", "a22", "b1", "b2", "cf1", "cf2", "tau_w", "cf3", "xh0"]
N_PARAMS = len(PARAM_NAMES)

# Parameter bounds (continuous-time formulation)
PARAM_BOUNDS = [
    (-25.0, 5.0),    # a11  – hidden-state self-dynamics (stable ⇒ mostly < 0)
    (-15.0, 15.0),   # a12  – v → hidden coupling
    (-30.0, 30.0),   # a21  – hidden → v coupling (key for non-minimum phase)
    (-25.0, 5.0),    # a22  – v self-dynamics
    (-20.0, 20.0),   # b1   – v_cmd → hidden
    (-5.0, 25.0),    # b2   – v_cmd → v (expect positive)
    (-3.0, 1.0),     # cf1  – Coulomb friction on v (expect ≤ 0)
    (-8.0, 2.0),     # cf2  – yaw-coupling speed loss (expect ≤ 0)
    (0.02, 8.0),     # τ_w  – w time constant (positive, seconds)
    (-2.0, 3.0),     # cf3  – Coulomb friction on w (positive in ẇ = … − cf3·sgn)
    (-2.0, 2.0),     # xh0  – initial hidden state (pitch proxy)
]


# ════════════════════════════════════════════════════════════════════════════════
#  Data loading (compatible with identify_wheel_leg_mimo.py NPZ format)
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class Series10Hz:
    name: str
    t: np.ndarray
    v: np.ndarray
    w: np.ndarray
    v_cmd: np.ndarray
    w_cmd: np.ndarray
    meta: Dict


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def butter_lowpass_sos(cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs_hz
    return signal.butter(order, cutoff_hz / nyq, btype="low", output="sos")


def lowpass_then_downsample(x: np.ndarray, *, fs_hz: float, cutoff_hz: float, q: int) -> np.ndarray:
    x = _as_1d(x)
    if len(x) < 8 * q:
        return x[::q].copy()
    sos = butter_lowpass_sos(cutoff_hz, fs_hz, order=4)
    xf = signal.sosfiltfilt(sos, x)
    return np.asarray(xf[::q], dtype=float)


def robust_fs_from_t(t: np.ndarray) -> float:
    t = _as_1d(t)
    if len(t) < 3:
        return 20.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    return 1.0 / max(float(np.median(dt)), 1e-6) if len(dt) > 0 else 20.0


def load_all_npz_to_10hz(
    data_dir: Path, *, cutoff_hz: float = 4.0, raw_to_mpc_q: int = 2
) -> List[Series10Hz]:
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {data_dir}")
    out: List[Series10Hz] = []
    for p in files:
        with np.load(p, allow_pickle=True) as z:
            t = _as_1d(z["t"])
            v = _as_1d(z["v_meas"])
            w = _as_1d(z["w_meas"])
            vc = _as_1d(z["v_cmd"])
            wc = _as_1d(z["w_cmd"])
            meta = z["meta"].item() if "meta" in z else {}
        fs = robust_fs_from_t(t)
        kw = dict(fs_hz=fs, cutoff_hz=cutoff_hz, q=raw_to_mpc_q)
        v10 = lowpass_then_downsample(v, **kw)
        w10 = lowpass_then_downsample(w, **kw)
        vc10 = lowpass_then_downsample(vc, **kw)
        wc10 = lowpass_then_downsample(wc, **kw)
        t10 = t[::raw_to_mpc_q].copy()
        t10 -= float(t10[0])
        n = min(len(t10), len(v10), len(w10), len(vc10), len(wc10))
        if n < 20:
            continue
        out.append(Series10Hz(
            name=p.stem, t=t10[:n], v=v10[:n], w=w10[:n],
            v_cmd=vc10[:n], w_cmd=wc10[:n], meta=meta,
        ))
    return out


# ════════════════════════════════════════════════════════════════════════════════
#  Model simulation
# ════════════════════════════════════════════════════════════════════════════════

def _smooth_sgn(x: float, eps: float = SGN_EPS) -> float:
    return math.tanh(x / eps)


@njit(cache=True)
def _smooth_sgn_nb(x: float, eps: float) -> float:
    return math.tanh(x / eps)


@njit(cache=True)
def simulate_w_only_nb(
    tau_w: float,
    cf3: float,
    w_cmd: np.ndarray,
    w0: float,
    N: int,
    dt: float,
    delay: int,
    sgn_eps: float,
) -> np.ndarray:
    w_pred = np.empty(N, dtype=np.float64)
    w_pred[0] = w0
    inv_tau = 1.0 / max(abs(tau_w), 1e-4)
    for k in range(N - 1):
        kd = k - delay
        wc = w_cmd[kd] if kd >= 0 else w_cmd[0]
        sw = _smooth_sgn_nb(w_pred[k], sgn_eps)
        dw = inv_tau * (wc - w_pred[k]) - cf3 * sw
        w_next = w_pred[k] + dt * dw
        w_pred[k + 1] = w_next
        if abs(w_next) > 80.0:
            for j in range(k + 1, N):
                w_pred[j] = np.nan
            break
    return w_pred


@njit(cache=True)
def simulate_v_with_meas_w_nb(
    a11: float,
    a12: float,
    a21: float,
    a22: float,
    b1: float,
    b2: float,
    cf1: float,
    cf2: float,
    xh0: float,
    v_cmd: np.ndarray,
    w_meas: np.ndarray,
    v0: float,
    N: int,
    dt: float,
    delay: int,
    sgn_eps: float,
) -> np.ndarray:
    v_pred = np.empty(N, dtype=np.float64)
    v_pred[0] = v0
    x_h = xh0
    for k in range(N - 1):
        v = v_pred[k]
        w = w_meas[k]
        kd = k - delay
        vc = v_cmd[kd] if kd >= 0 else v_cmd[0]
        sv = _smooth_sgn_nb(v, sgn_eps)
        dx_h = a11 * x_h + a12 * v + b1 * vc
        dv = a21 * x_h + a22 * v + b2 * vc + cf1 * sv + cf2 * v * abs(w)
        x_h = x_h + dt * dx_h
        v_next = v + dt * dv
        v_pred[k + 1] = v_next
        if abs(v_next) > 50.0 or abs(x_h) > 200.0:
            for j in range(k + 1, N):
                v_pred[j] = np.nan
            break
    return v_pred


@njit(cache=True)
def simulate_greybox_nb(
    a11: float,
    a12: float,
    a21: float,
    a22: float,
    b1: float,
    b2: float,
    cf1: float,
    cf2: float,
    tau_w: float,
    cf3: float,
    xh0: float,
    v_cmd: np.ndarray,
    w_cmd: np.ndarray,
    v0: float,
    w0: float,
    N: int,
    dt: float,
    delay: int,
    sgn_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_pred = np.empty(N, dtype=np.float64)
    w_pred = np.empty(N, dtype=np.float64)
    xh_trace = np.empty(N, dtype=np.float64)

    v_pred[0] = v0
    w_pred[0] = w0
    xh_trace[0] = xh0

    x_h = xh0
    inv_tau = 1.0 / max(abs(tau_w), 1e-4)

    for k in range(N - 1):
        v = v_pred[k]
        w = w_pred[k]
        kd = k - delay
        vc = v_cmd[kd] if kd >= 0 else v_cmd[0]
        wc = w_cmd[kd] if kd >= 0 else w_cmd[0]

        sv = _smooth_sgn_nb(v, sgn_eps)
        dx_h = a11 * x_h + a12 * v + b1 * vc
        dv = a21 * x_h + a22 * v + b2 * vc + cf1 * sv + cf2 * v * abs(w)
        x_h_new = x_h + dt * dx_h
        v_new = v + dt * dv

        sw = _smooth_sgn_nb(w, sgn_eps)
        dw = inv_tau * (wc - w) - cf3 * sw
        w_new = w + dt * dw

        x_h = x_h_new
        v_pred[k + 1] = v_new
        w_pred[k + 1] = w_new
        xh_trace[k + 1] = x_h

        if abs(v_new) > 50.0 or abs(w_new) > 80.0 or abs(x_h) > 200.0:
            for j in range(k + 1, N):
                v_pred[j] = np.nan
                w_pred[j] = np.nan
                xh_trace[j] = np.nan
            break

    return v_pred, w_pred, xh_trace



def simulate_greybox(
    params: np.ndarray,
    v_cmd: np.ndarray,
    w_cmd: np.ndarray,
    v0: float,
    w0: float,
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Python wrapper around the numba-compiled simulator."""
    a11, a12, a21, a22, b1, b2, cf1, cf2, tau_w, cf3 = params[:10]
    xh0 = float(params[10]) if len(params) > 10 else 0.0
    return simulate_greybox_nb(
        float(a11), float(a12), float(a21), float(a22), float(b1), float(b2),
        float(cf1), float(cf2), float(tau_w), float(cf3), float(xh0),
        v_cmd, w_cmd, float(v0), float(w0), int(N), DT, INPUT_DELAY, SGN_EPS,
    )


# ════════════════════════════════════════════════════════════════════════════════
#  Trajectory integration helpers
# ════════════════════════════════════════════════════════════════════════════════

def _cumintegrate(y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal-rule integration with dt = DT.

    Returns s with s[0]=0 and s[k] approximating the integral of y from 0 to k*DT
    using the trapezoidal rule.
    """
    y = _as_1d(y)
    n = len(y)
    if n == 0:
        return y.copy()
    if n == 1:
        return np.zeros(1, dtype=float)
    s = np.empty(n, dtype=float)
    s[0] = 0.0
    s[1:] = np.cumsum(0.5 * (y[:-1] + y[1:])) * DT
    return s


def trajectory_2d(v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D trajectory: x, y, phi from v and w."""
    # Integrate heading and position using trapezoidal integration for better accuracy at 10Hz.
    phi = _cumintegrate(w)
    x = _cumintegrate(v * np.cos(phi))
    y = _cumintegrate(v * np.sin(phi))
    return x, y, phi


# ════════════════════════════════════════════════════════════════════════════════
#  Loss functions
# ════════════════════════════════════════════════════════════════════════════════

BURN = 10  # burn-in samples to skip for metrics


def _safe_var(x: np.ndarray, minimum: float = 1e-6) -> float:
    return max(float(np.var(x)), minimum)


@dataclass(frozen=True)
class PackedSeries:
    starts: np.ndarray  # int64 [M]
    lens: np.ndarray    # int64 [M]
    v: np.ndarray
    w: np.ndarray
    v_cmd: np.ndarray
    w_cmd: np.ndarray
    var_v: np.ndarray
    var_w: np.ndarray
    var_s: np.ndarray
    var_phi: np.ndarray


def pack_series_for_numba(series_list: List[Series10Hz], burn: int = BURN) -> PackedSeries:
    m = len(series_list)
    lens = np.asarray([len(s.v) for s in series_list], dtype=np.int64)
    starts = np.empty(m, dtype=np.int64)
    total_n = int(np.sum(lens))

    v = np.empty(total_n, dtype=np.float64)
    w = np.empty(total_n, dtype=np.float64)
    v_cmd = np.empty(total_n, dtype=np.float64)
    w_cmd = np.empty(total_n, dtype=np.float64)

    var_v = np.empty(m, dtype=np.float64)
    var_w = np.empty(m, dtype=np.float64)
    var_s = np.empty(m, dtype=np.float64)
    var_phi = np.empty(m, dtype=np.float64)

    idx = 0
    for i, s in enumerate(series_list):
        n = int(lens[i])
        starts[i] = idx
        v[idx : idx + n] = s.v
        w[idx : idx + n] = s.w
        v_cmd[idx : idx + n] = s.v_cmd
        w_cmd[idx : idx + n] = s.w_cmd

        sm = _cumintegrate(s.v)
        phim = _cumintegrate(s.w)
        var_v[i] = _safe_var(s.v[burn:])
        var_w[i] = _safe_var(s.w[burn:])
        var_s[i] = _safe_var(sm[burn:])
        var_phi[i] = _safe_var(phim[burn:])
        idx += n

    return PackedSeries(
        starts=starts,
        lens=lens,
        v=v,
        w=w,
        v_cmd=v_cmd,
        w_cmd=w_cmd,
        var_v=var_v,
        var_w=var_w,
        var_s=var_s,
        var_phi=var_phi,
    )


@njit(cache=True, parallel=True)
def loss_w_only_packed(
    tau_w: float,
    cf3: float,
    w_meas: np.ndarray,
    w_cmd: np.ndarray,
    starts: np.ndarray,
    lens: np.ndarray,
    var_phi: np.ndarray,
    burn: int,
    dt: float,
    delay: int,
    sgn_eps: float,
) -> float:
    m = lens.shape[0]
    loss_each = np.empty(m, dtype=np.float64)
    invalid = np.zeros(m, dtype=np.uint8)
    for i in prange(m):
        start = int(starts[i])
        n = int(lens[i])
        n_eff = n - burn
        if n_eff <= 0:
            loss_each[i] = 0.0
            continue

        inv_tau = 1.0 / max(abs(tau_w), 1e-4)
        w0 = w_meas[start]
        w_pred = w0
        prev_wp = w0
        prev_wm = w0
        phi_p = 0.0
        phi_m = 0.0
        sum_ew2 = 0.0
        sum_ephi2 = 0.0
        wcmd0 = w_cmd[start]
        valid = True

        for k in range(n - 1):
            kd = k - delay
            wc = w_cmd[start + kd] if kd >= 0 else wcmd0
            sw = _smooth_sgn_nb(w_pred, sgn_eps)
            dw = inv_tau * (wc - w_pred) - cf3 * sw
            w_next = w_pred + dt * dw
            wm_next = w_meas[start + k + 1]

            phi_p += 0.5 * dt * (prev_wp + w_next)
            phi_m += 0.5 * dt * (prev_wm + wm_next)

            if (k + 1) >= burn:
                ew = w_next - wm_next
                ephi = phi_p - phi_m
                sum_ew2 += ew * ew
                sum_ephi2 += ephi * ephi

            w_pred = w_next
            prev_wp = w_next
            prev_wm = wm_next

            if abs(w_next) > 80.0 or not math.isfinite(w_next):
                valid = False
                break

        if not valid:
            loss_each[i] = 1e12
            invalid[i] = 1
        else:
            mse_w = sum_ew2 / n_eff
            mse_phi = sum_ephi2 / n_eff
            loss_each[i] = mse_w + 0.3 * mse_phi / max(var_phi[i], 1e-6)

    if np.sum(invalid) > 0:
        return 1e12
    return float(np.mean(loss_each))


@njit(cache=True, parallel=True)
def loss_v_with_meas_w_packed(
    v_params: np.ndarray,
    v_meas: np.ndarray,
    w_meas: np.ndarray,
    v_cmd: np.ndarray,
    starts: np.ndarray,
    lens: np.ndarray,
    var_s: np.ndarray,
    w_pos: float,
    burn: int,
    dt: float,
    delay: int,
    sgn_eps: float,
) -> float:
    a11, a12, a21, a22, b1, b2, cf1, cf2 = (
        float(v_params[0]), float(v_params[1]), float(v_params[2]), float(v_params[3]),
        float(v_params[4]), float(v_params[5]), float(v_params[6]), float(v_params[7]),
    )
    xh0 = float(v_params[8])

    m = lens.shape[0]
    loss_each = np.empty(m, dtype=np.float64)
    invalid = np.zeros(m, dtype=np.uint8)
    for i in prange(m):
        start = int(starts[i])
        n = int(lens[i])
        n_eff = n - burn
        if n_eff <= 0:
            loss_each[i] = 0.0
            continue

        v0 = v_meas[start]
        x_h = xh0
        v_pred = v0
        prev_vp = v0
        prev_vm = v0
        s_p = 0.0
        s_m = 0.0
        sum_ev2 = 0.0
        sum_es2 = 0.0
        vcmd0 = v_cmd[start]
        valid = True

        for k in range(n - 1):
            w_k = w_meas[start + k]
            kd = k - delay
            vc = v_cmd[start + kd] if kd >= 0 else vcmd0
            sv = _smooth_sgn_nb(v_pred, sgn_eps)
            dx_h = a11 * x_h + a12 * v_pred + b1 * vc
            dv = a21 * x_h + a22 * v_pred + b2 * vc + cf1 * sv + cf2 * v_pred * abs(w_k)
            x_h = x_h + dt * dx_h
            v_next = v_pred + dt * dv

            vm_next = v_meas[start + k + 1]
            s_p += 0.5 * dt * (prev_vp + v_next)
            s_m += 0.5 * dt * (prev_vm + vm_next)

            if (k + 1) >= burn:
                ev = v_next - vm_next
                es = s_p - s_m
                sum_ev2 += ev * ev
                sum_es2 += es * es

            v_pred = v_next
            prev_vp = v_next
            prev_vm = vm_next

            if abs(v_next) > 50.0 or abs(x_h) > 200.0 or (not math.isfinite(v_next)):
                valid = False
                break

        if not valid:
            loss_each[i] = 1e12
            invalid[i] = 1
        else:
            mse_v = sum_ev2 / n_eff
            mse_s = sum_es2 / n_eff
            loss_each[i] = mse_v + w_pos * mse_s / max(var_s[i], 1e-6)

    if np.sum(invalid) > 0:
        return 1e12
    return float(np.mean(loss_each))


@njit(cache=True, parallel=True)
def loss_joint_packed(
    params: np.ndarray,
    v_meas: np.ndarray,
    w_meas: np.ndarray,
    v_cmd: np.ndarray,
    w_cmd: np.ndarray,
    starts: np.ndarray,
    lens: np.ndarray,
    var_v: np.ndarray,
    var_w: np.ndarray,
    var_s: np.ndarray,
    var_phi: np.ndarray,
    w_vel: float,
    w_pos: float,
    burn: int,
    dt: float,
    delay: int,
    sgn_eps: float,
) -> float:
    a11, a12, a21, a22, b1, b2, cf1, cf2, tau_w, cf3 = (
        float(params[0]), float(params[1]), float(params[2]), float(params[3]),
        float(params[4]), float(params[5]), float(params[6]), float(params[7]),
        float(params[8]), float(params[9]),
    )
    xh0 = float(params[10])

    m = lens.shape[0]
    totals = np.zeros(m, dtype=np.float64)
    counts = np.zeros(m, dtype=np.float64)
    invalid = np.zeros(m, dtype=np.uint8)

    for i in prange(m):
        start = int(starts[i])
        n = int(lens[i])
        n_eff = n - burn
        if n_eff <= 0:
            continue

        inv_tau = 1.0 / max(abs(tau_w), 1e-4)
        v0 = v_meas[start]
        w0 = w_meas[start]
        x_h = xh0
        v_pred = v0
        w_pred = w0

        prev_vp = v0
        prev_vm = v0
        prev_wp = w0
        prev_wm = w0

        s_p = 0.0
        s_m = 0.0
        phi_p = 0.0
        phi_m = 0.0

        sum_ev2 = 0.0
        sum_ew2 = 0.0
        sum_es2 = 0.0
        sum_ephi2 = 0.0

        vcmd0 = v_cmd[start]
        wcmd0 = w_cmd[start]
        valid = True

        for k in range(n - 1):
            kd = k - delay
            vc = v_cmd[start + kd] if kd >= 0 else vcmd0
            wc = w_cmd[start + kd] if kd >= 0 else wcmd0

            sv = _smooth_sgn_nb(v_pred, sgn_eps)
            dx_h = a11 * x_h + a12 * v_pred + b1 * vc
            dv = a21 * x_h + a22 * v_pred + b2 * vc + cf1 * sv + cf2 * v_pred * abs(w_pred)
            x_h = x_h + dt * dx_h
            v_next = v_pred + dt * dv

            sw = _smooth_sgn_nb(w_pred, sgn_eps)
            dw = inv_tau * (wc - w_pred) - cf3 * sw
            w_next = w_pred + dt * dw

            vm_next = v_meas[start + k + 1]
            wm_next = w_meas[start + k + 1]

            s_p += 0.5 * dt * (prev_vp + v_next)
            s_m += 0.5 * dt * (prev_vm + vm_next)
            phi_p += 0.5 * dt * (prev_wp + w_next)
            phi_m += 0.5 * dt * (prev_wm + wm_next)

            if (k + 1) >= burn:
                ev = v_next - vm_next
                ew = w_next - wm_next
                es = s_p - s_m
                ephi = phi_p - phi_m
                sum_ev2 += ev * ev
                sum_ew2 += ew * ew
                sum_es2 += es * es
                sum_ephi2 += ephi * ephi

            v_pred = v_next
            w_pred = w_next
            prev_vp = v_next
            prev_vm = vm_next
            prev_wp = w_next
            prev_wm = wm_next

            if (
                abs(v_next) > 50.0 or abs(w_next) > 80.0 or abs(x_h) > 200.0
                or (not math.isfinite(v_next)) or (not math.isfinite(w_next))
            ):
                valid = False
                break

        if not valid:
            totals[i] = 1e12
            counts[i] = 1.0
            invalid[i] = 1
        else:
            mse_v = (sum_ev2 / n_eff) / max(var_v[i], 1e-6)
            mse_w = (sum_ew2 / n_eff) / max(var_w[i], 1e-6)
            mse_s = (sum_es2 / n_eff) / max(var_s[i], 1e-6)
            mse_phi = (sum_ephi2 / n_eff) / max(var_phi[i], 1e-6)
            loss_i = w_vel * (mse_v + mse_w) + w_pos * (mse_s + mse_phi)
            totals[i] = n_eff * loss_i
            counts[i] = n_eff

    if np.sum(invalid) > 0:
        return 1e12
    total = np.sum(totals)
    cnt = np.sum(counts)
    return float(total / max(cnt, 1.0))


# ════════════════════════════════════════════════════════════════════════════════
#  Optimization (three-phase)
# ════════════════════════════════════════════════════════════════════════════════

def _de(func, bounds, seeds=(42, 137, 314), maxiter=600, popsize=30, **kw):
    """Run differential_evolution with multiple random seeds, return best."""
    best = None
    for s in seeds:
        r = differential_evolution(
            func, bounds, seed=s, maxiter=maxiter, popsize=popsize,
            tol=1e-10, atol=1e-10, mutation=(0.5, 1.5), recombination=0.9,
            polish=False, **kw,
        )
        if best is None or r.fun < best.fun:
            best = r
    return best


def _refine(func, x0, rounds=2):
    """Alternate Nelder-Mead / Powell refinement."""
    best_x, best_f = x0.copy(), func(x0)
    for _ in range(rounds):
        for method in ("Nelder-Mead", "Powell"):
            opts = {"maxiter": 5000}
            if method == "Nelder-Mead":
                opts.update(xatol=1e-10, fatol=1e-12, adaptive=True)
            else:
                opts.update(ftol=1e-12)
            r = minimize(func, best_x, method=method, options=opts)
            if r.fun < best_f:
                best_x, best_f = r.x.copy(), r.fun
    return best_x, best_f


def phase1_fit_w(series: List[Series10Hz], verbose: bool = True) -> Tuple[float, float]:
    """Fit w model independently (2 parameters)."""
    if verbose:
        print("═══ Phase 1: Fit ω model ═══")
    bounds_w = [(0.02, 8.0), (-2.0, 3.0)]
    pack = pack_series_for_numba(series)
    obj = lambda p: loss_w_only_packed(
        float(p[0]), float(p[1]),
        pack.w, pack.w_cmd,
        pack.starts, pack.lens,
        pack.var_phi,
        BURN, DT, INPUT_DELAY, SGN_EPS,
    )
    r = _de(obj, bounds_w, seeds=range(5), maxiter=300, popsize=25)
    x, f = _refine(obj, r.x, rounds=3)
    if verbose:
        print(f"  τ_w = {x[0]:.6f}  cf3 = {x[1]:.6f}  loss = {f:.8f}")
    return float(x[0]), float(x[1])


def phase2_fit_v(
    series: List[Series10Hz], w_pos: float = 0.3, verbose: bool = True,
) -> np.ndarray:
    """Fit v model using measured w for coupling (8+1 parameters: v_params + xh0)."""
    if verbose:
        print("═══ Phase 2: Fit v model (measured ω for coupling) ═══")
    bounds_v = list(PARAM_BOUNDS[:8]) + [PARAM_BOUNDS[10]]  # 8 v-params + xh0
    pack = pack_series_for_numba(series)
    obj = lambda p: loss_v_with_meas_w_packed(
        p,
        pack.v, pack.w, pack.v_cmd,
        pack.starts, pack.lens,
        pack.var_s,
        float(w_pos),
        BURN, DT, INPUT_DELAY, SGN_EPS,
    )
    r = _de(obj, bounds_v, seeds=range(5), maxiter=500, popsize=25)
    x, f = _refine(obj, r.x, rounds=3)
    if verbose:
        names_v = PARAM_NAMES[:8] + ["xh0"]
        for nm, val in zip(names_v, x):
            print(f"  {nm:6s} = {val:12.6f}")
        print(f"  loss = {f:.8f}")
    return x


def phase3_joint(
    series: List[Series10Hz],
    init_params: np.ndarray,
    w_vel: float = 1.0,
    w_pos: float = 0.3,
    verbose: bool = True,
) -> np.ndarray:
    """Joint refinement of all 11 parameters with coupled simulation."""
    if verbose:
        print("═══ Phase 3: Joint refinement ═══")
    pack = pack_series_for_numba(series)
    obj = lambda p: loss_joint_packed(
        p,
        pack.v, pack.w, pack.v_cmd, pack.w_cmd,
        pack.starts, pack.lens,
        pack.var_v, pack.var_w, pack.var_s, pack.var_phi,
        float(w_vel), float(w_pos),
        BURN, DT, INPUT_DELAY, SGN_EPS,
    )

    # local refinement (fast, no DE)
    best_x, best_f = _refine(obj, init_params, rounds=5)

    # try a few perturbations to escape local minima
    import itertools
    for trial in range(5):
        rng = np.random.RandomState(42 + trial)
        perturb = init_params.copy()
        for i in range(len(perturb)):
            lo, hi = PARAM_BOUNDS[i]
            perturb[i] += rng.uniform(-0.1, 0.1) * (hi - lo)
            perturb[i] = np.clip(perturb[i], lo, hi)
        x2, f2 = _refine(obj, perturb, rounds=3)
        if f2 < best_f:
            best_x, best_f = x2, f2

    # final polish
    best_x, best_f = _refine(obj, best_x, rounds=3)
    if f2 < best_f:
        best_x, best_f = x2, f2
        if verbose:
            print("  (wide-band DE found better solution)")

    # final polish
    best_x, best_f = _refine(obj, best_x, rounds=3)

    if verbose:
        print("  Final parameters:")
        for nm, val in zip(PARAM_NAMES, best_x):
            print(f"    {nm:6s} = {val:12.6f}")
        print(f"  joint loss = {best_f:.8f}")
    return best_x


# ════════════════════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    params: np.ndarray, series_list: List[Series10Hz], burn: int = BURN,
) -> List[Dict]:
    results = []
    for s in series_list:
        N = len(s.v)
        vp, wp, xh = simulate_greybox(params, s.v_cmd, s.w_cmd, s.v[0], s.w[0], N)

        ev = vp[burn:] - s.v[burn:]
        ew = wp[burn:] - s.w[burn:]
        rmse_v = float(np.sqrt(np.mean(ev ** 2)))
        rmse_w = float(np.sqrt(np.mean(ew ** 2)))
        mae_v = float(np.mean(np.abs(ev)))
        mae_w = float(np.mean(np.abs(ew)))

        # displacement / heading
        sp = _cumintegrate(vp)
        sm = _cumintegrate(s.v)
        phip = _cumintegrate(wp)
        phim = _cumintegrate(s.w)
        rmse_s = float(np.sqrt(np.mean((sp[burn:] - sm[burn:]) ** 2)))
        rmse_phi = float(np.sqrt(np.mean((phip[burn:] - phim[burn:]) ** 2)))

        # BIC (2-output Gaussian, unknown variance)
        n = len(ev)
        rss = float(np.sum(ev ** 2) + np.sum(ew ** 2))
        bic = N_PARAMS * math.log(max(n, 1)) + n * math.log(max(rss / n, 1e-18))

        results.append(dict(
            name=s.name, scenario=str(s.meta.get("scenario", "")),
            rmse_v=rmse_v, rmse_w=rmse_w, mae_v=mae_v, mae_w=mae_w,
            rmse_s=rmse_s, rmse_phi=rmse_phi, bic=bic,
            v_pred=vp, w_pred=wp, x_h=xh,
            s_pred=sp, s_meas=sm, phi_pred=phip, phi_meas=phim,
        ))
    return results


# ════════════════════════════════════════════════════════════════════════════════
#  State-space export for MPC
# ════════════════════════════════════════════════════════════════════════════════

def build_discrete_ss(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Build discrete-time state-space matrices for MPC.

    State: [x_h, v_act, w_act, dv(=v_cmd_{k-1}), dw(=w_cmd_{k-1})]
    Input: [v_cmd, w_cmd]
    Output: [v_act, w_act]

    Linear part only (nonlinear terms exported separately in metadata).
    """
    a11, a12, a21, a22, b1, b2, cf1, cf2, tau_w, cf3 = params[:10]
    xh0 = float(params[10]) if len(params) > 10 else 0.0
    dt = DT
    inv_tau = 1.0 / max(abs(tau_w), 1e-4)

    nx, nu, ny = 5, 2, 2
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    C = np.zeros((ny, nx))
    D = np.zeros((ny, nu))

    # x_h[k+1] = (1+dt·a11)x_h + dt·a12·v + dt·b1·dv
    A[0, 0] = 1.0 + dt * a11
    A[0, 1] = dt * a12
    A[0, 3] = dt * b1

    # v[k+1] = dt·a21·x_h + (1+dt·a22)v + dt·b2·dv  (+ nonlinear terms)
    A[1, 0] = dt * a21
    A[1, 1] = 1.0 + dt * a22
    A[1, 3] = dt * b2

    # w[k+1] = (1-dt/τ)w + (dt/τ)dw  (+ nonlinear terms)
    A[2, 2] = 1.0 - dt * inv_tau
    A[2, 4] = dt * inv_tau

    # delay states: dv[k+1] = v_cmd[k], dw[k+1] = w_cmd[k]
    B[3, 0] = 1.0
    B[4, 1] = 1.0

    # outputs
    C[0, 1] = 1.0  # v_act
    C[1, 2] = 1.0  # w_act

    meta = {
        "model_type": "greybox_pitch_hidden_state",
        "dt": DT,
        "input_delay_steps": INPUT_DELAY,
        "sgn_eps": SGN_EPS,
        "state_order": ["x_h(pitch_hidden)", "v_act", "w_act",
                        "dv(=v_cmd[k-1])", "dw(=w_cmd[k-1])"],
        "input_order": ["v_cmd", "w_cmd"],
        "output_order": ["v_act", "w_act"],
        "continuous_params": {nm: float(v) for nm, v in zip(PARAM_NAMES, params)},
        "xh0": xh0,
        "nonlinear_terms": {
            "v_equation": f"+ {cf1:.6f}*sgn(v) + {cf2:.6f}*v*|w|",
            "w_equation": f"- {cf3:.6f}*sgn(w)",
            "cf1": float(cf1),
            "cf2": float(cf2),
            "cf3": float(cf3),
        },
        "v_model": {
            "type": "2nd_order_greybox_with_hidden_pitch",
            "A_ct": [[float(a11), float(a12)], [float(a21), float(a22)]],
            "B_ct": [float(b1), float(b2)],
        },
        "w_model": {
            "type": "1st_order_lag_plus_friction",
            "tau_w": float(tau_w),
            "cf3": float(cf3),
        },
    }
    return A, B, C, D, meta


# ════════════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════════════

def generate_plots(
    series: List[Series10Hz],
    results: List[Dict],
    plots_dir: Path,
    params: np.ndarray,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for s, r in zip(series, results):
        vp = r["v_pred"]
        wp = r["w_pred"]
        xh = r["x_h"]

        fig = plt.figure(figsize=(16, 22))
        gs = fig.add_gridspec(6, 2, hspace=0.38, wspace=0.28)

        # ── Row 1: velocity tracking ──
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(s.t, s.v_cmd, "g--", lw=1, alpha=0.6, label="v_cmd")
        ax.plot(s.t, s.v, "b-", lw=1.2, label="v_meas")
        ax.plot(s.t, vp, "r-", lw=1.2, label="v_pred")
        ax.set_ylabel("v (m/s)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.set_title(f"{s.name} — v tracking (RMSE={r['rmse_v']:.4f})")

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(s.t, s.w_cmd, "g--", lw=1, alpha=0.6, label="w_cmd")
        ax.plot(s.t, s.w, "b-", lw=1.2, label="w_meas")
        ax.plot(s.t, wp, "r-", lw=1.2, label="w_pred")
        ax.set_ylabel("w (rad/s)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.set_title(f"ω tracking (RMSE={r['rmse_w']:.4f})")

        # ── Row 2: residuals ──
        ev = vp - s.v
        ew = wp - s.w
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(s.t, ev, "r-", lw=0.7); ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.fill_between(s.t, ev, 0, alpha=0.25, color="red")
        ax.set_ylabel("v residual"); ax.set_title("v residual (pred−meas)"); ax.grid(True, alpha=0.25)

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(s.t, ew, "r-", lw=0.7); ax.axhline(0, color="k", ls="--", lw=0.5)
        ax.fill_between(s.t, ew, 0, alpha=0.25, color="orange")
        ax.set_ylabel("w residual"); ax.set_title("ω residual (pred−meas)"); ax.grid(True, alpha=0.25)

        # ── Row 3: residual histograms ──
        ax = fig.add_subplot(gs[2, 0])
        ax.hist(ev[BURN:], bins=60, density=True, alpha=0.7, color="steelblue")
        ax.set_xlabel("v residual"); ax.set_ylabel("density"); ax.set_title("v residual distribution")
        ax.axvline(0, color="k", ls="--", lw=0.5); ax.grid(True, alpha=0.25)

        ax = fig.add_subplot(gs[2, 1])
        ax.hist(ew[BURN:], bins=60, density=True, alpha=0.7, color="darkorange")
        ax.set_xlabel("w residual"); ax.set_ylabel("density"); ax.set_title("ω residual distribution")
        ax.axvline(0, color="k", ls="--", lw=0.5); ax.grid(True, alpha=0.25)

        # ── Row 4: displacement / heading ──
        ax = fig.add_subplot(gs[3, 0])
        ax.plot(s.t, r["s_meas"], "b-", lw=1.2, label="s_meas")
        ax.plot(s.t, r["s_pred"], "r--", lw=1.2, label="s_pred")
        ax.set_ylabel("s (m)"); ax.legend(fontsize=8)
        ax.set_title(f"Displacement (RMSE_s={r['rmse_s']:.4f})"); ax.grid(True, alpha=0.25)

        ax = fig.add_subplot(gs[3, 1])
        ax.plot(s.t, r["phi_meas"], "b-", lw=1.2, label="φ_meas")
        ax.plot(s.t, r["phi_pred"], "r--", lw=1.2, label="φ_pred")
        ax.set_ylabel("φ (rad)"); ax.legend(fontsize=8)
        ax.set_title(f"Heading (RMSE_φ={r['rmse_phi']:.4f})"); ax.grid(True, alpha=0.25)

        # ── Row 5: hidden state ──
        ax = fig.add_subplot(gs[4, 0])
        ax.plot(s.t, xh, "m-", lw=1.0, label="x_h (pitch hidden)")
        ax.set_ylabel("x_h"); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        ax.set_title("Hidden pitch state")
        ax.set_xlabel("t (s)")

        # ── Row 5 right: autocorrelation of residuals ──
        ax = fig.add_subplot(gs[4, 1])
        ev_valid = ev[BURN:]
        lags = min(50, len(ev_valid) // 2)
        acf = np.correlate(ev_valid - ev_valid.mean(), ev_valid - ev_valid.mean(), mode="full")
        acf = acf[len(acf) // 2:]
        acf = acf[:lags + 1] / (acf[0] + 1e-12)
        ax.bar(range(lags + 1), acf, color="steelblue", alpha=0.7, width=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(1.96/np.sqrt(len(ev_valid)), color="r", ls="--", lw=0.8, label="95% CI")
        ax.axhline(-1.96/np.sqrt(len(ev_valid)), color="r", ls="--", lw=0.8)
        ax.set_xlabel("lag"); ax.set_ylabel("ACF"); ax.legend(fontsize=8)
        ax.set_title("v residual autocorrelation"); ax.grid(True, alpha=0.25)

        # ── Row 6: 2D trajectory ──
        ax = fig.add_subplot(gs[5, :])
        xm, ym, _ = trajectory_2d(s.v, s.w)
        xp, yp, _ = trajectory_2d(vp, wp)
        ax.plot(xm, ym, "b-", lw=1.5, label="measured")
        ax.plot(xp, yp, "r--", lw=1.5, label="predicted")
        ax.plot(xm[0], ym[0], "go", ms=8, label="start")
        ax.plot(xm[-1], ym[-1], "bs", ms=7, label="end(meas)")
        ax.plot(xp[-1], yp[-1], "rs", ms=7, label="end(pred)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.legend(fontsize=8); ax.set_aspect("equal")
        ax.set_title("2D Trajectory"); ax.grid(True, alpha=0.25)

        fig.savefig(plots_dir / f"{s.name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=str, default="identify_data")
    ap.add_argument("--out-dir", type=str, default="identify_result")
    ap.add_argument("--cutoff-hz", type=float, default=4.0,
                    help="Anti-alias lowpass cutoff (Hz) before 20→10Hz downsample")
    ap.add_argument("--w-vel", type=float, default=1.0, help="Weight for velocity RMSE term")
    ap.add_argument("--w-pos", type=float, default=4.0, help="Weight for displacement integral term")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = out_root / stamp
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data dir: {data_dir}")
    print(f"Output:   {out_dir}")
    print(f"Weights:  w_vel={args.w_vel}, w_pos={args.w_pos}")
    print(f"Cutoff:   {args.cutoff_hz} Hz\n")

    # ── Load data ──
    series = load_all_npz_to_10hz(data_dir, cutoff_hz=args.cutoff_hz)
    if not series:
        raise RuntimeError("No usable data after preprocessing")
    for s in series:
        print(f"  Loaded {s.name}: N={len(s.v)}, duration={s.t[-1]:.1f}s")
    print()

    # ── Phase 1: w model ──
    tau_w, cf3 = phase1_fit_w(series)
    print()

    # ── Phase 2: v model (using measured w) ──
    v_params = phase2_fit_v(series, w_pos=args.w_pos)
    print()

    # ── Phase 3: joint refinement ──
    # v_params has 9 entries: [a11..cf2, xh0], insert tau_w, cf3 before xh0
    init_all = np.concatenate([v_params[:8], [tau_w, cf3], [v_params[8]]])
    final_params = phase3_joint(series, init_all, w_vel=args.w_vel, w_pos=args.w_pos)
    print()

    # ── Compute metrics ──
    results = compute_metrics(final_params, series)

    # ── Build state-space ──
    A, B, C, D, meta = build_discrete_ss(final_params)

    # ── Save model ──
    npz_path = out_dir / "model_greybox_10hz.npz"
    np.savez(npz_path, A=A, B=B, C=C, D=D, params=final_params,
             meta=json.dumps(meta, ensure_ascii=False))

    txt_path = out_dir / "model_greybox_10hz.txt"
    with txt_path.open("w") as f:
        f.write("# Grey-box 10Hz MIMO model for MPC\n")
        f.write(f"# Model: hidden pitch state + nonlinear friction/coupling\n")
        f.write(f"# w_vel={args.w_vel}, w_pos={args.w_pos}, cutoff={args.cutoff_hz}Hz\n\n")
        f.write("# Continuous-time parameters:\n")
        for nm, val in zip(PARAM_NAMES, final_params):
            f.write(f"#   {nm:6s} = {val:12.6f}\n")
        f.write("\n# v model (continuous):\n")
        f.write("#   ẋ_h   = a11·x_h + a12·v + b1·v_cmd\n")
        f.write("#   v̇_act = a21·x_h + a22·v + b2·v_cmd + cf1·sgn(v) + cf2·v·|w|\n")
        f.write("# w model (continuous):\n")
        f.write("#   ẇ_act = (1/τ)·(w_cmd − w_act) − cf3·sgn(w_act)\n\n")
        np.set_printoptions(precision=6, suppress=True, linewidth=140)
        f.write("# Discrete-time state-space (Euler, dt=0.1s):\n")
        f.write("# State: [x_h, v_act, w_act, dv(=v_cmd[k-1]), dw(=w_cmd[k-1])]\n\n")
        f.write("A =\n" + str(A) + "\n\n")
        f.write("B =\n" + str(B) + "\n\n")
        f.write("C =\n" + str(C) + "\n\n")
        f.write("D =\n" + str(D) + "\n\n")
        f.write("# Full metadata:\n")
        f.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    # ── Save per-file metrics CSV ──
    csv_path = out_dir / "per_file_metrics.csv"
    with csv_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "scenario", "rmse_v", "mae_v", "rmse_w", "mae_w",
                      "rmse_s", "rmse_phi", "bic"])
        for r in results:
            wr.writerow([r["name"], r["scenario"],
                         r["rmse_v"], r["mae_v"], r["rmse_w"], r["mae_w"],
                         r["rmse_s"], r["rmse_phi"], r["bic"]])

    # ── Generate plots ──
    print("Generating plots...")
    generate_plots(series, results, plots_dir, final_params)

    # ── Console summary ──
    print("\n" + "=" * 60)
    print("  Grey-box Identification Summary (10Hz)")
    print("=" * 60)
    print(f"Datasets: {len(series)}")
    print(f"\nIdentified parameters (continuous-time):")
    for nm, val in zip(PARAM_NAMES, final_params):
        print(f"  {nm:6s} = {val:12.6f}")

    # eigenvalue analysis of v subsystem
    Av = np.array([[final_params[0], final_params[1]],
                    [final_params[2], final_params[3]]])
    eigs = np.linalg.eigvals(Av)
    print(f"\nv subsystem A_ct eigenvalues: {eigs}")
    print(f"  (stable if Re < 0: Re = {[f'{e.real:.4f}' for e in eigs]})")
    xh0_val = float(final_params[10]) if len(final_params) > 10 else 0.0
    print(f"  xh0 (initial hidden state) = {xh0_val:.6f}")

    for r in results:
        print(f"\n[{r['name']}]")
        print(f"  RMSE_v = {r['rmse_v']:.4f}   MAE_v = {r['mae_v']:.4f}")
        print(f"  RMSE_w = {r['rmse_w']:.4f}   MAE_w = {r['mae_w']:.4f}")
        print(f"  RMSE_s = {r['rmse_s']:.4f}   RMSE_φ = {r['rmse_phi']:.4f}")
        print(f"  BIC    = {r['bic']:.3f}")

    print(f"\nResults saved to: {out_dir}")
    print(f"  Model: {npz_path}")
    print(f"  Text:  {txt_path}")
    print(f"  Plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
