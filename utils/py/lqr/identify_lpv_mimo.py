"""Identify a continuous-time LPV state-space model for wheel-leg MPC.

This script is independent from identify_greybox_mimo.py. It reads NPZ logs
recorded by rec_identify_data.py, downsamples them to the 20 Hz MPC rate, then
identifies a low-dimensional continuous-time LPV model and discretizes it at
each rollout step.

Prediction state for MPC:
    x = [x_h, v, w]

Identification also carries delayed command states so the exported linear model
keeps the current MPC convention:
    x_id = [x_h, v, w, dv, dw], dv=v_cmd[k-1], dw=w_cmd[k-1]

Scheduling variable:
    z   = leg_h * cos(leg_psi)
    rho = clip((z - z_ref) / z_scale, -rho_clip, rho_clip)

Continuous-time LPV model:
    [x_h_dot, v_dot]^T = (Ac0 + rho*Ac1) [x_h, v]^T
                         + (Bc0 + rho*Bc1) v_cmd_delay
                         + Gnl * (cf1*tanh(v/eps) + cf2*v*abs(w))

    w_dot = -(lam0 + rho*lam1) * w
            + (kw0 + rho*kw1) * w_cmd_delay
            - (cw0 + rho*cw1) * tanh(w/eps)

leg_psi is not a predicted state and is not a scheduling variable. It is used
only as an auxiliary x_h observability signal via a weak proxy loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import signal
    from scipy.optimize import differential_evolution, minimize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scipy is required. Install with: uv pip install scipy") from exc

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("numba is required. Install with: uv pip install numba") from exc


DT = 0.05
MPC_RATE_HZ = int(round(1.0 / DT))
INPUT_DELAY = 1
SGN_EPS = 0.05
BURN = 10

PARAM_NAMES = [
    "ca00", "ca01", "ca10", "ca11", "cb0", "cb1",
    "dca00", "dca01", "dca10", "dca11", "dcb0", "dcb1",
    "gxh", "gv", "cf1", "cf2",
    "w_lam0", "w_k0", "w_cf0", "w_lam1", "w_k1", "w_cf1",
    "xh0_bias", "xh0_psi", "xh0_v",
    "psi_bias", "psi_gain", "psi_v",
]

PARAM_BOUNDS = [
    (-25.0, 8.0),     # ca00
    (-20.0, 20.0),    # ca01
    (-40.0, 40.0),    # ca10
    (-25.0, 10.0),    # ca11
    (-20.0, 20.0),    # cb0
    (-10.0, 30.0),    # cb1
    (-20.0, 20.0),    # dca00
    (-20.0, 20.0),    # dca01
    (-40.0, 40.0),    # dca10
    (-20.0, 20.0),    # dca11
    (-15.0, 15.0),    # dcb0
    (-15.0, 15.0),    # dcb1
    (-3.0, 3.0),      # gxh
    (-3.0, 3.0),      # gv
    (-3.0, 3.0),      # cf1
    (-12.0, 6.0),     # cf2
    (0.05, 40.0),     # w_lam0
    (-20.0, 40.0),    # w_k0
    (-8.0, 8.0),      # w_cf0
    (-20.0, 20.0),    # w_lam1
    (-20.0, 20.0),    # w_k1
    (-8.0, 8.0),      # w_cf1
    (-8.0, 8.0),      # xh0_bias
    (-15.0, 15.0),    # xh0_psi
    (-10.0, 10.0),    # xh0_v
    (-0.8, 0.8),      # psi_bias
    (-12.0, 12.0),    # psi_gain
    (-6.0, 6.0),      # psi_v
]


@dataclass
class SeriesMPC:
    name: str
    t: np.ndarray
    v: np.ndarray
    w: np.ndarray
    v_cmd: np.ndarray
    w_cmd: np.ndarray
    leg_h: np.ndarray
    leg_psi: np.ndarray
    z: np.ndarray
    rho: np.ndarray
    meta: Dict


@dataclass(frozen=True)
class PackedSeries:
    starts: np.ndarray
    lens: np.ndarray
    v: np.ndarray
    w: np.ndarray
    v_cmd: np.ndarray
    w_cmd: np.ndarray
    leg_psi: np.ndarray
    rho: np.ndarray
    edge_weight: np.ndarray
    var_s: np.ndarray
    var_phi: np.ndarray
    var_psi: np.ndarray


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def robust_fs_from_t(t: np.ndarray) -> float:
    t = _as_1d(t)
    if len(t) < 3:
        return MPC_RATE_HZ
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return MPC_RATE_HZ
    return 1.0 / max(float(np.median(dt)), 1e-6)


def butter_lowpass_sos(cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    nyq = max(0.5 * fs_hz, 1e-6)
    wn = min(max(cutoff_hz / nyq, 1e-4), 0.98)
    return signal.butter(order, wn, btype="low", output="sos")


def lowpass_then_downsample(x: np.ndarray, *, fs_hz: float, cutoff_hz: float, q: int) -> np.ndarray:
    x = _as_1d(x)
    if len(x) < 8 * q or cutoff_hz <= 0.0:
        return x[::q].copy()
    sos = butter_lowpass_sos(cutoff_hz, fs_hz, order=4)
    xf = signal.sosfiltfilt(sos, x)
    return np.asarray(xf[::q], dtype=float)


def _safe_var(x: np.ndarray, minimum: float = 1e-6) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return minimum
    return max(float(np.nanvar(x)), minimum)


def _cumintegrate(y: np.ndarray, dt: float = DT) -> np.ndarray:
    y = _as_1d(y)
    out = np.zeros_like(y)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out


def delayed_signal(u: np.ndarray, delay: int = INPUT_DELAY) -> np.ndarray:
    u = _as_1d(u)
    if delay <= 0 or u.size == 0:
        return u.copy()
    out = np.empty_like(u)
    out[:delay] = u[0]
    out[delay:] = u[:-delay]
    return out


def compute_edge_weight(v_cmd: np.ndarray, *, horizon_steps: int = 16, gain: float = 2.0) -> np.ndarray:
    v_cmd = _as_1d(v_cmd)
    n = len(v_cmd)
    weight = np.ones(n, dtype=float)
    if n < 3:
        return weight
    dv = np.abs(np.diff(v_cmd, prepend=v_cmd[0]))
    threshold = max(0.08, 0.35 * float(np.nanpercentile(dv, 95)))
    edges = np.flatnonzero(dv > threshold)
    for idx in edges:
        end = min(n, int(idx) + horizon_steps)
        for k in range(int(idx), end):
            decay = 1.0 - (k - int(idx)) / max(horizon_steps, 1)
            weight[k] += gain * decay
    return weight


def load_all_npz_to_mpc_rate(
    data_dir: Path,
    *,
    cutoff_hz: float = 8.0,
    raw_to_mpc_q: Optional[int] = None,
    rho_clip: float = 1.5,
) -> Tuple[List[SeriesMPC], Dict[str, float]]:
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    raw: List[Dict[str, object]] = []
    all_z: List[np.ndarray] = []
    for p in files:
        with np.load(p, allow_pickle=True) as zf:
            required = ["t", "v_meas", "w_meas", "v_cmd", "w_cmd", "leg_h_meas", "leg_psi_meas"]
            missing = [k for k in required if k not in zf]
            if missing:
                print(f"skip {p.name}: missing {missing}")
                continue
            t = _as_1d(zf["t"])
            v = _as_1d(zf["v_meas"])
            w = _as_1d(zf["w_meas"])
            vc = _as_1d(zf["v_cmd"])
            wc = _as_1d(zf["w_cmd"])
            leg_h = _as_1d(zf["leg_h_meas"])
            leg_psi = _as_1d(zf["leg_psi_meas"])
            meta = zf["meta"].item() if "meta" in zf else {}

        fs = robust_fs_from_t(t)
        q = max(1, round(fs * DT)) if raw_to_mpc_q is None else max(1, int(raw_to_mpc_q))
        kw = dict(fs_hz=fs, cutoff_hz=cutoff_hz, q=q)
        v_ds = lowpass_then_downsample(v, **kw)
        w_ds = lowpass_then_downsample(w, **kw)
        vc_ds = lowpass_then_downsample(vc, **kw)
        wc_ds = lowpass_then_downsample(wc, **kw)
        h_ds = lowpass_then_downsample(leg_h, **kw)
        psi_ds = lowpass_then_downsample(leg_psi, **kw)
        t_ds = t[::q].copy()
        if t_ds.size > 0:
            t_ds -= float(t_ds[0])
        n = min(len(t_ds), len(v_ds), len(w_ds), len(vc_ds), len(wc_ds), len(h_ds), len(psi_ds))
        if n < 25:
            continue
        z_sched = h_ds[:n] * np.cos(psi_ds[:n])
        finite = np.isfinite(z_sched)
        if not np.any(finite):
            continue
        all_z.append(z_sched[finite])
        raw.append(dict(
            name=p.stem,
            t=t_ds[:n],
            v=v_ds[:n],
            w=w_ds[:n],
            v_cmd=vc_ds[:n],
            w_cmd=wc_ds[:n],
            leg_h=h_ds[:n],
            leg_psi=psi_ds[:n],
            z=z_sched,
            meta=meta,
        ))

    if not raw:
        raise RuntimeError(f"No usable .npz files in {data_dir}")

    z_all = np.concatenate(all_z)
    z_ref = float(np.nanmedian(z_all))
    z_p10 = float(np.nanpercentile(z_all, 10))
    z_p90 = float(np.nanpercentile(z_all, 90))
    z_scale = max(0.5 * (z_p90 - z_p10), 0.03)

    series: List[SeriesMPC] = []
    for item in raw:
        z_sched = np.asarray(item["z"], dtype=float)
        rho = np.clip((z_sched - z_ref) / z_scale, -rho_clip, rho_clip)
        series.append(SeriesMPC(
            name=str(item["name"]),
            t=np.asarray(item["t"], dtype=float),
            v=np.asarray(item["v"], dtype=float),
            w=np.asarray(item["w"], dtype=float),
            v_cmd=np.asarray(item["v_cmd"], dtype=float),
            w_cmd=np.asarray(item["w_cmd"], dtype=float),
            leg_h=np.asarray(item["leg_h"], dtype=float),
            leg_psi=np.asarray(item["leg_psi"], dtype=float),
            z=z_sched,
            rho=rho,
            meta=dict(item["meta"]),
        ))

    sched = {
        "z_ref": z_ref,
        "z_scale": z_scale,
        "rho_clip": float(rho_clip),
        "z_p10": z_p10,
        "z_p90": z_p90,
    }
    return series, sched


def pack_series(series_list: Sequence[SeriesMPC], burn: int = BURN) -> PackedSeries:
    lens = np.asarray([len(s.v) for s in series_list], dtype=np.int64)
    starts = np.empty(len(series_list), dtype=np.int64)
    total_n = int(np.sum(lens))
    v = np.empty(total_n, dtype=np.float64)
    w = np.empty(total_n, dtype=np.float64)
    v_cmd = np.empty(total_n, dtype=np.float64)
    w_cmd = np.empty(total_n, dtype=np.float64)
    leg_psi = np.empty(total_n, dtype=np.float64)
    rho = np.empty(total_n, dtype=np.float64)
    edge_weight = np.empty(total_n, dtype=np.float64)
    var_s = np.empty(len(series_list), dtype=np.float64)
    var_phi = np.empty(len(series_list), dtype=np.float64)
    var_psi = np.empty(len(series_list), dtype=np.float64)

    idx = 0
    for i, s in enumerate(series_list):
        n = int(lens[i])
        starts[i] = idx
        v[idx:idx + n] = s.v
        w[idx:idx + n] = s.w
        v_cmd[idx:idx + n] = s.v_cmd
        w_cmd[idx:idx + n] = s.w_cmd
        leg_psi[idx:idx + n] = s.leg_psi
        rho[idx:idx + n] = s.rho
        edge_weight[idx:idx + n] = compute_edge_weight(s.v_cmd)
        var_s[i] = _safe_var(_cumintegrate(s.v)[burn:])
        var_phi[i] = _safe_var(_cumintegrate(s.w)[burn:])
        var_psi[i] = _safe_var(s.leg_psi[burn:])
        idx += n

    return PackedSeries(starts, lens, v, w, v_cmd, w_cmd, leg_psi, rho, edge_weight, var_s, var_phi, var_psi)


def ridge_lstsq(X: np.ndarray, y: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    xtx = X.T @ X
    xtx.flat[:: xtx.shape[0] + 1] += ridge
    xty = X.T @ y
    return np.linalg.solve(xtx, xty)


def estimate_initial_guess(series: Sequence[SeriesMPC]) -> np.ndarray:
    x0 = np.zeros(len(PARAM_NAMES), dtype=float)

    v_all = np.concatenate([s.v[BURN:] for s in series])
    psi_all = np.concatenate([s.leg_psi[BURN:] for s in series])
    Xpsi = np.column_stack([np.ones_like(v_all), v_all])
    psi_bias, psi_v = ridge_lstsq(Xpsi, psi_all, ridge=1e-4)

    raw_proxy_all = psi_all - psi_bias - psi_v * v_all
    proxy_scale = max(float(np.std(raw_proxy_all)), 0.05)
    psi_gain = proxy_scale
    x0[25] = float(psi_bias)
    x0[26] = float(psi_gain)
    x0[27] = float(psi_v)
    x0[22] = float(-psi_bias / psi_gain)
    x0[23] = float(1.0 / psi_gain)
    x0[24] = float(-psi_v / psi_gain)

    dxh_list: List[np.ndarray] = []
    dv_list: List[np.ndarray] = []
    dw_list: List[np.ndarray] = []
    fx_list: List[np.ndarray] = []
    fw_list: List[np.ndarray] = []
    for s in series:
        vc_d = delayed_signal(s.v_cmd)
        wc_d = delayed_signal(s.w_cmd)
        xh_proxy = (s.leg_psi - psi_bias - psi_v * s.v) / psi_gain
        sv = np.tanh(s.v / SGN_EPS)
        sw = np.tanh(s.w / SGN_EPS)
        rho = s.rho
        idx = slice(1, len(s.v) - 1)
        dxh = np.gradient(xh_proxy, DT)[idx]
        dv = np.gradient(s.v, DT)[idx]
        dw = np.gradient(s.w, DT)[idx]
        Fx = np.column_stack([
            xh_proxy[idx],
            s.v[idx],
            vc_d[idx],
            rho[idx] * xh_proxy[idx],
            rho[idx] * s.v[idx],
            rho[idx] * vc_d[idx],
            sv[idx],
            s.v[idx] * np.abs(s.w[idx]),
        ])
        Fw = np.column_stack([
            -s.w[idx],
            wc_d[idx],
            -sw[idx],
            -rho[idx] * s.w[idx],
            rho[idx] * wc_d[idx],
            -rho[idx] * sw[idx],
        ])
        dxh_list.append(dxh)
        dv_list.append(dv)
        dw_list.append(dw)
        fx_list.append(Fx)
        fw_list.append(Fw)

    y_xh = np.concatenate(dxh_list)
    y_v = np.concatenate(dv_list)
    y_w = np.concatenate(dw_list)
    Fx = np.vstack(fx_list)
    Fw = np.vstack(fw_list)

    theta_xh = ridge_lstsq(Fx, y_xh, ridge=2e-3)
    theta_v = ridge_lstsq(Fx, y_v, ridge=2e-3)
    theta_w = ridge_lstsq(Fw, y_w, ridge=2e-3)

    x0[0] = float(theta_xh[0])
    x0[1] = float(theta_xh[1])
    x0[4] = float(theta_xh[2])
    x0[6] = float(theta_xh[3])
    x0[7] = float(theta_xh[4])
    x0[10] = float(theta_xh[5])

    x0[2] = float(theta_v[0])
    x0[3] = float(theta_v[1])
    x0[5] = float(theta_v[2])
    x0[8] = float(theta_v[3])
    x0[9] = float(theta_v[4])
    x0[11] = float(theta_v[5])

    nl_map = np.array([
        [theta_xh[6], theta_xh[7]],
        [theta_v[6], theta_v[7]],
    ], dtype=float)
    u, svals, vt = np.linalg.svd(nl_map, full_matrices=False)
    if svals[0] > 1e-6:
        left = u[:, 0] * math.sqrt(float(svals[0]))
        right = vt[0, :] * math.sqrt(float(svals[0]))
        if right[1] > 0.0:
            left = -left
            right = -right
        x0[12] = float(left[0])
        x0[13] = float(left[1])
        x0[14] = float(right[0])
        x0[15] = float(right[1])
    else:
        x0[12] = 0.0
        x0[13] = 0.05
        x0[14] = 0.0
        x0[15] = -0.1

    x0[16] = float(np.clip(theta_w[0], 0.2, 8.0))
    x0[17] = float(theta_w[1])
    x0[18] = float(theta_w[2])
    x0[19] = float(theta_w[3])
    x0[20] = float(theta_w[4])
    x0[21] = float(theta_w[5])

    lo = np.asarray([b[0] for b in PARAM_BOUNDS], dtype=float)
    hi = np.asarray([b[1] for b in PARAM_BOUNDS], dtype=float)
    return np.clip(x0, lo, hi)


@njit(cache=False)
def _smooth_sgn(x: float, eps: float) -> float:
    return math.tanh(x / eps)


@njit(cache=False)
def _zoh_v_matrices(
    a00: float,
    a01: float,
    a10: float,
    a11: float,
    b0: float,
    b1: float,
    g0: float,
    g1: float,
    dt: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    m00 = a00 * dt
    m01 = a01 * dt
    m10 = a10 * dt
    m11 = a11 * dt
    tr_m = m00 + m11
    det_m = m00 * m11 - m01 * m10
    disc = tr_m * tr_m - 4.0 * det_m
    eps = 1e-12

    if disc > eps:
        s = math.sqrt(disc)
        lam1 = 0.5 * (tr_m + s)
        lam2 = 0.5 * (tr_m - s)
        e1 = math.exp(lam1)
        e2 = math.exp(lam2)
        beta = (e1 - e2) / (lam1 - lam2)
        alpha = e1 - beta * lam1
    elif disc < -eps:
        p = 0.5 * tr_m
        q = 0.5 * math.sqrt(-disc)
        ep = math.exp(p)
        beta = ep * math.sin(q) / q
        alpha = ep * (math.cos(q) - p * math.sin(q) / q)
    else:
        lam = 0.5 * tr_m
        el = math.exp(lam)
        beta = el
        alpha = el * (1.0 - lam)

    ad00 = alpha + beta * m00
    ad01 = beta * m01
    ad10 = beta * m10
    ad11 = alpha + beta * m11

    det_a = a00 * a11 - a01 * a10
    c = alpha - 1.0
    if abs(det_a) > 1e-10:
        inv_det = 1.0 / det_a
        g00 = c * a11 * inv_det + beta * dt
        g01 = c * (-a01) * inv_det
        g10 = c * (-a10) * inv_det
        g11 = c * a00 * inv_det + beta * dt
    else:
        g00 = dt + 0.5 * dt * dt * a00
        g01 = 0.5 * dt * dt * a01
        g10 = 0.5 * dt * dt * a10
        g11 = dt + 0.5 * dt * dt * a11

    bd0 = g00 * b0 + g01 * b1
    bd1 = g10 * b0 + g11 * b1
    gd0 = g00 * g0 + g01 * g1
    gd1 = g10 * g0 + g11 * g1
    return ad00, ad01, ad10, ad11, bd0, bd1, gd0, gd1


@njit(cache=False)
def _w_step(lambda_w: float, k_w: float, c_w: float, w: float, wc: float, dt: float) -> float:
    lam = max(lambda_w, 1e-5)
    alpha = math.exp(-lam * dt)
    integ = (1.0 - alpha) / lam
    return alpha * w + integ * (k_w * wc - c_w * _smooth_sgn(w, SGN_EPS))


@njit(cache=False)
def _spectral_radius_2x2(a00: float, a01: float, a10: float, a11: float) -> float:
    tr = a00 + a11
    det = a00 * a11 - a01 * a10
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        s = math.sqrt(disc)
        l1 = 0.5 * (tr + s)
        l2 = 0.5 * (tr - s)
        return max(abs(l1), abs(l2))
    return math.sqrt(max(det, 0.0))


@njit(cache=False)
def _regularization_penalty(params: np.ndarray) -> float:
    penalty = 0.0
    slope_idx = np.array([6, 7, 8, 9, 10, 11, 19, 20, 21], dtype=np.int64)
    for i in range(slope_idx.shape[0]):
        val = params[slope_idx[i]]
        penalty += 2e-4 * val * val

    for j in range(7):
        rho = -1.5 + 0.5 * j
        a00 = params[0] + rho * params[6]
        a01 = params[1] + rho * params[7]
        a10 = params[2] + rho * params[8]
        a11 = params[3] + rho * params[9]
        ad00, ad01, ad10, ad11, _, _, _, _ = _zoh_v_matrices(a00, a01, a10, a11, 0.0, 0.0, 0.0, 0.0, DT)
        sr = _spectral_radius_2x2(ad00, ad01, ad10, ad11)
        if sr > 1.02:
            d = sr - 1.02
            penalty += 4e4 * d * d
        if sr > 1.15:
            d2 = sr - 1.15
            penalty += 1e7 * d2 * d2
        lam = params[16] + rho * params[19]
        if lam < 0.10:
            d3 = 0.10 - lam
            penalty += 1e6 * d3 * d3
        cf = params[18] + rho * params[21]
        if cf < -4.0:
            d4 = cf + 4.0
            penalty += 1e4 * d4 * d4
    return penalty


@njit(cache=False, parallel=True)
def loss_packed(
    params: np.ndarray,
    v_meas: np.ndarray,
    w_meas: np.ndarray,
    v_cmd: np.ndarray,
    w_cmd: np.ndarray,
    leg_psi: np.ndarray,
    rho_arr: np.ndarray,
    edge_weight: np.ndarray,
    starts: np.ndarray,
    lens: np.ndarray,
    var_s: np.ndarray,
    var_phi: np.ndarray,
    var_psi: np.ndarray,
    burn: int,
    delay: int,
    w_pos: float,
    w_yaw: float,
    w_transient: float,
    w_psi: float,
) -> float:
    m = lens.shape[0]
    losses = np.empty(m, dtype=np.float64)
    invalid = np.zeros(m, dtype=np.uint8)

    for i in prange(m):
        start = int(starts[i])
        n = int(lens[i])
        n_eff = n - burn
        if n_eff <= 2:
            losses[i] = 0.0
            continue

        v_pred = v_meas[start]
        w_pred = w_meas[start]
        xh = params[22] + params[23] * leg_psi[start] + params[24] * v_meas[start]
        prev_vp = v_pred
        prev_vm = v_pred
        prev_wp = w_pred
        prev_wm = w_pred
        s_p = 0.0
        s_m = 0.0
        phi_p = 0.0
        phi_m = 0.0
        sum_ev2 = 0.0
        sum_ew2 = 0.0
        sum_es2 = 0.0
        sum_ephi2 = 0.0
        sum_edge_ev2 = 0.0
        sum_psi2 = 0.0
        sum_edge_w = 0.0
        vcmd0 = v_cmd[start]
        wcmd0 = w_cmd[start]
        ok = True

        for k in range(n - 1):
            kd = k - delay
            vc = v_cmd[start + kd] if kd >= 0 else vcmd0
            wc = w_cmd[start + kd] if kd >= 0 else wcmd0
            rho = 0.5 * (rho_arr[start + k] + rho_arr[start + k + 1])

            a00 = params[0] + rho * params[6]
            a01 = params[1] + rho * params[7]
            a10 = params[2] + rho * params[8]
            a11 = params[3] + rho * params[9]
            b0 = params[4] + rho * params[10]
            b1 = params[5] + rho * params[11]
            ad00, ad01, ad10, ad11, bd0, bd1, gd0, gd1 = _zoh_v_matrices(
                a00, a01, a10, a11, b0, b1, params[12], params[13], DT
            )

            nl = params[14] * _smooth_sgn(v_pred, SGN_EPS) + params[15] * v_pred * abs(w_pred)
            xh_next = ad00 * xh + ad01 * v_pred + bd0 * vc + gd0 * nl
            v_next = ad10 * xh + ad11 * v_pred + bd1 * vc + gd1 * nl

            lam = params[16] + rho * params[19]
            kw = params[17] + rho * params[20]
            cf = params[18] + rho * params[21]
            w_next = _w_step(lam, kw, cf, w_pred, wc, DT)

            vm_next = v_meas[start + k + 1]
            wm_next = w_meas[start + k + 1]
            s_p += 0.5 * DT * (prev_vp + v_next)
            s_m += 0.5 * DT * (prev_vm + vm_next)
            phi_p += 0.5 * DT * (prev_wp + w_next)
            phi_m += 0.5 * DT * (prev_wm + wm_next)

            if k + 1 >= burn:
                ev = v_next - vm_next
                ew = w_next - wm_next
                es = s_p - s_m
                ephi = phi_p - phi_m
                ewgt = edge_weight[start + k + 1]
                psi_proxy = params[25] + params[26] * xh_next + params[27] * v_next
                epsi = psi_proxy - leg_psi[start + k + 1]
                sum_ev2 += ev * ev
                sum_ew2 += ew * ew
                sum_es2 += es * es
                sum_ephi2 += ephi * ephi
                sum_edge_ev2 += ewgt * ev * ev
                sum_edge_w += ewgt
                sum_psi2 += epsi * epsi

            xh = xh_next
            v_pred = v_next
            w_pred = w_next
            prev_vp = v_next
            prev_vm = vm_next
            prev_wp = w_next
            prev_wm = wm_next

            if abs(v_next) > 30.0 or abs(w_next) > 80.0 or abs(xh_next) > 80.0:
                ok = False
                break
            if not (math.isfinite(v_next) and math.isfinite(w_next) and math.isfinite(xh_next)):
                ok = False
                break

        if not ok:
            losses[i] = 1e12
            invalid[i] = 1
        else:
            mse_v = sum_ev2 / n_eff
            mse_w = sum_ew2 / n_eff
            mse_s = sum_es2 / n_eff
            mse_phi = sum_ephi2 / n_eff
            mse_edge = sum_edge_ev2 / max(sum_edge_w, 1.0)
            mse_psi = sum_psi2 / n_eff
            losses[i] = (
                mse_v + 0.60 * mse_w
                + w_pos * mse_s / max(var_s[i], 1e-6)
                + w_yaw * mse_phi / max(var_phi[i], 1e-6)
                + w_transient * mse_edge
                + w_psi * mse_psi / max(var_psi[i], 1e-6)
            )

    if np.sum(invalid) > 0:
        return 1e12
    return float(np.mean(losses) + _regularization_penalty(params))


def fit_model(args: argparse.Namespace, packed: PackedSeries, series: Sequence[SeriesMPC]) -> Tuple[np.ndarray, float]:
    def obj(x: np.ndarray) -> float:
        return loss_packed(
            np.asarray(x, dtype=np.float64),
            packed.v,
            packed.w,
            packed.v_cmd,
            packed.w_cmd,
            packed.leg_psi,
            packed.rho,
            packed.edge_weight,
            packed.starts,
            packed.lens,
            packed.var_s,
            packed.var_phi,
            packed.var_psi,
            BURN,
            INPUT_DELAY,
            args.w_pos,
            args.w_yaw,
            args.w_transient,
            args.w_psi,
        )

    x_init = estimate_initial_guess(series)
    _ = obj(x_init)
    bounds = list(PARAM_BOUNDS)

    best_x = x_init.copy()
    best_f = float(obj(best_x))

    if args.de_maxiter > 0:
        print(f"Global search: differential_evolution maxiter={args.de_maxiter}, popsize={args.de_popsize}")
        de = differential_evolution(
            obj,
            bounds=bounds,
            maxiter=args.de_maxiter,
            popsize=args.de_popsize,
            polish=False,
            updating="immediate",
            workers=1,
            seed=args.seed,
            tol=0.015,
            recombination=0.75,
        )
        if float(de.fun) < best_f:
            best_x = np.asarray(de.x, dtype=float)
            best_f = float(de.fun)

    rng = np.random.default_rng(args.seed + 17)
    starts = [best_x, x_init]
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    for _ in range(max(0, args.random_starts)):
        jitter = rng.normal(0.0, 0.10, size=len(best_x))
        starts.append(np.clip(best_x + jitter, lo, hi))

    print("Local refinement: L-BFGS-B")
    for sx in starts:
        res = minimize(obj, sx, method="L-BFGS-B", bounds=bounds, options={"maxiter": args.local_maxiter, "ftol": 1e-9})
        if float(res.fun) < best_f:
            best_x = np.asarray(res.x, dtype=float)
            best_f = float(res.fun)

    return best_x, best_f


def simulate_series(params: np.ndarray, s: SeriesMPC) -> Dict[str, np.ndarray]:
    n = len(s.v)
    xh = np.empty(n, dtype=float)
    vp = np.empty(n, dtype=float)
    wp = np.empty(n, dtype=float)
    psi_proxy = np.empty(n, dtype=float)

    vp[0] = float(s.v[0])
    wp[0] = float(s.w[0])
    xh[0] = float(params[22] + params[23] * s.leg_psi[0] + params[24] * s.v[0])
    psi_proxy[0] = float(params[25] + params[26] * xh[0] + params[27] * vp[0])

    vcmd0 = float(s.v_cmd[0])
    wcmd0 = float(s.w_cmd[0])
    for k in range(n - 1):
        kd = k - INPUT_DELAY
        vc = float(s.v_cmd[kd]) if kd >= 0 else vcmd0
        wc = float(s.w_cmd[kd]) if kd >= 0 else wcmd0
        rho = float(0.5 * (s.rho[k] + s.rho[k + 1]))

        a00 = float(params[0] + rho * params[6])
        a01 = float(params[1] + rho * params[7])
        a10 = float(params[2] + rho * params[8])
        a11 = float(params[3] + rho * params[9])
        b0 = float(params[4] + rho * params[10])
        b1 = float(params[5] + rho * params[11])
        ad00, ad01, ad10, ad11, bd0, bd1, gd0, gd1 = _zoh_v_matrices(
            a00, a01, a10, a11, b0, b1, float(params[12]), float(params[13]), DT
        )
        nl = float(params[14] * math.tanh(vp[k] / SGN_EPS) + params[15] * vp[k] * abs(wp[k]))
        xh[k + 1] = ad00 * xh[k] + ad01 * vp[k] + bd0 * vc + gd0 * nl
        vp[k + 1] = ad10 * xh[k] + ad11 * vp[k] + bd1 * vc + gd1 * nl

        lam = float(params[16] + rho * params[19])
        kw = float(params[17] + rho * params[20])
        cf = float(params[18] + rho * params[21])
        wp[k + 1] = _w_step(lam, kw, cf, wp[k], wc, DT)
        psi_proxy[k + 1] = params[25] + params[26] * xh[k + 1] + params[27] * vp[k + 1]

    return {"v_pred": vp, "w_pred": wp, "x_h": xh, "psi_proxy": psi_proxy}


def rollout_metrics(params: np.ndarray, series: Sequence[SeriesMPC], burn: int = BURN) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for s in series:
        pred = simulate_series(params, s)
        vp = pred["v_pred"]
        wp = pred["w_pred"]
        ev = vp[burn:] - s.v[burn:]
        ew = wp[burn:] - s.w[burn:]
        sp = _cumintegrate(vp)
        sm = _cumintegrate(s.v)
        phip = _cumintegrate(wp)
        phim = _cumintegrate(s.w)
        psi_err = pred["psi_proxy"][burn:] - s.leg_psi[burn:]
        edge_w = compute_edge_weight(s.v_cmd)[burn:]
        edge_rmse_v = float(np.sqrt(np.average(ev * ev, weights=edge_w))) if ev.size else 0.0
        out.append(dict(
            name=s.name,
            tag=str(s.meta.get("tag", s.meta.get("scenario", ""))),
            rmse_v=float(np.sqrt(np.mean(ev * ev))) if ev.size else 0.0,
            rmse_w=float(np.sqrt(np.mean(ew * ew))) if ew.size else 0.0,
            mae_v=float(np.mean(np.abs(ev))) if ev.size else 0.0,
            mae_w=float(np.mean(np.abs(ew))) if ew.size else 0.0,
            rmse_s=float(np.sqrt(np.mean((sp[burn:] - sm[burn:]) ** 2))) if ev.size else 0.0,
            rmse_phi=float(np.sqrt(np.mean((phip[burn:] - phim[burn:]) ** 2))) if ew.size else 0.0,
            edge_rmse_v=edge_rmse_v,
            rmse_psi_proxy=float(np.sqrt(np.mean(psi_err * psi_err))) if psi_err.size else 0.0,
            z_min=float(np.nanmin(s.z)),
            z_max=float(np.nanmax(s.z)),
            rho_min=float(np.nanmin(s.rho)),
            rho_max=float(np.nanmax(s.rho)),
            s_pred=sp,
            s_meas=sm,
            phi_pred=phip,
            phi_meas=phim,
            **pred,
        ))
    return out


def aggregate_metrics(results: Sequence[Dict[str, object]], series: Sequence[SeriesMPC]) -> Dict[str, float]:
    ev_all: List[np.ndarray] = []
    ew_all: List[np.ndarray] = []
    vm_all: List[np.ndarray] = []
    wm_all: List[np.ndarray] = []
    for r, s in zip(results, series):
        ev_all.append(np.asarray(r["v_pred"], dtype=float)[BURN:] - s.v[BURN:])
        ew_all.append(np.asarray(r["w_pred"], dtype=float)[BURN:] - s.w[BURN:])
        vm_all.append(s.v[BURN:])
        wm_all.append(s.w[BURN:])
    ev = np.concatenate(ev_all) if ev_all else np.zeros(0)
    ew = np.concatenate(ew_all) if ew_all else np.zeros(0)
    vm = np.concatenate(vm_all) if vm_all else np.zeros(0)
    wm = np.concatenate(wm_all) if wm_all else np.zeros(0)
    return {
        "rmse_v": float(np.sqrt(np.mean(ev * ev))) if ev.size else 0.0,
        "rmse_w": float(np.sqrt(np.mean(ew * ew))) if ew.size else 0.0,
        "r2_v": float(1.0 - np.sum(ev * ev) / max(np.sum((vm - np.mean(vm)) ** 2), 1e-12)) if ev.size else 0.0,
        "r2_w": float(1.0 - np.sum(ew * ew) / max(np.sum((wm - np.mean(wm)) ** 2), 1e-12)) if ew.size else 0.0,
        "n_samples": float(ev.size),
    }


def observer_gains(params: np.ndarray, target_pole: float = 0.55) -> Dict[str, float]:
    a00, a01, a10, a11, _, _, _, _ = _zoh_v_matrices(
        float(params[0]), float(params[1]), float(params[2]), float(params[3]), 0.0, 0.0, 0.0, 0.0, DT
    )
    psi_gain = float(params[26])
    lv = (a00 - target_pole) / a10 if abs(a10) > 1e-8 else 0.0
    lpsi = 0.15 / psi_gain if abs(psi_gain) > 1e-8 else 0.0
    return {"target_pole_v_only": target_pole, "L_v": float(lv), "L_psi": float(lpsi), "Ad00_at_rho0": float(a00), "Ad10_at_rho0": float(a10)}


def format_matrix(mat: np.ndarray) -> str:
    return "\n".join("  [" + ", ".join(f"{v: .10e}" for v in row) + "]" for row in mat)


def generate_kinematic_model_yaml(
    params: np.ndarray,
    sched: Dict[str, float],
    *,
    obs_v_innovation_max: float = 0.25,
    obs_omega_innovation_max: float = 2.5,
    obs_psi_innovation_max: float = 0.35,
) -> str:
    obs = observer_gains(params)
    lines = [
        "/**:",
        "  ros__parameters:",
        "    kinematic_model:",
        "      # LPV scheduling: rho = clip((leg_h*cos(leg_psi) - z_ref) / z_scale, -rho_clip, rho_clip)",
        f"      z_ref: {sched['z_ref']}",
        f"      z_scale: {sched['z_scale']}",
        f"      rho_clip: {sched['rho_clip']}",
        "      # Smooth sign approximation epsilon used in tanh(x / sgn_eps)",
        f"      sgn_eps: {SGN_EPS}",
        "",
        "      # [x_h_dot, v_dot]^T = (Ac0 + rho*Ac1)[x_h, v]^T + (Bc0 + rho*Bc1) dv + Gnl*nl",
    ]
    for name in ["ca00", "ca01", "ca10", "ca11", "cb0", "cb1"]:
        lines.append(f"      {name}: {float(params[PARAM_NAMES.index(name)])}")
    lines.extend([
        "",
        "      # Rho slopes for the continuous-time LPV matrices",
    ])
    for name in ["dca00", "dca01", "dca10", "dca11", "dcb0", "dcb1"]:
        lines.append(f"      {name}: {float(params[PARAM_NAMES.index(name)])}")
    lines.extend([
        "",
        "      # Nonlinear term: nl = cf1*tanh(v/sgn_eps) + cf2*v*abs(w)",
    ])
    for name in ["gxh", "gv", "cf1", "cf2"]:
        lines.append(f"      {name}: {float(params[PARAM_NAMES.index(name)])}")
    lines.extend([
        "",
        "      # Yaw channel: w_dot = -(w_lam0 + rho*w_lam1) * w + (w_k0 + rho*w_k1) * dw - (w_cf0 + rho*w_cf1) * tanh(w/sgn_eps)",
    ])
    for name in ["w_lam0", "w_k0", "w_cf0", "w_lam1", "w_k1", "w_cf1"]:
        lines.append(f"      {name}: {float(params[PARAM_NAMES.index(name)])}")
    lines.extend([
        "",
        "      # Hidden-state initialization / leg_psi proxy observer",
    ])
    for name in ["xh0_bias", "xh0_psi", "xh0_v", "psi_bias", "psi_gain", "psi_v"]:
        lines.append(f"      {name}: {float(params[PARAM_NAMES.index(name)])}")
    lines.append(f"      obs_lv: {float(obs['L_v'])}")
    lines.append(f"      obs_lpsi: {float(obs['L_psi'])}")
    lines.extend([
        "",
        "      # Preliminary one-step innovation gates; validate on representative Mature/NORMAL logs.",
        f"      obs_v_innovation_max: {obs_v_innovation_max}",
        f"      obs_omega_innovation_max: {obs_omega_innovation_max}",
        f"      obs_psi_innovation_max: {obs_psi_innovation_max}",
    ])
    return "\n".join(lines)


def save_model_txt(
    out_path: Path,
    params: np.ndarray,
    loss_value: float,
    sched: Dict[str, float],
    agg: Dict[str, float],
    results: Sequence[Dict[str, object]],
) -> None:
    ac0 = np.array([
        [params[0], params[1], 0.0, params[4], 0.0],
        [params[2], params[3], 0.0, params[5], 0.0],
        [0.0, 0.0, -params[16], 0.0, params[17]],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=float)
    ac1 = np.array([
        [params[6], params[7], 0.0, params[10], 0.0],
        [params[8], params[9], 0.0, params[11], 0.0],
        [0.0, 0.0, -params[19], 0.0, params[20]],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=float)
    ad00, ad01, ad10, ad11, bd0, bd1, gd0, gd1 = _zoh_v_matrices(
        float(params[0]), float(params[1]), float(params[2]), float(params[3]), float(params[4]), float(params[5]), float(params[12]), float(params[13]), DT
    )
    lam0 = float(params[16])
    alpha_w0 = math.exp(-max(lam0, 1e-5) * DT)
    integ_w0 = (1.0 - alpha_w0) / max(lam0, 1e-5)
    obs = observer_gains(params)
    lines = [
        "Continuous-time LPV wheel-leg identification result",
        f"generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"dt: {DT}",
        f"mpc_rate_hz: {MPC_RATE_HZ}",
        f"input_delay_steps: {INPUT_DELAY}",
        "",
        "Predicted MPC state: [x_h, v, w]",
        "Identification/export state: [x_h, v, w, dv, dw] where dv/dw are one-step delayed commands",
        "Scheduling variable: z = leg_h * cos(leg_psi)",
        f"rho = clip((z - {sched['z_ref']:.10g}) / {sched['z_scale']:.10g}, -{sched['rho_clip']:.4g}, {sched['rho_clip']:.4g})",
        "Discretization during rollout: per-step exact ZOH using rho_mid = 0.5*(rho_k + rho_{k+1})",
        "leg_psi usage: auxiliary x_h observability/proxy only; not a predicted state and not a scheduling variable",
        "",
        f"objective: {loss_value:.10e}",
        "aggregate_metrics:",
    ]
    for k, v in agg.items():
        lines.append(f"  {k}: {v:.10g}")
    lines.extend(["", "parameters:"])
    for name, value in zip(PARAM_NAMES, params):
        lines.append(f"  {name}: {value:.12g}")
    lines.extend([
        "",
        "Continuous-time LPV equations:",
        "  [x_h_dot, v_dot]^T = (Ac0 + rho*Ac1)[x_h, v]^T + (Bc0 + rho*Bc1) dv + Gnl*nl",
        "  nl = cf1*tanh(v/SGN_EPS) + cf2*v*abs(w)",
        "  w_dot = -(w_lam0 + rho*w_lam1) * w + (w_k0 + rho*w_k1) * dw - (w_cf0 + rho*w_cf1) * tanh(w/SGN_EPS)",
        "  dv[k+1] = v_cmd[k], dw[k+1] = w_cmd[k]",
        "",
        "Ac0_at_rho_0:",
        format_matrix(ac0),
        "",
        "Ac1_rho_slope:",
        format_matrix(ac1),
        "",
        "Discrete snapshot at rho=0 for [x_h, v, w, dv, dw]:",
        format_matrix(np.array([
            [ad00, ad01, 0.0, bd0, 0.0],
            [ad10, ad11, 0.0, bd1, 0.0],
            [0.0, 0.0, alpha_w0, 0.0, integ_w0 * params[17]],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=float)),
        "",
        f"Nonlinear discrete gains at rho=0: Gnl_xh={gd0:.10e}, Gnl_v={gd1:.10e}, Gnl_w={(-integ_w0 * params[18]):.10e}",
        "",
        "Initial hidden-state model:",
        "  x_h[0] = xh0_bias + xh0_psi*leg_psi[0] + xh0_v*v[0]",
        "leg_psi proxy model:",
        "  leg_psi_hat = psi_bias + psi_gain*x_h + psi_v*v",
        "observer_report_for_future_MPC_integration:",
        "  xh_pred = model x_h prediction before correction",
        "  v_pred = model v prediction before correction",
        "  psi_proxy_pred = psi_bias + psi_gain*xh_pred + psi_v*v_pred",
        "  xh_hat_next = xh_pred + L_v*(v_meas-v_pred) + L_psi*(leg_psi_meas-psi_proxy_pred)",
    ])
    for k, v in obs.items():
        lines.append(f"  {k}: {v:.12g}")
    lines.extend(["", "per_file_summary:"])
    for r in results:
        lines.append(
            f"  {r['name']}: rmse_v={float(r['rmse_v']):.6g}, rmse_w={float(r['rmse_w']):.6g}, "
            f"rmse_s={float(r['rmse_s']):.6g}, edge_rmse_v={float(r['edge_rmse_v']):.6g}, "
            f"z=[{float(r['z_min']):.5g},{float(r['z_max']):.5g}]"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_metrics_csv(path: Path, results: Sequence[Dict[str, object]]) -> None:
    keys = [
        "name", "tag", "rmse_v", "rmse_w", "mae_v", "mae_w", "rmse_s", "rmse_phi",
        "edge_rmse_v", "rmse_psi_proxy", "z_min", "z_max", "rho_min", "rho_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})


def save_npz(path: Path, params: np.ndarray, sched: Dict[str, float], agg: Dict[str, float]) -> None:
    np.savez_compressed(
        path,
        params=params,
        param_names=np.asarray(PARAM_NAMES),
        schedule=json.dumps(sched),
        aggregate_metrics=json.dumps(agg),
        dt=DT,
        input_delay_steps=INPUT_DELAY,
        sgn_eps=SGN_EPS,
    )


def generate_plots(series: Sequence[SeriesMPC], results: Sequence[Dict[str, object]], plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    all_vm: List[np.ndarray] = []
    all_vp: List[np.ndarray] = []
    all_wm: List[np.ndarray] = []
    all_wp: List[np.ndarray] = []
    all_z: List[np.ndarray] = []

    for s, r in zip(series, results):
        vp = np.asarray(r["v_pred"], dtype=float)
        wp = np.asarray(r["w_pred"], dtype=float)
        xh = np.asarray(r["x_h"], dtype=float)
        psi_proxy = np.asarray(r["psi_proxy"], dtype=float)

        all_vm.append(s.v[BURN:])
        all_vp.append(vp[BURN:])
        all_wm.append(s.w[BURN:])
        all_wp.append(wp[BURN:])
        all_z.append(s.z[BURN:])

        fig = plt.figure(figsize=(15, 18))
        gs = fig.add_gridspec(6, 1, hspace=0.34)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(s.t, s.v_cmd, "k--", lw=0.9, alpha=0.65, label="v_cmd")
        ax.plot(s.t, s.v, "b", lw=1.1, label="v_meas")
        ax.plot(s.t, vp, "r", lw=1.1, label="v_rollout")
        ax.set_ylabel("v [m/s]")
        ax.set_title(f"{s.name}: v rollout, RMSE={float(r['rmse_v']):.4f}, edge={float(r['edge_rmse_v']):.4f}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(s.t, s.w_cmd, "k--", lw=0.9, alpha=0.65, label="w_cmd")
        ax.plot(s.t, s.w, "b", lw=1.1, label="w_meas")
        ax.plot(s.t, wp, "r", lw=1.1, label="w_rollout")
        ax.set_ylabel("w [rad/s]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[2, 0])
        ax.plot(s.t, vp - s.v, "r", lw=0.8, label="v error")
        ax.plot(s.t, wp - s.w, "m", lw=0.8, label="w error")
        ax.axhline(0.0, color="k", lw=0.6, ls="--")
        ax.set_ylabel("error")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        ax = fig.add_subplot(gs[3, 0])
        ax.plot(s.t, np.asarray(r["s_meas"], dtype=float), "b", lw=1.0, label="s_meas")
        ax.plot(s.t, np.asarray(r["s_pred"], dtype=float), "r--", lw=1.0, label="s_rollout")
        ax2 = ax.twinx()
        ax2.plot(s.t, s.z, "g", lw=0.8, alpha=0.6, label="z schedule")
        ax.set_ylabel("s [m]")
        ax2.set_ylabel("z [m]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")

        ax = fig.add_subplot(gs[4, 0])
        ax.plot(s.t, xh, "m", lw=1.0, label="x_h latent")
        ax2 = ax.twinx()
        ax2.plot(s.t, s.leg_psi, "b", lw=0.8, alpha=0.7, label="leg_psi")
        ax2.plot(s.t, psi_proxy, "r--", lw=0.8, alpha=0.8, label="psi_proxy(x_h,v)")
        ax.set_ylabel("x_h")
        ax2.set_ylabel("leg_psi [rad]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        ax2.legend(fontsize=8, loc="upper right")

        ax = fig.add_subplot(gs[5, 0])
        ax.plot(s.t, s.leg_h, "c", lw=0.9, label="leg_h")
        ax.plot(s.t, s.z, "g", lw=0.9, label="leg_h*cos(leg_psi)")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("leg / schedule [m]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(plots_dir / f"{s.name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    vm = np.concatenate(all_vm) if all_vm else np.zeros(0)
    vp = np.concatenate(all_vp) if all_vp else np.zeros(0)
    wm = np.concatenate(all_wm) if all_wm else np.zeros(0)
    wp = np.concatenate(all_wp) if all_wp else np.zeros(0)
    zz = np.concatenate(all_z) if all_z else np.zeros(0)

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1)
    sc = ax.scatter(vm, vp, c=zz, s=6, alpha=0.45, cmap="viridis")
    lim = [min(float(np.min(vm)), float(np.min(vp))) if vm.size else -1.0, max(float(np.max(vm)), float(np.max(vp))) if vm.size else 1.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("v_meas [m/s]")
    ax.set_ylabel("v_rollout [m/s]")
    ax.set_title("Velocity fit scatter")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="z = leg_h*cos(leg_psi) [m]")

    ax = fig.add_subplot(1, 2, 2)
    sc = ax.scatter(wm, wp, c=zz, s=6, alpha=0.45, cmap="viridis")
    lim = [min(float(np.min(wm)), float(np.min(wp))) if wm.size else -1.0, max(float(np.max(wm)), float(np.max(wp))) if wm.size else 1.0]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("w_meas [rad/s]")
    ax.set_ylabel("w_rollout [rad/s]")
    ax.set_title("Yaw-rate fit scatter")
    ax.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax, label="z = leg_h*cos(leg_psi) [m]")
    fig.savefig(plots_dir / "scatter_fit.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Identify continuous-time LPV MIMO model from rec_identify_data.py logs")
    ap.add_argument("--data-dir", type=str, default="identify_data")
    ap.add_argument("--out-dir", type=str, default="identify_lpv_result")
    ap.add_argument("--cutoff-hz", type=float, default=8.0, help="low-pass cutoff before downsampling; <=0 disables filtering")
    ap.add_argument("--raw-to-mpc-q", type=int, default=0, help="manual downsample factor; 0 auto from timestamp")
    ap.add_argument("--rho-clip", type=float, default=1.5, help="clipping for scheduling variable rho to avoid excessive extrapolation")
    ap.add_argument("--w-pos", type=float, default=1.0, help="weight for displacement multi-step error")
    ap.add_argument("--w-yaw", type=float, default=0.2, help="weight for heading integral error")
    ap.add_argument("--w-transient", type=float, default=0.8, help="extra v error weight around v_cmd edges")
    ap.add_argument("--w-psi", type=float, default=0.2, help="weak auxiliary leg_psi proxy loss for x_h observability")
    ap.add_argument("--obs-v-innovation-max", type=float, default=0.25, help="preliminary observer velocity innovation gate in m/s")
    ap.add_argument("--obs-omega-innovation-max", type=float, default=2.5, help="preliminary observer yaw-rate innovation gate in rad/s")
    ap.add_argument("--obs-psi-innovation-max", type=float, default=0.35, help="preliminary observer leg-angle proxy innovation gate in rad")
    ap.add_argument("--de-maxiter", type=int, default=100, help="differential evolution iterations; 0 skips global search")
    ap.add_argument("--de-popsize", type=int, default=32, help="differential evolution population size")
    ap.add_argument("--local-maxiter", type=int, default=800, help="local optimization iterations for each random start")
    ap.add_argument("--random-starts", type=int, default=4, help="number of random jittered starts around the best DE result for local refinement")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility of global search and random starts")
    args = ap.parse_args()
    innovation_gates = (
        args.obs_v_innovation_max,
        args.obs_omega_innovation_max,
        args.obs_psi_innovation_max,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in innovation_gates):
        ap.error("observer innovation gates must be finite and positive")

    data_dir = Path(args.data_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / stamp
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_q = None if args.raw_to_mpc_q <= 0 else args.raw_to_mpc_q
    series, sched = load_all_npz_to_mpc_rate(data_dir, cutoff_hz=args.cutoff_hz, raw_to_mpc_q=raw_q, rho_clip=args.rho_clip)
    print(f"Loaded {len(series)} files, {sum(len(s.v) for s in series)} samples at {MPC_RATE_HZ} Hz")
    print(f"Schedule z_ref={sched['z_ref']:.5f}, z_scale={sched['z_scale']:.5f}, z_p10={sched['z_p10']:.5f}, z_p90={sched['z_p90']:.5f}")

    packed = pack_series(series)
    params, loss_value = fit_model(args, packed, series)
    results = rollout_metrics(params, series)
    agg = aggregate_metrics(results, series)

    save_model_txt(out_dir / f"model_lpv_{MPC_RATE_HZ}hz.txt", params, loss_value, sched, agg, results)
    save_metrics_csv(out_dir / "per_file_metrics.csv", results)
    save_npz(out_dir / f"model_lpv_{MPC_RATE_HZ}hz.npz", params, sched, agg)
    yaml_str = generate_kinematic_model_yaml(
        params,
        sched,
        obs_v_innovation_max=args.obs_v_innovation_max,
        obs_omega_innovation_max=args.obs_omega_innovation_max,
        obs_psi_innovation_max=args.obs_psi_innovation_max,
    )
    (out_dir / "kinematic_model.yaml").write_text(yaml_str + "\n", encoding="utf-8")
    generate_plots(series, results, plots_dir)

    print("Done")
    print(f"  result_dir: {out_dir}")
    print(f"  model_txt:  {out_dir / f'model_lpv_{MPC_RATE_HZ}hz.txt'}")
    print(f"  metrics:    {out_dir / 'per_file_metrics.csv'}")
    print(f"  YAML:       {out_dir / 'kinematic_model.yaml'}")
    print(f"  plots:      {plots_dir}")
    print(f"  aggregate:  rmse_v={agg['rmse_v']:.5f}, rmse_w={agg['rmse_w']:.5f}, r2_v={agg['r2_v']:.4f}, r2_w={agg['r2_w']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
