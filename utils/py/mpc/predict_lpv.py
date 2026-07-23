#!/usr/bin/env python3
"""Predict and visualize LPV model on a single NPZ recording.

Reads a kinematic_model.yaml (exported by identify_lpv_mimo.py) and a single
.npz file (recorded by rec_identify_data.py), runs the continuous-time LPV
rollout, then generates diagnostic plots and summary metrics.

Usage:
  python3 predict_lpv.py --model path/to/kinematic_model.yaml --npz path/to/data.npz
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    raise RuntimeError("PyYAML is required. Install with: uv pip install pyyaml")

try:
    from scipy import signal
except ImportError as exc:
    raise RuntimeError("scipy is required. Install with: uv pip install scipy") from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("matplotlib is required. Install with: uv pip install matplotlib") from exc


DT_DEFAULT = 0.05
INPUT_DELAY_DEFAULT = 1
SGN_EPS_DEFAULT = 0.05
BURN_DEFAULT = 10

PARAM_NAMES = [
    "ca00", "ca01", "ca10", "ca11", "cb0", "cb1",
    "dca00", "dca01", "dca10", "dca11", "dcb0", "dcb1",
    "gxh", "gv", "cf1", "cf2",
    "w_lam0", "w_k0", "w_cf0", "w_lam1", "w_k1", "w_cf1",
    "xh0_bias", "xh0_psi", "xh0_v",
    "psi_bias", "psi_gain", "psi_v",
]


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _cumintegrate(y: np.ndarray, dt: float) -> np.ndarray:
    y = _as_1d(y)
    out = np.zeros_like(y)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
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


def robust_fs_from_t(t: np.ndarray) -> float:
    t = _as_1d(t)
    if len(t) < 3:
        return 20.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if len(dt) == 0:
        return 20.0
    return 1.0 / max(float(np.median(dt)), 1e-6)


def smooth_sgn(x: float, eps: float) -> float:
    return math.tanh(x / eps)


def zoh_v_matrices(
    a00: float, a01: float, a10: float, a11: float,
    b0: float, b1: float, g0: float, g1: float,
    dt: float,
) -> Tuple[float, ...]:
    m00 = a00 * dt
    m01 = a01 * dt
    m10 = a10 * dt
    m11 = a11 * dt
    tr_m = m00 + m11
    det_m = m00 * m11 - m01 * m10
    disc = tr_m * tr_m - 4.0 * det_m
    eps_m = 1e-12

    if disc > eps_m:
        s = math.sqrt(disc)
        lam1 = 0.5 * (tr_m + s)
        lam2 = 0.5 * (tr_m - s)
        e1 = math.exp(lam1)
        e2 = math.exp(lam2)
        beta = (e1 - e2) / (lam1 - lam2)
        alpha = e1 - beta * lam1
    elif disc < -eps_m:
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


def w_step(lambda_w: float, k_w: float, c_w: float, w: float, wc: float, dt: float, sgn_eps: float) -> float:
    lam = max(lambda_w, 1e-5)
    alpha = math.exp(-lam * dt)
    integ = (1.0 - alpha) / lam
    return alpha * w + integ * (k_w * wc - c_w * smooth_sgn(w, sgn_eps))


def load_kinematic_model(yaml_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    km = raw.get("/**", {}).get("ros__parameters", {}).get("kinematic_model", {})
    if not km:
        raise ValueError(f"Could not find /**.ros__parameters.kinematic_model in {yaml_path}")

    param_names_set = set(PARAM_NAMES)
    params = {}
    for k, v in km.items():
        if k in param_names_set:
            params[k] = float(v)

    missing = param_names_set - set(params.keys())
    if missing:
        raise ValueError(f"Missing parameters in YAML: {missing}")

    sched = {
        "z_ref": float(km.get("z_ref", 0.0)),
        "z_scale": float(km.get("z_scale", 0.1)),
        "rho_clip": float(km.get("rho_clip", 1.5)),
        "sgn_eps": float(km.get("sgn_eps", SGN_EPS_DEFAULT)),
    }

    return params, sched


def load_and_preprocess_npz(
    npz_path: Path,
    sched: Dict[str, float],
    *,
    dt: float = DT_DEFAULT,
    cutoff_hz: float = 8.0,
    raw_to_mpc_q: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as zf:
        required = ["t", "v_meas", "w_meas", "v_cmd", "w_cmd", "leg_h_meas", "leg_psi_meas"]
        missing = [k for k in required if k not in zf]
        if missing:
            raise ValueError(f"NPZ missing keys: {missing}")

        t_raw = _as_1d(zf["t"])
        v_raw = _as_1d(zf["v_meas"])
        w_raw = _as_1d(zf["w_meas"])
        vc_raw = _as_1d(zf["v_cmd"])
        wc_raw = _as_1d(zf["w_cmd"])
        h_raw = _as_1d(zf["leg_h_meas"])
        psi_raw = _as_1d(zf["leg_psi_meas"])
        s_raw = _as_1d(zf["s_meas"]) if "s_meas" in zf else None
        yaw_raw = _as_1d(zf["yaw_meas"]) if "yaw_meas" in zf else None

    fs = robust_fs_from_t(t_raw)
    q = max(1, round(fs * dt)) if raw_to_mpc_q is None else max(1, int(raw_to_mpc_q))
    kw = dict(fs_hz=fs, cutoff_hz=cutoff_hz, q=q)

    t = t_raw[::q].copy()
    if t.size > 0:
        t -= float(t[0])

    v = lowpass_then_downsample(v_raw, **kw)
    w = lowpass_then_downsample(w_raw, **kw)
    vc = lowpass_then_downsample(vc_raw, **kw)
    wc = lowpass_then_downsample(wc_raw, **kw)
    h = lowpass_then_downsample(h_raw, **kw)
    psi = lowpass_then_downsample(psi_raw, **kw)
    s_meas = lowpass_then_downsample(s_raw, **kw) if s_raw is not None else None
    yaw = lowpass_then_downsample(yaw_raw, **kw) if yaw_raw is not None else None

    n = min(len(t), len(v), len(w), len(vc), len(wc), len(h), len(psi))
    t = t[:n]
    v = v[:n]
    w = w[:n]
    vc = vc[:n]
    wc = wc[:n]
    h = h[:n]
    psi = psi[:n]
    if s_meas is not None:
        s_meas = s_meas[:n]
    if yaw is not None:
        yaw = yaw[:n]

    z = h * np.cos(psi)
    rho = np.clip((z - sched["z_ref"]) / sched["z_scale"], -sched["rho_clip"], sched["rho_clip"])

    result = {
        "t": t, "v": v, "w": w,
        "v_cmd": vc, "w_cmd": wc,
        "leg_h": h, "leg_psi": psi, "z": z, "rho": rho,
    }
    if s_meas is not None:
        result["s_meas"] = s_meas
    if yaw is not None:
        result["yaw"] = yaw
    return result


def predict(
    params: Dict[str, float],
    data: Dict[str, np.ndarray],
    *,
    dt: float = DT_DEFAULT,
    input_delay: int = INPUT_DELAY_DEFAULT,
    sgn_eps: float = SGN_EPS_DEFAULT,
) -> Dict[str, np.ndarray]:
    n = len(data["v"])
    xh = np.empty(n, dtype=float)
    vp = np.empty(n, dtype=float)
    wp = np.empty(n, dtype=float)
    psi_proxy = np.empty(n, dtype=float)

    vp[0] = float(data["v"][0])
    wp[0] = float(data["w"][0])
    xh[0] = float(params["xh0_bias"] + params["xh0_psi"] * data["leg_psi"][0]
                  + params["xh0_v"] * data["v"][0])
    psi_proxy[0] = float(params["psi_bias"] + params["psi_gain"] * xh[0]
                         + params["psi_v"] * vp[0])

    vcmd0 = float(data["v_cmd"][0])
    wcmd0 = float(data["w_cmd"][0])
    for k in range(n - 1):
        kd = k - input_delay
        vc = float(data["v_cmd"][kd]) if kd >= 0 else vcmd0
        wc = float(data["w_cmd"][kd]) if kd >= 0 else wcmd0
        rho = float(0.5 * (data["rho"][k] + data["rho"][k + 1]))

        a00 = float(params["ca00"] + rho * params["dca00"])
        a01 = float(params["ca01"] + rho * params["dca01"])
        a10 = float(params["ca10"] + rho * params["dca10"])
        a11 = float(params["ca11"] + rho * params["dca11"])
        b0 = float(params["cb0"] + rho * params["dcb0"])
        b1 = float(params["cb1"] + rho * params["dcb1"])
        ad00, ad01, ad10, ad11, bd0, bd1, gd0, gd1 = zoh_v_matrices(
            a00, a01, a10, a11, b0, b1,
            float(params["gxh"]), float(params["gv"]), dt,
        )
        nl = float(params["cf1"] * smooth_sgn(vp[k], sgn_eps)
                   + params["cf2"] * vp[k] * abs(wp[k]))
        xh[k + 1] = ad00 * xh[k] + ad01 * vp[k] + bd0 * vc + gd0 * nl
        vp[k + 1] = ad10 * xh[k] + ad11 * vp[k] + bd1 * vc + gd1 * nl

        lam = float(params["w_lam0"] + rho * params["w_lam1"])
        kw = float(params["w_k0"] + rho * params["w_k1"])
        cf = float(params["w_cf0"] + rho * params["w_cf1"])
        wp[k + 1] = w_step(lam, kw, cf, wp[k], wc, dt, sgn_eps)
        psi_proxy[k + 1] = (params["psi_bias"] + params["psi_gain"] * xh[k + 1]
                            + params["psi_v"] * vp[k + 1])

    return {"v_pred": vp, "w_pred": wp, "x_h": xh, "psi_proxy": psi_proxy}


def compute_metrics(
    data: Dict[str, np.ndarray],
    pred: Dict[str, np.ndarray],
    burn: int = BURN_DEFAULT,
    dt: float = DT_DEFAULT,
) -> Dict[str, float]:
    vp = pred["v_pred"]
    wp = pred["w_pred"]
    n_eff = len(data["v"]) - burn
    if n_eff <= 0:
        return {}

    ev = vp[burn:] - data["v"][burn:]
    ew = wp[burn:] - data["w"][burn:]
    sp = _cumintegrate(vp, dt)
    sm = _cumintegrate(data["v"], dt)
    phip = _cumintegrate(wp, dt)
    phim = _cumintegrate(data["w"], dt)
    psi_err = pred["psi_proxy"][burn:] - data["leg_psi"][burn:]
    edge_w = compute_edge_weight(data["v_cmd"])[burn:]
    edge_norm = max(float(np.sum(edge_w)), 1e-12)

    vm = data["v"][burn:]
    wm = data["w"][burn:]

    return {
        "n_samples": float(n_eff),
        "rmse_v": float(np.sqrt(np.mean(ev * ev))),
        "rmse_w": float(np.sqrt(np.mean(ew * ew))),
        "mae_v": float(np.mean(np.abs(ev))),
        "mae_w": float(np.mean(np.abs(ew))),
        "rmse_s": float(np.sqrt(np.mean((sp[burn:] - sm[burn:]) ** 2))),
        "rmse_phi": float(np.sqrt(np.mean((phip[burn:] - phim[burn:]) ** 2))),
        "edge_rmse_v": float(np.sqrt(np.average(ev * ev, weights=edge_w))),
        "rmse_psi_proxy": float(np.sqrt(np.mean(psi_err * psi_err))),
        "max_abs_ev": float(np.max(np.abs(ev))),
        "max_abs_ew": float(np.max(np.abs(ew))),
        "p95_abs_ev": float(np.percentile(np.abs(ev), 95)),
        "p95_abs_ew": float(np.percentile(np.abs(ew), 95)),
        "r2_v": float(1.0 - np.sum(ev * ev) / max(np.sum((vm - np.mean(vm)) ** 2), 1e-12)),
        "r2_w": float(1.0 - np.sum(ew * ew) / max(np.sum((wm - np.mean(wm)) ** 2), 1e-12)),
        "mean_ev": float(np.mean(ev)),
        "mean_ew": float(np.mean(ew)),
        "std_ev": float(np.std(ev)),
        "std_ew": float(np.std(ew)),
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    print("=" * 60)
    print("Prediction Metrics")
    print("=" * 60)
    print(f"  Samples (after burn):     {metrics.get('n_samples', 0):.0f}")
    print(f"  v RMSE:                   {metrics.get('rmse_v', 0):.6f}  [m/s]")
    print(f"  w RMSE:                   {metrics.get('rmse_w', 0):.6f}  [rad/s]")
    print(f"  v MAE:                    {metrics.get('mae_v', 0):.6f}  [m/s]")
    print(f"  w MAE:                    {metrics.get('mae_w', 0):.6f}  [rad/s]")
    print(f"  v R²:                     {metrics.get('r2_v', 0):.4f}")
    print(f"  w R²:                     {metrics.get('r2_w', 0):.4f}")
    print(f"  v bias (mean error):      {metrics.get('mean_ev', 0):.6f}  [m/s]")
    print(f"  w bias (mean error):      {metrics.get('mean_ew', 0):.6f}  [rad/s]")
    print(f"  v error std:              {metrics.get('std_ev', 0):.6f}  [m/s]")
    print(f"  w error std:              {metrics.get('std_ew', 0):.6f}  [rad/s]")
    print(f"  v max |error|:            {metrics.get('max_abs_ev', 0):.6f}  [m/s]")
    print(f"  w max |error|:            {metrics.get('max_abs_ew', 0):.6f}  [rad/s]")
    print(f"  v 95% |error|:            {metrics.get('p95_abs_ev', 0):.6f}  [m/s]")
    print(f"  w 95% |error|:            {metrics.get('p95_abs_ew', 0):.6f}  [rad/s]")
    print(f"  s (displacement) RMSE:    {metrics.get('rmse_s', 0):.6f}  [m]")
    print(f"  heading RMSE:             {metrics.get('rmse_phi', 0):.6f}  [rad]")
    print(f"  edge-weighted v RMSE:     {metrics.get('edge_rmse_v', 0):.6f}  [m/s]")
    print(f"  leg_psi proxy RMSE:       {metrics.get('rmse_psi_proxy', 0):.6f}  [rad]")
    print("-" * 60)


def generate_plots(
    data: Dict[str, np.ndarray],
    pred: Dict[str, np.ndarray],
    metrics: Dict[str, float],
    out_base: Path,
    title_tag: str = "",
    burn: int = BURN_DEFAULT,
    dt: float = DT_DEFAULT,
) -> None:
    t = data["t"]
    vp = pred["v_pred"]
    wp = pred["w_pred"]
    xh = pred["x_h"]
    psi_proxy = pred["psi_proxy"]
    vm = data["v"]
    wm = data["w"]
    sp = _cumintegrate(vp, dt)
    sm = _cumintegrate(vm, dt)
    phip = _cumintegrate(wp, dt)
    phim = _cumintegrate(wm, dt)
    z = data.get("z", t * 0)

    fig = plt.figure(figsize=(16, 22))
    title = f"LPV Model Prediction{f' — {title_tag}' if title_tag else ''}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(8, 1, hspace=0.35)

    # 1 — v
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, data["v_cmd"], "k--", lw=0.9, alpha=0.65, label="v_cmd")
    ax.plot(t, vm, "b", lw=1.1, label="v_meas")
    ax.plot(t, vp, "r", lw=1.1, label="v_pred")
    ax.set_ylabel("v [m/s]")
    ax.set_title(f"Velocity — RMSE={metrics.get('rmse_v', 0):.4f}, "
                 f"R²={metrics.get('r2_v', 0):.4f}, "
                 f"edge-RMSE={metrics.get('edge_rmse_v', 0):.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # 2 — w
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, data["w_cmd"], "k--", lw=0.9, alpha=0.65, label="w_cmd")
    ax.plot(t, wm, "b", lw=1.1, label="w_meas")
    ax.plot(t, wp, "r", lw=1.1, label="w_pred")
    ax.set_ylabel("w [rad/s]")
    ax.set_title(f"Yaw Rate — RMSE={metrics.get('rmse_w', 0):.4f}, "
                 f"R²={metrics.get('r2_w', 0):.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # 3 — errors
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, vp - vm, "r", lw=0.8, label="v error")
    ax.plot(t, wp - wm, "m", lw=0.8, label="w error")
    ax.axhline(0.0, color="k", lw=0.6, ls="--")
    ax.axhline(metrics.get("rmse_v", 0), color="r", lw=0.5, ls=":", alpha=0.6,
               label=f"±v RMSE={metrics.get('rmse_v', 0):.3f}")
    ax.axhline(-metrics.get("rmse_v", 0), color="r", lw=0.5, ls=":", alpha=0.6)
    ax.set_ylabel("error")
    ax.set_title(f"Prediction Errors — v MAE={metrics.get('mae_v', 0):.4f}, "
                 f"w MAE={metrics.get('mae_w', 0):.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # 4 — displacement
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t, sm, "b", lw=1.0, label="s_meas (integrated v)")
    ax.plot(t, sp, "r--", lw=1.0, label="s_pred (integrated v_pred)")
    ax2 = ax.twinx()
    ax2.plot(t, z, "g", lw=0.8, alpha=0.6, label="z schedule")
    ax.set_ylabel("s [m]")
    ax2.set_ylabel("z [m]")
    ax.set_title(f"Displacement — RMSE={metrics.get('rmse_s', 0):.4f} m")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")

    # 5 — heading
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(t, phim, "b", lw=1.0, label="heading_meas (integrated w)")
    ax.plot(t, phip, "r--", lw=1.0, label="heading_pred (integrated w_pred)")
    ax.set_ylabel("heading [rad]")
    ax.set_title(f"Heading — RMSE={metrics.get('rmse_phi', 0):.4f} rad")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # 6 — x_h + leg_psi proxy
    ax = fig.add_subplot(gs[5, 0])
    ax.plot(t, xh, "m", lw=1.0, label="x_h latent")
    ax2 = ax.twinx()
    ax2.plot(t, data["leg_psi"], "b", lw=0.8, alpha=0.7, label="leg_psi_meas")
    ax2.plot(t, psi_proxy, "r--", lw=0.8, alpha=0.8, label="psi_proxy(x_h, v)")
    ax.set_ylabel("x_h")
    ax2.set_ylabel("leg_psi [rad]")
    ax.set_title(f"Leg Psi Proxy — RMSE={metrics.get('rmse_psi_proxy', 0):.4f} rad")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")

    # 7 — leg / schedule
    ax = fig.add_subplot(gs[6, 0])
    ax.plot(t, data["leg_h"], "c", lw=0.9, label="leg_h")
    ax.plot(t, data["leg_psi"], "b", lw=0.9, label="leg_psi")
    ax.plot(t, z, "g", lw=0.9, label="leg_h*cos(leg_psi) = z")
    ax.set_ylabel("leg / schedule")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    # 8 — rho schedule
    ax = fig.add_subplot(gs[7, 0])
    ax.plot(t, data["rho"], "purple", lw=0.9, label="rho")
    ax.axhline(0.0, color="k", lw=0.5, ls="--")
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.axhline(-1.0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("rho")
    ax.set_title(f"Scheduling variable rho — "
                 f"min={float(np.min(data['rho'])):.3f}, "
                 f"max={float(np.max(data['rho'])):.3f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.savefig(out_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Time series: {out_base.with_suffix('.png')}")

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"Fit Scatter{f' — {title_tag}' if title_tag else ''}", fontsize=13)

    for ax, xm, xp, name in [
        (axes[0], vm[burn:], vp[burn:], "v"),
        (axes[1], wm[burn:], wp[burn:], "w"),
    ]:
        zc = data["z"][burn:] if "z" in data else np.zeros(len(xm))
        sc = ax.scatter(xm, xp, c=zc, s=6, alpha=0.45, cmap="viridis")
        lo = min(float(np.min(xm)), float(np.min(xp))) if xm.size else -1.0
        hi = max(float(np.max(xm)), float(np.max(xp))) if xm.size else 1.0
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"{name}_meas")
        ax.set_ylabel(f"{name}_pred")
        ax.set_title(f"{name} — R²={metrics.get(f'r2_{name}', 0):.4f}")
        ax.grid(True, alpha=0.25)
        fig.colorbar(sc, ax=ax, label="z [m]")

    fig.savefig(out_base.with_name(f"{out_base.stem}_scatter.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter:    {out_base.with_name(f'{out_base.stem}_scatter.png')}")

    # Error histogram
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Error Distribution{f' — {title_tag}' if title_tag else ''}", fontsize=13)

    ev_b = vp[burn:] - vm[burn:]
    ew_b = wp[burn:] - wm[burn:]
    for ax, err, label in [
        (axes[0], ev_b, "v error [m/s]"),
        (axes[1], ew_b, "w error [rad/s]"),
    ]:
        ax.hist(err, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        mu = float(np.mean(err))
        sd = float(np.std(err))
        ax.axvline(mu, color="r", lw=1.2, ls=":", label=f"mean={mu:.4f}")
        ax.axvline(mu - sd, color="orange", lw=0.8, ls=":", label=f"±1σ={sd:.4f}")
        ax.axvline(mu + sd, color="orange", lw=0.8, ls=":")
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"μ={mu:.4f}, σ={sd:.4f}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_base.with_name(f"{out_base.stem}_error_dist.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Error dist: {out_base.with_name(f'{out_base.stem}_error_dist.png')}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Predict and visualize LPV model on a single NPZ recording."
    )
    ap.add_argument("--model", type=str, required=True,
                    help="Path to kinematic_model.yaml from identify_lpv_mimo.py")
    ap.add_argument("--npz", type=str, required=True,
                    help="Path to a single .npz file from rec_identify_data.py")
    ap.add_argument("--out-dir", type=str, default="predict_lpv_result",
                    help="Output directory for plots and metrics")
    ap.add_argument("--dt", type=float, default=DT_DEFAULT,
                    help=f"MPC timestep (default {DT_DEFAULT})")
    ap.add_argument("--input-delay", type=int, default=INPUT_DELAY_DEFAULT,
                    help=f"Input delay steps (default {INPUT_DELAY_DEFAULT})")
    ap.add_argument("--burn", type=int, default=BURN_DEFAULT,
                    help=f"Samples to discard from start for metrics (default {BURN_DEFAULT})")
    ap.add_argument("--cutoff-hz", type=float, default=8.0,
                    help="Lowpass cutoff for downsampling (default 8.0); ≤0 to disable")
    ap.add_argument("--raw-to-mpc-q", type=int, default=0,
                    help="Manual downsample factor; 0 = auto from timestamps")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional tag for plot titles")
    args = ap.parse_args()

    model_path = Path(args.model)
    npz_path = Path(args.npz)
    if not model_path.exists():
        print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
        return 1
    if not npz_path.exists():
        print(f"ERROR: npz file not found: {npz_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = float(args.dt)
    input_delay = int(args.input_delay)
    burn = int(args.burn)

    print(f"Loading model from: {model_path}")
    params, sched = load_kinematic_model(model_path)
    print(f"  z_ref={sched['z_ref']:.5g}, z_scale={sched['z_scale']:.5g}, "
          f"rho_clip={sched['rho_clip']:.4g}")
    sgn_eps = sched["sgn_eps"]
    param_arr = np.array([params[n] for n in PARAM_NAMES], dtype=float)
    print(f"  params: min={float(np.min(param_arr)):.4g}, "
          f"max={float(np.max(param_arr)):.4g}, "
          f"norm={float(np.linalg.norm(param_arr)):.4g}")

    print(f"Loading data from: {npz_path}")
    raw_q = None if args.raw_to_mpc_q <= 0 else args.raw_to_mpc_q
    data = load_and_preprocess_npz(
        npz_path, sched, dt=dt, cutoff_hz=args.cutoff_hz, raw_to_mpc_q=raw_q,
    )
    n = len(data["t"])
    t_span = data["t"][-1] - data["t"][0] if n > 1 else 0.0
    print(f"  {n} samples, {t_span:.1f} s @ {round(1.0 / dt)} Hz")

    print("Running model prediction (free rollout)...")
    pred = predict(params, data, dt=dt, input_delay=input_delay, sgn_eps=sgn_eps)

    print("Computing metrics...")
    metrics = compute_metrics(data, pred, burn=burn, dt=dt)
    print_metrics(metrics)

    out_base = out_dir / f"{npz_path.stem}_predict"
    print(f"Generating plots in: {out_dir}")
    generate_plots(
        data, pred, metrics, out_base,
        title_tag=args.tag or npz_path.stem,
        burn=burn, dt=dt,
    )

    metrics_path = out_base.with_suffix(".txt")
    with open(metrics_path, "w") as f:
        f.write(f"model: {model_path}\n")
        f.write(f"npz: {npz_path}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"input_delay: {input_delay}\n")
        f.write(f"burn: {burn}\n")
        f.write(f"sgn_eps: {sgn_eps}\n")
        f.write(f"n_samples: {n}\n\n")
        f.write("Metrics:\n")
        for k, v in sorted(metrics.items()):
            f.write(f"  {k}: {v:.10g}\n")
    print(f"  Metrics:    {metrics_path}")

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
