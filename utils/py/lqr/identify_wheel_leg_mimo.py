#!/usr/bin/env python3
"""End-to-end identification for wheel-leg v/w command-to-measure dynamics (for MPC).

Data:
  - Raw recordings are in identify_data/*.npz (recorded by identify_wheel_leg_record.py)
  - Raw sample rate ~20Hz (dt~0.05s)
  - MPC runs at 10Hz (dt_mpc=0.1s)

Pipeline:
  1) Anti-alias low-pass filter then downsample 20Hz -> 10Hz
  2) Delay alignment: base comm delay L=1 MPC step (0.1s)
     We model as a 1-step input delay state (no command-history backtracking).
  3) Decoupled identification:
     - w channel: first-order lag, stable, DC gain ~ 1
     - v channel: ARX structure search
         n in [1..5]
         coupling candidates: w, |w|, w^2, v*w
         criterion: BIC
         constraint: spectral radius rho(A_v) < 1.0
  4) Export 10Hz MIMO state-space A,B,C,D for MPC
  5) Predict on downsampled data, plot per dataset

Outputs:
  Saved to identify_result/<run_stamp>/
    - leaderboard.csv
    - per_file_metrics.csv
    - model_mimo_10hz.npz (A,B,C,D + metadata)
    - model_mimo_10hz.txt (human-readable)
    - plots/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import signal  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scipy (scipy.signal) to run.") from exc


NPZ_KEYS = {
    "t": "t",
    "v": "v_meas",
    "w": "w_meas",
    "v_cmd": "v_cmd",
    "w_cmd": "w_cmd",
    "meta": "meta",
}


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
    x = np.asarray(x).astype(float)
    return x.reshape(-1)


def butter_lowpass_sos(cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs_hz
    if cutoff_hz <= 0.0 or cutoff_hz >= nyq:
        raise ValueError(f"Invalid cutoff_hz={cutoff_hz} for fs_hz={fs_hz}")
    wn = cutoff_hz / nyq
    return signal.butter(order, wn, btype="low", output="sos")


def lowpass_then_downsample(x: np.ndarray, *, fs_hz: float, cutoff_hz: float, q: int) -> np.ndarray:
    """Anti-alias low-pass + downsample by integer factor q.

    Uses zero-phase SOS filtering (sosfiltfilt) to avoid phase distortion.
    """
    x = _as_1d(x)
    if len(x) < 8 * q:
        return x[::q].copy()
    sos = butter_lowpass_sos(cutoff_hz=cutoff_hz, fs_hz=fs_hz, order=4)
    xf = signal.sosfiltfilt(sos, x)
    return np.asarray(xf[::q], dtype=float)


def robust_fs_from_t(t: np.ndarray) -> float:
    t = _as_1d(t)
    if len(t) < 3:
        return 20.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return 20.0
    dt_med = float(np.median(dt))
    if dt_med <= 1e-6:
        return 20.0
    return 1.0 / dt_med


def rolling_var(x: np.ndarray, win: int) -> np.ndarray:
    x = _as_1d(x)
    win = int(max(3, win))
    w = np.ones(win, dtype=float) / float(win)
    mean = np.convolve(x, w, mode="same")
    mean2 = np.convolve(x * x, w, mode="same")
    var = mean2 - mean * mean
    return np.maximum(var, 0.0)


def activity_weight(u: np.ndarray, y: np.ndarray, *, win: int = 25, min_w: float = 0.05) -> np.ndarray:
    """Down-weight near-silent segments (low variance in u/y)."""
    u = _as_1d(u)
    y = _as_1d(y)
    vu = rolling_var(u, win)
    vy = rolling_var(y, win)
    a = vy + 0.25 * vu
    med = float(np.median(a))
    scale = med if med > 1e-12 else float(np.mean(a) + 1e-12)
    w = a / (scale + 1e-12)
    w = np.clip(w, min_w, 1.0)
    return w


def load_all_npz_to_10hz(
    data_dir: Path,
    *,
    cutoff_hz: float = 4.0,
    raw_to_mpc_q: int = 2,
) -> List[Series10Hz]:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No npz files found in {data_dir}")

    out: List[Series10Hz] = []
    for p in files:
        with np.load(p, allow_pickle=True) as z:
            t = _as_1d(z[NPZ_KEYS["t"]])
            v = _as_1d(z[NPZ_KEYS["v"]])
            w = _as_1d(z[NPZ_KEYS["w"]])
            v_cmd = _as_1d(z[NPZ_KEYS["v_cmd"]])
            w_cmd = _as_1d(z[NPZ_KEYS["w_cmd"]])
            meta = z[NPZ_KEYS["meta"]].item() if NPZ_KEYS["meta"] in z else {}

        fs_raw = robust_fs_from_t(t)
        v10 = lowpass_then_downsample(v, fs_hz=fs_raw, cutoff_hz=cutoff_hz, q=raw_to_mpc_q)
        w10 = lowpass_then_downsample(w, fs_hz=fs_raw, cutoff_hz=cutoff_hz, q=raw_to_mpc_q)
        vcmd10 = lowpass_then_downsample(v_cmd, fs_hz=fs_raw, cutoff_hz=cutoff_hz, q=raw_to_mpc_q)
        wcmd10 = lowpass_then_downsample(w_cmd, fs_hz=fs_raw, cutoff_hz=cutoff_hz, q=raw_to_mpc_q)
        t10 = t[::raw_to_mpc_q].copy()
        t10 = t10 - float(t10[0])

        n = min(len(t10), len(v10), len(w10), len(vcmd10), len(wcmd10))
        if n < 10:
            continue
        t10 = t10[:n]
        out.append(
            Series10Hz(
                name=p.stem,
                t=t10,
                v=v10[:n],
                w=w10[:n],
                v_cmd=vcmd10[:n],
                w_cmd=wcmd10[:n],
                meta=meta,
            )
        )
    return out


def weighted_ls(X: np.ndarray, y: np.ndarray, w: np.ndarray, ridge: float = 1e-9) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = _as_1d(y)
    w = _as_1d(w)
    w = np.clip(w, 1e-12, np.inf)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    XtX = Xw.T @ Xw
    XtX = XtX + ridge * np.eye(XtX.shape[0])
    Xty = Xw.T @ yw
    return np.linalg.solve(XtX, Xty)


def rmse_mae(y: np.ndarray, yhat: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float]:
    y = _as_1d(y)
    yhat = _as_1d(yhat)
    e = yhat - y
    if w is None:
        return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))
    wv = _as_1d(w)
    wv = np.clip(wv, 1e-12, np.inf)
    s = float(np.sum(wv))
    rmse = math.sqrt(float(np.sum(wv * (e * e)) / s))
    mae = float(np.sum(wv * np.abs(e)) / s)
    return rmse, mae


def bic_from_weighted_rss(rss_w: float, n_eff: float, k: int) -> float:
    n_eff = float(max(n_eff, 1.0))
    rss_w = float(max(rss_w, 1e-18))
    return float(n_eff * math.log(rss_w / n_eff) + k * math.log(n_eff))


@dataclass
class WModel:
    alpha: float
    beta: float
    rmse: float
    mae: float


def fit_w_first_order(all_series: Sequence[Series10Hz]) -> WModel:
    y_list: List[float] = []
    x_list: List[Tuple[float, float]] = []
    w_list: List[float] = []

    for s in all_series:
        w_meas = s.w
        w_cmd = s.w_cmd
        ww = activity_weight(w_cmd, w_meas)
        # y = w[k+1], regressors are w[k] and w_cmd[k-1] (delay L=1)
        for k in range(1, len(w_meas) - 1):
            y_list.append(float(w_meas[k + 1]))
            x_list.append((float(w_meas[k]), float(w_cmd[k - 1])))
            w_list.append(float(ww[k]))

    y = np.asarray(y_list, dtype=float)
    X = np.asarray(x_list, dtype=float)
    ww = np.asarray(w_list, dtype=float)

    # Enforce DC gain ~1 by constraining beta = 1 - alpha.
    # Rewrite: (y - u) = alpha * (x - u), where u is delayed command.
    x_prev = X[:, 0]
    u_del = X[:, 1]
    yy = y - u_del
    xx = x_prev - u_del
    alpha = float(weighted_ls(xx.reshape(-1, 1), yy, ww, ridge=1e-8)[0])
    alpha = float(np.clip(alpha, -0.99, 0.99))
    beta = 1.0 - alpha

    yhat = alpha * x_prev + beta * u_del
    r, m = rmse_mae(y, yhat, ww)
    return WModel(alpha=alpha, beta=beta, rmse=r, mae=m)


@dataclass
class VCandidate:
    n: int
    coupling: Tuple[str, ...]
    theta: np.ndarray
    rmse: float
    mae: float
    bic: float
    rho: float


def coupling_features(
    coupling: Sequence[str],
    *,
    v_k: np.ndarray,
    w_k: np.ndarray,
) -> np.ndarray:
    cols: List[np.ndarray] = []
    for c in coupling:
        if c == "w":
            cols.append(w_k)
        elif c == "abs_w":
            cols.append(np.abs(w_k))
        elif c == "w2":
            cols.append(w_k * w_k)
        elif c == "v_w":
            cols.append(v_k * w_k)
        else:
            raise ValueError(f"Unknown coupling term: {c}")
    if not cols:
        return np.zeros((len(v_k), 0), dtype=float)
    return np.stack(cols, axis=1).astype(float)


def spectral_radius_companion(a: np.ndarray) -> float:
    a = _as_1d(a)
    n = len(a)
    if n <= 0:
        return 0.0
    A = np.zeros((n, n), dtype=float)
    A[0, :] = a
    if n > 1:
        A[1:, :-1] = np.eye(n - 1)
    eig = np.linalg.eigvals(A)
    return float(np.max(np.abs(eig)))


def iter_coupling_sets() -> List[Tuple[str, ...]]:
    base = ["w", "abs_w", "w2", "v_w"]
    out: List[Tuple[str, ...]] = [tuple()]
    for mask in range(1, 1 << len(base)):
        cs = tuple(base[i] for i in range(len(base)) if (mask & (1 << i)))
        out.append(cs)
    # Prefer simpler earlier (BIC already penalizes, but deterministic ordering helps reading)
    out.sort(key=lambda t: (len(t), t))
    return out


def build_v_regression_mats(series: Series10Hz, *, n: int, coupling: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = series.v
    w = series.w
    v_cmd = series.v_cmd
    wt = activity_weight(v_cmd, v)

    # Model (delay L=1):
    #   v[k+1] = a1*v[k] + ... + an*v[k-n+1] + b*v_cmd[k-1] + sum c_j * phi_j[k]
    # where phi_j are coupling features based on w[k] and/or v[k].
    k0 = max(n - 1, 1)
    k1 = len(v) - 2
    if k1 <= k0:
        return np.zeros((0, n + 1), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    rows: List[np.ndarray] = []
    y: List[float] = []
    ww: List[float] = []
    for k in range(k0, k1 + 1):
        y.append(float(v[k + 1]))
        ar = np.array([float(v[k - i]) for i in range(0, n)], dtype=float)  # v[k], v[k-1], ...
        udel = float(v_cmd[k - 1])
        v_k = float(v[k])
        w_k = float(w[k])
        phis = coupling_features(coupling, v_k=np.array([v_k]), w_k=np.array([w_k]))
        row = np.concatenate([ar, np.array([udel], dtype=float), phis.reshape(-1)], axis=0)
        rows.append(row)
        ww.append(float(wt[k]))

    X = np.stack(rows, axis=0)
    yy = np.asarray(y, dtype=float)
    wv = np.asarray(ww, dtype=float)
    return X, yy, wv


def fit_v_candidates(all_series: Sequence[Series10Hz], *, n_max: int = 5) -> List[VCandidate]:
    coupling_sets = iter_coupling_sets()
    out: List[VCandidate] = []

    for n in range(1, n_max + 1):
        for coupling in coupling_sets:
            Xs: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            ws: List[np.ndarray] = []
            for s in all_series:
                X, y, wv = build_v_regression_mats(s, n=n, coupling=coupling)
                if len(y) == 0:
                    continue
                Xs.append(X)
                ys.append(y)
                ws.append(wv)
            if not ys:
                continue

            X_all = np.concatenate(Xs, axis=0)
            y_all = np.concatenate(ys, axis=0)
            w_all = np.concatenate(ws, axis=0)

            theta = weighted_ls(X_all, y_all, w_all, ridge=1e-8)
            yhat = X_all @ theta
            r, m = rmse_mae(y_all, yhat, w_all)

            # Stability constraint on A_v (companion formed by AR coefficients only)
            a = theta[:n]
            rho = spectral_radius_companion(a)
            if not np.isfinite(rho) or rho >= 1.0:
                continue

            e = yhat - y_all
            rss_w = float(np.sum(w_all * (e * e)))
            n_eff = float(np.sum(w_all))
            k_params = int(len(theta))
            bic = bic_from_weighted_rss(rss_w, n_eff, k_params)

            out.append(
                VCandidate(
                    n=n,
                    coupling=tuple(coupling),
                    theta=theta,
                    rmse=r,
                    mae=m,
                    bic=bic,
                    rho=rho,
                )
            )
    out.sort(key=lambda c: c.bic)
    return out


def candidate_is_exportable_lti(c: VCandidate) -> bool:
    # Only linear w coupling can be embedded into an LTI MIMO state-space directly.
    # abs_w, w2, v_w are nonlinear in the state.
    return all(term in ("w",) for term in c.coupling)


def build_mimo_ss(*, v_best: VCandidate, w_model: WModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    if not candidate_is_exportable_lti(v_best):
        raise ValueError(f"Selected v model is not exportable to LTI state-space: coupling={v_best.coupling}")

    n = v_best.n
    # State: [v[k], v[k-1], ..., v[k-n+1], w[k], dv[k], dw[k]]
    # dv[k] = v_cmd[k-1], dw[k] = w_cmd[k-1] (1-step delay states)
    nx = n + 3
    nu = 2
    ny = 2
    A = np.zeros((nx, nx), dtype=float)
    B = np.zeros((nx, nu), dtype=float)
    C = np.zeros((ny, nx), dtype=float)
    D = np.zeros((ny, nu), dtype=float)

    theta = v_best.theta
    a = theta[:n]
    b = float(theta[n])
    c_w = 0.0
    if len(v_best.coupling) == 1 and v_best.coupling[0] == "w":
        c_w = float(theta[n + 1])

    # v update
    A[0, 0:n] = a
    A[0, n] = c_w  # coupling from w state
    A[0, n + 1] = b  # dv state

    # shift v history
    for i in range(1, n):
        A[i, i - 1] = 1.0

    # w update: w[k+1] = alpha*w[k] + beta*dw[k]
    A[n, n] = float(w_model.alpha)
    A[n, n + 2] = float(w_model.beta)

    # delay states
    # dv[k+1] = v_cmd[k]
    # dw[k+1] = w_cmd[k]
    B[n + 1, 0] = 1.0
    B[n + 2, 1] = 1.0

    # outputs
    C[0, 0] = 1.0  # v
    C[1, n] = 1.0  # w

    meta = {
        "state_order": [
            *[f"v(k-{i})" for i in range(0, n)],
            "w(k)",
            "dv(k)=v_cmd(k-1)",
            "dw(k)=w_cmd(k-1)",
        ],
        "input_order": ["v_cmd", "w_cmd"],
        "output_order": ["v_meas", "w_meas"],
        "v_model": {
            "n": int(v_best.n),
            "coupling": list(v_best.coupling),
            "theta": v_best.theta.tolist(),
            "rmse": float(v_best.rmse),
            "mae": float(v_best.mae),
            "bic": float(v_best.bic),
            "rho": float(v_best.rho),
        },
        "w_model": {
            "alpha": float(w_model.alpha),
            "beta": float(w_model.beta),
            "rmse": float(w_model.rmse),
            "mae": float(w_model.mae),
        },
    }
    return A, B, C, D, meta


def simulate_ss(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, u: np.ndarray, x0: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(-1)
    nx = A.shape[0]
    if x.shape[0] != nx:
        raise ValueError("x0 dimension mismatch")
    yhat = []
    for k in range(u.shape[0]):
        y = C @ x + D @ u[k]
        yhat.append(y)
        x = A @ x + B @ u[k]
    return np.stack(yhat, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="identify_data")
    ap.add_argument("--out-dir", type=str, default="identify_result")
    ap.add_argument("--cutoff-hz", type=float, default=8.0)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = out_root / run_stamp
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    series = load_all_npz_to_10hz(data_dir, cutoff_hz=float(args.cutoff_hz), raw_to_mpc_q=2)
    if not series:
        raise RuntimeError("No usable series after preprocessing")

    # 1) w channel
    w_model = fit_w_first_order(series)

    # 2) v channel candidates
    v_cands = fit_v_candidates(series, n_max=int(args.n_max))
    if not v_cands:
        raise RuntimeError("No feasible v candidates found (check constraints / data)")
    v_best_overall = v_cands[0]
    exportable = [c for c in v_cands if candidate_is_exportable_lti(c)]
    if not exportable:
        raise RuntimeError("No exportable (LTI) v model found. Best candidates all include nonlinear couplings.")
    v_best = exportable[0]

    # 3) Build MIMO state-space
    A, B, C, D, meta = build_mimo_ss(v_best=v_best, w_model=w_model)

    # 4) Save leaderboard
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = out_dir / "leaderboard.csv"
    with leaderboard_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "n", "coupling", "bic", "rmse", "mae", "rho", "exportable_lti"])
        for i, c in enumerate(v_cands[: max(50, int(args.topk))], start=1):
            wr.writerow(
                [
                    i,
                    c.n,
                    "+".join(c.coupling) if c.coupling else "(none)",
                    c.bic,
                    c.rmse,
                    c.mae,
                    c.rho,
                    int(candidate_is_exportable_lti(c)),
                ]
            )

    # 5) Save model
    npz_path = out_dir / "model_mimo_10hz.npz"
    np.savez(npz_path, A=A, B=B, C=C, D=D, meta=json.dumps(meta, ensure_ascii=False))
    txt_path = out_dir / "model_mimo_10hz.txt"
    with txt_path.open("w") as f:
        f.write("# 10Hz MIMO state-space (exported for MPC)\n")
        f.write(f"# v_best_overall: n={v_best_overall.n}, coupling={v_best_overall.coupling}, BIC={v_best_overall.bic:.3f}, rmse={v_best_overall.rmse:.4f}, rho={v_best_overall.rho:.6f}\n")
        f.write(f"# v_exported:     n={v_best.n}, coupling={v_best.coupling}, BIC={v_best.bic:.3f}, rmse={v_best.rmse:.4f}, rho={v_best.rho:.6f}\n")
        f.write(f"# w_model: alpha={w_model.alpha:.6f}, beta={w_model.beta:.6f}, rmse={w_model.rmse:.4f}\n\n")
        np.set_printoptions(precision=6, suppress=True, linewidth=140)
        f.write("A =\n")
        f.write(str(A))
        f.write("\n\nB =\n")
        f.write(str(B))
        f.write("\n\nC =\n")
        f.write(str(C))
        f.write("\n\nD =\n")
        f.write(str(D))
        f.write("\n\n# meta\n")
        f.write(json.dumps(meta, ensure_ascii=False, indent=2))
        f.write("\n")

    # 6) Predict + plot per dataset
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_path = out_dir / "per_file_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "scenario", "rmse_v", "mae_v", "rmse_w", "mae_w", "n", "coupling"])

        for s in series:
            u = np.stack([s.v_cmd, s.w_cmd], axis=1)

            # initial state: fill v history with first sample
            n = v_best.n
            x0 = np.zeros((n + 3,), dtype=float)
            x0[0:n] = float(s.v[0])
            x0[n] = float(s.w[0])
            x0[n + 1] = float(s.v_cmd[0])  # dv (unknown prev cmd) -> assume same as first
            x0[n + 2] = float(s.w_cmd[0])

            yhat = simulate_ss(A, B, C, D, u, x0)
            v_hat = yhat[:, 0]
            w_hat = yhat[:, 1]

            # ignore the first second (10 samples) in metrics to reduce init bias
            burn = min(10, len(s.v) // 4)
            wv = activity_weight(s.v_cmd, s.v)
            ww = activity_weight(s.w_cmd, s.w)
            rv, mv = rmse_mae(s.v[burn:], v_hat[burn:], wv[burn:])
            rw, mw = rmse_mae(s.w[burn:], w_hat[burn:], ww[burn:])

            scenario = str(s.meta.get("scenario", ""))
            wr.writerow([s.name, scenario, rv, mv, rw, mw, v_best.n, "+".join(v_best.coupling) if v_best.coupling else "(none)"])

            fig = plt.figure(figsize=(12, 7))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(s.t, s.v_cmd, label="v_cmd", linewidth=1.2)
            ax1.plot(s.t, s.v, label="v_act(v_meas)", linewidth=1.4)
            ax1.plot(s.t, v_hat, label="v_hat", linewidth=1.4)
            ax1.set_ylabel("v")
            ax1.grid(True, alpha=0.25)
            ax1.legend(loc="best")
            ax1.set_title(f"{s.name}  scenario={scenario}")

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(s.t, s.w_cmd, label="w_cmd", linewidth=1.2)
            ax2.plot(s.t, s.w, label="w_act(w_meas)", linewidth=1.4)
            ax2.plot(s.t, w_hat, label="w_hat", linewidth=1.4)
            ax2.set_ylabel("w")
            ax2.set_xlabel("t (s)")
            ax2.grid(True, alpha=0.25)
            ax2.legend(loc="best")

            fig.tight_layout()
            fig_path = plots_dir / f"{s.name}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)

    # Console summary
    print("===== Identification Summary (10Hz) =====")
    print(f"# datasets used: {len(series)}")
    print(f"w model: alpha={w_model.alpha:.6f}, beta={w_model.beta:.6f}, rmse={w_model.rmse:.4f}, mae={w_model.mae:.4f}")
    print(
        "v best overall (BIC): "
        f"n={v_best_overall.n}, coupling={v_best_overall.coupling}, BIC={v_best_overall.bic:.3f}, "
        f"rmse={v_best_overall.rmse:.4f}, mae={v_best_overall.mae:.4f}, rho={v_best_overall.rho:.6f}"
    )
    print(
        "v exported (LTI):     "
        f"n={v_best.n}, coupling={v_best.coupling}, BIC={v_best.bic:.3f}, "
        f"rmse={v_best.rmse:.4f}, mae={v_best.mae:.4f}, rho={v_best.rho:.6f}"
    )
    print("\nTop BIC leaderboard:")
    for i, c in enumerate(v_cands[: int(args.topk)], start=1):
        tag = "LTI" if candidate_is_exportable_lti(c) else "nonlinear"
        coup = "+".join(c.coupling) if c.coupling else "(none)"
        print(f"{i:2d}. n={c.n}  coup={coup:14s}  BIC={c.bic:10.3f}  rmse={c.rmse:7.4f}  rho={c.rho:8.6f}  {tag}")

    print("\nExported matrices (A,B,C,D) saved to:")
    print(str(npz_path))
    print(str(txt_path))
    print("All results saved under:")
    print(str(out_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
