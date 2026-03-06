#!/usr/bin/env python3
"""Power model identification for wheel-leg balanced robot.

从 power_identify_data/ 中的 npz 文件读取数据，辨识底盘功率模型：

  P(t) = c₀ + c₁·v·a + c₂·ω·α + c₃·a² + c₄·α²
       + c₅·|v| + c₆·|ω| + c₇·v² + c₈·ω² + c₉·|a| + c₁₀·|α|
       + c₁₁·|v·ω|

物理含义：
  c₀      : 基线功率（平衡 + 电控待机）
  c₁·v·a  : 平移加速的机械功率 P = m·a·v
  c₂·ω·α  : 偏航加速的机械功率 P = I·α·ω
  c₃·a²   : 轮电机电阻损耗 ∝ 力矩² ∝ 加速度²
  c₄·α²   : 偏航电机电阻损耗
  c₅·|v|  : 滚动摩擦功率损耗
  c₆·|ω|  : 偏航摩擦功率损耗
  c₇·v²   : 速度二次项（粘滞阻力等）
  c₈·ω²   : 角速度二次项
  c₉·|a|  : 平移加速时的静摩擦功率
  c₁₀·|α| : 偏航加速时的静摩擦功率
  c₁₁·|v·ω|: 平移-偏航耦合功率（离心效应、差速转向）

辨识方法：
    1. 低通滤波 + 降采样到 MPC 频率
  2. 数值微分得 a=dv/dt, α=dω/dt
  3. 构建特征矩阵 Φ (线性回归)
  4. 可选 Ridge/Lasso 正则化
  5. 强制部分系数为物理合理符号 (bounded least squares)
  6. 交叉验证评估

输出 (保存到 power_identify_result/<run_stamp>/):
  - power_model.npz        (系数 + 元数据)
  - power_model.txt        (可读文本)
  - per_file_metrics.csv
  - plots/*.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import signal
    from scipy.optimize import lsq_linear
except ImportError as exc:
    raise RuntimeError("scipy is required: pip install scipy") from exc


# ════════════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════════════

DT_TARGET = 0.05    # 降采样目标步长 20Hz
SGN_EPS = 0.05     # smooth-|x| ≈ sqrt(x² + eps²)

# 特征名
FEATURE_NAMES = [
    "1 (bias)",        # c0:  基线功率
    "v·a",             # c1:  平移机械功率
    "ω·α",             # c2:  偏航机械功率
    "a²",              # c3:  加速度电阻损耗
    "α²",              # c4:  角加速度电阻损耗
    "|v|",             # c5:  滚动摩擦
    "|ω|",             # c6:  偏航摩擦
    "v²",              # c7:  粘滞阻力
    "ω²",              # c8:  角速度二次
    "|a|",             # c9:  平移加速静摩擦
    "|α|",             # c10: 偏航加速静摩擦
    "|v·ω|",           # c11: 耦合项
]
N_FEATURES = len(FEATURE_NAMES)

# 系数 bounds: (lower, upper) —— 基于物理直觉
# None 表示不限制
COEFF_BOUNDS_PHYS = [
    (0.0, None),      # c0:  基线功率 >= 0
    (None, None),     # c1:  v·a       可正可负（取决于刹车能否回馈）
    (None, None),     # c2:  ω·α       同上
    (0.0, None),      # c3:  a²        电阻损耗 >= 0
    (0.0, None),      # c4:  α²        >= 0
    (0.0, None),      # c5:  |v|       摩擦功 >= 0
    (0.0, None),      # c6:  |ω|       >= 0
    (0.0, None),      # c7:  v²        >= 0
    (0.0, None),      # c8:  ω²        >= 0
    (0.0, None),      # c9:  |a|       >= 0
    (0.0, None),      # c10: |α|       >= 0
    (0.0, None),      # c11: |v·ω|     >= 0
]


# ════════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class PowerSeries:
    name: str
    t: np.ndarray        # time [s], shape (N,)
    v: np.ndarray        # measured velocity [m/s]
    w: np.ndarray        # measured angular velocity [rad/s]
    pwr: np.ndarray      # measured chassis power [W]
    energy: np.ndarray   # remaining energy [J]
    pwr_limit: np.ndarray  # referee power limit [W]
    leg_mode: np.ndarray   # leg mode
    meta: Dict


def _as_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def butter_lowpass_sos(cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs_hz
    return signal.butter(order, cutoff_hz / nyq, btype="low", output="sos")


def lowpass_then_downsample(x: np.ndarray, *, fs_hz: float,
                             cutoff_hz: float, q: int) -> np.ndarray:
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


def load_all_npz(data_dir: Path, *, cutoff_hz: float = 3.0,
                  raw_to_target_q: int = 2) -> List[PowerSeries]:
    """Load and downsample all power npz files in data_dir."""
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    out: List[PowerSeries] = []
    for p in files:
        with np.load(p, allow_pickle=True) as z:
            keys = list(z.keys())
            # 验证必要字段
            if "curr_chassis_pwr" not in keys:
                print(f"  跳过 {p.name}: 缺少 curr_chassis_pwr 字段（非功率录制文件）")
                continue

            t = _as_1d(z["t"])
            v = _as_1d(z["v_meas"])
            w = _as_1d(z["w_meas"])
            pwr = _as_1d(z["curr_chassis_pwr"])
            energy = _as_1d(z["remaining_energy"])
            pwr_limit = _as_1d(z["rfr_pwr_limit"])
            leg_mode = _as_1d(z["leg_mode"]) if "leg_mode" in keys else np.zeros_like(t)
            meta = z["meta"].item() if "meta" in z else {}

        fs = robust_fs_from_t(t)
        q = max(1, round(fs * DT_TARGET))  # dynamically compute decimation factor
        kw = dict(fs_hz=fs, cutoff_hz=cutoff_hz, q=q)

        v10 = lowpass_then_downsample(v, **kw)
        w10 = lowpass_then_downsample(w, **kw)
        pwr10 = lowpass_then_downsample(pwr, **kw)
        energy10 = lowpass_then_downsample(energy, **kw)
        pwr_limit10 = lowpass_then_downsample(pwr_limit, **kw)
        leg_mode10 = leg_mode[::q].copy()  # mode不做滤波
        t10 = t[::q].copy()
        t10 -= float(t10[0])

        n = min(len(t10), len(v10), len(w10), len(pwr10), len(energy10), len(pwr_limit10), len(leg_mode10))
        if n < 20:
            print(f"  跳过 {p.name}: 数据太少 ({n} 点)")
            continue

        out.append(PowerSeries(
            name=p.stem,
            t=t10[:n], v=v10[:n], w=w10[:n],
            pwr=pwr10[:n], energy=energy10[:n],
            pwr_limit=pwr_limit10[:n], leg_mode=leg_mode10[:n],
            meta=meta,
        ))
        print(f"  loaded {p.name}: {n} samples @ {1.0/DT_TARGET:.0f}Hz, "
              f"pwr range [{pwr10[:n].min():.0f}, {pwr10[:n].max():.0f}]W")

    return out


# ════════════════════════════════════════════════════════════════════════════════
#  Feature Construction
# ════════════════════════════════════════════════════════════════════════════════

def compute_derivatives(x: np.ndarray, dt: float) -> np.ndarray:
    """Central differences with forward/backward at endpoints."""
    dx = np.zeros_like(x)
    if len(x) < 3:
        return dx
    dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx


def smooth_abs(x: np.ndarray, eps: float = SGN_EPS) -> np.ndarray:
    """Smooth approximation of |x|."""
    return np.sqrt(x ** 2 + eps ** 2)


def build_feature_matrix(s: PowerSeries) -> Tuple[np.ndarray, np.ndarray]:
    """Build (Φ, y) where y = pwr and Φ has columns for each feature.

    Returns:
        Phi: shape (N, N_FEATURES)
        y:   shape (N,)
    """
    dt = DT_TARGET
    v = s.v
    w = s.w
    a = compute_derivatives(v, dt)     # dv/dt
    alpha = compute_derivatives(w, dt)  # dω/dt

    N = len(v)
    Phi = np.zeros((N, N_FEATURES))

    Phi[:, 0] = 1.0              # bias
    Phi[:, 1] = v * a             # v·a  (mechanical translation power)
    Phi[:, 2] = w * alpha          # ω·α  (mechanical rotation power)
    Phi[:, 3] = a ** 2             # a²   (motor loss)
    Phi[:, 4] = alpha ** 2         # α²   (motor loss)
    Phi[:, 5] = smooth_abs(v)     # |v|  (rolling friction)
    Phi[:, 6] = smooth_abs(w)     # |ω|  (yaw friction)
    Phi[:, 7] = v ** 2             # v²
    Phi[:, 8] = w ** 2             # ω²
    Phi[:, 9] = smooth_abs(a)     # |a|  (accel friction)
    Phi[:, 10] = smooth_abs(alpha)  # |α|
    Phi[:, 11] = smooth_abs(v * w)  # |v·ω| (coupling)

    return Phi, s.pwr.copy()


def build_global_regression(series_list: List[PowerSeries],
                            ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Stack all series into one big regression problem.

    Returns:
        Phi_all: (N_total, N_FEATURES)
        y_all:   (N_total,)
        splits:  list of series lengths for per-file metrics
    """
    Phi_parts, y_parts, splits = [], [], []
    for s in series_list:
        Phi, y = build_feature_matrix(s)
        # 过滤掉 leg_mode != 4 (Mature) 的数据点，因为其他模式功率特性不同
        mask = (s.leg_mode == 4)
        if mask.sum() < 10:
            print(f"  WARNING: {s.name} has <10 samples in Mature mode, using all data")
            mask = np.ones(len(y), dtype=bool)
        Phi_parts.append(Phi[mask])
        y_parts.append(y[mask])
        splits.append(int(mask.sum()))

    Phi_all = np.vstack(Phi_parts)
    y_all = np.concatenate(y_parts)
    return Phi_all, y_all, splits


# ════════════════════════════════════════════════════════════════════════════════
#  Identification
# ════════════════════════════════════════════════════════════════════════════════

def solve_bounded_lstsq(Phi: np.ndarray, y: np.ndarray,
                         alpha_ridge: float = 0.0) -> np.ndarray:
    """Solve bounded least squares with optional Ridge regularization.

    Bounds from COEFF_BOUNDS_PHYS enforce physical sign constraints.
    """
    n_feat = Phi.shape[1]
    lower = np.full(n_feat, -np.inf)
    upper = np.full(n_feat, np.inf)

    for i, (lo, hi) in enumerate(COEFF_BOUNDS_PHYS):
        if lo is not None:
            lower[i] = lo
        if hi is not None:
            upper[i] = hi

    if alpha_ridge > 0.0:
        # Ridge: min ||Phi·c - y||² + alpha·||c||²
        # ⟺ min || [Phi; sqrt(alpha)·I] · c - [y; 0] ||²
        n = Phi.shape[1]
        Phi_aug = np.vstack([Phi, np.sqrt(alpha_ridge) * np.eye(n)])
        y_aug = np.concatenate([y, np.zeros(n)])
    else:
        Phi_aug = Phi
        y_aug = y

    result = lsq_linear(Phi_aug, y_aug, bounds=(lower, upper),
                         method="bvls", max_iter=5000)
    return result.x


def compute_metrics(Phi: np.ndarray, y: np.ndarray,
                    coeffs: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    y_pred = Phi @ coeffs
    residual = y - y_pred
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "max_abs_err": float(np.max(np.abs(residual))),
        "mean_pwr": float(np.mean(y)),
        "std_pwr": float(np.std(y)),
    }


# ════════════════════════════════════════════════════════════════════════════════
#  Cross-validation
# ════════════════════════════════════════════════════════════════════════════════

def leave_one_out_cv(series_list: List[PowerSeries],
                     alpha_ridge: float = 0.0) -> Tuple[np.ndarray, List[Dict]]:
    """Leave-one-file-out cross validation.

    Returns:
        best_coeffs: from full-data fit
        cv_metrics: list of per-fold metrics dicts
    """
    # Full fit
    Phi_all, y_all, _ = build_global_regression(series_list)
    best_coeffs = solve_bounded_lstsq(Phi_all, y_all, alpha_ridge)

    cv_metrics = []
    n = len(series_list)
    if n < 2:
        return best_coeffs, cv_metrics

    for i in range(n):
        train_set = [s for j, s in enumerate(series_list) if j != i]
        test_set = [series_list[i]]

        Phi_train, y_train, _ = build_global_regression(train_set)
        Phi_test, y_test, _ = build_global_regression(test_set)

        c_train = solve_bounded_lstsq(Phi_train, y_train, alpha_ridge)
        m = compute_metrics(Phi_test, y_test, c_train)
        m["test_file"] = series_list[i].name
        cv_metrics.append(m)

    return best_coeffs, cv_metrics


# ════════════════════════════════════════════════════════════════════════════════
#  Output & Plotting
# ════════════════════════════════════════════════════════════════════════════════

def format_model_text(coeffs: np.ndarray, global_metrics: Dict,
                      cv_metrics: List[Dict]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  Wheel-Leg Robot Chassis Power Model")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Model form:")
    lines.append("  P(t) = c₀ + c₁·v·a + c₂·ω·α + c₃·a² + c₄·α²")
    lines.append("       + c₅·|v| + c₆·|ω| + c₇·v² + c₈·ω²")
    lines.append("       + c₉·|a| + c₁₀·|α| + c₁₁·|v·ω|")
    lines.append("")
    lines.append("where:")
    lines.append("  v     = measured chassis velocity [m/s]")
    lines.append("  ω     = measured chassis angular velocity [rad/s]")
    lines.append("  a     = dv/dt [m/s²]")
    lines.append("  α     = dω/dt [rad/s²]")
    lines.append("  P     = chassis electrical power [W]")
    lines.append("")
    lines.append("─── Identified Coefficients ───")
    for i, (name, c) in enumerate(zip(FEATURE_NAMES, coeffs)):
        lines.append(f"  c{i:2d} ({name:>12s}) = {c:12.6f}")
    lines.append("")
    lines.append("─── Global Fit Metrics ───")
    for k, v in global_metrics.items():
        lines.append(f"  {k:>15s} = {v:.4f}")
    if cv_metrics:
        lines.append("")
        lines.append("─── Cross-Validation (leave-one-file-out) ───")
        rmses = [m["rmse"] for m in cv_metrics]
        r2s = [m["r2"] for m in cv_metrics]
        lines.append(f"  CV RMSE:  mean={np.mean(rmses):.2f}  std={np.std(rmses):.2f}  "
                      f"range=[{np.min(rmses):.2f}, {np.max(rmses):.2f}]")
        lines.append(f"  CV R²:    mean={np.mean(r2s):.4f}  std={np.std(r2s):.4f}  "
                      f"range=[{np.min(r2s):.4f}, {np.max(r2s):.4f}]")
        for m in cv_metrics:
            lines.append(f"    fold {m.get('test_file', '?'):>40s}:  "
                          f"RMSE={m['rmse']:.2f}  R²={m['r2']:.4f}")
    lines.append("")
    lines.append("─── MPC Integration Notes ───")
    lines.append("  In MPC, at each prediction step k:")
    lines.append("    a(k) ≈ (v(k) - v(k-1)) / dt")
    lines.append("    α(k) ≈ (ω(k) - ω(k-1)) / dt")
    lines.append("    P(k) = Σ cᵢ·φᵢ(v(k), ω(k), a(k), α(k))")
    lines.append("    E(k+1) = E(k) + (P_charge - P(k)) × dt")
    lines.append("  where P_charge ≈ rfr_pwr_limit, E = remaining capacitor energy")
    lines.append("")
    return "\n".join(lines)


def try_setup_matplotlib():
    try:
        import matplotlib
        if os.environ.get("DISPLAY", "") == "":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plot_results(series_list: List[PowerSeries], coeffs: np.ndarray,
                 out_dir: Path) -> None:
    plt = try_setup_matplotlib()
    if plt is None:
        print("matplotlib 不可用，跳过绘图")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for s in series_list:
        Phi, y = build_feature_matrix(s)
        y_pred = Phi @ coeffs

        fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True)

        # 1. 速度
        ax = axes[0]
        ax.plot(s.t, s.v, "b", lw=1.0, label="v_meas")
        ax.set_ylabel("v [m/s]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_title(s.name, fontsize=10)

        # 2. 角速度
        ax = axes[1]
        ax.plot(s.t, s.w, "b", lw=1.0, label="ω_meas")
        ax.set_ylabel("ω [rad/s]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.2)

        # 3. 功率：实测 vs 预测
        ax = axes[2]
        ax.plot(s.t, y, "r", lw=1.0, alpha=0.7, label="P_meas")
        ax.plot(s.t, y_pred, "b--", lw=1.0, label="P_pred")
        ax.set_ylabel("Power [W]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.2)

        # 4. 残差
        ax = axes[3]
        residual = y - y_pred
        ax.plot(s.t, residual, "gray", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        ax.set_ylabel(f"Residual [W]\nRMSE={rmse:.1f}")
        ax.grid(True, alpha=0.2)

        # 5. 电容电量
        ax = axes[4]
        ax.plot(s.t, s.energy, "m", lw=1.0, label="remaining_energy")
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Energy [J]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(plot_dir / f"{s.name}.png", dpi=160)
        plt.close(fig)
        print(f"  plot: {s.name}.png")

    # ── 全局散点图：P_pred vs P_meas ──
    fig, ax = plt.subplots(figsize=(8, 8))
    all_y, all_yp = [], []
    for s in series_list:
        Phi, y = build_feature_matrix(s)
        yp = Phi @ coeffs
        all_y.append(y)
        all_yp.append(yp)
    all_y = np.concatenate(all_y)
    all_yp = np.concatenate(all_yp)
    ax.scatter(all_y, all_yp, s=1.5, alpha=0.3, c="steelblue")
    lo = min(all_y.min(), all_yp.min())
    hi = max(all_y.max(), all_yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("P_measured [W]")
    ax.set_ylabel("P_predicted [W]")
    ax.set_title("Power Model: Predicted vs Measured")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "scatter_global.png", dpi=160)
    plt.close(fig)

    # ── 特征重要性图 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    # Compute feature importance as |c_i| × std(φ_i)
    Phi_all = np.vstack([build_feature_matrix(s)[0] for s in series_list])
    phi_std = np.std(Phi_all, axis=0)
    importance = np.abs(coeffs) * phi_std
    importance_norm = importance / max(importance.sum(), 1e-12) * 100.0

    idx = np.argsort(importance_norm)[::-1]
    bars = ax.barh(range(N_FEATURES), importance_norm[idx], color="steelblue")
    ax.set_yticks(range(N_FEATURES))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in idx])
    ax.set_xlabel("Relative importance [%]")
    ax.set_title("Feature Importance (|coeff| × std(feature))")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    print(f"  plots saved to {plot_dir}")


# ════════════════════════════════════════════════════════════════════════════════
#  C++ Code Generation
# ════════════════════════════════════════════════════════════════════════════════

def generate_cpp_snippet(coeffs: np.ndarray) -> str:
    """Generate C++ code snippet for MPC power prediction."""
    lines = []
    lines.append("// ──── Auto-generated power model coefficients ────")
    lines.append("// P(v, w, a, alpha) = c[0]")
    lines.append("//   + c[1]*v*a + c[2]*w*alpha")
    lines.append("//   + c[3]*a*a + c[4]*alpha*alpha")
    lines.append("//   + c[5]*fabs(v) + c[6]*fabs(w)")
    lines.append("//   + c[7]*v*v + c[8]*w*w")
    lines.append("//   + c[9]*fabs(a) + c[10]*fabs(alpha)")
    lines.append("//   + c[11]*fabs(v*w)")
    lines.append("")
    lines.append(f"static constexpr int kPowerModelNumCoeffs = {N_FEATURES};")
    lines.append("static constexpr double kPowerModelCoeffs[kPowerModelNumCoeffs] = {")
    for i, c in enumerate(coeffs):
        comma = "," if i < len(coeffs) - 1 else ""
        lines.append(f"    {c:18.10e}{comma}  // c{i}: {FEATURE_NAMES[i]}")
    lines.append("};")
    lines.append("")
    lines.append("inline double predict_chassis_power(")
    lines.append("    double v, double w, double a, double alpha) {")
    lines.append("    const auto* c = kPowerModelCoeffs;")
    lines.append("    return c[0]")
    lines.append("        + c[1]*v*a + c[2]*w*alpha")
    lines.append("        + c[3]*a*a + c[4]*alpha*alpha")
    lines.append("        + c[5]*std::fabs(v) + c[6]*std::fabs(w)")
    lines.append("        + c[7]*v*v + c[8]*w*w")
    lines.append("        + c[9]*std::fabs(a) + c[10]*std::fabs(alpha)")
    lines.append("        + c[11]*std::fabs(v*w);")
    lines.append("}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Power model identification for wheel-leg robot")
    parser.add_argument("--data-dir", type=str, default="power_identify_data",
                        help="directory with .npz recordings (must include curr_chassis_pwr field)")
    parser.add_argument("--out-dir", type=str, default="",
                        help="output directory (default: power_identify_result/<timestamp>)")
    parser.add_argument("--cutoff-hz", type=float, default=4.0,
                        help="lowpass cutoff frequency for preprocessing")
    parser.add_argument("--regularize", type=str, default="ridge",
                        choices=["none", "ridge"],
                        help="regularization type")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="regularization strength (Ridge alpha)")
    parser.add_argument("--no-cv", action="store_true",
                        help="skip cross-validation")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip plotting")
    parser.add_argument("--no-bounds", action="store_true",
                        help="disable physical sign bounds on coefficients")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return 1

    print(f"Loading data from {data_dir}...")
    series_list = load_all_npz(data_dir, cutoff_hz=args.cutoff_hz)
    if not series_list:
        print("没有可用的功率录制数据。")
        return 1
    print(f"Loaded {len(series_list)} files, total samples: "
          f"{sum(len(s.t) for s in series_list)}")

    # ── 辨识 ──
    alpha_ridge = args.alpha if args.regularize == "ridge" else 0.0

    if args.no_bounds:
        # 临时清除 bounds
        for i in range(N_FEATURES):
            COEFF_BOUNDS_PHYS[i] = (None, None)

    print(f"\nFitting power model (regularize={args.regularize}, alpha={alpha_ridge})...")
    if args.no_cv or len(series_list) < 2:
        Phi_all, y_all, splits = build_global_regression(series_list)
        coeffs = solve_bounded_lstsq(Phi_all, y_all, alpha_ridge)
        cv_metrics = []
    else:
        print("Running leave-one-file-out cross-validation...")
        coeffs, cv_metrics = leave_one_out_cv(series_list, alpha_ridge)

    # ── 全局指标 ──
    Phi_all, y_all, splits = build_global_regression(series_list)
    global_metrics = compute_metrics(Phi_all, y_all, coeffs)

    # ── 每文件指标 ──
    per_file_metrics = []
    offset = 0
    for i, s in enumerate(series_list):
        n = splits[i]
        m = compute_metrics(Phi_all[offset:offset+n], y_all[offset:offset+n], coeffs)
        m["file"] = s.name
        per_file_metrics.append(m)
        offset += n

    # ── Print ──
    model_text = format_model_text(coeffs, global_metrics, cv_metrics)
    print("\n" + model_text)

    # ── Output directory ──
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("power_identify_result") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model npz
    npz_path = out_dir / "power_model.npz"
    np.savez_compressed(
        npz_path,
        coeffs=coeffs,
        feature_names=FEATURE_NAMES,
        global_metrics=global_metrics,
        cv_metrics=cv_metrics,
        dt=DT_TARGET,
    )
    print(f"Model saved: {npz_path}")

    # Save text
    txt_path = out_dir / "power_model.txt"
    txt_path.write_text(model_text)
    print(f"Text:  {txt_path}")

    # Save C++ snippet
    cpp_path = out_dir / "power_model_snippet.hpp"
    cpp_path.write_text(generate_cpp_snippet(coeffs))
    print(f"C++:   {cpp_path}")

    # Save per-file CSV
    csv_path = out_dir / "per_file_metrics.csv"
    if per_file_metrics:
        keys = list(per_file_metrics[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_file_metrics)
        print(f"CSV:   {csv_path}")

    # Plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_results(series_list, coeffs, out_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
