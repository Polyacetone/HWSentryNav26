"""Batch/offline fitting for wheel_leg_lqr identification data.

目标
- 自动读取 identify_data/*.npz（或指定目录/文件列表），对所有数据联合拟合一组共享参数。
- 角速度 omega：离散一阶滞后（ARX(1)）。
- 线速度 v：离散二阶 ARX(2) + 多种耦合项（w, w_cmd, dw, dw_cmd, dv_cmd 等），用 Ridge 抑制过拟合。
- 支持对输入 (v_cmd, w_cmd) 做纯延迟 d 的网格搜索。

使用示例
  python3 navigation_sentry_2026/utils/py/lqr/identify_wheel_leg_fit_offline_batch.py --list
  python3 navigation_sentry_2026/utils/py/lqr/identify_wheel_leg_fit_offline_batch.py --dir identify_data
  python3 navigation_sentry_2026/utils/py/lqr/identify_wheel_leg_fit_offline_batch.py --files identify_data/a.npz identify_data/b.npz

输出
- 终端：全局最优参数 + 每个文件的 RMSE
- 图：默认不画图；传 --plot 后为每个文件保存对比图到 identify_data/<stem>__fit_global.png（不覆盖，自动加编号）

说明
- 仅依赖 numpy（matplotlib 可选）。
- 为了能跨多个数据集联合拟合，参数估计采用“1-step 线性回归”（用测量的 v[k], v[k-1] 作为回归量）。
  评估时再用递推仿真（用预测的 w_hat 驱动 v_hat），更贴近真实闭环误差。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def try_setup_matplotlib():
    try:
        import matplotlib

        if os.environ.get("DISPLAY", "") == "":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 10000):
        cand = path.with_name(f"{stem}__{i:03d}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"无法生成不冲突文件名: {path}")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def clip_finite(t: np.ndarray, *arrs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    mask = np.isfinite(t)
    for x in arrs:
        mask &= np.isfinite(x)
    t2 = t[mask]
    out = [x[mask] for x in arrs]
    return t2, out


def apply_input_delay(t: np.ndarray, u: np.ndarray, delay: float) -> np.ndarray:
    """u_d(t) = u(t-delay)，线性插值；超出范围用边界延拓。"""
    d = float(delay)
    if abs(d) < 1e-12:
        return u
    t_shift = t - d
    return np.interp(t_shift, t, u, left=float(u[0]), right=float(u[-1]))


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(win, dtype=float) / float(win)
    y = np.convolve(x_pad, k, mode="valid")
    return y


def derivative_central(x: np.ndarray, dt: float) -> np.ndarray:
    """简单中心差分，端点用前/后向差分。"""
    dt = float(max(dt, 1e-9))
    dx = np.empty_like(x, dtype=float)
    if len(x) < 3:
        dx[:] = 0.0
        return dx
    dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx


def uniform_resample(t: np.ndarray, *arrs: np.ndarray, dt: Optional[float] = None) -> Tuple[np.ndarray, float, List[np.ndarray]]:
    """将不等间隔数据插值到均匀时间轴。"""
    if len(t) < 5:
        raise RuntimeError("数据点太少，无法重采样。")

    order = np.argsort(t)
    t = t[order]
    arrs = [a[order] for a in arrs]

    dt_raw = np.diff(t)
    dt_med = float(np.median(dt_raw[np.isfinite(dt_raw)]))
    dt_use = float(dt if dt is not None else dt_med)
    dt_use = max(dt_use, 1e-4)

    t0 = float(t[0])
    t1 = float(t[-1])
    n = int(np.floor((t1 - t0) / dt_use)) + 1
    if n < 10:
        raise RuntimeError("重采样后样本太少。")
    tu = t0 + dt_use * np.arange(n, dtype=float)

    out = [np.interp(tu, t, a, left=float(a[0]), right=float(a[-1])) for a in arrs]
    return tu, dt_use, out


def ridge_solve(X: np.ndarray, y: np.ndarray, lam: float, bias_idx: Optional[int] = None) -> np.ndarray:
    """Ridge: (X^T X + lam I)^{-1} X^T y。

    bias_idx: 若指定，则该列不做正则（常数项）。
    """
    lam = float(lam)
    XtX = X.T @ X
    Xty = X.T @ y
    reg = lam * np.eye(XtX.shape[0], dtype=float)
    if bias_idx is not None:
        reg[bias_idx, bias_idx] = 0.0
    A = XtX + reg
    return np.linalg.solve(A, Xty)


# ---------------------- Dataset ----------------------


@dataclass
class Dataset:
    path: Path
    scenario: str
    t: np.ndarray
    dt: float
    v: np.ndarray
    w: np.ndarray
    v_cmd: np.ndarray
    w_cmd: np.ndarray


def load_npz(path: Path) -> Dict:
    d = np.load(path, allow_pickle=True)
    out: Dict = {k: d[k] for k in d.files}
    if "meta" in out:
        meta = out["meta"]
        try:
            if isinstance(meta, np.ndarray) and meta.dtype == object:
                out["meta"] = meta.item()
        except Exception:
            pass
    return out


def build_dataset(path: Path, resample_dt: Optional[float], smooth_win: int) -> Dataset:
    d = load_npz(path)

    t = np.asarray(d["t"], dtype=float)
    v = np.asarray(d["v_meas"], dtype=float)
    w = np.asarray(d["w_meas"], dtype=float)
    v_cmd = np.asarray(d["v_cmd"], dtype=float)
    w_cmd = np.asarray(d["w_cmd"], dtype=float)

    t, arrs = clip_finite(t, v, w, v_cmd, w_cmd)
    v, w, v_cmd, w_cmd = arrs

    tu, dt_u, outs = uniform_resample(t, v, w, v_cmd, w_cmd, dt=resample_dt)
    v_u, w_u, vcmd_u, wcmd_u = outs

    if smooth_win > 1:
        v_u = moving_average(v_u, smooth_win)
        w_u = moving_average(w_u, smooth_win)
        vcmd_u = moving_average(vcmd_u, smooth_win)
        wcmd_u = moving_average(wcmd_u, smooth_win)

    meta = d.get("meta", {})
    scenario = "unknown"
    if isinstance(meta, dict):
        scenario = str(meta.get("scenario", "unknown"))

    return Dataset(
        path=path,
        scenario=scenario,
        t=tu,
        dt=dt_u,
        v=v_u,
        w=w_u,
        v_cmd=vcmd_u,
        w_cmd=wcmd_u,
    )


def list_npz_files(out_dir: Path) -> List[Path]:
    if not out_dir.exists():
        return []
    return sorted(out_dir.glob("*.npz"))


# ---------------------- Models (discrete-time) ----------------------


@dataclass
class GlobalFitResult:
    delay: float
    ridge_lambda: float
    omega_params: Dict[str, float]
    v_params: Dict[str, float]
    v_feature_names: List[str]
    w_rmse_total: float
    v_rmse_total: float
    score: float


def fit_omega_arx1_all(dsets: Sequence[Dataset], w_cmd_delayed: Sequence[np.ndarray], ridge_lam: float) -> np.ndarray:
    """联合拟合 omega[k+1] = a*w[k] + b*w_cmd[k] + c。"""
    rows = []
    ys = []
    for ds, wcmd in zip(dsets, w_cmd_delayed):
        w = ds.w
        if len(w) < 5:
            continue
        X = np.stack([w[:-1], wcmd[:-1], np.ones(len(w) - 1, dtype=float)], axis=1)
        y = w[1:]
        rows.append(X)
        ys.append(y)

    if not rows:
        raise RuntimeError("omega：没有可用于拟合的数据集。")

    Xall = np.concatenate(rows, axis=0)
    yall = np.concatenate(ys, axis=0)

    theta = ridge_solve(Xall, yall, ridge_lam, bias_idx=2)
    return theta  # [a, b, c]


def simulate_omega_arx1(w0: float, w_cmd: np.ndarray, theta: np.ndarray) -> np.ndarray:
    a, b, c = [float(x) for x in theta]
    n = len(w_cmd)
    w_hat = np.zeros(n, dtype=float)
    w_hat[0] = float(w0)
    for k in range(n - 1):
        w_hat[k + 1] = a * w_hat[k] + b * float(w_cmd[k]) + c
    return w_hat


def omega_params_from_discrete(a: float, dt: float) -> Dict[str, float]:
    """将离散系数 a 映射为等效一阶时间常数 tau（若 a 在 (0,1) 内）。"""
    out = {"a": float(a)}
    if 1e-6 < a < 0.999999:
        out["tau_equiv"] = float(-dt / np.log(a))
    return out


def build_v_features(
    v: np.ndarray,
    v_cmd: np.ndarray,
    w: np.ndarray,
    w_cmd: np.ndarray,
    dt: float,
    include: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """构建 v 的 ARX(2) 特征：

    目标：y = v[k+1]
    特征：v[k], v[k-1], v_cmd[k], v_cmd[k-1], 以及可选耦合项的 (k, k-1)

    include 可选项：
      - w, w_cmd, dw, dw_cmd, dv_cmd
    """
    if len(v) < 6:
        raise RuntimeError("v：样本太少，无法构建 ARX(2) 特征。")

    dt = float(max(dt, 1e-6))

    # 采用中心差分得到导数（已做平滑则噪声可控）
    dw = derivative_central(w, dt)
    dw_cmd = derivative_central(w_cmd, dt)
    dv_cmd = derivative_central(v_cmd, dt)

    # 从 k=1 到 n-2：因为需要 v[k-1] 且预测 v[k+1]
    k0 = 1
    k1 = len(v) - 2

    y = v[k0 + 1 : k1 + 1]  # v[k+1]

    cols = []
    names = []

    def add_pair(sig: np.ndarray, base: str):
        cols.append(sig[k0:k1])
        names.append(f"{base}[k]")
        cols.append(sig[k0 - 1 : k1 - 1])
        names.append(f"{base}[k-1]")

    # AR part
    cols.append(v[k0:k1])
    names.append("v[k]")
    cols.append(v[k0 - 1 : k1 - 1])
    names.append("v[k-1]")

    # cmd part
    cols.append(v_cmd[k0:k1])
    names.append("v_cmd[k]")
    cols.append(v_cmd[k0 - 1 : k1 - 1])
    names.append("v_cmd[k-1]")

    include_set = set(include)
    if "w" in include_set:
        add_pair(w, "w")
    if "w_cmd" in include_set:
        add_pair(w_cmd, "w_cmd")
    if "dw" in include_set:
        add_pair(dw, "dw")
    if "dw_cmd" in include_set:
        add_pair(dw_cmd, "dw_cmd")
    if "dv_cmd" in include_set:
        add_pair(dv_cmd, "dv_cmd")

    # bias
    cols.append(np.ones_like(y))
    names.append("bias")

    X = np.stack(cols, axis=1)
    return X, y, names


def fit_v_arx2_all(
    dsets: Sequence[Dataset],
    v_cmd_delayed: Sequence[np.ndarray],
    w_hat_all: Sequence[np.ndarray],
    w_cmd_delayed: Sequence[np.ndarray],
    ridge_lam: float,
    include: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """联合拟合 v 的 ARX(2)+耦合：y = X theta（1-step）。

    注意：回归量里用测量 v（保证可堆叠且线性）；耦合项用 w_hat（与最终仿真一致）。
    """
    rows = []
    ys = []
    feat_names: Optional[List[str]] = None

    for ds, vcmd, w_hat, wcmd in zip(dsets, v_cmd_delayed, w_hat_all, w_cmd_delayed):
        X, y, names = build_v_features(ds.v, vcmd, w_hat, wcmd, ds.dt, include=include)
        rows.append(X)
        ys.append(y)
        if feat_names is None:
            feat_names = names
        else:
            if feat_names != names:
                raise RuntimeError("特征名不一致（内部错误）。")

    if not rows:
        raise RuntimeError("v：没有可用于拟合的数据集。")

    Xall = np.concatenate(rows, axis=0)
    yall = np.concatenate(ys, axis=0)

    bias_idx = int(Xall.shape[1] - 1)
    theta = ridge_solve(Xall, yall, ridge_lam, bias_idx=bias_idx)
    return theta, (feat_names or [])


def simulate_v_arx2(
    v0: float,
    v1: float,
    v_cmd: np.ndarray,
    w_hat: np.ndarray,
    w_cmd: np.ndarray,
    dt: float,
    theta: np.ndarray,
    include: Sequence[str],
    feature_names: Sequence[str],
) -> np.ndarray:
    """用拟合到的 theta 递推仿真 v_hat。"""
    n = len(v_cmd)
    if n < 3:
        return np.full(n, float(v0), dtype=float)

    dt = float(max(dt, 1e-6))

    dw_hat = derivative_central(w_hat, dt)
    dw_cmd = derivative_central(w_cmd, dt)
    dv_cmd = derivative_central(v_cmd, dt)

    v_hat = np.zeros(n, dtype=float)
    v_hat[0] = float(v0)
    v_hat[1] = float(v1)

    include_set = set(include)

    # 为了避免构建大矩阵，这里按 feature_names 的顺序生成一行特征
    for k in range(1, n - 1):
        feat: List[float] = []
        for name in feature_names:
            if name == "v[k]":
                feat.append(float(v_hat[k]))
            elif name == "v[k-1]":
                feat.append(float(v_hat[k - 1]))
            elif name == "v_cmd[k]":
                feat.append(float(v_cmd[k]))
            elif name == "v_cmd[k-1]":
                feat.append(float(v_cmd[k - 1]))
            elif name == "w[k]" and "w" in include_set:
                feat.append(float(w_hat[k]))
            elif name == "w[k-1]" and "w" in include_set:
                feat.append(float(w_hat[k - 1]))
            elif name == "w_cmd[k]" and "w_cmd" in include_set:
                feat.append(float(w_cmd[k]))
            elif name == "w_cmd[k-1]" and "w_cmd" in include_set:
                feat.append(float(w_cmd[k - 1]))
            elif name == "dw[k]" and "dw" in include_set:
                feat.append(float(dw_hat[k]))
            elif name == "dw[k-1]" and "dw" in include_set:
                feat.append(float(dw_hat[k - 1]))
            elif name == "dw_cmd[k]" and "dw_cmd" in include_set:
                feat.append(float(dw_cmd[k]))
            elif name == "dw_cmd[k-1]" and "dw_cmd" in include_set:
                feat.append(float(dw_cmd[k - 1]))
            elif name == "dv_cmd[k]" and "dv_cmd" in include_set:
                feat.append(float(dv_cmd[k]))
            elif name == "dv_cmd[k-1]" and "dv_cmd" in include_set:
                feat.append(float(dv_cmd[k - 1]))
            elif name == "bias":
                feat.append(1.0)
            else:
                # feature_names 与 include 不匹配时的兜底
                feat.append(0.0)

        v_hat[k + 1] = float(np.dot(theta, np.asarray(feat, dtype=float)))

    return v_hat


# ---------------------- Global search ----------------------


def evaluate_candidate(
    dsets: Sequence[Dataset],
    delay: float,
    ridge_lam: float,
    include: Sequence[str],
) -> GlobalFitResult:
    # delayed commands (time-domain interpolation)
    v_cmd_d = [apply_input_delay(ds.t, ds.v_cmd, delay) for ds in dsets]
    w_cmd_d = [apply_input_delay(ds.t, ds.w_cmd, delay) for ds in dsets]

    # omega fit
    theta_w = fit_omega_arx1_all(dsets, w_cmd_d, ridge_lam)

    # omega simulate per dataset
    w_hat_all = [simulate_omega_arx1(float(ds.w[0]), wcmd, theta_w) for ds, wcmd in zip(dsets, w_cmd_d)]

    # v fit (1-step) using measured v and predicted w_hat
    theta_v, feat_names = fit_v_arx2_all(
        dsets,
        v_cmd_delayed=v_cmd_d,
        w_hat_all=w_hat_all,
        w_cmd_delayed=w_cmd_d,
        ridge_lam=ridge_lam,
        include=include,
    )

    # simulate and compute RMSE
    v_errs = []
    w_errs = []
    for ds, vcmd, wcmd, w_hat in zip(dsets, v_cmd_d, w_cmd_d, w_hat_all):
        v_hat = simulate_v_arx2(
            v0=float(ds.v[0]),
            v1=float(ds.v[1]) if len(ds.v) > 1 else float(ds.v[0]),
            v_cmd=vcmd,
            w_hat=w_hat,
            w_cmd=wcmd,
            dt=ds.dt,
            theta=theta_v,
            include=include,
            feature_names=feat_names,
        )

        # 跳过前两点（初始化）
        v_errs.append(np.mean((v_hat[2:] - ds.v[2:]) ** 2))
        w_errs.append(np.mean((w_hat[1:] - ds.w[1:]) ** 2))

    v_rmse_total = float(np.sqrt(np.mean(v_errs)))
    w_rmse_total = float(np.sqrt(np.mean(w_errs)))
    score = v_rmse_total + 0.3 * w_rmse_total

    # pack params
    dt_ref = float(np.median([ds.dt for ds in dsets]))
    omega_params = {"b": float(theta_w[1]), "c": float(theta_w[2]), **omega_params_from_discrete(float(theta_w[0]), dt_ref)}
    v_params = {name: float(val) for name, val in zip(feat_names, theta_v)}

    return GlobalFitResult(
        delay=float(delay),
        ridge_lambda=float(ridge_lam),
        omega_params=omega_params,
        v_params=v_params,
        v_feature_names=list(feat_names),
        w_rmse_total=w_rmse_total,
        v_rmse_total=v_rmse_total,
        score=score,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch offline fit for wheel_leg identification data")
    parser.add_argument("--dir", type=str, default="identify_data", help="数据目录")
    parser.add_argument("--files", nargs="*", default=[], help="指定 npz 文件列表（覆盖 --dir 扫描）")
    parser.add_argument("--list", action="store_true", help="列出数据目录下所有 npz 并退出")
    parser.add_argument("--plot", action="store_true", help="保存对比图")

    parser.add_argument("--resample-dt", type=float, default=0.0, help="重采样 dt（0 表示用各文件 dt 的中位数）")
    parser.add_argument("--smooth-win", type=int, default=5, help="移动平均平滑窗口（奇数，<=1 关闭）")

    parser.add_argument("--delay-max", type=float, default=0.35, help="输入延迟搜索最大值(s)")
    parser.add_argument("--delay-step", type=float, default=0.02, help="输入延迟搜索步长(s)")

    parser.add_argument(
        "--ridge",
        type=float,
        nargs="*",
        default=[0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
        help="Ridge lambda 候选列表",
    )

    parser.add_argument(
        "--coupling",
        type=str,
        default="w,w_cmd,dw,dw_cmd,dv_cmd",
        help="v 模型耦合项：逗号分隔，可选 w,w_cmd,dw,dw_cmd,dv_cmd；空字符串表示不使用耦合",
    )

    parser.add_argument("--top", type=int, default=8, help="打印前 N 个候选结果")
    parser.add_argument("--v-ylim", type=float, nargs=2, metavar=("VMIN", "VMAX"), default=(-1.5, 1.5), help="v plot y limits: min max")
    parser.add_argument("--w-ylim", type=float, nargs=2, metavar=("WMIN", "WMAX"), default=(-6.0, 6.0), help="omega plot y limits: min max")

    args = parser.parse_args()

    out_dir = Path(args.dir)
    if args.list:
        for p in list_npz_files(out_dir):
            print(p)
        return 0

    if args.files:
        paths = [Path(p) for p in args.files]
    else:
        paths = list_npz_files(out_dir)

    paths = [p for p in paths if p.exists() and p.suffix == ".npz"]
    if not paths:
        print("没有找到任何 npz 文件。")
        return 1

    resample_dt = float(args.resample_dt)
    resample_dt = None if resample_dt <= 0.0 else resample_dt
    smooth_win = int(args.smooth_win)

    dsets: List[Dataset] = []
    for p in paths:
        try:
            ds = build_dataset(p, resample_dt=resample_dt, smooth_win=smooth_win)
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue
        dsets.append(ds)

    if not dsets:
        print("没有任何数据集成功载入/预处理。")
        return 2

    include = [x.strip() for x in str(args.coupling).split(",") if x.strip()]

    delay_max = float(args.delay_max)
    delay_step = float(args.delay_step)
    delays = np.arange(0.0, max(delay_max + 1e-12, delay_step), delay_step)
    ridge_list = [float(x) for x in args.ridge]

    results: List[GlobalFitResult] = []
    for d in delays:
        for lam in ridge_list:
            try:
                res = evaluate_candidate(dsets, delay=float(d), ridge_lam=float(lam), include=include)
            except Exception as e:
                print(f"[skip] delay={d:.3f}, ridge={lam:g}: {e}")
                continue
            results.append(res)

    if not results:
        print("没有任何候选 (delay, ridge) 拟合成功。")
        return 3

    results.sort(key=lambda r: r.score)
    best = results[0]

    print("\n===== Batch Offline Fit (Global) =====")
    print("datasets =", len(dsets))
    print("coupling =", ",".join(include) if include else "(none)")

    print("\n--- Best Candidate ---")
    print("delay(s)     =", f"{best.delay:.4f}")
    print("ridge_lambda =", f"{best.ridge_lambda:g}")
    print(f"RMSE_total(v)={best.v_rmse_total:.4f}  RMSE_total(w)={best.w_rmse_total:.4f}  score={best.score:.4f}")

    print("\n--- omega params ---")
    print(json.dumps(best.omega_params, ensure_ascii=False, sort_keys=True))

    print("\n--- v params ---")
    print(json.dumps(best.v_params, ensure_ascii=False, sort_keys=False))

    print("\n--- Top Candidates ---")
    for r in results[: max(1, int(args.top))]:
        print(
            f"delay={r.delay:.3f} ridge={r.ridge_lambda:g}  RMSE(v)={r.v_rmse_total:.4f} RMSE(w)={r.w_rmse_total:.4f} score={r.score:.4f}"
        )

    # per-file evaluation with best
    v_cmd_d = [apply_input_delay(ds.t, ds.v_cmd, best.delay) for ds in dsets]
    w_cmd_d = [apply_input_delay(ds.t, ds.w_cmd, best.delay) for ds in dsets]

    # reconstruct theta vectors
    # omega theta from params
    # note: omega_params stores a,b,c; keep exact from re-fit for safety by re-running fit
    theta_w = fit_omega_arx1_all(dsets, w_cmd_d, best.ridge_lambda)
    w_hat_all = [simulate_omega_arx1(float(ds.w[0]), wcmd, theta_w) for ds, wcmd in zip(dsets, w_cmd_d)]

    theta_v, feat_names = fit_v_arx2_all(
        dsets,
        v_cmd_delayed=v_cmd_d,
        w_hat_all=w_hat_all,
        w_cmd_delayed=w_cmd_d,
        ridge_lam=best.ridge_lambda,
        include=include,
    )

    print("\n===== Per-file RMSE (using global params) =====")
    per_file = []
    for ds, vcmd, wcmd, w_hat in zip(dsets, v_cmd_d, w_cmd_d, w_hat_all):
        v_hat = simulate_v_arx2(
            v0=float(ds.v[0]),
            v1=float(ds.v[1]) if len(ds.v) > 1 else float(ds.v[0]),
            v_cmd=vcmd,
            w_hat=w_hat,
            w_cmd=wcmd,
            dt=ds.dt,
            theta=theta_v,
            include=include,
            feature_names=feat_names,
        )
        v_e = rmse(v_hat[2:], ds.v[2:])
        w_e = rmse(w_hat[1:], ds.w[1:])
        per_file.append((ds.path, ds.scenario, v_e, w_e))
        print(f"{ds.path.name} | scenario={ds.scenario} | RMSE(v)={v_e:.4f} RMSE(w)={w_e:.4f}")

    plt = try_setup_matplotlib()
    if not args.plot:
        return 0
    if plt is None:
        print("matplotlib 不可用，无法画图。")
        return 0

    for ds, vcmd, wcmd, w_hat in zip(dsets, v_cmd_d, w_cmd_d, w_hat_all):
        v_hat = simulate_v_arx2(
            v0=float(ds.v[0]),
            v1=float(ds.v[1]) if len(ds.v) > 1 else float(ds.v[0]),
            v_cmd=vcmd,
            w_hat=w_hat,
            w_cmd=wcmd,
            dt=ds.dt,
            theta=theta_v,
            include=include,
            feature_names=feat_names,
        )

        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(ds.t, vcmd, "k--", linewidth=1.0, label="v_cmd(delayed)")
        ax1.plot(ds.t, ds.v, "b", linewidth=1.2, label="v_meas")
        ax1.plot(ds.t, v_hat, "r", linewidth=1.2, label="v_hat(global)")
        ax1.set_ylabel("v [m/s]")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best")
        ax1.set_title(f"{ds.path.name} | scenario={ds.scenario} | delay={best.delay:.3f}s | ridge={best.ridge_lambda:g}")
        if getattr(args, "v_ylim", None) is not None:
            ax1.set_ylim(float(args.v_ylim[0]), float(args.v_ylim[1]))

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(ds.t, wcmd, "k--", linewidth=1.0, label="omega_cmd(delayed)")
        ax2.plot(ds.t, ds.w, "b", linewidth=1.2, label="omega_meas")
        ax2.plot(ds.t, w_hat, "r", linewidth=1.2, label="omega_hat(global)")
        ax2.set_ylabel("omega [rad/s]")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="best")
        if getattr(args, "w_ylim", None) is not None:
            ax2.set_ylim(float(args.w_ylim[0]), float(args.w_ylim[1]))

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(ds.t[2:], ds.v[2:] - v_hat[2:], "m", linewidth=1.0, label="v_err")
        ax3.plot(ds.t[1:], ds.w[1:] - w_hat[1:], "g", linewidth=1.0, label="w_err")
        ax3.set_xlabel("t [s]")
        ax3.set_ylabel("error")
        ax3.grid(True, alpha=0.25)
        ax3.legend(loc="best")

        fig.tight_layout()

        out_png = ensure_unique_path(ds.path.with_name(ds.path.stem + "__fit_global.png"))
        fig.savefig(out_png, dpi=160)
        print("saved:", str(out_png))
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
