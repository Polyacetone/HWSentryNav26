"""System identification for wheel_leg_lqr_follow_sim closed-loop velocity response.

目标：辨识近似模型参数 (tau_v, k_v, tau_omega)，并画图展示拟合效果。

近似模型（连续时间）：
  v_dot = a
  a_dot = (1/tau_v) * (k_v * (v_cmd - v) - a)
  omega_dot = (1/tau_omega) * (omega_cmd - omega)

采集方式：
- 通过 ROS2 发布 /path_follower/chassis_cmd (interfaces/msg/ChassisCmd)
- 订阅 /serial_bridge/chassis_status (interfaces/msg/ChassisStatus) 获取 v/omega 输出

重要注意：
- wheel_leg_lqr_follow_sim 会在 1000Hz 内部做 v/w 的加速度限幅（rate-limit），并做速度/角速度以及 |v*w| 乘积限幅。
- 为避免“被限幅导致拟合不佳”，本脚本生成的激励指令会严格满足这些约束，并预留 margin。
- 角速度指令时必须同步更新 theta（积分得到），否则 theta_ref 与 omega_ref 不一致，会引入额外动态污染辨识。

运行前：
1) 确保仿真节点已在跑（wheel_leg_lqr_follow_sim.py）。
2) source 工作空间环境（能 import rclpy 与 interfaces.msg）。

示例：
  source /home/yuki/sentry_2026/install/setup.bash
  python3 navigation_sentry_2026/utils/py/sim/identify_wheel_leg_lqr_delay_model.py

输出：
- 终端打印识别结果
- 保存图像：identify_wheel_leg_lqr_delay_model.png
- 保存数据：identify_wheel_leg_lqr_delay_model_data.npz
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Limits:
    # 必须与 wheel_leg_lqr_follow_sim.py 中 SimConfig 保持一致（或更保守）
    max_v: float = 1.5
    max_w: float = 6.0
    max_acc: float = 2.0
    max_ang_acc: float = 10.0
    max_v_w_product: float = 3.0


@dataclass
class ScenarioConfig:
    name: str
    duration: float
    generator: Callable[[float], Tuple[float, float]]


def clamp_v_w_product(v: float, w: float, limits: Limits) -> Tuple[float, float]:
    v = float(np.clip(v, -limits.max_v, limits.max_v))
    w = float(np.clip(w, -limits.max_w, limits.max_w))

    vw = abs(v * w)
    if vw > limits.max_v_w_product and vw > 1e-12:
        scale = limits.max_v_w_product / vw
        s = math.sqrt(scale)
        v *= s
        w *= s
    return v, w


def rate_limit(cur: float, target: float, max_rate: float, dt: float) -> float:
    if dt <= 0.0:
        return float(cur)
    delta = target - cur
    max_delta = max_rate * dt
    delta = float(np.clip(delta, -max_delta, max_delta))
    return float(cur + delta)


def build_scenarios(limits: Limits, margin: float) -> List[ScenarioConfig]:
    """构造多个混合激励场景。

    这里 generator(t) 输出 (v_cmd_raw, w_cmd_raw) 的“期望目标”，
    实际发布前还会经过 rate-limit + clamp_v_w_product（且带 margin），确保不触发仿真限幅。
    """

    v_peak = limits.max_v * margin
    w_peak = limits.max_w * margin

    def trapezoid(t: float, t_ramp: float, t_hold: float, peak: float) -> float:
        # 0 -> peak -> 0 的对称梯形
        T = 2 * t_ramp + t_hold
        tt = t % T
        if tt < t_ramp:
            return peak * (tt / t_ramp)
        if tt < t_ramp + t_hold:
            return peak
        return peak * (1.0 - (tt - (t_ramp + t_hold)) / t_ramp)

    scenarios: List[ScenarioConfig] = []

    # 0) 静止保持
    scenarios.append(
        ScenarioConfig(
            name="idle",
            duration=5.0,
            generator=lambda t: (0.0, 0.0),
        )
    )

    # 1) 纯平动：多次梯形加减速
    scenarios.append(
        ScenarioConfig(
            name="v_trapezoid",
            duration=10.0,
            generator=lambda t: (trapezoid(t, t_ramp=1.2, t_hold=0.6, peak=+0.9 * v_peak) - 0.45 * v_peak, 0.0),
        )
    )

    # 2) 纯转动：多次梯形加减速
    scenarios.append(
        ScenarioConfig(
            name="w_trapezoid",
            duration=10.0,
            generator=lambda t: (0.0, trapezoid(t, t_ramp=0.8, t_hold=0.5, peak=0.85 * w_peak) - 0.3 * w_peak),
        )
    )

    # 3) 正弦混合：平动 + 转动同时变化（后面会靠乘积钳制）
    scenarios.append(
        ScenarioConfig(
            name="vw_sine_mix",
            duration=14.0,
            generator=lambda t: (
                0.75 * v_peak * math.sin(2 * math.pi * 0.18 * t) + 0.15 * v_peak * math.sin(2 * math.pi * 0.47 * t),
                0.65 * w_peak * math.sin(2 * math.pi * 0.11 * t + 0.7) + 0.20 * w_peak * math.sin(2 * math.pi * 0.37 * t + 1.1),
            ),
        )
    )

    # 4) 交替耦合：先加速再转向，再同时变化
    # def coupled_script(t: float) -> Tuple[float, float]:
    #     if t < 4.0:
    #         return (0.8 * v_peak * (t / 4.0), 0.0)
    #     if t < 7.0:
    #         tt = t - 4.0
    #         return (0.8 * v_peak, 0.7 * w_peak * (tt / 3.0))
    #     if t < 11.0:
    #         tt = t - 7.0
    #         return (0.8 * v_peak * (1.0 - tt / 4.0), 0.7 * w_peak * (1.0 - 2.0 * tt / 4.0))
    #     tt = t - 11.0
    #     return (0.35 * v_peak * math.sin(2 * math.pi * 0.4 * tt), 0.55 * w_peak * math.sin(2 * math.pi * 0.25 * tt + 0.3))
    # scenarios.append(ScenarioConfig(name="coupled_script", duration=16.0, generator=coupled_script))

    return scenarios


def fit_tau_omega(t: np.ndarray, omega: np.ndarray, omega_cmd: np.ndarray) -> float:
    """最小二乘拟合 omega_dot = (1/tau) * (omega_cmd - omega)."""
    dt = np.diff(t)
    domega = np.diff(omega)

    # 对齐到 k (0..N-2): y_k = domega/dt, x_k = omega_cmd[k] - omega[k]
    y = domega / np.maximum(dt, 1e-9)
    x = omega_cmd[:-1] - omega[:-1]

    mask = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-4) & (dt > 1e-4)
    if np.count_nonzero(mask) < 20:
        raise RuntimeError("omega 可用数据太少（可能 omega_cmd 变化不足或数据没收齐）。")

    x_m = x[mask]
    y_m = y[mask]

    p = float(np.dot(x_m, y_m) / np.dot(x_m, x_m))  # p = 1/tau
    if p <= 1e-9:
        raise RuntimeError(f"拟合得到 1/tau_omega={p} 非正，数据/场景可能不合适。")
    return 1.0 / p


def fit_tau_k_v(t: np.ndarray, v: np.ndarray, v_cmd: np.ndarray) -> Tuple[float, float]:
    """最小二乘拟合：

    a_dot = (k/tau)*(v_cmd - v) - (1/tau)*a

    用差分近似 a 与 a_dot，从而对 p1=k/tau, p2=1/tau 做线性最小二乘。
    """

    # a_i 定义在采样点 i：a_i ≈ (v[i]-v[i-1])/(t[i]-t[i-1])
    dt1 = np.diff(t)
    dv = np.diff(v)
    a = dv / np.maximum(dt1, 1e-9)  # length N-1, aligned to i=1..N-1

    # a_dot_i aligned to i=1..N-2
    da = np.diff(a)
    dt2 = t[2:] - t[1:-1]
    a_dot = da / np.maximum(dt2, 1e-9)

    # Build regression at i=1..N-2
    v_i = v[1:-1]
    vcmd_i = v_cmd[1:-1]
    a_i = a[:-1]

    y = a_dot
    x1 = vcmd_i - v_i
    x2 = a_i

    mask = (
        np.isfinite(y)
        & np.isfinite(x1)
        & np.isfinite(x2)
        & (np.abs(x1) + np.abs(x2) > 1e-6)
        & (dt2 > 1e-4)
    )

    if np.count_nonzero(mask) < 30:
        raise RuntimeError("v 可用数据太少（可能 v_cmd 变化不足或数据没收齐）。")

    A = np.stack([x1[mask], -x2[mask]], axis=1)  # y = p1*x1 + p2*(-x2)
    b = y[mask]

    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    p1 = float(p[0])
    p2 = float(p[1])

    if p2 <= 1e-9:
        raise RuntimeError(f"拟合得到 1/tau_v={p2} 非正，数据/场景可能不合适。")

    tau_v = 1.0 / p2
    k_v = p1 / p2
    return tau_v, k_v


def simulate_ident_model(t: np.ndarray, v_cmd: np.ndarray, w_cmd: np.ndarray, tau_v: float, k_v: float, tau_w: float,
                        v0: float, omega0: float) -> Tuple[np.ndarray, np.ndarray]:
    """用识别得到的模型在采样时刻上仿真输出。"""
    n = len(t)
    v_hat = np.zeros(n, dtype=float)
    w_hat = np.zeros(n, dtype=float)

    v_hat[0] = float(v0)
    w_hat[0] = float(omega0)

    # 初始加速度用首段差分估计更稳一点
    if n >= 2:
        dt0 = max(float(t[1] - t[0]), 1e-6)
        a = float((v_hat[0] - v0) / dt0)  # 0
    else:
        a = 0.0

    for i in range(n - 1):
        dt = float(t[i + 1] - t[i])
        dt = max(dt, 1e-6)

        # v/a second-order lag
        v_err = float(v_cmd[i] - v_hat[i])
        v_dot = a
        a_dot = (1.0 / tau_v) * (k_v * v_err - a)
        v_hat[i + 1] = v_hat[i] + v_dot * dt
        a = a + a_dot * dt

        # omega first-order lag
        w_err = float(w_cmd[i] - w_hat[i])
        w_dot = (1.0 / tau_w) * w_err
        w_hat[i + 1] = w_hat[i] + w_dot * dt

    return v_hat, w_hat


def try_setup_matplotlib():
    # 在无显示环境下使用 Agg
    try:
        import matplotlib

        if os.environ.get("DISPLAY", "") == "":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def main() -> int:
    # 延迟导入 ROS，避免在未 source 环境下直接报错
    try:
        import rclpy
        from rclpy.node import Node
        from interfaces.msg import ChassisCmd, ChassisStatus
    except Exception as e:
        print("无法导入 ROS2 依赖（rclpy / interfaces.msg）。请先 source install/setup.bash。\n", e)
        return 2

    limits = Limits()

    # 为了尽可能不触发仿真限幅，这里用更保守的 margin
    # 同时发布侧还会做 rate-limit，确保 dv/dt、dw/dt 满足加速度约束
    margin = 0.75

    cmd_hz = 10.0  # 发布指令频率；越高越容易让内部 rate-limit 不生效
    cmd_dt = 1.0 / cmd_hz

    scenarios = build_scenarios(limits, margin)

    # 预热一段时间让仿真稳定（姿态/腿长等调整）
    warmup_sec = 1.0

    # 记录数组（命令与状态分别记录，后续按时间插值对齐）
    cmd_t: List[float] = []
    cmd_v: List[float] = []
    cmd_w: List[float] = []
    cmd_theta: List[float] = []

    st_t: List[float] = []
    st_v: List[float] = []
    st_w: List[float] = []

    class IdentNode(Node):
        def __init__(self):
            super().__init__("wheel_leg_lqr_ident")
            self.pub = self.create_publisher(ChassisCmd, "/path_follower/chassis_cmd", 1)
            self.sub = self.create_subscription(ChassisStatus, "/serial_bridge/chassis_status", self._status_cb, 1)

            self._t0_wall = time.time()
            self._t_prev_cmd = self._t0_wall

            self._scenario_idx = -1
            self._scenario_t0 = self._t0_wall
            self._theta = 0.0

            self._v_cur = 0.0
            self._w_cur = 0.0

            self._timer = self.create_timer(cmd_dt, self._on_timer)

            total = warmup_sec + sum(s.duration for s in scenarios)
            self.get_logger().info(
                f"ident start: cmd_hz={cmd_hz}, warmup={warmup_sec}s, scenarios={len(scenarios)}, total~{total:.1f}s"
            )

        def _status_cb(self, msg: ChassisStatus) -> None:
            t_now = time.time() - self._t0_wall
            st_t.append(float(t_now))
            st_v.append(float(msg.velocity))
            st_w.append(float(msg.omega))

        def _publish_cmd(self, v: float, w: float, theta: float) -> None:
            msg = ChassisCmd()
            msg.velocity = float(v)
            msg.omega = float(w)
            msg.theta = float(theta)
            msg.slow_spin = False
            msg.fast_spin = False
            self.pub.publish(msg)

        def _on_timer(self) -> None:
            wall = time.time()
            dt = max(wall - self._t_prev_cmd, 1e-6)
            self._t_prev_cmd = wall

            elapsed = wall - self._t0_wall

            # warmup
            if elapsed < warmup_sec:
                v_tgt, w_tgt = 0.0, 0.0
            else:
                # locate scenario
                t_in = elapsed - warmup_sec
                idx = 0
                acc = 0.0
                while idx < len(scenarios) and (acc + scenarios[idx].duration) < t_in:
                    acc += scenarios[idx].duration
                    idx += 1

                if idx >= len(scenarios):
                    # done
                    self._publish_cmd(0.0, 0.0, self._theta)
                    self.get_logger().info("ident done: stop publishing")
                    raise SystemExit

                if idx != self._scenario_idx:
                    self._scenario_idx = idx
                    self._scenario_t0 = wall
                    self.get_logger().info(f"scenario[{idx}] = {scenarios[idx].name}")

                local_t = t_in - acc
                v_raw, w_raw = scenarios[idx].generator(local_t)

                # 1) rate-limit（用更保守的 max_rate，减少触发仿真内部 rate-limit 的概率）
                v_tgt = rate_limit(self._v_cur, v_raw, max_rate=limits.max_acc * margin, dt=dt)
                w_tgt = rate_limit(self._w_cur, w_raw, max_rate=limits.max_ang_acc * margin, dt=dt)

                # 2) clamp with margin
                v_tgt, w_tgt = clamp_v_w_product(v_tgt, w_tgt, Limits(
                    max_v=limits.max_v * margin,
                    max_w=limits.max_w * margin,
                    max_acc=limits.max_acc,
                    max_ang_acc=limits.max_ang_acc,
                    max_v_w_product=limits.max_v_w_product * margin,
                ))

            # update theta to be consistent with omega reference
            self._theta = wrap_to_pi(self._theta + float(w_tgt) * dt)

            self._v_cur = float(v_tgt)
            self._w_cur = float(w_tgt)

            t_rel = wall - self._t0_wall
            cmd_t.append(float(t_rel))
            cmd_v.append(float(self._v_cur))
            cmd_w.append(float(self._w_cur))
            cmd_theta.append(float(self._theta))

            self._publish_cmd(self._v_cur, self._w_cur, self._theta)

    # 运行采集
    rclpy.init()
    node = IdentNode()

    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if len(st_t) < 80:
        print(f"状态数据太少: {len(st_t)} 点。请确认 wheel_leg_lqr_follow_sim 在运行，且 /serial_bridge/chassis_status 有数据。")
        return 3

    # 对齐：将 cmd 插值到 status 时间轴
    t = np.asarray(st_t, dtype=float)
    v_meas = np.asarray(st_v, dtype=float)
    w_meas = np.asarray(st_w, dtype=float)

    cmd_t_np = np.asarray(cmd_t, dtype=float)
    cmd_v_np = np.asarray(cmd_v, dtype=float)
    cmd_w_np = np.asarray(cmd_w, dtype=float)

    # 去掉时间倒序/重复点
    order = np.argsort(t)
    t = t[order]
    v_meas = v_meas[order]
    w_meas = w_meas[order]

    # 插值
    v_cmd = np.interp(t, cmd_t_np, cmd_v_np)
    w_cmd = np.interp(t, cmd_t_np, cmd_w_np)

    # 辨识
    tau_omega = fit_tau_omega(t, w_meas, w_cmd)
    tau_v, k_v = fit_tau_k_v(t, v_meas, v_cmd)

    # 仿真并评估
    v_hat, w_hat = simulate_ident_model(
        t=t,
        v_cmd=v_cmd,
        w_cmd=w_cmd,
        tau_v=tau_v,
        k_v=k_v,
        tau_w=tau_omega,
        v0=float(v_meas[0]),
        omega0=float(w_meas[0]),
    )

    v_rmse = float(np.sqrt(np.mean((v_hat - v_meas) ** 2)))
    w_rmse = float(np.sqrt(np.mean((w_hat - w_meas) ** 2)))

    print("\n===== Identification Result =====")
    print(f"tau_v     = {tau_v:.4f}  [s]")
    print(f"k_v       = {k_v:.4f}  [-]")
    print(f"tau_omega = {tau_omega:.4f}  [s]")
    print(f"RMSE(v)   = {v_rmse:.4f}  [m/s]")
    print(f"RMSE(w)   = {w_rmse:.4f}  [rad/s]")

    out_npz = "identify_wheel_leg_lqr_delay_model_data.npz"
    np.savez(
        out_npz,
        t=t,
        v_cmd=v_cmd,
        w_cmd=w_cmd,
        v_meas=v_meas,
        w_meas=w_meas,
        v_hat=v_hat,
        w_hat=w_hat,
        tau_v=tau_v,
        k_v=k_v,
        tau_omega=tau_omega,
    )

    plt = try_setup_matplotlib()
    if plt is None:
        print("matplotlib 不可用，已保存 npz 数据：", out_npz)
        return 0

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, v_cmd, "k--", linewidth=1.0, label="v_cmd")
    ax1.plot(t, v_meas, "b", linewidth=1.3, label="v_meas")
    ax1.plot(t, v_hat, "r", linewidth=1.2, label="v_hat")
    ax1.set_ylabel("v [m/s]")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    ax1.set_title(f"Fit: tau_v={tau_v:.3f}s, k_v={k_v:.3f}, tau_omega={tau_omega:.3f}s")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, w_cmd, "k--", linewidth=1.0, label="omega_cmd")
    ax2.plot(t, w_meas, "b", linewidth=1.3, label="omega_meas")
    ax2.plot(t, w_hat, "r", linewidth=1.2, label="omega_hat")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("omega [rad/s]")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()
    out_png = "identify_wheel_leg_lqr_delay_model.png"
    fig.savefig(out_png, dpi=160)
    print("saved:", out_png)

    # 如果有显示则弹窗
    if os.environ.get("DISPLAY", "") != "":
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
