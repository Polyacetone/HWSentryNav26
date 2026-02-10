"""Record data for wheel_leg_lqr_follow_sim identification.

目标：发布激励指令并录制 /serial_bridge/chassis_status，结束后保存 npz + png。

特性：
- 场景由可独立组合的 v/w profiles 构成（支持 v 匀加速 + w 正弦等组合）
- 支持预置 scenario（兼容之前一键跑常用组合）
- 发布侧做 rate-limit + 速度/角速度/|v*w| 钳制（带 margin），尽量避免触发仿真内部限幅
- 启动时查询 TF，使用 imu_world<-chassis_link 的姿态计算 theta0（查不到直接报错退出）
- 输出文件不覆盖：统一保存到 identify_data/ 下，按时间戳+场景名命名

运行示例：
  source /home/yuki/sentry_2026/install/setup.bash

  # 查看预置组合
  python3 navigation_sentry_2026/utils/py/lqr/identify_wheel_leg_record.py --list-scenarios

  # 查看可组合的 1D profiles
  python3 navigation_sentry_2026/utils/py/lqr/identify_wheel_leg_record.py --list-v-profiles
  python3 navigation_sentry_2026/utils/py/lqr/identify_wheel_leg_record.py --list-w-profiles

  # 预置场景
  python3 navigation_sentry_2026/utils/py/lqr/identify_wheel_leg_record.py --scenario v_const_w_sine

  # 自由组合：v 匀加速/匀减速，同时 w 正弦
  python3 navigation_sentry_2026/utils/py/lqr/identify_wheel_leg_record.py --v-profile v_ramp --w-profile w_sine

输出：
- identify_data/<timestamp>__<scenario>__<tag>.npz
- identify_data/<timestamp>__<scenario>__<tag>.png
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Limits:
    # 建议与 wheel_leg_lqr_follow_sim.py 中 SimConfig 一致（或更保守）
    max_v: float = 1.5
    max_w: float = 6.0
    max_acc: float = 2.5
    max_ang_acc: float = 6.0
    max_v_w_product: float = 3.0


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


# ---------------------- Profile DSL (segments, 1D) ----------------------


@dataclass(frozen=True)
class Segment1D:
    name: str
    duration: float
    generator: Callable[[float], float]  # local_t -> value


@dataclass(frozen=True)
class Profile1D:
    name: str
    segments: Tuple[Segment1D, ...]

    @property
    def duration(self) -> float:
        return float(sum(s.duration for s in self.segments))

    def sample(self, local_t: float) -> float:
        t = float(local_t)
        for seg in self.segments:
            if t <= seg.duration + 1e-12:
                return float(seg.generator(t))
            t -= seg.duration
        return float(self.segments[-1].generator(self.segments[-1].duration))


@dataclass(frozen=True)
class Scenario:
    """2D scenario composed by independent v/w profiles."""

    name: str
    v_profile: Profile1D
    w_profile: Profile1D

    @property
    def duration(self) -> float:
        return float(max(self.v_profile.duration, self.w_profile.duration))

    def sample(self, local_t: float) -> Tuple[float, float]:
        t = float(local_t)
        return float(self.v_profile.sample(t)), float(self.w_profile.sample(t))


def seg_hold(duration: float, value: float, name: str = "hold") -> Segment1D:
    return Segment1D(name=name, duration=float(duration), generator=lambda t: float(value))


def seg_ramp(duration: float, x0: float, x1: float, name: str = "ramp") -> Segment1D:
    dur = float(duration)

    def _gen(t: float) -> float:
        if dur <= 1e-9:
            return float(x1)
        s = float(np.clip(float(t) / dur, 0.0, 1.0))
        x = (1.0 - s) * x0 + s * x1
        return float(x)

    return Segment1D(name=name, duration=dur, generator=_gen)


def seg_sine(duration: float, amp: float, bias: float, hz: float, phase: float = 0.0, name: str = "sine") -> Segment1D:
    dur = float(duration)

    def _gen(t: float) -> float:
        tt = float(t)
        x = bias + amp * math.sin(2.0 * math.pi * float(hz) * tt + float(phase))
        return float(x)

    return Segment1D(name=name, duration=dur, generator=_gen)


def seg_chirp(duration: float, f0: float, f1: float, amp: float, name: str = "chirp") -> Segment1D:
    """线性扫频信号: 从 f0 Hz 扫到 f1 Hz"""
    dur = float(duration)
    def _gen(t: float) -> float:
        # 频率随时间线性增加: f(t) = f0 + (f1-f0)*t/T
        # 相位是频率的积分: phi(t) = 2*pi * [f0*t + 0.5*(f1-f0)*t^2/T]
        phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) * (t**2) / dur)
        return float(amp * math.sin(phase))
    return Segment1D(name=name, duration=dur, generator=_gen)


def seg_stairs(duration: float, steps: List[float], name: str = "stairs") -> Segment1D:
    """阶梯步进信号: 将 duration 平均分配给多个幅值"""
    dur = float(duration)
    n = len(steps)
    def _gen(t: float) -> float:
        idx = min(int(t / (dur / n)), n - 1)
        return float(steps[idx])
    return Segment1D(name=name, duration=dur, generator=_gen)


def seg_pulse(duration: float, amp: float, width: float, name: str = "pulse") -> Segment1D:
    """脉冲信号: 在中间位置触发一个宽度为 width 的脉冲"""
    dur = float(duration)
    def _gen(t: float) -> float:
        if (dur/2 - width/2) <= t <= (dur/2 + width/2):
            return float(amp)
        return 0.0
    return Segment1D(name=name, duration=dur, generator=_gen)


def build_v_profiles(limits: Limits, margin: float) -> Dict[str, Profile1D]:
    v_peak = limits.max_v * margin
    profiles: Dict[str, Profile1D] = {}

    profiles["idle"] = Profile1D(
        name="idle",
        segments=(seg_hold(duration=10.0, value=0.0, name="idle"),),
    )

    profiles["v_const"] = Profile1D(
        name="v_const",
        segments=(
            seg_hold(duration=1.0, value=0.0, name="idle"),
            seg_ramp(duration=1.0, x0=0.0, x1=v_peak, name="v_accel"),
            seg_hold(duration=6.0, value=v_peak, name="v_hold"),
            seg_ramp(duration=1.0, x0=v_peak, x1=0.0, name="v_decel"),
            seg_hold(duration=1.0, value=0.0, name="idle"),
        ),
    )

    profiles["v_ramp"] = Profile1D(
        name="v_ramp",
        segments=(
            seg_hold(duration=1.0, value=0.0, name="idle"),
            seg_ramp(duration=1.0, x0=0.0, x1=v_peak, name="v_accel"),
            seg_ramp(duration=2.0, x0=v_peak, x1=-v_peak, name="v_decel"),
            seg_ramp(duration=2.0, x0=-v_peak, x1=v_peak, name="v_accel"),
            seg_ramp(duration=2.0, x0=v_peak, x1=-v_peak, name="v_decel"),
            seg_ramp(duration=1.0, x0=v_peak, x1=0.0, name="v_recover"),
            seg_hold(duration=1.0, value=0.0, name="idle"),
        ),
    )

    profiles["v_sine"] = Profile1D(
        name="v_sine",
        segments=(
            seg_hold(duration=2.0, value=0.0, name="idle"),
            seg_sine(duration=6.0, amp=v_peak, bias=0.0, hz=0.25, phase=0.0, name="v_sine"),
            seg_hold(duration=2.0, value=0.0, name="idle"),
        ),
    )

    # 1. 基础阶梯：测线性度和死区
    profiles["v_stairs"] = Profile1D(
        name="v_stairs",
        segments=(
            seg_hold(duration=2.0, value=0.0),
            seg_stairs(duration=6.0, steps=[v_peak*0.25, v_peak*0.5, v_peak*0.75, v_peak]),
            seg_hold(duration=2.0, value=0.0),
        )
    )

    # 2. 扫频：测带宽和谐振
    profiles["v_chirp"] = Profile1D(
        name="v_chirp",
        segments=(
            seg_hold(duration=2.0, value=0.0),
            seg_chirp(duration=6.0, f0=0.1, f1=0.5, amp=v_peak*0.6),
            seg_hold(duration=2.0, value=0.0),
        )
    )

    return profiles


def build_w_profiles(limits: Limits, margin: float) -> Dict[str, Profile1D]:
    w_peak = limits.max_w * margin
    profiles: Dict[str, Profile1D] = {}

    profiles["idle"] = Profile1D(
        name="idle",
        segments=(seg_hold(duration=10.0, value=0.0, name="idle"),),
    )

    profiles["w_const"] = Profile1D(
        name="w_const",
        segments=(
            seg_hold(duration=1.0, value=0.0, name="idle"),
            seg_ramp(duration=1.0, x0=0.0, x1=w_peak, name="w_accel"),
            seg_hold(duration=6.0, value=w_peak, name="w_hold"),
            seg_ramp(duration=1.0, x0=w_peak, x1=0.0, name="w_decel"),
            seg_hold(duration=1.0, value=0.0, name="idle"),
        ),
    )

    profiles["w_ramp"] = Profile1D(
        name="w_ramp",
        segments=(
            seg_hold(duration=1.0, value=0.0, name="idle"),
            seg_ramp(duration=1.0, x0=0.0, x1=w_peak, name="w_accel"),
            seg_ramp(duration=2.0, x0=w_peak, x1=-w_peak, name="w_decel"),
            seg_ramp(duration=2.0, x0=-w_peak, x1=w_peak, name="w_accel"),
            seg_ramp(duration=2.0, x0=w_peak, x1=-w_peak, name="w_decel"),
            seg_ramp(duration=1.0, x0=-w_peak, x1=0.0, name="w_recover"),
            seg_hold(duration=1.0, value=0.0, name="idle"),
        ),
    )

    profiles["w_sine"] = Profile1D(
        name="w_sine",
        segments=(
            seg_hold(duration=2.0, value=0.0, name="idle"),
            seg_sine(duration=6.0, amp=w_peak, bias=0.0, hz=0.25, phase=0.0, name="w_sine"),
            seg_hold(duration=2.0, value=0.0, name="idle"),
        ),
    )

    # 1. 转向扫频 (0.2Hz -> 8Hz)
    profiles["w_chirp"] = Profile1D(
        name="w_chirp",
        segments=(
            seg_hold(duration=2.0, value=0.0),
            seg_chirp(duration=6.0, f0=0.2, f1=1.0, amp=w_peak*0.6),
            seg_hold(duration=2.0, value=0.0),
        )
    )

    # 2. 快速左右切换
    profiles["w_switch"] = Profile1D(
        name="w_switch",
        segments=(
            seg_hold(duration=2.0, value=0.0),
            seg_stairs(duration=6.0, steps=[w_peak*0.5, -w_peak*0.6, w_peak*0.7, -w_peak*0.8]),
            seg_hold(duration=2.0, value=0.0),
        )
    )

    return profiles


def build_preset_scenarios(v_profiles: Dict[str, Profile1D], w_profiles: Dict[str, Profile1D]) -> Dict[str, Scenario]:
    scenarios: Dict[str, Scenario] = {}

    def _mk(name: str, v_name: str, w_name: str) -> Scenario:
        return Scenario(name=name, v_profile=v_profiles[v_name], w_profile=w_profiles[w_name])

    scenarios["v_const"] = _mk("v_const", "v_const", "idle")
    scenarios["v_ramp"] = _mk("v_ramp", "v_ramp", "idle")
    scenarios["v_sine"] = _mk("v_sine", "v_sine", "idle")
    scenarios["v_stairs"] = _mk("v_stairs", "v_stairs", "idle")
    scenarios["v_chirp"] = _mk("v_chirp", "v_chirp", "idle")

    scenarios["w_const"] = _mk("w_const", "idle", "w_const")
    scenarios["w_ramp"] = _mk("w_ramp", "idle", "w_ramp")
    scenarios["w_sine"] = _mk("w_sine", "idle", "w_sine")
    scenarios["w_switch"] = _mk("w_switch", "idle", "w_switch")
    scenarios["w_chirp"] = _mk("w_chirp", "idle", "w_chirp")

    scenarios["v_const_w_const"] = _mk("v_const_w_const", "v_const", "w_const")
    scenarios["v_const_w_sine"] = _mk("v_const_w_sine", "v_const", "w_sine")
    scenarios["v_sine_w_const"] = _mk("v_sine_w_const", "v_sine", "w_const")
    scenarios["v_ramp_w_ramp"] = _mk("v_ramp_w_ramp", "v_ramp", "w_ramp")
    scenarios["v_stairs_w_switch"] = _mk("v_stairs_w_switch", "v_stairs", "w_switch")
    scenarios["v_chirp_w_chirp"] = _mk("v_chirp_w_chirp", "v_chirp", "w_chirp")

    return scenarios


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


def select_scenario_interactive(names: Sequence[str]) -> str:
    print("\n可用场景：")
    for i, name in enumerate(names):
        print(f"  [{i:02d}] {name}")

    while True:
        s = input("请选择场景编号或名称：").strip()
        if s == "":
            continue
        if s in names:
            return s
        try:
            idx = int(s)
            if 0 <= idx < len(names):
                return names[idx]
        except Exception:
            pass
        print("输入无效，请重试。")


def select_profile_interactive(kind: str, names: Sequence[str], default_name: str = "idle") -> str:
    print(f"\n可用 {kind} profiles：")
    for i, name in enumerate(names):
        mark = " (default)" if name == default_name else ""
        print(f"  [{i:02d}] {name}{mark}")

    prompt = f"请选择 {kind} profile 编号或名称（回车默认 {default_name}）："
    while True:
        s = input(prompt).strip()
        if s == "":
            if default_name in names:
                return default_name
            return names[0]
        if s in names:
            return s
        try:
            idx = int(s)
            if 0 <= idx < len(names):
                return names[idx]
        except Exception:
            pass
        print("输入无效，请重试。")


def quat_xy_xaxis_yaw(x: float, y: float, z: float, w: float) -> float:
    """Compute yaw of body x-axis projection on world xy plane."""

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    zw = z * w

    r00 = ww + xx - yy - zz
    r10 = 2.0 * (xy + zw)
    return wrap_to_pi(math.atan2(r10, r00))


def lookup_initial_theta_from_tf(node, world_frame: str, chassis_frame: str, timeout_sec: float) -> float:
    """Query TF and compute initial theta0.

    theta0 is chassis_link x-axis projection angle on imu_world xy plane.
    Must succeed; otherwise raises RuntimeError.
    """

    try:
        import tf2_ros
    except Exception as e:
        raise RuntimeError("无法导入 tf2_ros；请确认已安装并 source ROS2 环境") from e

    buffer = tf2_ros.Buffer()
    _listener = tf2_ros.TransformListener(buffer, node)

    t_start = time.time()
    last_err: Optional[BaseException] = None
    while True:
        try:
            trans = buffer.lookup_transform(world_frame, chassis_frame, rclpy.time.Time())
            q = trans.transform.rotation
            theta0 = quat_xy_xaxis_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
            node.get_logger().info(
                f"tf theta0: {world_frame}<-{chassis_frame}: yaw={theta0:.6f} rad ({theta0 * 180.0 / math.pi:.2f} deg)"
            )
            return float(theta0)
        except Exception as e:
            last_err = e
            if time.time() - t_start >= float(timeout_sec):
                break
            try:
                import rclpy

                rclpy.spin_once(node, timeout_sec=0.05)
            except Exception:
                time.sleep(0.05)

    raise RuntimeError(
        f"TF 查询失败（{timeout_sec:.2f}s 内未获得 {world_frame} <- {chassis_frame}）: {last_err}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Record identification data for wheel_leg_lqr_follow_sim")
    parser.add_argument("--scenario", type=str, default="", help="预置场景名称；与 v/w profile 互斥（优先使用 scenario）")
    parser.add_argument("--list-scenarios", action="store_true", help="列出所有预置场景并退出")
    parser.add_argument("--v-profile", type=str, default="", help="v profile 名称（可与 w profile 自由组合）")
    parser.add_argument("--w-profile", type=str, default="", help="w profile 名称（可与 v profile 自由组合）")
    parser.add_argument("--list-v-profiles", action="store_true", help="列出所有 v profiles 并退出")
    parser.add_argument("--list-w-profiles", action="store_true", help="列出所有 w profiles 并退出")
    parser.add_argument("--tag", type=str, default="", help="附加标签，会写入文件名")
    parser.add_argument("--cmd-hz", type=float, default=10.0, help="发布指令频率")
    parser.add_argument("--warmup", type=float, default=5.0, help="预热时长(s)")
    parser.add_argument("--margin", type=float, default=0.6, help="限幅裕度(0~1)，越小越保守")
    parser.add_argument("--out-dir", type=str, default="identify_data", help="输出目录")
    parser.add_argument("--tf-world", type=str, default="imu_world", help="TF world frame（默认 imu_world）")
    parser.add_argument("--tf-chassis", type=str, default="chassis_link", help="TF chassis frame（默认 chassis_link）")
    parser.add_argument("--tf-timeout", type=float, default=2.0, help="TF 查询超时(s)，超时直接报错退出")
    args = parser.parse_args()

    limits = Limits()
    margin = float(args.margin)
    cmd_hz = float(args.cmd_hz)
    cmd_dt = 1.0 / cmd_hz
    warmup_sec = float(args.warmup)

    v_profiles = build_v_profiles(limits, margin)
    w_profiles = build_w_profiles(limits, margin)
    preset_scenarios = build_preset_scenarios(v_profiles, w_profiles)

    v_names = sorted(v_profiles.keys())
    w_names = sorted(w_profiles.keys())
    scenario_names = sorted(preset_scenarios.keys())

    if args.list_v_profiles:
        for n in v_names:
            print(n)
        return 0

    if args.list_w_profiles:
        for n in w_names:
            print(n)
        return 0

    if args.list_scenarios:
        for n in scenario_names:
            print(n)
        return 0

    # 延迟导入 ROS，避免未 source 环境就报错
    try:
        import rclpy
        from rclpy.node import Node
        from interfaces.msg import ChassisCmd, ChassisStatus
    except Exception as e:
        print("无法导入 ROS2 依赖（rclpy / interfaces.msg）。请先 source install/setup.bash。\n", e)
        return 2

    # Choose scenario
    scenario_name = str(args.scenario).strip()
    v_profile_name = str(args.v_profile).strip()
    w_profile_name = str(args.w_profile).strip()

    scenario: Scenario
    if scenario_name != "":
        if scenario_name not in preset_scenarios:
            print(f"未知场景: {scenario_name}")
            print("可用：", ", ".join(scenario_names))
            return 1
        scenario = preset_scenarios[scenario_name]
    else:
        if v_profile_name == "" and w_profile_name == "":
            chosen = select_scenario_interactive(scenario_names)
            scenario = preset_scenarios[chosen]
        else:
            if v_profile_name == "":
                v_profile_name = select_profile_interactive("v", v_names, default_name="idle")
            if w_profile_name == "":
                w_profile_name = select_profile_interactive("w", w_names, default_name="idle")

            if v_profile_name not in v_profiles:
                print(f"未知 v profile: {v_profile_name}")
                print("可用：", ", ".join(v_names))
                return 1
            if w_profile_name not in w_profiles:
                print(f"未知 w profile: {w_profile_name}")
                print("可用：", ", ".join(w_names))
                return 1

            scenario = Scenario(
                name=f"v={v_profile_name}__w={w_profile_name}",
                v_profile=v_profiles[v_profile_name],
                w_profile=w_profiles[w_profile_name],
            )

    total_sec = warmup_sec + scenario.duration
    print(
        f"\n即将开始录制：scenario={scenario.name}, warmup={warmup_sec:.2f}s, duration={scenario.duration:.2f}s, total~{total_sec:.2f}s"
    )

    # 记录数组（命令与状态分别记录，后续按时间插值对齐）
    cmd_t: List[float] = []
    cmd_v: List[float] = []
    cmd_w: List[float] = []
    cmd_theta: List[float] = []

    st_t: List[float] = []
    st_v: List[float] = []
    st_w: List[float] = []

    class RecordNode(Node):
        def __init__(self):
            super().__init__("wheel_leg_lqr_ident_record")
            self.pub = self.create_publisher(ChassisCmd, "/path_follower/chassis_cmd", 1)
            self.sub = self.create_subscription(ChassisStatus, "/serial_bridge/chassis_status", self._status_cb, 1)

            self._t0_wall = time.time()
            self._t_prev_cmd = self._t0_wall

            self._theta0 = lookup_initial_theta_from_tf(
                self,
                world_frame=str(args.tf_world),
                chassis_frame=str(args.tf_chassis),
                timeout_sec=float(args.tf_timeout),
            )
            self._theta = float(self._theta0)

            self._v_cur = 0.0
            self._w_cur = 0.0

            self._recording_started = False
            self._timer = self.create_timer(cmd_dt, self._on_timer)
            self.get_logger().info(
                f"record start: cmd_hz={cmd_hz:g}, warmup={warmup_sec:g}s, scenario={scenario.name}, total~{total_sec:.1f}s, theta0={self._theta0:.6f}"
            )

        def _status_cb(self, msg: ChassisStatus) -> None:
            t_now = time.time() - self._t0_wall
            if t_now < warmup_sec:
                return
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

            if elapsed < warmup_sec:
                v_raw, w_raw = 0.0, 0.0
            else:
                v_raw, w_raw = scenario.sample(elapsed - warmup_sec)

            v_tgt = rate_limit(self._v_cur, float(v_raw), max_rate=limits.max_acc * margin, dt=dt)
            w_tgt = rate_limit(self._w_cur, float(w_raw), max_rate=limits.max_ang_acc * margin, dt=dt)

            v_tgt, w_tgt = clamp_v_w_product(
                v_tgt,
                w_tgt,
                Limits(
                    max_v=limits.max_v * margin,
                    max_w=limits.max_w * margin,
                    max_acc=limits.max_acc,
                    max_ang_acc=limits.max_ang_acc,
                    max_v_w_product=limits.max_v_w_product * margin,
                ),
            )

            # trapezoidal integration for theta: use average of previous and current angular rates
            self._theta = wrap_to_pi(self._theta + 0.5 * (float(self._w_cur) + float(w_tgt)) * dt)
            self._v_cur = float(v_tgt)
            self._w_cur = float(w_tgt)

            t_rel = wall - self._t0_wall
            if elapsed >= warmup_sec:
                if not self._recording_started:
                    self.get_logger().info("warmup finished: start recording")
                    self._recording_started = True
                cmd_t.append(float(t_rel))
                cmd_v.append(float(self._v_cur))
                cmd_w.append(float(self._w_cur))
                cmd_theta.append(float(self._theta))

            self._publish_cmd(self._v_cur, self._w_cur, self._theta)

            if elapsed >= total_sec:
                self._publish_cmd(0.0, 0.0, self._theta)
                self.get_logger().info("record done: stop publishing")
                raise SystemExit

    rclpy.init()
    node = RecordNode()
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

    t = np.asarray(st_t, dtype=float)
    v_meas = np.asarray(st_v, dtype=float)
    w_meas = np.asarray(st_w, dtype=float)
    order = np.argsort(t)
    t = t[order]
    v_meas = v_meas[order]
    w_meas = w_meas[order]

    cmd_t_np = np.asarray(cmd_t, dtype=float)
    cmd_v_np = np.asarray(cmd_v, dtype=float)
    cmd_w_np = np.asarray(cmd_w, dtype=float)
    v_cmd = np.interp(t, cmd_t_np, cmd_v_np)
    w_cmd = np.interp(t, cmd_t_np, cmd_w_np)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    tag = str(args.tag).strip()
    tag_part = f"__{tag}" if tag else ""
    base = out_dir / f"{ts}__{scenario.name}{tag_part}"

    npz_path = ensure_unique_path(base.with_suffix(".npz"))
    png_path = npz_path.with_suffix(".png")

    meta = {
        "scenario": scenario.name,
        "v_profile": scenario.v_profile.name,
        "w_profile": scenario.w_profile.name,
        "v_segments": [{"name": s.name, "duration": s.duration} for s in scenario.v_profile.segments],
        "w_segments": [{"name": s.name, "duration": s.duration} for s in scenario.w_profile.segments],
        "limits": asdict(limits),
        "margin": margin,
        "cmd_hz": cmd_hz,
        "warmup_sec": warmup_sec,
        "tf_world": str(args.tf_world),
        "tf_chassis": str(args.tf_chassis),
        "tf_timeout": float(args.tf_timeout),
        "theta0": float(getattr(node, "_theta0", 0.0)),
        "record_wall_time": time.time(),
    }

    np.savez_compressed(
        npz_path,
        meta=meta,
        # status timeline (primary)
        t=t,
        v_meas=v_meas,
        w_meas=w_meas,
        # cmd aligned to status timeline
        v_cmd=v_cmd,
        w_cmd=w_cmd,
        # raw cmd timeline (optional)
        cmd_t=np.asarray(cmd_t, dtype=float),
        cmd_v=np.asarray(cmd_v, dtype=float),
        cmd_w=np.asarray(cmd_w, dtype=float),
        cmd_theta=np.asarray(cmd_theta, dtype=float),
    )

    plt = try_setup_matplotlib()
    if plt is None:
        print("matplotlib 不可用，仅保存数据：", str(npz_path))
        return 0

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, v_cmd, "k--", linewidth=1.0, label="v_cmd")
    ax1.plot(t, v_meas, "b", linewidth=1.2, label="v_meas")
    ax1.set_ylabel("v [m/s]")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    ax1.set_title(f"Recorded: {scenario.name}  (cmd_hz={cmd_hz:g}, margin={margin:g})")
    ax1.set_ylim(-limits.max_v, limits.max_v)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, w_cmd, "k--", linewidth=1.0, label="omega_cmd")
    ax2.plot(t, w_meas, "b", linewidth=1.2, label="omega_meas")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("omega [rad/s]")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")
    ax2.set_ylim(-limits.max_w, limits.max_w)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)

    print("saved:", str(npz_path))
    print("saved:", str(png_path))

    if os.environ.get("DISPLAY", "") != "":
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
