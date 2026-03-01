"""Record chassis power data for power model identification.

被动录制脚本：订阅 /path_follower/chassis_cmd 和 /serial_bridge/chassis_status，
记录速度、角速度、功率、电容电量等完整数据，用于后续功率模型辨识。

该脚本本身不发布任何指令，适合在正常导航运行时录制真实功率数据。

输出：
  power_identify_data/<timestamp>__<tag>.npz
  power_identify_data/<timestamp>__<tag>.png
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Record chassis power data for identification")
    parser.add_argument("--tag", type=str, default="", help="附加标签写入文件名")
    parser.add_argument("--out-dir", type=str, default="power_identify_data", help="输出目录")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="录制时长(s)；0 表示直到 Ctrl-C")
    parser.add_argument("--warmup", type=float, default=1.0,
                        help="预热时长(s)，预热期间不记录数据")
    args = parser.parse_args()

    warmup_sec = float(args.warmup)

    # ─── 数据容器 ───
    st_t: List[float] = []
    st_v: List[float] = []
    st_w: List[float] = []
    st_leg_mode: List[int] = []
    st_remaining_energy: List[int] = []
    st_curr_chassis_pwr: List[int] = []
    st_rfr_pwr_limit: List[int] = []

    # ─── ROS 导入 ───
    try:
        import rclpy
        from rclpy.node import Node
        from interfaces.msg import ChassisStatus
    except Exception as e:
        print("无法导入 ROS2 依赖，请先 source install/setup.bash。\n", e)
        return 2

    class PowerRecordNode(Node):
        def __init__(self, warmup: float):
            super().__init__("power_identify_rec")
            self.sub_status = self.create_subscription(
                ChassisStatus, "/serial_bridge/chassis_status", self._status_cb, 1)
            self._t0 = time.time()
            self._warmup = warmup
            self._recording_started = warmup <= 0.0
            self.get_logger().info(f"power record node started (warmup={warmup}s)")

        def _status_cb(self, msg: ChassisStatus) -> None:
            t = time.time() - self._t0
            if t < self._warmup:
                return
            st_t.append(float(t))
            st_v.append(float(msg.velocity))
            st_w.append(float(msg.omega))
            st_leg_mode.append(int(msg.leg_mode))
            st_remaining_energy.append(int(msg.remaining_energy))
            st_curr_chassis_pwr.append(int(msg.curr_chassis_pwr))
            st_rfr_pwr_limit.append(int(msg.rfr_pwr_limit))

    rclpy.init()
    node = PowerRecordNode(warmup=warmup_sec)
    start_time = time.time()
    try:
        if args.duration > 0.0:
            deadline = start_time + args.duration + warmup_sec
            while rclpy.ok() and time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    if len(st_t) == 0:
        print("未收到任何 chassis_status 消息，未生成输出。")
        return 1

    # ─── 整理数据 ───
    t_st = np.asarray(st_t, dtype=float)
    v_meas = np.asarray(st_v, dtype=float)
    w_meas = np.asarray(st_w, dtype=float)
    leg_mode = np.asarray(st_leg_mode, dtype=np.uint8)
    remaining_energy = np.asarray(st_remaining_energy, dtype=np.int16)
    curr_chassis_pwr = np.asarray(st_curr_chassis_pwr, dtype=np.int16)
    rfr_pwr_limit = np.asarray(st_rfr_pwr_limit, dtype=np.int16)

    order_st = np.argsort(t_st)
    t_st = t_st[order_st]
    v_meas = v_meas[order_st]
    w_meas = w_meas[order_st]
    leg_mode = leg_mode[order_st]
    remaining_energy = remaining_energy[order_st]
    curr_chassis_pwr = curr_chassis_pwr[order_st]
    rfr_pwr_limit = rfr_pwr_limit[order_st]

    # ─── 插值指令到状态时间轴 ───
    # commands are not needed for power fitting; omit them entirely

    t = t_st

    # ─── 保存 ───
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    tag = str(args.tag).strip()
    tag_part = f"__{tag}" if tag else ""
    base = out_dir / f"{ts}{tag_part}"

    npz_path = ensure_unique_path(base.with_suffix(".npz"))
    png_path = npz_path.with_suffix(".png")

    meta = {
        "tag": tag,
        "record_wall_time": time.time(),
        "duration": args.duration,
        "warmup": warmup_sec,
    }

    np.savez_compressed(
        npz_path,
        meta=meta,
        # 状态时间轴（主时间线）
        t=t,
        v_meas=v_meas,
        w_meas=w_meas,
        leg_mode=leg_mode,
        remaining_energy=remaining_energy,
        curr_chassis_pwr=curr_chassis_pwr,
        rfr_pwr_limit=rfr_pwr_limit,
    )

    # ─── 绘图 ───
    plt = try_setup_matplotlib()
    if plt is None:
        print("matplotlib 不可用，仅保存数据：", str(npz_path))
        return 0

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    ax = axes[0]
    ax.plot(t, v_meas, "b", lw=1.2, label="v_meas")
    ax.set_ylabel("v [m/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(f"Power Recording (tag={tag})")

    ax = axes[1]
    ax.plot(t, w_meas, "b", lw=1.2, label="ω_meas")
    ax.set_ylabel("ω [rad/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t, curr_chassis_pwr.astype(float), "r", lw=1.2, label="chassis_pwr")
    ax.plot(t, rfr_pwr_limit.astype(float), "g--", lw=1.0, label="rfr_pwr_limit")
    ax.set_ylabel("Power [W]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax = axes[3]
    ax.plot(t, remaining_energy.astype(float), "m", lw=1.2, label="remaining_energy")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Energy [J]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)

    print("saved:", str(npz_path))
    print("saved:", str(png_path))

    if os.environ.get("DISPLAY", "") != "":
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
