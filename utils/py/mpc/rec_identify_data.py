"""Record chassis_cmd and chassis_status for later analysis.

This script listens to the same topics that the original `identify_pub_and_rec.py`
published to, but it *does not* send any command.  It simply subscribes to
`/nav_executor/chassis_cmd` and `/serial_bridge/chassis_status` and stores all
messages along with monotonic receive timestamps.  When the node is shut down
(e.g. via Ctrl‑C) the collected data is saved in the same `identify_data/`
directory as the original script.

Usage examples:

  source /home/yuki/sentry_2026/install/setup.bash

  # record until manually stopped
  python3 navigation_sentry_2026/utils/py/mpc/rec_identify_data.py --tag test1

  # add a duration (seconds), then the script exits automatically
  python3 navigation_sentry_2026/utils/py/mpc/rec_identify_data.py --duration 15.0

The output files are named using a timestamp and optional tag:

    identify_data/<timestamp>__<tag>.npz
    identify_data/<timestamp>__<tag>.png

"""

from __future__ import annotations

import argparse
import json
import math
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
        import matplotlib.pyplot as plt  # noqa: F401

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
    parser = argparse.ArgumentParser(description="Record chassis_cmd and chassis_status")
    parser.add_argument("--tag", type=str, default="", help="optional tag to put in filename")
    parser.add_argument("--out-dir", type=str, default="identify_data", help="output directory")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="optional recording duration in seconds; 0 means run until interrupted",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=3.0,
        help="warmup time in seconds before data is recorded",
    )
    args = parser.parse_args()

    warmup_sec = float(args.warmup)
    if not math.isfinite(args.duration) or args.duration < 0.0:
        parser.error("--duration must be finite and nonnegative")
    if not math.isfinite(warmup_sec) or warmup_sec < 0.0:
        parser.error("--warmup must be finite and nonnegative")

    cmd_t: List[float] = []
    cmd_v: List[float] = []
    cmd_w: List[float] = []
    st_t: List[float] = []
    st_v: List[float] = []
    st_w: List[float] = []
    st_leg_h: List[float] = []
    st_leg_psi: List[float] = []
    st_s: List[float] = []
    st_yaw: List[float] = []
    st_pitch: List[float] = []

    # delayed ROS imports
    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener
        from interfaces.msg import ChassisCmd, ChassisStatus
    except Exception as e:  # pragma: no cover - environment issue
        print("无法导入 ROS2 依赖，请先 source install/setup.bash。\n", e)
        return 2

    class RecordNode(Node):
        def __init__(self, warmup: float):
            super().__init__("wheel_leg_mpc_ident_rec")
            self.sub_cmd = self.create_subscription(ChassisCmd, "/nav_executor/chassis_cmd", self._cmd_cb, 1)
            self.sub_status = self.create_subscription(
                ChassisStatus, "/serial_bridge/chassis_status", self._status_cb, 1
            )
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
            self._t0 = time.monotonic()
            self._warmup = warmup
            self._recording_started = warmup <= 0.0
            self._s_acc = 0.0
            self._last_pos = None
            self.get_logger().info(f"record node started (warmup={warmup}s)")

        def _cmd_cb(self, msg: ChassisCmd) -> None:
            t = time.monotonic() - self._t0
            if t < self._warmup:
                return
            if not self._recording_started:
                self.get_logger().info("warmup finished: start recording commands and status")
                self._recording_started = True
            cmd_t.append(float(t))
            cmd_v.append(float(msg.velocity))
            cmd_w.append(float(msg.omega))

        def _status_cb(self, msg: ChassisStatus) -> None:
            t = time.monotonic() - self._t0
            if t < self._warmup:
                return
            st_t.append(float(t))
            st_v.append(float(msg.velocity))
            st_w.append(float(msg.omega))
            st_leg_h.append(float(msg.leg_h))
            st_leg_psi.append(float(msg.leg_psi))

            # Query odom<-chassis_link exactly on primary timeline samples.
            s_meas = float("nan")
            yaw_meas = float("nan")
            pitch_meas = float("nan")
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    "odom", "chassis_link", Time(), timeout=Duration(seconds=0.02)
                )
                tr = tf_msg.transform.translation
                q = tf_msg.transform.rotation

                pos = np.array([float(tr.x), float(tr.y), float(tr.z)], dtype=float)
                if self._last_pos is not None:
                    self._s_acc += float(np.linalg.norm(pos - self._last_pos))
                self._last_pos = pos
                s_meas = float(self._s_acc)

                # Forward axis in odom from quaternion; roll is intentionally ignored.
                qx = float(q.x)
                qy = float(q.y)
                qz = float(q.z)
                qw = float(q.w)
                fx = 1.0 - 2.0 * (qy * qy + qz * qz)
                fy = 2.0 * (qx * qy + qz * qw)
                fz = 2.0 * (qx * qz - qy * qw)

                yaw_meas = math.atan2(fy, fx)
                pitch_meas = math.atan2(-fz, math.hypot(fx, fy))
            except Exception:
                pass

            st_s.append(s_meas)
            st_yaw.append(yaw_meas)
            st_pitch.append(pitch_meas)

    rclpy.init()
    node = RecordNode(warmup=warmup_sec)
    start_time = time.monotonic()
    try:
        if args.duration > 0.0:
            # spin for max duration
            deadline = start_time + warmup_sec + args.duration
            while rclpy.ok() and time.monotonic() < deadline:
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
        print("未收到 chassis_status，未生成输出。")
        return 1

    # Sort the primary status timeline and the independently received commands.
    t_st = np.asarray(st_t, dtype=float)
    v_meas = np.asarray(st_v, dtype=float)
    w_meas = np.asarray(st_w, dtype=float)
    leg_h_meas = np.asarray(st_leg_h, dtype=float)
    leg_psi_meas = np.asarray(st_leg_psi, dtype=float)
    s_meas = np.asarray(st_s, dtype=float)
    yaw_meas = np.asarray(st_yaw, dtype=float)
    pitch_meas = np.asarray(st_pitch, dtype=float)
    order_st = np.argsort(t_st)
    t_st = t_st[order_st]
    v_meas = v_meas[order_st]
    w_meas = w_meas[order_st]
    leg_h_meas = leg_h_meas[order_st]
    leg_psi_meas = leg_psi_meas[order_st]
    s_meas = s_meas[order_st]
    yaw_meas = yaw_meas[order_st]
    pitch_meas = pitch_meas[order_st]

    t_cmd = np.asarray(cmd_t, dtype=float)
    v_cmd_raw = np.asarray(cmd_v, dtype=float)
    w_cmd_raw = np.asarray(cmd_w, dtype=float)
    order_cmd = np.argsort(t_cmd)
    t_cmd = t_cmd[order_cmd]
    v_cmd_raw = v_cmd_raw[order_cmd]
    w_cmd_raw = w_cmd_raw[order_cmd]

    # Commands are zero-order held. Linear interpolation would create command
    # values that were never sent and leak future command changes into fitting.
    if t_cmd.size > 0:
        command_index = np.searchsorted(t_cmd, t_st, side="right") - 1
        command_valid = command_index >= 0
        safe_index = np.maximum(command_index, 0)
        v_cmd = np.where(command_valid, v_cmd_raw[safe_index], np.nan)
        w_cmd = np.where(command_valid, w_cmd_raw[safe_index], np.nan)
    else:
        v_cmd = np.full_like(t_st, np.nan, dtype=float)
        w_cmd = np.full_like(t_st, np.nan, dtype=float)
    t = t_st

    finite = np.all(np.isfinite(np.column_stack([
        t, v_meas, w_meas, leg_h_meas, leg_psi_meas, v_cmd, w_cmd,
    ])), axis=1)
    if np.sum(finite) < 2:
        print("有效状态/命令样本不足，未生成输出。")
        return 1
    t = t[finite]
    t -= t[0]
    v_meas = v_meas[finite]
    w_meas = w_meas[finite]
    leg_h_meas = leg_h_meas[finite]
    leg_psi_meas = leg_psi_meas[finite]
    s_meas = s_meas[finite]
    yaw_meas = yaw_meas[finite]
    pitch_meas = pitch_meas[finite]
    v_cmd = v_cmd[finite]
    w_cmd = w_cmd[finite]

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
        metadata_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
        # status timeline (primary)
        t=t,
        v_meas=v_meas,
        w_meas=w_meas,
        leg_h_meas=leg_h_meas,
        leg_psi_meas=leg_psi_meas,
        s_meas=s_meas,
        yaw_meas=yaw_meas,
        pitch_meas=pitch_meas,
        # cmd aligned to status timeline
        v_cmd=v_cmd,
        w_cmd=w_cmd,
    )

    plt = try_setup_matplotlib()
    if plt is None:
        print("matplotlib 不可用，仅保存数据：", str(npz_path))
        return 0

    fig = plt.figure(figsize=(12, 18))
    ax1 = fig.add_subplot(7, 1, 1)
    ax1.plot(t, v_cmd, "k--", linewidth=1.0, label="v_cmd")
    ax1.plot(t, v_meas, "b", linewidth=1.2, label="v_meas")
    ax1.set_ylabel("v [m/s]")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    ax1.set_title(f"Recorded (tag={tag})")

    ax2 = fig.add_subplot(7, 1, 2)
    ax2.plot(t, w_cmd, "k--", linewidth=1.0, label="omega_cmd")
    ax2.plot(t, w_meas, "b", linewidth=1.2, label="omega_meas")
    ax2.set_ylabel("omega [rad/s]")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(7, 1, 3)
    ax3.plot(t, leg_h_meas, "b", linewidth=1.2, label="leg_h_meas")
    ax3.set_ylabel("leg_h [m]")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(7, 1, 4)
    ax4.plot(t, leg_psi_meas, "b", linewidth=1.2, label="leg_psi_meas")
    ax4.set_ylabel("leg_psi [rad]")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best")

    ax5 = fig.add_subplot(7, 1, 5)
    ax5.plot(t, s_meas, "g", linewidth=1.2, label="s_meas")
    ax5.set_ylabel("s [m]")
    ax5.grid(True, alpha=0.25)
    ax5.legend(loc="best")

    ax6 = fig.add_subplot(7, 1, 6)
    ax6.plot(t, yaw_meas, "m", linewidth=1.2, label="yaw_meas")
    ax6.set_ylabel("yaw [rad]")
    ax6.grid(True, alpha=0.25)
    ax6.legend(loc="best")

    ax7 = fig.add_subplot(7, 1, 7)
    ax7.plot(t, pitch_meas, "c", linewidth=1.2, label="pitch_meas")
    ax7.set_xlabel("t [s]")
    ax7.set_ylabel("pitch [rad]")
    ax7.grid(True, alpha=0.25)
    ax7.legend(loc="best")

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)

    print("saved:", str(npz_path))
    print("saved:", str(png_path))

    if os.environ.get("DISPLAY", "") != "":
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
