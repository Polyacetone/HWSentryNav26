"""Record chassis_cmd and chassis_status for later analysis.

This script listens to the same topics that the original `identify_pub_and_rec.py`
published to, but it *does not* send any command.  It simply subscribes to
`/path_follower/chassis_cmd` and `/serial_bridge/chassis_status` and stores all
messages along with their wall‑clock timestamps.  When the node is shut down
(e.g. via Ctrl‑C) the collected data is saved in the same `identify_data/`
directory as the original script.

Usage examples:

  source /home/yuki/sentry_2026/install/setup.bash

  # record until manually stopped
  python3 navigation_sentry_2026/utils/py/lqr/identify_rec.py --tag test1

  # add a duration (seconds), then the script exits automatically
  python3 navigation_sentry_2026/utils/py/lqr/identify_rec.py --duration 15.0

The output files are named using a timestamp and optional tag:

    identify_data/<timestamp>__<tag>.npz
    identify_data/<timestamp>__<tag>.png

"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

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
    parser.add_argument("--tag", type=str, default="", help="additional tag to put in filename")
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

    cmd_t: List[float] = []
    cmd_v: List[float] = []
    cmd_w: List[float] = []
    st_t: List[float] = []
    st_v: List[float] = []
    st_w: List[float] = []

    # delayed ROS imports
    try:
        import rclpy
        from rclpy.node import Node
        from interfaces.msg import ChassisCmd, ChassisStatus
    except Exception as e:  # pragma: no cover - environment issue
        print("无法导入 ROS2 依赖，请先 source install/setup.bash。\n", e)
        return 2

    class RecordNode(Node):
        def __init__(self, warmup: float):
            super().__init__("wheel_leg_lqr_ident_rec")
            self.sub_cmd = self.create_subscription(ChassisCmd, "/path_follower/chassis_cmd", self._cmd_cb, 1)
            self.sub_status = self.create_subscription(
                ChassisStatus, "/serial_bridge/chassis_status", self._status_cb, 1
            )
            self._t0 = time.time()
            self._warmup = warmup
            self._recording_started = warmup <= 0.0
            self.get_logger().info(f"record node started (warmup={warmup}s)")

        def _cmd_cb(self, msg: ChassisCmd) -> None:
            t = time.time() - self._t0
            if t < self._warmup:
                return
            if not self._recording_started:
                self.get_logger().info("warmup finished: start recording commands and status")
                self._recording_started = True
            cmd_t.append(float(t))
            cmd_v.append(float(msg.velocity))
            cmd_w.append(float(msg.omega))

        def _status_cb(self, msg: ChassisStatus) -> None:
            t = time.time() - self._t0
            if t < self._warmup:
                return
            st_t.append(float(t))
            st_v.append(float(msg.velocity))
            st_w.append(float(msg.omega))

    rclpy.init()
    node = RecordNode(warmup=warmup_sec)
    start_time = time.time()
    try:
        if args.duration > 0.0:
            # spin for max duration
            deadline = start_time + args.duration
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

    if len(cmd_t) == 0 and len(st_t) == 0:
        print("未收到任何消息，未生成输出。")
        return 1

    # sort status timeline (primary) and also sort command timeline for interpolation
    t_st = np.asarray(st_t, dtype=float)
    v_meas = np.asarray(st_v, dtype=float)
    w_meas = np.asarray(st_w, dtype=float)
    order_st = np.argsort(t_st)
    t_st = t_st[order_st]
    v_meas = v_meas[order_st]
    w_meas = w_meas[order_st]

    t_cmd = np.asarray(cmd_t, dtype=float)
    v_cmd_raw = np.asarray(cmd_v, dtype=float)
    w_cmd_raw = np.asarray(cmd_w, dtype=float)
    order_cmd = np.argsort(t_cmd)
    t_cmd = t_cmd[order_cmd]
    v_cmd_raw = v_cmd_raw[order_cmd]
    w_cmd_raw = w_cmd_raw[order_cmd]

    # interpolate command onto the status timeline (same as identify_pub_and_rec)
    v_cmd = np.interp(t_st, t_cmd, v_cmd_raw)
    w_cmd = np.interp(t_st, t_cmd, w_cmd_raw)
    t = t_st

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
        # status timeline (primary)
        t=t,
        v_meas=v_meas,
        w_meas=w_meas,
        # cmd aligned to status timeline
        v_cmd=v_cmd,
        w_cmd=w_cmd,
        # raw cmd timeline (optional)
        cmd_t=t_cmd,
        cmd_v=v_cmd_raw,
        cmd_w=w_cmd_raw,
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
    ax1.set_title(f"Recorded (tag={tag})")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, w_cmd, "k--", linewidth=1.0, label="omega_cmd")
    ax2.plot(t, w_meas, "b", linewidth=1.2, label="omega_meas")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("omega [rad/s]")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)

    print("saved:", str(npz_path))
    print("saved:", str(png_path))

    if os.environ.get("DISPLAY", "") != "":
        plt.show()

    return 0


if __name__ == "__main__":
    main()
