#!/usr/bin/env python3
"""
录制 LiDAR-IMU 外参标定所需数据。

订阅话题:
  /serial_bridge/imu_raw   - 外部 BMI088 IMU (m/s², rad/s)
  /mid360_driver/imu        - MID360 内部 IMU   (g, rad/s)
  /mid360_driver/lidar      - MID360 点云 (PointCloud2)

使用方法:
  source /home/yuki/sentry_2026/install/setup.bash
  python3 navigation_sentry_2026/utils/py/calib/record_calib_data.py --duration 60 --tag test1

建议标定动作:
  1. 先静止 3-5 秒（用于重力对齐）
  2. 缓慢绕各轴旋转（激发全部 6-DOF 可观测性）
  3. 交替做平移和旋转运动
  4. 避免剧烈冲击（防止 IMU 饱和）
  5. 结束前再静止 3-5 秒
"""

from __future__ import annotations

import argparse
import os
import struct
import time
from pathlib import Path
from typing import List

import numpy as np

# ─── IMU 饱和阈值 ─────────────────────────────────────────────
_GRAVITY = 9.80665  # m/s²
# MID360 内部 IMU（话题单位: g, rad/s）
MID360_ACC_SAT_G = 39.5 / _GRAVITY   # ≈ 4.028 g
MID360_GYRO_SAT  = 34.5              # rad/s
# 外部下位机 IMU（话题单位: m/s², rad/s）
EXT_ACC_SAT      = 235.0             # m/s²
EXT_GYRO_SAT     = 34.5             # rad/s
# 检测阈值 = 饱和值 × 此比例（留 2% 裕量）
_SAT_RATIO = 0.99


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 10000):
        cand = path.with_name(f"{stem}__{i:03d}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"无法生成不冲突文件名: {path}")


def parse_pointcloud2(msg) -> np.ndarray:
    """将 PointCloud2 消息解析为 (N, 5) 数组 [x, y, z, intensity, timestamp]。

    MID360 发布的 PointCloud2 每个点 24 字节:
      x (float32), y (float32), z (float32), intensity (float32), timestamp (float64)
    """
    data = bytes(msg.data)
    n_points = msg.width * msg.height
    point_step = msg.point_step
    if n_points == 0:
        return np.empty((0, 5), dtype=np.float64)

    points = np.empty((n_points, 5), dtype=np.float64)
    # 快速解析: 用 numpy 直接从 buffer 读取
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_points, point_step)
    points[:, 0] = np.frombuffer(raw[:, 0:4].tobytes(), dtype=np.float32)   # x
    points[:, 1] = np.frombuffer(raw[:, 4:8].tobytes(), dtype=np.float32)   # y
    points[:, 2] = np.frombuffer(raw[:, 8:12].tobytes(), dtype=np.float32)  # z
    points[:, 3] = np.frombuffer(raw[:, 12:16].tobytes(), dtype=np.float32) # intensity
    points[:, 4] = np.frombuffer(raw[:, 16:24].tobytes(), dtype=np.float64) # timestamp
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="录制 LiDAR-IMU 标定数据")
    parser.add_argument("--tag", type=str, default="", help="文件名附加标签")
    parser.add_argument("--out-dir", type=str, default="calib_data", help="输出目录")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="录制时长 (秒)，0 表示手动 Ctrl-C 停止")
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="预热时长 (秒)，等待话题稳定")
    parser.add_argument("--max-scans", type=int, default=0,
                        help="最大点云帧数，0 表示不限制")
    args = parser.parse_args()

    # --- 数据容器 ---
    ext_imu_t: List[float] = []
    ext_imu_acc: List[List[float]] = []
    ext_imu_gyro: List[List[float]] = []

    mid_imu_t: List[float] = []
    mid_imu_acc: List[List[float]] = []
    mid_imu_gyro: List[List[float]] = []

    lidar_scans: List[np.ndarray] = []  # 每帧点云 (N, 5)
    lidar_stamps: List[float] = []       # 每帧头时间戳

    # --- ROS2 ---
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Imu, PointCloud2
    except Exception as e:
        print(f"无法导入 ROS2 依赖，请先 source install/setup.bash。\n{e}")
        return 2

    class RecordNode(Node):
        def __init__(self, warmup: float, max_scans: int):
            super().__init__("calib_recorder")
            self._t0 = time.time()
            self._warmup = warmup
            self._recording = warmup <= 0.0
            self._max_scans = max_scans

            qos_best = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )

            self.create_subscription(
                Imu, "/serial_bridge/imu_raw", self._ext_imu_cb, qos_best)
            self.create_subscription(
                Imu, "/mid360_driver/imu", self._mid_imu_cb, qos_best)
            self.create_subscription(
                PointCloud2, "/mid360_driver/lidar", self._lidar_cb, qos_best)

            self._ext_count = 0
            self._mid_count = 0
            self._lidar_count = 0
            self._ext_sat_count = 0
            self._mid_sat_count = 0
            self._ext_last_sat_warn = -999.0
            self._mid_last_sat_warn = -999.0
            self._timer = self.create_timer(2.0, self._status_cb)
            self.get_logger().info(
                f"标定数据录制节点启动 (预热 {warmup}s)")

        def _wall_t(self):
            return time.time() - self._t0

        def _check_warmup(self) -> bool:
            if self._recording:
                return True
            if self._wall_t() >= self._warmup:
                self._recording = True
                self.get_logger().info("预热完成，开始录制数据")
                return True
            return False

        def _ext_imu_cb(self, msg: Imu) -> None:
            if not self._check_warmup():
                return
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            acc  = [msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z]
            gyro = [msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z]
            # 饱和检测（外部IMU 单位: m/s², rad/s）
            acc_max  = max(abs(v) for v in acc)
            gyro_max = max(abs(v) for v in gyro)
            if (acc_max  >= EXT_ACC_SAT  * _SAT_RATIO or
                    gyro_max >= EXT_GYRO_SAT * _SAT_RATIO):
                self._ext_sat_count += 1
                wall = self._wall_t()
                if wall - self._ext_last_sat_warn >= 1.0:
                    self.get_logger().warn(
                        f"[外部IMU饱和] acc={acc_max:.1f}m/s²(限{EXT_ACC_SAT}), "
                        f"gyro={gyro_max:.3f}rad/s(限{EXT_GYRO_SAT}) "
                        f"| 累计饱和 {self._ext_sat_count} 帧"
                    )
                    self._ext_last_sat_warn = wall
            ext_imu_t.append(t)
            ext_imu_acc.append(acc)
            ext_imu_gyro.append(gyro)
            self._ext_count += 1

        def _mid_imu_cb(self, msg: Imu) -> None:
            if not self._check_warmup():
                return
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            acc  = [msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z]
            gyro = [msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z]
            # 饱和检测（MID360 acc 单位: g; gyro 单位: rad/s）
            acc_max  = max(abs(v) for v in acc)
            gyro_max = max(abs(v) for v in gyro)
            if (acc_max  >= MID360_ACC_SAT_G * _SAT_RATIO or
                    gyro_max >= MID360_GYRO_SAT  * _SAT_RATIO):
                self._mid_sat_count += 1
                wall = self._wall_t()
                if wall - self._mid_last_sat_warn >= 1.0:
                    acc_ms2 = acc_max * _GRAVITY
                    self.get_logger().warn(
                        f"[MID360 IMU饱和] acc={acc_ms2:.1f}m/s²(限39.5), "
                        f"gyro={gyro_max:.3f}rad/s(限{MID360_GYRO_SAT}) "
                        f"| 累计饱和 {self._mid_sat_count} 帧"
                    )
                    self._mid_last_sat_warn = wall
            mid_imu_t.append(t)
            mid_imu_acc.append(acc)
            mid_imu_gyro.append(gyro)
            self._mid_count += 1

        def _lidar_cb(self, msg: PointCloud2) -> None:
            if not self._check_warmup():
                return
            if self._max_scans > 0 and len(lidar_scans) >= self._max_scans:
                return
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            pts = parse_pointcloud2(msg)
            if pts.shape[0] > 0:
                lidar_stamps.append(t)
                lidar_scans.append(pts)
                self._lidar_count += 1

        def _status_cb(self) -> None:
            t = self._wall_t()
            self.get_logger().info(
                f"[{t:.1f}s] 外部IMU: {self._ext_count}"  
                f"(饱和{self._ext_sat_count}次), "
                f"MID360 IMU: {self._mid_count}"
                f"(饱和{self._mid_sat_count}次), "
                f"LiDAR: {self._lidar_count} 帧")

    rclpy.init()
    node = RecordNode(warmup=args.warmup, max_scans=args.max_scans)
    start = time.time()
    try:
        if args.duration > 0.0:
            deadline = start + args.warmup + args.duration
            while rclpy.ok() and time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
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

    # --- 保存数据 ---
    if len(ext_imu_t) == 0 and len(mid_imu_t) == 0:
        print("未收到任何 IMU 数据，退出。")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"__{args.tag}" if args.tag else ""
    out_path = ensure_unique_path(out_dir / f"calib_{ts}{tag}.npz")

    save_dict = {
        "ext_imu_t": np.asarray(ext_imu_t, dtype=np.float64),
        "ext_imu_acc": np.asarray(ext_imu_acc, dtype=np.float64),
        "ext_imu_gyro": np.asarray(ext_imu_gyro, dtype=np.float64),
        "mid_imu_t": np.asarray(mid_imu_t, dtype=np.float64),
        "mid_imu_acc": np.asarray(mid_imu_acc, dtype=np.float64),
        "mid_imu_gyro": np.asarray(mid_imu_gyro, dtype=np.float64),
        "lidar_stamps": np.asarray(lidar_stamps, dtype=np.float64),
    }
    # 点云单独存储（变长）
    for i, scan in enumerate(lidar_scans):
        save_dict[f"scan_{i:05d}"] = scan.astype(np.float32)

    save_dict["n_scans"] = np.array(len(lidar_scans), dtype=np.int32)

    np.savez_compressed(str(out_path), **save_dict)

    ext_sat = node._ext_sat_count
    mid_sat = node._mid_sat_count
    ext_total = max(len(ext_imu_t), 1)
    mid_total = max(len(mid_imu_t), 1)
    print(f"\n{'='*60}")
    print(f"录制完成:")
    print(f"  外部 IMU 样本数: {len(ext_imu_t)}")
    print(f"  外部 IMU 饱和帧: {ext_sat} ({100.0*ext_sat/ext_total:.2f}%)")
    if ext_sat > 0:
        print(f"  ⚠ 外部IMU出现饱和，建议减小运动幅度或检查标定是否有效")
    print(f"  MID360 IMU 样本数: {len(mid_imu_t)}")
    print(f"  MID360 IMU 饱和帧: {mid_sat} ({100.0*mid_sat/mid_total:.2f}%)")
    if mid_sat > 0:
        print(f"  ⚠ MID360 IMU出现饱和，建议减小运动幅度")
    print(f"  LiDAR 点云帧数: {len(lidar_scans)}")
    if lidar_scans:
        total_pts = sum(s.shape[0] for s in lidar_scans)
        print(f"  点云总点数: {total_pts:,}")
    print(f"  输出文件: {out_path}")
    print(f"  文件大小: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
