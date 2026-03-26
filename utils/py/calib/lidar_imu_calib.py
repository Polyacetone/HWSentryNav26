#!/usr/bin/env python3
"""
Spatiotemporal LiDAR-IMU Extrinsic Calibration Tool
====================================================

Calibrates the rigid body transformation T_lidar_imu (pose of IMU expressed
in the LiDAR frame) and the temporal offset td between a LiDAR and an IMU
using rosbag2 data.

Algorithm
---------
1. Read LiDAR (PointCloud2) and IMU (Imu) data from a ROS 2 rosbag.
2. Estimate frame-to-frame LiDAR odometry via point-to-plane ICP.
3. Integrate IMU angular velocity for corresponding time intervals.
4. Solve rotation via the quaternion hand-eye method  (AX = XB).
5. Solve translation via the lever-arm least-squares formulation.
6. Jointly refine rotation, translation and time offset td.

Output format matches small_glim config:
    T_lidar_imu: [tx, ty, tz, qx, qy, qz, qw]

Dependencies
------------
    pip install open3d numpy scipy
    # ROS 2 packages (already available in a sourced workspace):
    #   rosbag2_py, rclpy, sensor_msgs

Calibration Data Requirements
-----------------------------
Record a bag (30–90 s) with **diverse rotational** motions around all three
axes.  Pure yaw / pitch / roll sequences as well as combined motions are
ideal.  Avoid long straight-line translations without rotation.

Usage
-----
    # Source your ROS 2 workspace first, then:
    python3 lidar_imu_calib.py /path/to/rosbag_dir \\
        --imu-topic  /serial_bridge/imu_raw \\
        --lidar-topic /mid360_driver/lidar
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# ── Open3D (required) ────────────────────────────────────────────
try:
    import open3d as o3d
except ImportError:
    sys.exit("open3d is required.  Install with:  pip install open3d")

# ── ROS 2 bag reading ───────────────────────────────────────────
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu, PointCloud2

    _BACKEND = "ros2"
except ImportError:
    _BACKEND = None

if _BACKEND is None:
    try:
        from rosbags.rosbag2 import Reader as _RosbagReader
        from rosbags.typesys import get_typestore, Stores

        _BACKEND = "rosbags"
    except ImportError:
        pass

if _BACKEND is None:
    sys.exit(
        "Either ROS 2 (rosbag2_py + rclpy + sensor_msgs) or the 'rosbags' "
        "pip package is required.\n"
        "  pip install rosbags   (pure-Python, works outside a ROS env)"
    )


# ═════════════════════════════════════════════════════════════════
#  Data Structures
# ═════════════════════════════════════════════════════════════════
@dataclass
class ImuSample:
    stamp: float  # seconds
    acc: np.ndarray  # (3,)  m/s²
    gyro: np.ndarray  # (3,)  rad/s


@dataclass
class LidarScan:
    stamp: float  # seconds (header stamp)
    points: np.ndarray  # (N, 3) float64


@dataclass
class RelativeMotion:
    """Relative SE(3) between two consecutive LiDAR scans."""

    t0: float
    t1: float
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)
    fitness: float  # ICP fitness score


# ═════════════════════════════════════════════════════════════════
#  PointCloud2 Parser
# ═════════════════════════════════════════════════════════════════
def _pointcloud2_to_xyz(fields, point_step: int, width: int, height: int,
                        data: bytes) -> np.ndarray:
    """Extract Nx3 float64 array from PointCloud2 raw data."""
    offsets = {f_name: f_off for f_name, f_off in fields}
    n = width * height
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    raw = np.frombuffer(bytearray(data), dtype=np.uint8)
    xyz = np.empty((n, 3), dtype=np.float32)
    for i, axis in enumerate(("x", "y", "z")):
        off = offsets[axis]
        # Slice and copy to ensure contiguous buffer for .view()
        xyz[:, i] = np.frombuffer(
            raw.reshape(n, point_step)[:, off : off + 4].copy().tobytes(),
            dtype="<f4",
        )

    result = xyz.astype(np.float64)
    # Remove NaN / Inf points
    valid = np.isfinite(result).all(axis=1)
    return result[valid]


def _parse_pc2_ros2(msg) -> np.ndarray:
    """Parse a sensor_msgs/msg/PointCloud2 (rclpy) to Nx3."""
    fields = [(f.name, f.offset) for f in msg.fields]
    return _pointcloud2_to_xyz(fields, msg.point_step, msg.width,
                               msg.height, bytes(msg.data))


def _parse_pc2_rosbags(msg) -> np.ndarray:
    """Parse a rosbags-deserialized PointCloud2 to Nx3."""
    fields = [(f.name, f.offset) for f in msg.fields]
    return _pointcloud2_to_xyz(fields, msg.point_step, msg.width,
                               msg.height, bytes(msg.data))


# ═════════════════════════════════════════════════════════════════
#  Rosbag Reading
# ═════════════════════════════════════════════════════════════════
def _read_bag_ros2(
    path: str,
    imu_topic: str,
    lidar_topic: str,
    max_scans: int,
    skip_scans: int,
) -> Tuple[List[ImuSample], List[LidarScan]]:
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=path, storage_id="")
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage, converter)

    topics_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if imu_topic not in topics_types:
        sys.exit(f"IMU topic '{imu_topic}' not found in bag. "
                 f"Available: {list(topics_types.keys())}")
    if lidar_topic not in topics_types:
        sys.exit(f"LiDAR topic '{lidar_topic}' not found in bag. "
                 f"Available: {list(topics_types.keys())}")

    filter_ = rosbag2_py.StorageFilter(topics=[imu_topic, lidar_topic])
    reader.set_filter(filter_)

    imus: List[ImuSample] = []
    scans: List[LidarScan] = []
    scan_count = 0

    while reader.has_next():
        topic, data, _ts = reader.read_next()

        if topic == imu_topic:
            msg = deserialize_message(data, Imu)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            acc = np.array([msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z])
            gyro = np.array([msg.angular_velocity.x,
                             msg.angular_velocity.y,
                             msg.angular_velocity.z])
            imus.append(ImuSample(stamp, acc, gyro))

        elif topic == lidar_topic:
            scan_count += 1
            if scan_count <= skip_scans:
                continue
            if max_scans and len(scans) >= max_scans:
                continue
            msg = deserialize_message(data, PointCloud2)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            pts = _parse_pc2_ros2(msg)
            if pts.shape[0] > 50:
                scans.append(LidarScan(stamp, pts))

    return imus, scans


def _read_bag_rosbags(
    path: str,
    imu_topic: str,
    lidar_topic: str,
    max_scans: int,
    skip_scans: int,
) -> Tuple[List[ImuSample], List[LidarScan]]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    imus: List[ImuSample] = []
    scans: List[LidarScan] = []
    scan_count = 0

    with _RosbagReader(path) as reader:
        conns = {c.topic: c for c in reader.connections}
        if imu_topic not in conns:
            sys.exit(f"IMU topic '{imu_topic}' not found. "
                     f"Available: {list(conns.keys())}")
        if lidar_topic not in conns:
            sys.exit(f"LiDAR topic '{lidar_topic}' not found. "
                     f"Available: {list(conns.keys())}")

        for conn, _ts, rawdata in reader.messages(
            connections=[conns[imu_topic], conns[lidar_topic]]
        ):
            if conn.topic == imu_topic:
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                acc = np.array([msg.linear_acceleration.x,
                                msg.linear_acceleration.y,
                                msg.linear_acceleration.z])
                gyro = np.array([msg.angular_velocity.x,
                                 msg.angular_velocity.y,
                                 msg.angular_velocity.z])
                imus.append(ImuSample(stamp, acc, gyro))

            elif conn.topic == lidar_topic:
                scan_count += 1
                if scan_count <= skip_scans:
                    continue
                if max_scans and len(scans) >= max_scans:
                    continue
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                pts = _parse_pc2_rosbags(msg)
                if pts.shape[0] > 50:
                    scans.append(LidarScan(stamp, pts))

    return imus, scans


def read_bag(
    path: str,
    imu_topic: str,
    lidar_topic: str,
    max_scans: int = 300,
    skip_scans: int = 0,
) -> Tuple[List[ImuSample], List[LidarScan]]:
    if _BACKEND == "ros2":
        return _read_bag_ros2(path, imu_topic, lidar_topic, max_scans,
                              skip_scans)
    else:
        return _read_bag_rosbags(path, imu_topic, lidar_topic, max_scans,
                                 skip_scans)


# ═════════════════════════════════════════════════════════════════
#  LiDAR Odometry (ICP)
# ═════════════════════════════════════════════════════════════════
def _to_o3d(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def compute_lidar_motions(
    scans: List[LidarScan],
    voxel_size: float = 0.5,
    max_corr_dist: float = 1.0,
    min_fitness: float = 0.3,
) -> List[RelativeMotion]:
    """
    Estimate relative SE(3) between consecutive LiDAR scans with
    point-to-plane ICP.

    Returns A_i where:  p_{i} = A_i · p_{i+1}
    i.e. A_i = T_lidar_i^{-1 ← world} · T_world_← lidar_{i+1}
    """
    motions: List[RelativeMotion] = []
    prev_pcd: Optional[o3d.geometry.PointCloud] = None
    prev_stamp: float = 0.0

    for idx, scan in enumerate(scans):
        pcd = _to_o3d(scan.points)
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 3, max_nn=30
            )
        )

        if prev_pcd is not None:
            # source = current (j), target = prev (i)
            # result.transformation transforms source → target
            reg = o3d.pipelines.registration.registration_icp(
                source=pcd,
                target=prev_pcd,
                max_correspondence_distance=max_corr_dist,
                init=np.eye(4),
                estimation_method=(
                    o3d.pipelines.registration
                    .TransformationEstimationPointToPlane()
                ),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=80,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                ),
            )
            T = reg.transformation  # 4x4, transforms j-coords → i-coords
            fitness = reg.fitness
            if fitness >= min_fitness:
                R = T[:3, :3].copy()
                t = T[:3, 3].copy()
                motions.append(
                    RelativeMotion(prev_stamp, scan.stamp, R, t, fitness)
                )

            if (idx + 1) % 20 == 0:
                print(f"  ICP: {idx + 1}/{len(scans)}  "
                      f"accepted={len(motions)}  fitness={fitness:.3f}")

        prev_pcd = pcd
        prev_stamp = scan.stamp

    return motions


# ═════════════════════════════════════════════════════════════════
#  IMU Buffer with Fast Range Integration
# ═════════════════════════════════════════════════════════════════
class ImuBuffer:
    """Sorted IMU sample buffer with fast gyroscope integration."""

    def __init__(self, samples: List[ImuSample]):
        samples.sort(key=lambda s: s.stamp)
        self.stamps = np.array([s.stamp for s in samples])
        self.gyros = np.array([s.gyro for s in samples])  # (N, 3)
        self.accs = np.array([s.acc for s in samples])  # (N, 3)

    def integrate_rotation(self, t0: float, t1: float) -> np.ndarray:
        """
        Integrate angular velocity from t0 to t1.

        Returns R such that  R_imu(t1) = R_imu(t0) · R,
        i.e. R is the relative rotation in the *body* frame.
        """
        i_start = bisect_left(self.stamps, t0)
        i_end = bisect_right(self.stamps, t1)

        R = np.eye(3)
        prev_t = t0

        for k in range(i_start, i_end):
            dt = self.stamps[k] - prev_t
            if dt <= 0:
                continue
            omega = self.gyros[k]
            rotvec = omega * dt
            angle = np.linalg.norm(rotvec)
            if angle > 1e-12:
                R = R @ Rotation.from_rotvec(rotvec).as_matrix()
            prev_t = self.stamps[k]

        # Tail extrapolation to t1
        dt = t1 - prev_t
        if dt > 1e-9 and i_end > 0:
            omega = self.gyros[min(i_end, len(self.gyros) - 1)]
            rotvec = omega * dt
            angle = np.linalg.norm(rotvec)
            if angle > 1e-12:
                R = R @ Rotation.from_rotvec(rotvec).as_matrix()

        return R

    def gravity_direction(self, t0: float, t1: float) -> np.ndarray:
        """Average accelerometer direction in [t0, t1] (≈ gravity when still)."""
        i0 = bisect_left(self.stamps, t0)
        i1 = bisect_right(self.stamps, t1)
        if i1 <= i0:
            return np.array([0.0, 0.0, 1.0])
        avg = self.accs[i0:i1].mean(axis=0)
        n = np.linalg.norm(avg)
        return avg / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


# ═════════════════════════════════════════════════════════════════
#  Hand-Eye Calibration:  A · X = X · B   (rotation)
# ═════════════════════════════════════════════════════════════════
def _quat_left(q):
    """Left quaternion multiplication matrix (Hamilton, [w,x,y,z])."""
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x,  w, -z,  y],
        [y,  z,  w, -x],
        [z, -y,  x,  w],
    ])


def _quat_right(q):
    """Right quaternion multiplication matrix (Hamilton, [w,x,y,z])."""
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x,  w,  z, -y],
        [y, -z,  w,  x],
        [z,  y, -x,  w],
    ])


def _rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion [w, x, y, z]."""
    q_xyzw = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def solve_hand_eye_rotation(
    R_A_list: List[np.ndarray],
    R_B_list: List[np.ndarray],
) -> np.ndarray:
    """
    Solve  R_A · R_X = R_X · R_B  for  R_X  via the quaternion null-space
    method.

    Parameters
    ----------
    R_A_list : LiDAR relative rotations
    R_B_list : IMU relative rotations (same intervals)

    Returns
    -------
    R_X : (3, 3) rotation matrix of X = T_lidar_imu
    """
    M = np.zeros((4, 4))
    for R_A, R_B in zip(R_A_list, R_B_list):
        q_A = _rot_to_quat_wxyz(R_A)
        q_B = _rot_to_quat_wxyz(R_B)
        C = _quat_left(q_A) - _quat_right(q_B)  # (4, 4)
        M += C.T @ C

    eigvals, eigvecs = np.linalg.eigh(M)
    q_X = eigvecs[:, 0]  # smallest eigenvalue
    if q_X[0] < 0:
        q_X = -q_X
    q_X /= np.linalg.norm(q_X)

    # Report condition (ratio of two smallest eigenvalues)
    if len(R_A_list) >= 3:
        ratio = eigvals[0] / (eigvals[1] + 1e-15)
        if ratio > 0.5:
            print(f"  WARNING: hand-eye eigenvalue ratio={ratio:.3f} "
                  "(close to 1 suggests degenerate motions)")

    R_X = Rotation.from_quat([q_X[1], q_X[2], q_X[3], q_X[0]]).as_matrix()
    return R_X


# ═════════════════════════════════════════════════════════════════
#  Translation Estimation (lever-arm least squares)
# ═════════════════════════════════════════════════════════════════
def solve_translation(
    R_A_list: List[np.ndarray],
    t_A_list: List[np.ndarray],
    R_X: np.ndarray,
    min_rotation_deg: float = 5.0,
) -> np.ndarray:
    """
    Solve  (R_A - I) · t_X ≈ t_A  for t_X via least squares.

    This lever-arm approximation is valid when the IMU translation during
    each inter-frame interval is small compared to the lever-arm effect.
    We select only pairs with sufficient rotation to keep the system
    well-conditioned.
    """
    A_blocks = []
    b_blocks = []

    for R_A, t_A in zip(R_A_list, t_A_list):
        angle = np.arccos(np.clip((np.trace(R_A) - 1) / 2, -1, 1))
        if np.degrees(angle) < min_rotation_deg:
            continue
        A_blocks.append(R_A - np.eye(3))
        b_blocks.append(t_A)

    if len(A_blocks) < 3:
        print("  WARNING: fewer than 3 high-rotation pairs available "
              "for translation estimation, result may be inaccurate")
        # Fall back to all pairs
        A_blocks = [R - np.eye(3) for R in R_A_list]
        b_blocks = list(t_A_list)

    A_mat = np.vstack(A_blocks)  # (3K, 3)
    b_vec = np.concatenate(b_blocks)  # (3K,)

    t_X, _, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    if rank < 3:
        print(f"  WARNING: translation system rank={rank} (need 3). "
              "Include more rotational motions.")
    return t_X


# ═════════════════════════════════════════════════════════════════
#  Spatiotemporal Refinement
# ═════════════════════════════════════════════════════════════════
def refine_calibration(
    motions: List[RelativeMotion],
    imu_buf: ImuBuffer,
    R_init: np.ndarray,
    t_init: np.ndarray,
    td_init: float = 0.0,
    optimize_td: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Jointly refine rotation, translation and time offset.

    Cost = Σ_i  w_rot · ‖log(R_A_i R_X R_B_i^T R_X^T)‖²
         + Σ_i  w_trans · ‖(R_A_i - I) t_X - t_A_i‖²

    The rotation part re-integrates IMU with the shifted time offset.
    """
    stamps_pairs = [(m.t0, m.t1) for m in motions]
    R_A_list = [m.R for m in motions]
    t_A_list = [m.t for m in motions]

    def cost(params):
        rvec = params[:3]
        tvec = params[3:6]
        td = params[6] if optimize_td else td_init

        R_X = Rotation.from_rotvec(rvec).as_matrix()
        total = 0.0

        for i, (s0, s1) in enumerate(stamps_pairs):
            R_B = imu_buf.integrate_rotation(s0 + td, s1 + td)

            # Rotation residual
            R_err = R_A_list[i] @ R_X @ R_B.T @ R_X.T
            rot_err = Rotation.from_matrix(R_err).magnitude()
            total += rot_err ** 2

            # Translation residual (lever-arm, weighted lower)
            t_pred = (R_A_list[i] - np.eye(3)) @ tvec
            t_err = t_A_list[i] - t_pred
            total += 0.05 * np.dot(t_err, t_err)

        return total

    x0 = np.concatenate([
        Rotation.from_matrix(R_init).as_rotvec(),
        t_init,
        [td_init],
    ])

    ndim = 7 if optimize_td else 6
    result = minimize(
        cost,
        x0[:ndim] if not optimize_td else x0,
        method="Nelder-Mead",
        options={"maxiter": 30000, "xatol": 1e-9, "fatol": 1e-13,
                 "adaptive": True},
    )

    R_opt = Rotation.from_rotvec(result.x[:3]).as_matrix()
    t_opt = result.x[3:6]
    td_opt = result.x[6] if optimize_td else td_init

    return R_opt, t_opt, td_opt


# ═════════════════════════════════════════════════════════════════
#  Evaluation Helpers
# ═════════════════════════════════════════════════════════════════
def evaluate_residuals(
    motions: List[RelativeMotion],
    imu_buf: ImuBuffer,
    R_X: np.ndarray,
    t_X: np.ndarray,
    td: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-pair rotation (deg) and translation (m) residuals."""
    rot_errs = []
    trans_errs = []

    for m in motions:
        R_B = imu_buf.integrate_rotation(m.t0 + td, m.t1 + td)
        R_err = m.R @ R_X @ R_B.T @ R_X.T
        rot_errs.append(np.degrees(Rotation.from_matrix(R_err).magnitude()))

        t_pred = (m.R - np.eye(3)) @ t_X
        trans_errs.append(np.linalg.norm(m.t - t_pred))

    return np.array(rot_errs), np.array(trans_errs)


def format_result(R: np.ndarray, t: np.ndarray) -> str:
    """Format as small_glim config string: [tx, ty, tz, qx, qy, qz, qw]."""
    q = Rotation.from_matrix(R).as_quat()  # [x, y, z, w] (scipy default)
    return (f"[{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}, "
            f"{q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}]")


# ═════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Spatiotemporal LiDAR-IMU extrinsic calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("bag", help="Path to ROS 2 rosbag directory")
    parser.add_argument("--imu-topic", default="/serial_bridge/imu_raw",
                        help="IMU topic name (default: /serial_bridge/imu_raw)")
    parser.add_argument("--lidar-topic", default="/mid360_driver/lidar",
                        help="LiDAR topic name (default: /mid360_driver/lidar)")
    parser.add_argument("--max-scans", type=int, default=1000,
                        help="Maximum number of LiDAR scans to use (default: 1000)")
    parser.add_argument("--skip-scans", type=int, default=10,
                        help="Skip the first N scans (default: 10)")
    parser.add_argument("--voxel-size", type=float, default=0.5,
                        help="Voxel downsampling resolution for ICP (m) (default: 0.5)")
    parser.add_argument("--max-corr-dist", type=float, default=1.0,
                        help="ICP max correspondence distance (m) (default: 1.0)")
    parser.add_argument("--min-rotation", type=float, default=2.0,
                        help="Min rotation (deg) to include a pair (default: 2.0)")
    parser.add_argument("--no-refine-td", action="store_true",
                        help="Disable time-offset refinement")
    args = parser.parse_args()

    bag_path = str(Path(args.bag).resolve())

    # ── Step 1: Read data ──
    print(f"[1/5] Reading rosbag: {bag_path}")
    print(f"       IMU topic:   {args.imu_topic}")
    print(f"       LiDAR topic: {args.lidar_topic}")
    t_start = time.time()
    imus, scans = read_bag(bag_path, args.imu_topic, args.lidar_topic,
                           args.max_scans, args.skip_scans)
    print(f"       IMU samples: {len(imus)}")
    print(f"       LiDAR scans: {len(scans)}")
    if len(scans) < 5:
        sys.exit("ERROR: Too few LiDAR scans. Need at least 5.")
    if len(imus) < 20:
        sys.exit("ERROR: Too few IMU samples.")
    print(f"       Time range:  {scans[0].stamp:.3f} → {scans[-1].stamp:.3f} "
          f"({scans[-1].stamp - scans[0].stamp:.1f} s)")
    print(f"       Elapsed: {time.time() - t_start:.1f} s")

    # ── Step 2: LiDAR odometry ──
    print(f"\n[2/5] Computing LiDAR odometry (ICP, voxel={args.voxel_size}m)")
    t_start = time.time()
    motions = compute_lidar_motions(scans, args.voxel_size, args.max_corr_dist)
    print(f"       Accepted motion pairs: {len(motions)}")
    if len(motions) < 5:
        sys.exit("ERROR: Too few accepted ICP pairs. Try lowering --min-fitness.")
    print(f"       Elapsed: {time.time() - t_start:.1f} s")

    # Filter by minimum rotation
    rot_angles = [np.degrees(np.arccos(np.clip((np.trace(m.R) - 1) / 2, -1, 1)))
                  for m in motions]
    filtered = [m for m, a in zip(motions, rot_angles)
                if a >= args.min_rotation]
    print(f"       Pairs with rotation ≥ {args.min_rotation}°: {len(filtered)}")
    if len(filtered) < 5:
        print("       WARNING: Few high-rotation pairs. "
              "Using all pairs for rotation calibration.")
        filtered = motions

    # ── Step 3: IMU integration ──
    print(f"\n[3/5] Integrating IMU gyroscope")
    t_start = time.time()
    imu_buf = ImuBuffer(imus)

    R_A_list = [m.R for m in filtered]
    t_A_list = [m.t for m in filtered]
    R_B_list = [imu_buf.integrate_rotation(m.t0, m.t1) for m in filtered]
    print(f"       Integrated {len(R_B_list)} rotation pairs")
    print(f"       Elapsed: {time.time() - t_start:.1f} s")

    # ── Step 4: Hand-eye calibration ──
    print(f"\n[4/5] Hand-eye calibration (AX = XB)")
    t_start = time.time()

    R_X = solve_hand_eye_rotation(R_A_list, R_B_list)
    t_X = solve_translation(R_A_list, t_A_list, R_X,
                            min_rotation_deg=max(args.min_rotation, 5.0))

    print(f"       Initial R_X (euler ZYX, deg): "
          f"{Rotation.from_matrix(R_X).as_euler('ZYX', degrees=True)}")
    print(f"       Initial t_X (m): [{t_X[0]:.5f}, {t_X[1]:.5f}, {t_X[2]:.5f}]")

    rot_err, trans_err = evaluate_residuals(motions, imu_buf, R_X, t_X, 0.0)
    print(f"       Rotation residual:  "
          f"mean={rot_err.mean():.3f}° median={np.median(rot_err):.3f}° "
          f"max={rot_err.max():.3f}°")
    print(f"       Translation residual: "
          f"mean={trans_err.mean():.5f}m max={trans_err.max():.5f}m")
    print(f"       Elapsed: {time.time() - t_start:.1f} s")

    # ── Step 5: Spatiotemporal refinement ──
    optimize_td = not args.no_refine_td
    print(f"\n[5/5] Spatiotemporal refinement "
          f"(optimize_td={'yes' if optimize_td else 'no'})")
    t_start = time.time()

    R_opt, t_opt, td_opt = refine_calibration(
        motions, imu_buf, R_X, t_X, td_init=0.0,
        optimize_td=optimize_td,
    )

    rot_err, trans_err = evaluate_residuals(motions, imu_buf,
                                            R_opt, t_opt, td_opt)
    print(f"       Rotation residual:  "
          f"mean={rot_err.mean():.3f}° median={np.median(rot_err):.3f}° "
          f"max={rot_err.max():.3f}°")
    print(f"       Translation residual: "
          f"mean={trans_err.mean():.5f}m max={trans_err.max():.5f}m")
    print(f"       Time offset td: {td_opt * 1000:.3f} ms")
    print(f"       Elapsed: {time.time() - t_start:.1f} s")

    # ── Final result ──
    euler = Rotation.from_matrix(R_opt).as_euler("ZYX", degrees=True)
    print("\n" + "=" * 60)
    print("  CALIBRATION RESULT")
    print("=" * 60)
    print(f"  Rotation (euler ZYX, deg): "
          f"[{euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f}]")
    print(f"  Translation (m):           "
          f"[{t_opt[0]:.5f}, {t_opt[1]:.5f}, {t_opt[2]:.5f}]")
    print(f"  Time offset (ms):          {td_opt * 1000:.3f}")
    print()
    print("  small_glim config format (paste into params_sensors.yaml):")
    print(f"      T_lidar_imu: {format_result(R_opt, t_opt)}")
    if optimize_td:
        print(f"      # imu_time_offset: {td_opt:.6f}  "
              "(add to node.imu_time_offset in params_node.yaml)")
    print("=" * 60)


if __name__ == "__main__":
    main()
