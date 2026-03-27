#!/usr/bin/env python3
"""
离线 LiDAR-IMU 外参标定脚本。

标定外部 IMU (BMI088) 与倒置 MID360 LiDAR 之间的变换 T_lidar_imu。

方法:
  Phase 1 — 重力对齐 + 角速度匹配 → 初始旋转 R
  Phase 2 — 杠杆臂 (lever arm) → 初始平移 t
  Phase 3 — 扫描配准 (GICP) + IMU 预积分 → hand-eye AX=XB
  Phase 4 — 联合非线性优化 (所有约束)
  Phase 5 — 信息矩阵 & 置信度评估

使用方法:
  python3 navigation_sentry_2026/utils/py/calib/calibrate_lidar_imu.py \\
      --data calib_data/calib_20260327_120000.npz \\
      --T-init "-0.09873 -0.02404 0.11875 -1.0 0.0 0.0 0.0"

输出 T_lidar_imu 格式: [tx, ty, tz, qx, qy, qz, qw]
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# ──────────────────────────────────────────────────────────────
#  常量
# ──────────────────────────────────────────────────────────────
GRAVITY = 9.80665  # m/s²
G_UNIT_TO_MS2 = GRAVITY  # MID360 加速度 g → m/s²


# ──────────────────────────────────────────────────────────────
#  Numba 加速核心计算
# ──────────────────────────────────────────────────────────────
try:
    import numba as nb
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("[WARN] numba 未安装，将使用纯 numpy (较慢)")

if HAS_NUMBA:
    @nb.njit(parallel=True)
    def _voxel_downsample_impl(pts: np.ndarray, voxel_size: float) -> np.ndarray:
        """体素下采样，返回每个体素的质心。"""
        inv = 1.0 / voxel_size
        n = pts.shape[0]
        # 计算体素键
        keys = np.empty((n, 3), dtype=np.int64)
        for i in nb.prange(n):
            keys[i, 0] = np.int64(np.floor(pts[i, 0] * inv))
            keys[i, 1] = np.int64(np.floor(pts[i, 1] * inv))
            keys[i, 2] = np.int64(np.floor(pts[i, 2] * inv))
        # 使用简单哈希
        big_prime1 = np.int64(73856093)
        big_prime2 = np.int64(19349669)
        big_prime3 = np.int64(83492791)
        table_size = np.int64(max(n * 2, 1024))
        hash_arr = np.empty(n, dtype=np.int64)
        for i in nb.prange(n):
            h = (keys[i, 0] * big_prime1) ^ (keys[i, 1] * big_prime2) ^ (keys[i, 2] * big_prime3)
            hash_arr[i] = h % table_size
        # 排序后按组聚合
        order = np.argsort(hash_arr)
        # 第一遍: 统计体素数 (保守估计)
        # 用 key 三元组精确去重
        result = np.empty((n, 3), dtype=np.float64)
        count = 0
        i = 0
        while i < n:
            idx = order[i]
            cx = pts[idx, 0]
            cy = pts[idx, 1]
            cz = pts[idx, 2]
            kx = keys[idx, 0]
            ky = keys[idx, 1]
            kz = keys[idx, 2]
            cnt = 1
            j = i + 1
            while j < n:
                jdx = order[j]
                if hash_arr[jdx] != hash_arr[idx]:
                    break
                if keys[jdx, 0] == kx and keys[jdx, 1] == ky and keys[jdx, 2] == kz:
                    cx += pts[jdx, 0]
                    cy += pts[jdx, 1]
                    cz += pts[jdx, 2]
                    cnt += 1
                    j += 1
                else:
                    break
            result[count, 0] = cx / cnt
            result[count, 1] = cy / cnt
            result[count, 2] = cz / cnt
            count += 1
            i = j
        return result[:count]

    @nb.njit(parallel=True)
    def _compute_normals_impl(pts: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
        """基于 KNN 的法向量估计。"""
        n = pts.shape[0]
        normals = np.empty((n, 3), dtype=np.float64)
        for i in nb.prange(n):
            # 取近邻
            nn_idx = indices[i, :k]
            # 计算协方差矩阵
            cx, cy, cz = 0.0, 0.0, 0.0
            cnt = 0
            for j in range(k):
                if nn_idx[j] >= 0 and nn_idx[j] < n:
                    cx += pts[nn_idx[j], 0]
                    cy += pts[nn_idx[j], 1]
                    cz += pts[nn_idx[j], 2]
                    cnt += 1
            if cnt == 0:
                normals[i, 0] = 0.0
                normals[i, 1] = 0.0
                normals[i, 2] = 1.0
                continue
            mx = cx / cnt
            my = cy / cnt
            mz = cz / cnt
            # 3x3 协方差
            cov00, cov01, cov02 = 0.0, 0.0, 0.0
            cov11, cov12, cov22 = 0.0, 0.0, 0.0
            for j in range(k):
                if nn_idx[j] >= 0 and nn_idx[j] < n:
                    dx = pts[nn_idx[j], 0] - mx
                    dy = pts[nn_idx[j], 1] - my
                    dz = pts[nn_idx[j], 2] - mz
                    cov00 += dx * dx
                    cov01 += dx * dy
                    cov02 += dx * dz
                    cov11 += dy * dy
                    cov12 += dy * dz
                    cov22 += dz * dz
            # 最小特征值对应的特征向量 = 法向量
            # 使用解析法求 3x3 对称矩阵的最小特征值特征向量
            # 迭代幂法的逆版本 (inverse iteration)
            # 先用 Jacobi 方法的简化版
            a = np.array([
                [cov00, cov01, cov02],
                [cov01, cov11, cov12],
                [cov02, cov12, cov22],
            ])
            # 简单方法: 直接求三个特征值
            # 用特征多项式
            p = -(cov00 + cov11 + cov22)
            q = cov00 * cov11 + cov00 * cov22 + cov11 * cov22 - cov01**2 - cov02**2 - cov12**2
            r = (cov00 * cov11 * cov22 + 2 * cov01 * cov02 * cov12
                 - cov00 * cov12**2 - cov11 * cov02**2 - cov22 * cov01**2)
            # λ^3 + p*λ^2 + q*λ + r = 0
            # Cardano / trigonometric method
            pp = q - p * p / 3.0
            qq = r - p * q / 3.0 + 2.0 * p * p * p / 27.0
            det = qq * qq / 4.0 + pp * pp * pp / 27.0
            if det < 0:
                m = 2.0 * np.sqrt(-pp / 3.0)
                theta = np.arccos(3.0 * qq / (pp * m + 1e-30)) / 3.0
                l0 = m * np.cos(theta) - p / 3.0
                l1 = m * np.cos(theta - 2.0 * np.pi / 3.0) - p / 3.0
                l2 = m * np.cos(theta - 4.0 * np.pi / 3.0) - p / 3.0
                lmin = min(l0, l1, l2)
            else:
                lmin = 0.0
            # Inverse iteration
            shift = lmin - 1e-6 * (abs(lmin) + 1e-10)
            b00 = a[0, 0] - shift
            b01 = a[0, 1]
            b02 = a[0, 2]
            b11 = a[1, 1] - shift
            b12 = a[1, 2]
            b22 = a[2, 2] - shift
            # 解 (A - shift*I) x = [1,1,1]
            # 手工 3x3 求解
            d = (b00 * (b11 * b22 - b12 * b12)
                 - b01 * (b01 * b22 - b12 * b02)
                 + b02 * (b01 * b12 - b11 * b02))
            if abs(d) < 1e-20:
                normals[i, 0] = 0.0
                normals[i, 1] = 0.0
                normals[i, 2] = 1.0
                continue
            inv_d = 1.0 / d
            rhs0, rhs1, rhs2 = 1.0, 1.0, 1.0
            x0 = inv_d * ((b11 * b22 - b12 * b12) * rhs0
                          + (b02 * b12 - b01 * b22) * rhs1
                          + (b01 * b12 - b02 * b11) * rhs2)
            x1 = inv_d * ((b12 * b02 - b01 * b22) * rhs0
                          + (b00 * b22 - b02 * b02) * rhs1
                          + (b01 * b02 - b00 * b12) * rhs2)
            x2 = inv_d * ((b01 * b12 - b02 * b11) * rhs0
                          + (b02 * b01 - b00 * b12) * rhs1
                          + (b00 * b11 - b01 * b01) * rhs2)
            norm = np.sqrt(x0 * x0 + x1 * x1 + x2 * x2)
            if norm < 1e-20:
                normals[i, 0] = 0.0
                normals[i, 1] = 0.0
                normals[i, 2] = 1.0
            else:
                normals[i, 0] = x0 / norm
                normals[i, 1] = x1 / norm
                normals[i, 2] = x2 / norm
        return normals

    @nb.njit(parallel=True)
    def _point_to_plane_residuals(
        src: np.ndarray,           # (M, 3)
        tgt: np.ndarray,           # (N, 3)
        tgt_normals: np.ndarray,   # (N, 3)
        nn_idx: np.ndarray,        # (M,) indices into tgt
        R: np.ndarray,             # (3, 3)
        t: np.ndarray,             # (3,)
    ) -> np.ndarray:
        """计算 point-to-plane 残差。"""
        m = src.shape[0]
        res = np.empty(m, dtype=np.float64)
        for i in nb.prange(m):
            # 变换源点
            px = R[0, 0] * src[i, 0] + R[0, 1] * src[i, 1] + R[0, 2] * src[i, 2] + t[0]
            py = R[1, 0] * src[i, 0] + R[1, 1] * src[i, 1] + R[1, 2] * src[i, 2] + t[1]
            pz = R[2, 0] * src[i, 0] + R[2, 1] * src[i, 1] + R[2, 2] * src[i, 2] + t[2]
            j = nn_idx[i]
            dx = px - tgt[j, 0]
            dy = py - tgt[j, 1]
            dz = pz - tgt[j, 2]
            res[i] = dx * tgt_normals[j, 0] + dy * tgt_normals[j, 1] + dz * tgt_normals[j, 2]
        return res

    @nb.njit(parallel=True)
    def _preintegrate_imu(
        acc: np.ndarray,   # (N, 3)
        gyro: np.ndarray,  # (N, 3)
        dt_arr: np.ndarray # (N-1,)
    ) -> Tuple:
        """IMU 预积分 (简化版: 只积分角速度得旋转增量)。
        返回 (delta_R_flat, delta_v, delta_p) 各为 (9,), (3,), (3,)
        """
        n = acc.shape[0]
        # 旋转增量 (行优先 3x3)
        R = np.eye(3)
        v = np.zeros(3)
        p = np.zeros(3)
        for i in range(n - 1):
            dt = dt_arr[i]
            wx, wy, wz = gyro[i, 0], gyro[i, 1], gyro[i, 2]
            # 旋转矩阵增量: R_new = R * exp(w * dt)
            angle = np.sqrt(wx * wx + wy * wy + wz * wz) * dt
            if angle < 1e-10:
                dR = np.eye(3)
            else:
                ax = wx * dt / angle * dt  # 轴
                # 重新计算
                theta = np.sqrt(wx * wx + wy * wy + wz * wz) * dt
                kx = wx / (theta / dt + 1e-30)
                ky = wy / (theta / dt + 1e-30)
                kz = wz / (theta / dt + 1e-30)
                ct = np.cos(theta)
                st = np.sin(theta)
                vt = 1.0 - ct
                dR = np.array([
                    [ct + kx * kx * vt,       kx * ky * vt - kz * st, kx * kz * vt + ky * st],
                    [ky * kx * vt + kz * st,  ct + ky * ky * vt,      ky * kz * vt - kx * st],
                    [kz * kx * vt - ky * st,  kz * ky * vt + kx * st, ct + kz * kz * vt],
                ])
            # 加速度在当前帧
            a_local = np.array([acc[i, 0], acc[i, 1], acc[i, 2]])
            a_nav = np.zeros(3)
            for r in range(3):
                for c in range(3):
                    a_nav[r] += R[r, c] * a_local[c]
            p = p + v * dt + 0.5 * a_nav * dt * dt
            v = v + a_nav * dt
            R_new = np.zeros((3, 3))
            for r in range(3):
                for c in range(3):
                    for k in range(3):
                        R_new[r, c] += R[r, k] * dR[k, c]
            R = R_new
        return R.ravel(), v, p

else:
    # Fallback: 纯 numpy 版本
    def _voxel_downsample_impl(pts, voxel_size):
        inv = 1.0 / voxel_size
        keys = np.floor(pts[:, :3] * inv).astype(np.int64)
        _, idx = np.unique(
            keys[:, 0] * 73856093 + keys[:, 1] * 19349669 + keys[:, 2] * 83492791,
            return_index=True)
        return pts[idx, :3].astype(np.float64)

    def _compute_normals_impl(pts, indices, k):
        n = pts.shape[0]
        normals = np.zeros((n, 3))
        for i in range(n):
            nn = pts[indices[i, :k]]
            cov = np.cov(nn.T)
            if cov.ndim < 2:
                normals[i] = [0, 0, 1]
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]
        return normals

    def _point_to_plane_residuals(src, tgt, tgt_normals, nn_idx, R, t):
        transformed = (R @ src.T).T + t
        diff = transformed - tgt[nn_idx]
        return np.sum(diff * tgt_normals[nn_idx], axis=1)

    def _preintegrate_imu(acc, gyro, dt_arr):
        n = acc.shape[0]
        R = np.eye(3)
        v = np.zeros(3)
        p = np.zeros(3)
        for i in range(n - 1):
            dt = dt_arr[i]
            w = gyro[i]
            theta = np.linalg.norm(w) * dt
            if theta < 1e-10:
                dR = np.eye(3)
            else:
                k = w / np.linalg.norm(w)
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                dR = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
            a_nav = R @ acc[i]
            p += v * dt + 0.5 * a_nav * dt * dt
            v += a_nav * dt
            R = R @ dR
        return R.ravel(), v, p


# ──────────────────────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────────────────────

def rotvec_to_mat(rv: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rv).as_matrix()


def mat_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """返回 [qx, qy, qz, qw]"""
    return Rotation.from_matrix(R).as_quat()


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """输入 [qx, qy, qz, qw]"""
    return Rotation.from_quat(q).as_matrix()


def se3_to_params(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """SE(3) → 6-DOF (rotvec[3], translation[3])"""
    return np.concatenate([mat_to_rotvec(R), t])


def params_to_se3(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """6-DOF → (R, t)"""
    return rotvec_to_mat(params[:3]), params[3:6].copy()


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ──────────────────────────────────────────────────────────────
#  数据加载
# ──────────────────────────────────────────────────────────────

@dataclass
class CalibData:
    ext_imu_t: np.ndarray   # (N_ext,)
    ext_imu_acc: np.ndarray  # (N_ext, 3) m/s²
    ext_imu_gyro: np.ndarray # (N_ext, 3) rad/s
    mid_imu_t: np.ndarray    # (N_mid,)
    mid_imu_acc: np.ndarray  # (N_mid, 3) m/s² (已转换)
    mid_imu_gyro: np.ndarray # (N_mid, 3) rad/s
    lidar_stamps: np.ndarray # (N_scan,)
    scans: List[np.ndarray]  # 每帧 (N_pts, 3)


def load_data(path: str) -> CalibData:
    print(f"加载数据: {path}")
    d = np.load(path, allow_pickle=False)
    n_scans = int(d["n_scans"])
    scans = []
    for i in range(n_scans):
        s = d[f"scan_{i:05d}"]
        scans.append(s[:, :3].astype(np.float64))  # 只取 xyz

    # MID360 加速度从 g 转换为 m/s²
    mid_acc = d["mid_imu_acc"].astype(np.float64) * G_UNIT_TO_MS2

    return CalibData(
        ext_imu_t=d["ext_imu_t"].astype(np.float64),
        ext_imu_acc=d["ext_imu_acc"].astype(np.float64),
        ext_imu_gyro=d["ext_imu_gyro"].astype(np.float64),
        mid_imu_t=d["mid_imu_t"].astype(np.float64),
        mid_imu_acc=mid_acc,
        mid_imu_gyro=d["mid_imu_gyro"].astype(np.float64),
        lidar_stamps=d["lidar_stamps"].astype(np.float64),
        scans=scans,
    )


# ──────────────────────────────────────────────────────────────
#  时间同步
# ──────────────────────────────────────────────────────────────

def sync_imu_data(
    t1: np.ndarray, d1: np.ndarray,
    t2: np.ndarray, d2: np.ndarray,
    max_dt: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按最近邻时间戳同步两组 IMU 数据。
    返回 (synced_t, synced_d1, synced_d2)
    """
    idx2 = np.searchsorted(t2, t1, side="left")
    idx2 = np.clip(idx2, 0, len(t2) - 1)

    # 也检查 idx2-1
    dt_right = np.abs(t2[idx2] - t1)
    idx2_left = np.clip(idx2 - 1, 0, len(t2) - 1)
    dt_left = np.abs(t2[idx2_left] - t1)
    use_left = dt_left < dt_right
    best_idx = np.where(use_left, idx2_left, idx2)
    best_dt = np.minimum(dt_left, dt_right)

    mask = best_dt < max_dt
    return t1[mask], d1[mask], d2[best_idx[mask]]


def sync_three(
    t1: np.ndarray, acc1: np.ndarray, gyro1: np.ndarray,
    t2: np.ndarray, acc2: np.ndarray, gyro2: np.ndarray,
    max_dt: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """同步两组 IMU 的加速度和角速度。"""
    idx2 = np.searchsorted(t2, t1, side="left")
    idx2 = np.clip(idx2, 0, len(t2) - 1)
    idx2_left = np.clip(idx2 - 1, 0, len(t2) - 1)
    dt_right = np.abs(t2[idx2] - t1)
    dt_left = np.abs(t2[idx2_left] - t1)
    use_left = dt_left < dt_right
    best_idx = np.where(use_left, idx2_left, idx2)
    best_dt = np.minimum(dt_left, dt_right)
    mask = best_dt < max_dt

    return (t1[mask],
            acc1[mask], gyro1[mask],
            acc2[best_idx[mask]], gyro2[best_idx[mask]])


# ──────────────────────────────────────────────────────────────
#  静止段检测
# ──────────────────────────────────────────────────────────────

def detect_stationary_segments(
    gyro: np.ndarray,
    timestamps: np.ndarray,
    gyro_thresh: float = 0.05,    # rad/s
    min_duration: float = 0.5,     # 秒
) -> List[Tuple[int, int]]:
    """检测角速度幅值低于阈值的静止段。"""
    gyro_norm = np.linalg.norm(gyro, axis=1)
    is_static = gyro_norm < gyro_thresh
    segments = []
    start = None
    for i in range(len(is_static)):
        if is_static[i] and start is None:
            start = i
        elif not is_static[i] and start is not None:
            if timestamps[i - 1] - timestamps[start] >= min_duration:
                segments.append((start, i))
            start = None
    if start is not None and timestamps[-1] - timestamps[start] >= min_duration:
        segments.append((start, len(is_static)))
    return segments


# ──────────────────────────────────────────────────────────────
#  Phase 1: 重力 + 角速度 → 初始旋转
# ──────────────────────────────────────────────────────────────

def estimate_rotation_gravity_gyro(
    sync_t: np.ndarray,
    ext_acc: np.ndarray,   # (N, 3) m/s²
    ext_gyro: np.ndarray,  # (N, 3) rad/s
    mid_acc: np.ndarray,   # (N, 3) m/s² (已转换)
    mid_gyro: np.ndarray,  # (N, 3) rad/s
    T_lidar_livox_imu: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    利用重力方向一致性和角速度匹配估计 R_ext_livox (从 livox_imu 到 ext_imu 的旋转)。

    然后 R_lidar_ext = R_lidar_livox * R_livox_ext = R_lidar_livox * R_ext_livox^T

    使用 Wahba 问题的 SVD 解法。
    """
    R_L_I, t_L_I = T_lidar_livox_imu  # T from livox_imu to lidar

    # 收集向量对 (在各自 IMU 坐标系下)
    # 重力约束: 静止时加速度方向
    ext_gyro_norm = np.linalg.norm(ext_gyro, axis=1)
    static_mask = ext_gyro_norm < 0.05

    vectors_ext = []
    vectors_mid = []
    weights = []

    if np.sum(static_mask) > 50:
        # 使用静止段的平均重力
        g_ext = np.mean(ext_acc[static_mask], axis=0)
        g_mid = np.mean(mid_acc[static_mask], axis=0)
        g_ext_n = g_ext / (np.linalg.norm(g_ext) + 1e-10)
        g_mid_n = g_mid / (np.linalg.norm(g_mid) + 1e-10)
        vectors_ext.append(g_ext_n)
        vectors_mid.append(g_mid_n)
        weights.append(10.0)  # 重力约束权重高

    # 角速度约束: 高角速度时方向应一致
    dynamic_mask = ext_gyro_norm > 0.3
    if np.sum(dynamic_mask) > 20:
        # 取多个角速度方向对
        dyn_ext_gyro = ext_gyro[dynamic_mask]
        dyn_mid_gyro = mid_gyro[dynamic_mask]
        # 采样以避免过多重复方向
        step = max(1, len(dyn_ext_gyro) // 100)
        for k in range(0, len(dyn_ext_gyro), step):
            ge = dyn_ext_gyro[k]
            gm = dyn_mid_gyro[k]
            ne = np.linalg.norm(ge)
            nm = np.linalg.norm(gm)
            if ne > 0.1 and nm > 0.1:
                vectors_ext.append(ge / ne)
                vectors_mid.append(gm / nm)
                weights.append(1.0)

    if len(vectors_ext) < 2:
        print("[WARN] 向量对不足，旋转估计可能不可靠")
        return np.eye(3)

    # Wahba 问题: 求 R 使得 sum(w_i * ||v_ext_i - R * v_mid_i||^2) 最小
    # 等价于: R = U * diag(1,1,det(U)*det(V)) * V^T 其中 B = sum(w_i * v_ext_i * v_mid_i^T), B = U S V^T
    V_ext = np.array(vectors_ext)
    V_mid = np.array(vectors_mid)
    W = np.array(weights)

    B = np.zeros((3, 3))
    for i in range(len(W)):
        B += W[i] * np.outer(V_ext[i], V_mid[i])

    U, S, Vt = np.linalg.svd(B)
    d = np.linalg.det(U) * np.linalg.det(Vt)
    R_ext_mid = U @ np.diag([1, 1, d]) @ Vt

    # R_ext_mid: 从 livox_imu 到 ext_imu 的旋转
    # R_lidar_ext = R_L_I @ R_ext_mid^T
    R_L_E = R_L_I @ R_ext_mid.T

    return R_L_E


# ──────────────────────────────────────────────────────────────
#  Phase 2: 杠杆臂 → 初始平移
# ──────────────────────────────────────────────────────────────

def estimate_translation_lever_arm(
    sync_t: np.ndarray,
    ext_acc: np.ndarray,    # (N, 3) m/s²
    ext_gyro: np.ndarray,   # (N, 3) rad/s
    mid_acc: np.ndarray,    # (N, 3) m/s²
    mid_gyro: np.ndarray,   # (N, 3) rad/s
    R_E_I: np.ndarray,      # 从 livox_imu 到 ext_imu 的旋转
    T_lidar_livox_imu: Tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    利用杠杆臂原理估计平移。

    在 ext_imu 坐标系下:
      a_E = R_E_I * a_I + alpha_E × r + omega_E × (omega_E × r)

    其中 r = t_E_to_I_in_E = 从 E 到 I 在 E 系中的向量

    重排为线性方程: A * r = b
      A_i = [alpha_E_i]× + [omega_E_i]× @ [omega_E_i]×
      b_i = a_E_i - R_E_I * a_I_i
    """
    n = len(sync_t)
    if n < 20:
        print("[WARN] 数据点不足，平移无法估计")
        return np.zeros(3)

    # 计算角加速度 (有限差分)
    dt = np.diff(sync_t)
    alpha = np.diff(ext_gyro, axis=0) / dt[:, None]

    # 使用中间时刻
    omega = 0.5 * (ext_gyro[:-1] + ext_gyro[1:])
    a_ext = 0.5 * (ext_acc[:-1] + ext_acc[1:])
    a_mid = 0.5 * (mid_acc[:-1] + mid_acc[1:])

    # 过滤: 只用有足够角速度的时刻
    omega_norm = np.linalg.norm(omega, axis=1)
    mask = omega_norm > 0.2  # 需要一定角速度才有杠杆臂效应
    if np.sum(mask) < 10:
        print("[WARN] 动态段不足，平移估计可能不准")
        mask = np.ones(len(omega), dtype=bool)

    omega_m = omega[mask]
    alpha_m = alpha[mask]
    a_ext_m = a_ext[mask]
    a_mid_m = a_mid[mask]
    m = len(omega_m)

    # 构建线性系统
    A = np.zeros((m * 3, 3))
    b = np.zeros(m * 3)
    for i in range(m):
        W = skew(omega_m[i])
        Al = skew(alpha_m[i])
        A[3*i:3*i+3, :] = Al + W @ W
        b[3*i:3*i+3] = a_ext_m[i] - R_E_I @ a_mid_m[i]

    # 正则化最小二乘
    r, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # r 是 t_E_to_I_in_E: 从 ext_imu 到 livox_imu 在 ext_imu 系中的向量
    # 我们需要 t_L_E (lidar 系中 ext_imu 原点的位置)
    R_L_I, t_L_I = T_lidar_livox_imu
    R_L_E_approx = R_L_I @ R_E_I.T  # 暂不精确，但近似可用
    # t_L_E = t_L_I - R_L_E @ r  (因为 r = R_E_L @ (t_L_I - t_L_E) ≈ R_L_E^T @ (t_L_I - t_L_E))
    # 所以 t_L_E = t_L_I - R_L_E @ r
    # 更准确: t_E_to_I_in_E = R_L_E^T @ (t_L_I - t_L_E)
    # => t_L_E = t_L_I - R_L_E @ r
    t_L_E = t_L_I - R_L_E_approx @ r

    return t_L_E


# ──────────────────────────────────────────────────────────────
#  点云处理
# ──────────────────────────────────────────────────────────────

def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts
    return _voxel_downsample_impl(np.ascontiguousarray(pts[:, :3], dtype=np.float64), voxel_size)


def estimate_normals(pts: np.ndarray, k: int = 20) -> np.ndarray:
    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=k)
    return _compute_normals_impl(np.ascontiguousarray(pts, dtype=np.float64), indices.astype(np.int64), k)


# ──────────────────────────────────────────────────────────────
#  Phase 3: 扫描配准 (point-to-plane ICP)
# ──────────────────────────────────────────────────────────────

def icp_point_to_plane(
    src: np.ndarray,
    tgt: np.ndarray,
    tgt_normals: np.ndarray,
    max_iters: int = 30,
    dist_thresh: float = 0.5,
    tol: float = 1e-6,
    init_R: Optional[np.ndarray] = None,
    init_t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Point-to-plane ICP，返回 (R, t, fitness)。"""
    R = init_R if init_R is not None else np.eye(3)
    t = init_t if init_t is not None else np.zeros(3)

    tgt_tree = cKDTree(tgt)

    for iteration in range(max_iters):
        # 变换源点
        src_tf = (R @ src.T).T + t

        # 最近邻
        dists, nn_idx = tgt_tree.query(src_tf, k=1)

        # 距离过滤
        mask = dists < dist_thresh
        if np.sum(mask) < 10:
            break

        src_m = src[mask]
        nn_m = nn_idx[mask]

        # Point-to-plane 线性化
        # 残差: ((R*s + t) - tgt[nn]) · n = 0
        # 对于小增量 δ = (δω, δt):
        #   ((I + [δω]×) * R * s + t + δt - tgt[nn]) · n = 0
        # => ([R*s]× · δω + δt - (tgt[nn] - R*s - t)) · n = 0
        # => n^T * [R*s]× * δω + n^T * δt = n^T * (tgt[nn] - R*s - t)
        src_tf_m = (R @ src_m.T).T + t
        diff = tgt[nn_m] - src_tf_m
        n_pts = src_m.shape[0]

        # 构建线性系统
        A = np.zeros((n_pts, 6))
        b_vec = np.zeros(n_pts)
        for i in range(n_pts):
            ni = tgt_normals[nn_m[i]]
            si = src_tf_m[i]
            # [si]× @ ni → cross(si, ni)
            c = np.cross(si, ni)
            A[i, :3] = c
            A[i, 3:6] = ni
            b_vec[i] = ni @ diff[i]

        # 解
        x, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        delta_w = x[:3]
        delta_t = x[3:6]

        # 更新
        dR = rotvec_to_mat(delta_w)
        R = dR @ R
        t = dR @ t + delta_t

        if np.linalg.norm(delta_w) < tol and np.linalg.norm(delta_t) < tol:
            break

    # 计算 fitness
    src_tf = (R @ src.T).T + t
    dists, _ = tgt_tree.query(src_tf, k=1)
    fitness = np.mean(dists < dist_thresh)

    return R, t, fitness


def compute_scan_relative_poses(
    scans: List[np.ndarray],
    voxel_size: float = 0.1,
    normal_k: int = 20,
    icp_dist: float = 0.5,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """对连续扫描对进行 ICP 配准，返回相对位姿列表。"""
    print(f"  计算 {len(scans)-1} 对扫描配准...")
    results = []
    prev_ds = voxel_downsample(scans[0], voxel_size)
    prev_normals = estimate_normals(prev_ds, normal_k)

    for i in range(1, len(scans)):
        curr_ds = voxel_downsample(scans[i], voxel_size)
        curr_normals = estimate_normals(curr_ds, normal_k)

        R, t, fitness = icp_point_to_plane(
            curr_ds, prev_ds, prev_normals,
            dist_thresh=icp_dist,
        )
        results.append((R, t, fitness))

        if (i % 10 == 0) or (i == len(scans) - 1):
            print(f"    [{i}/{len(scans)-1}] fitness={fitness:.3f}")

        prev_ds = curr_ds
        prev_normals = curr_normals

    return results


# ──────────────────────────────────────────────────────────────
#  IMU 预积分
# ──────────────────────────────────────────────────────────────

def compute_imu_preintegration(
    imu_t: np.ndarray,
    imu_acc: np.ndarray,
    imu_gyro: np.ndarray,
    scan_stamps: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """为每对连续扫描计算 IMU 预积分。"""
    results = []
    for i in range(len(scan_stamps) - 1):
        t0, t1 = scan_stamps[i], scan_stamps[i + 1]
        mask = (imu_t >= t0) & (imu_t <= t1)
        if np.sum(mask) < 2:
            results.append((np.eye(3).ravel(), np.zeros(3), np.zeros(3)))
            continue
        acc_seg = np.ascontiguousarray(imu_acc[mask], dtype=np.float64)
        gyro_seg = np.ascontiguousarray(imu_gyro[mask], dtype=np.float64)
        dt_arr = np.diff(imu_t[mask]).astype(np.float64)
        dR, dv, dp = _preintegrate_imu(acc_seg, gyro_seg, dt_arr)
        results.append((dR.reshape(3, 3), dv, dp))
    return results


# ──────────────────────────────────────────────────────────────
#  Phase 3: Hand-eye AX = XB
# ──────────────────────────────────────────────────────────────

def hand_eye_calibration(
    lidar_poses: List[Tuple[np.ndarray, np.ndarray]],
    imu_poses: List[Tuple[np.ndarray, np.ndarray]],
    R_init: np.ndarray,
    t_init: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hand-eye 标定: A_i @ X = X @ B_i
    A_i: LiDAR 相对位姿 (从扫描配准)
    B_i: IMU 相对位姿 (从预积分)
    X = T_lidar_ext_imu

    使用 Tsai-Lenz 方法改进。
    """
    n = len(lidar_poses)
    if n < 2:
        return R_init, t_init

    # --- 旋转部分: 求解 R_A @ R_X = R_X @ R_B ---
    # 使用四元数方法
    M = np.zeros((4 * n, 4))
    for i in range(n):
        R_A = lidar_poses[i][0]
        R_B = imu_poses[i][0]
        q_a = Rotation.from_matrix(R_A).as_quat()  # [x, y, z, w]
        q_b = Rotation.from_matrix(R_B).as_quat()
        # (q_a - q_b) @ q_x = 0 AND (q_a + q_b) @ q_x = 0 的混合
        # 使用 Kronecker product 方法
        # left(q_a) - right(q_b) 的零空间
        L = _quat_left_mult(q_a) - _quat_right_mult(q_b)
        M[4*i:4*i+4, :] = L

    _, S, Vt = np.linalg.svd(M)
    q_x = Vt[-1]  # 最小奇异值对应的右奇异向量
    # 确保 w > 0
    if q_x[3] < 0:
        q_x = -q_x
    R_X = quat_to_mat(q_x)

    # --- 平移部分: (R_A - I) @ t_X = R_X @ t_B - t_A ---
    C = np.zeros((3 * n, 3))
    d = np.zeros(3 * n)
    for i in range(n):
        R_A, t_A = lidar_poses[i]
        R_B, t_B = imu_poses[i]
        C[3*i:3*i+3, :] = R_A - np.eye(3)
        d[3*i:3*i+3] = R_X @ t_B - t_A
    t_X, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    return R_X, t_X


def _quat_left_mult(q: np.ndarray) -> np.ndarray:
    """四元数左乘矩阵 [x, y, z, w] 格式。"""
    x, y, z, w = q
    return np.array([
        [ w, -z,  y,  x],
        [ z,  w, -x,  y],
        [-y,  x,  w,  z],
        [-x, -y, -z,  w],
    ])


def _quat_right_mult(q: np.ndarray) -> np.ndarray:
    """四元数右乘矩阵 [x, y, z, w] 格式。"""
    x, y, z, w = q
    return np.array([
        [ w,  z, -y,  x],
        [-z,  w,  x,  y],
        [ y, -x,  w,  z],
        [-x, -y, -z,  w],
    ])


# ──────────────────────────────────────────────────────────────
#  Phase 4: 联合非线性优化
# ──────────────────────────────────────────────────────────────

def build_residuals(
    params: np.ndarray,  # [rotvec(3), t(3)]
    sync_t: np.ndarray,
    ext_acc: np.ndarray,
    ext_gyro: np.ndarray,
    mid_acc: np.ndarray,
    mid_gyro: np.ndarray,
    lidar_rel_poses: List[Tuple[np.ndarray, np.ndarray, float]],
    imu_preint: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    T_lidar_livox_imu: Tuple[np.ndarray, np.ndarray],
    w_gravity: float = 50.0,
    w_gyro: float = 5.0,
    w_lever: float = 2.0,
    w_handeye_rot: float = 20.0,
    w_handeye_trans: float = 10.0,
) -> np.ndarray:
    """构建所有约束的残差向量。"""
    R_L_E, t_L_E = params_to_se3(params)
    R_L_I, t_L_I = T_lidar_livox_imu

    # R_E_I: livox_imu 到 ext_imu 的旋转
    R_E_I = R_L_E.T @ R_L_I

    residuals = []

    # ─── 1. 重力方向一致性 ───
    gyro_norm = np.linalg.norm(ext_gyro, axis=1)
    static_mask = gyro_norm < 0.05
    if np.sum(static_mask) > 20:
        g_ext = np.mean(ext_acc[static_mask], axis=0)
        g_mid = np.mean(mid_acc[static_mask], axis=0)
        g_ext_n = g_ext / (np.linalg.norm(g_ext) + 1e-10)
        g_mid_n = g_mid / (np.linalg.norm(g_mid) + 1e-10)
        r_grav = w_gravity * (R_E_I @ g_mid_n - g_ext_n)
        residuals.append(r_grav)

    # ─── 2. 角速度一致性 (采样) ───
    dynamic_mask = gyro_norm > 0.1
    if np.sum(dynamic_mask) > 10:
        idx_dyn = np.where(dynamic_mask)[0]
        step = max(1, len(idx_dyn) // 200)
        for k in range(0, len(idx_dyn), step):
            i = idx_dyn[k]
            r_gyro = w_gyro * (R_E_I @ mid_gyro[i] - ext_gyro[i])
            residuals.append(r_gyro)

    # ─── 3. 杠杆臂加速度 (采样) ───
    if len(sync_t) > 20:
        dt = np.diff(sync_t)
        alpha = np.diff(ext_gyro, axis=0) / (dt[:, None] + 1e-10)
        omega_mid_pts = 0.5 * (ext_gyro[:-1] + ext_gyro[1:])
        a_ext_mid = 0.5 * (ext_acc[:-1] + ext_acc[1:])
        a_mid_mid = 0.5 * (mid_acc[:-1] + mid_acc[1:])

        # t_E_to_I_in_E
        r_E_to_I = R_L_E.T @ (t_L_I - t_L_E)

        omega_norm_m = np.linalg.norm(omega_mid_pts, axis=1)
        lever_mask = omega_norm_m > 0.2
        idx_lever = np.where(lever_mask)[0]
        step = max(1, len(idx_lever) // 150)
        for k in range(0, len(idx_lever), step):
            i = idx_lever[k]
            w_vec = omega_mid_pts[i]
            a_vec = alpha[i]
            predicted = R_E_I @ a_mid_mid[i] + np.cross(a_vec, r_E_to_I) + np.cross(w_vec, np.cross(w_vec, r_E_to_I))
            r_lever = w_lever * (predicted - a_ext_mid[i])
            residuals.append(r_lever)

    # ─── 4. Hand-eye 约束 (扫描配准 vs IMU 预积分) ───
    n_pairs = min(len(lidar_rel_poses), len(imu_preint))
    for i in range(n_pairs):
        R_A, t_A, fitness = lidar_rel_poses[i]
        R_B, dv, dp = imu_preint[i]
        if fitness < 0.3:
            continue
        weight_scale = fitness  # 以 fitness 作为权重

        # AX = XB → R_A @ R_X = R_X @ R_B
        # 旋转残差
        R_err = R_A @ R_L_E - R_L_E @ R_B
        r_rot = w_handeye_rot * weight_scale * mat_to_rotvec(R_err.T @ R_err + np.eye(3))
        # 简化: 用 log(R_A @ R_X @ R_B^T @ R_X^T) 作为旋转残差
        R_residual = R_A @ R_L_E @ R_B.T @ R_L_E.T
        r_rot_vec = w_handeye_rot * weight_scale * mat_to_rotvec(R_residual)
        residuals.append(r_rot_vec)

        # 平移残差: (R_A - I) @ t_X = R_X @ t_B - t_A
        # 这里 t_B 来自 IMU 预积分的位置增量 dp
        r_trans = w_handeye_trans * weight_scale * (
            (R_A - np.eye(3)) @ t_L_E - (R_L_E @ dp - t_A)
        )
        residuals.append(r_trans)

    if not residuals:
        return np.zeros(1)

    return np.concatenate(residuals)


def joint_optimize(
    R_init: np.ndarray,
    t_init: np.ndarray,
    sync_t: np.ndarray,
    ext_acc: np.ndarray,
    ext_gyro: np.ndarray,
    mid_acc: np.ndarray,
    mid_gyro: np.ndarray,
    lidar_rel_poses: List[Tuple[np.ndarray, np.ndarray, float]],
    imu_preint: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    T_lidar_livox_imu: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """联合非线性最小二乘优化。"""
    x0 = se3_to_params(R_init, t_init)

    def cost_fn(params):
        return build_residuals(
            params, sync_t, ext_acc, ext_gyro, mid_acc, mid_gyro,
            lidar_rel_poses, imu_preint, T_lidar_livox_imu,
        )

    print("  运行联合优化...")
    result = least_squares(
        cost_fn, x0,
        method="lm",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=500,
        verbose=1,
    )

    R_opt, t_opt = params_to_se3(result.x)

    info = {
        "cost": result.cost,
        "optimality": result.optimality,
        "nfev": result.nfev,
        "success": result.success,
        "message": result.message,
        "jacobian": result.jac,
        "residuals": result.fun,
    }
    return R_opt, t_opt, info


# ──────────────────────────────────────────────────────────────
#  Phase 5: 置信度评估
# ──────────────────────────────────────────────────────────────

def compute_confidence(jac: np.ndarray, residuals: np.ndarray) -> dict:
    """
    从 Jacobian 和残差计算信息矩阵、协方差、特征值。

    信息矩阵 H = J^T @ J
    协方差 Σ ≈ σ² * (J^T @ J)^{-1}  其中 σ² = ||r||² / (m - n)
    """
    m, n = jac.shape  # m: 残差数, n: 参数数 (6)
    H = jac.T @ jac   # 信息矩阵 (Hessian 近似)

    # 残差方差
    dof = max(m - n, 1)
    sigma2 = np.sum(residuals ** 2) / dof

    # 协方差矩阵
    try:
        cov = sigma2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(H)

    # 特征分解
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    eigvals_cov, eigvecs_cov = np.linalg.eigh(cov)

    # 标准差 (rotation in rad, translation in m)
    std_devs = np.sqrt(np.maximum(np.diag(cov), 0))

    # 条件数
    cond = eigvals_H[-1] / max(eigvals_H[0], 1e-20)

    # 各方向的可观测性 (特征值越大越可观测)
    observability = eigvals_H / (np.max(eigvals_H) + 1e-20)

    return {
        "information_matrix": H,
        "covariance": cov,
        "sigma2": sigma2,
        "eigenvalues_info": eigvals_H,
        "eigenvectors_info": eigvecs_H,
        "eigenvalues_cov": eigvals_cov,
        "std_devs": std_devs,
        "condition_number": cond,
        "observability": observability,
    }


def print_confidence(conf: dict, R: np.ndarray, t: np.ndarray) -> None:
    """格式化打印置信度信息。"""
    print("\n" + "=" * 60)
    print("  置信度分析")
    print("=" * 60)

    std = conf["std_devs"]
    print(f"\n  参数标准差 (1σ):")
    print(f"    旋转 rx: {np.degrees(std[0]):.4f}°")
    print(f"    旋转 ry: {np.degrees(std[1]):.4f}°")
    print(f"    旋转 rz: {np.degrees(std[2]):.4f}°")
    print(f"    平移 tx: {std[3]*1000:.3f} mm")
    print(f"    平移 ty: {std[4]*1000:.3f} mm")
    print(f"    平移 tz: {std[5]*1000:.3f} mm")

    print(f"\n  信息矩阵特征值:")
    for i, ev in enumerate(conf["eigenvalues_info"]):
        obs = conf["observability"][i]
        bar = "█" * max(1, int(obs * 30))
        print(f"    λ_{i}: {ev:12.2f}  可观测性: {obs:.4f}  {bar}")

    print(f"\n  条件数: {conf['condition_number']:.2f}")
    if conf["condition_number"] > 1e6:
        print("  ⚠ 条件数过大，部分方向可观测性差!")
    elif conf["condition_number"] > 1e3:
        print("  ⚠ 条件数较大，建议增加激励运动")

    # 总体置信度评分
    min_obs = np.min(conf["observability"])
    if min_obs > 0.01:
        grade = "优秀"
    elif min_obs > 0.001:
        grade = "良好"
    elif min_obs > 0.0001:
        grade = "一般"
    else:
        grade = "差 - 建议重新采集数据"
    print(f"\n  总体标定质量: {grade}")
    print(f"  最小可观测性: {min_obs:.6f}")


# ──────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────

def parse_transform(s: str) -> Tuple[np.ndarray, np.ndarray]:
    """解析 'tx ty tz qx qy qz qw' 格式的变换。"""
    vals = list(map(float, s.strip().split()))
    assert len(vals) == 7, f"需要 7 个值 (tx ty tz qx qy qz qw)，得到 {len(vals)}"
    t = np.array(vals[:3])
    q = np.array(vals[3:7])  # [qx, qy, qz, qw]
    R = quat_to_mat(q)
    return R, t


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LiDAR-IMU 外参标定",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="录制数据 .npz 文件路径")
    parser.add_argument(
        "--T-init",
        type=str,
        default="-0.09873 -0.02404 0.11875 -1.0 0.0 0.0 0.0",
        help="T_lidar_imu 初始值 (tx ty tz qx qy qz qw)",
    )
    parser.add_argument(
        "--T-livox-imu-lidar",
        type=str,
        default="0.006 -0.012 0.008 0.0 0.0 0.0 1.0",
        help="MID360 内部 T_lidar_livox_imu (tx ty tz qx qy qz qw)",
    )
    parser.add_argument("--voxel-size", type=float, default=0.1, help="点云下采样体素大小 (m)")
    parser.add_argument("--icp-dist", type=float, default=0.5, help="ICP 最大对应距离 (m)")
    parser.add_argument("--max-scan-pairs", type=int, default=200, help="最大扫描对数")
    parser.add_argument("--skip-icp", action="store_true", help="跳过点云配准 (仅用 IMU 约束)")
    parser.add_argument("--output", type=str, default="", help="输出结果文件路径 (.npz)")
    args = parser.parse_args()

    t_start = time.time()

    # ─── 加载数据 ───
    data = load_data(args.data)
    print(f"  外部 IMU: {len(data.ext_imu_t)} 样本")
    print(f"  MID360 IMU: {len(data.mid_imu_t)} 样本")
    print(f"  LiDAR 扫描: {len(data.scans)} 帧")

    if len(data.ext_imu_t) < 100 or len(data.mid_imu_t) < 100:
        print("[ERROR] IMU 数据太少，无法标定")
        return 1

    # ─── 解析已知变换 ───
    R_init, t_init = parse_transform(args.T_init)
    R_L_I, t_L_I = parse_transform(args.T_livox_imu_lidar)
    T_LI = (R_L_I, t_L_I)

    print(f"\n初始 T_lidar_imu:")
    print(f"  t = [{t_init[0]:.5f}, {t_init[1]:.5f}, {t_init[2]:.5f}] m")
    q_init = mat_to_quat(R_init)
    print(f"  q = [{q_init[0]:.5f}, {q_init[1]:.5f}, {q_init[2]:.5f}, {q_init[3]:.5f}]")

    # ─── 时间同步 ───
    print("\n[Phase 0] 时间同步...")
    sync_t, ext_acc, ext_gyro, mid_acc, mid_gyro = sync_three(
        data.ext_imu_t, data.ext_imu_acc, data.ext_imu_gyro,
        data.mid_imu_t, data.mid_imu_acc, data.mid_imu_gyro,
    )
    print(f"  同步后 IMU 对数: {len(sync_t)}")

    # 检测静止段
    static_segs = detect_stationary_segments(ext_gyro, sync_t)
    total_static = sum(sync_t[e-1] - sync_t[s] for s, e in static_segs)
    print(f"  检测到 {len(static_segs)} 个静止段，总时长 {total_static:.1f}s")

    # ─── Phase 1: 旋转估计 ───
    print("\n[Phase 1] 重力+角速度 → 旋转估计...")
    R_L_E_grav = estimate_rotation_gravity_gyro(
        sync_t, ext_acc, ext_gyro, mid_acc, mid_gyro, T_LI,
    )
    rv_grav = mat_to_rotvec(R_L_E_grav)
    print(f"  重力+角速度估计 rotvec: [{rv_grav[0]:.5f}, {rv_grav[1]:.5f}, {rv_grav[2]:.5f}]")

    # 与初始值比较，选择更可靠的
    rv_init = mat_to_rotvec(R_init)
    angle_diff = np.degrees(np.linalg.norm(rv_grav - rv_init))
    print(f"  与初始值角度差: {angle_diff:.2f}°")
    if angle_diff > 20:
        print("  ⚠ 旋转差异较大，使用初始值作为基准并融合数据估计")
        # 加权融合: 更信任初始值
        R_phase1 = R_init
    else:
        # 用重力估计修正
        R_phase1 = R_L_E_grav

    # ─── Phase 2: 平移估计 ───
    print("\n[Phase 2] 杠杆臂 → 平移估计...")
    R_E_I_est = R_phase1.T @ R_L_I
    t_L_E_lever = estimate_translation_lever_arm(
        sync_t, ext_acc, ext_gyro, mid_acc, mid_gyro,
        R_E_I_est, T_LI,
    )
    print(f"  杠杆臂估计 t: [{t_L_E_lever[0]:.5f}, {t_L_E_lever[1]:.5f}, {t_L_E_lever[2]:.5f}] m")
    t_diff = np.linalg.norm(t_L_E_lever - t_init)
    print(f"  与初始值距离: {t_diff*1000:.1f} mm")
    if t_diff > 0.05:
        print("  ⚠ 平移差异较大，使用初始值")
        t_phase2 = t_init.copy()
    else:
        t_phase2 = 0.5 * (t_L_E_lever + t_init)

    # ─── Phase 3: 点云配准 + Hand-eye ───
    lidar_rel_poses = []
    imu_preint = []

    if not args.skip_icp and len(data.scans) >= 2:
        print("\n[Phase 3] 点云配准 + IMU 预积分 → Hand-eye...")

        # 限制扫描对数
        n_use = min(len(data.scans), args.max_scan_pairs + 1)
        step = max(1, len(data.scans) // n_use)
        selected_idx = list(range(0, len(data.scans), step))[:n_use]
        selected_scans = [data.scans[i] for i in selected_idx]
        selected_stamps = data.lidar_stamps[selected_idx]

        print(f"  使用 {len(selected_scans)} 帧扫描 (步长 {step})")

        # 点云配准
        lidar_rel_poses = compute_scan_relative_poses(
            selected_scans,
            voxel_size=args.voxel_size,
            icp_dist=args.icp_dist,
        )

        # IMU 预积分 (使用外部 IMU)
        print("  计算 IMU 预积分...")
        imu_preint = compute_imu_preintegration(
            data.ext_imu_t, data.ext_imu_acc, data.ext_imu_gyro,
            selected_stamps,
        )

        # Hand-eye 标定
        if len(lidar_rel_poses) >= 3:
            lidar_pairs = [(r, t, f) for r, t, f in lidar_rel_poses if f > 0.3]
            imu_pairs = [(R, dv, dp) for (R, dv, dp) in imu_preint[:len(lidar_pairs)]]
            good_lidar = [(r, t) for r, t, f in lidar_pairs]
            good_imu = [(R, dp) for R, dv, dp in imu_pairs]

            if len(good_lidar) >= 3:
                R_he, t_he = hand_eye_calibration(
                    good_lidar, good_imu, R_phase1, t_phase2,
                )
                rv_he = mat_to_rotvec(R_he)
                print(f"  Hand-eye 估计 rotvec: [{rv_he[0]:.5f}, {rv_he[1]:.5f}, {rv_he[2]:.5f}]")
                print(f"  Hand-eye 估计 t: [{t_he[0]:.5f}, {t_he[1]:.5f}, {t_he[2]:.5f}] m")

                # 融合
                angle_diff_he = np.degrees(np.linalg.norm(mat_to_rotvec(R_he) - mat_to_rotvec(R_phase1)))
                if angle_diff_he < 10:
                    R_phase1 = R_he
                    t_phase2 = t_he
                else:
                    print(f"  ⚠ Hand-eye 与前序估计差异 {angle_diff_he:.1f}°, 保留前序结果")
    else:
        if args.skip_icp:
            print("\n[Phase 3] 跳过点云配准 (--skip-icp)")
        else:
            print("\n[Phase 3] 点云帧不足，跳过配准")

    # ─── Phase 4: 联合优化 ───
    print("\n[Phase 4] 联合非线性优化...")
    R_opt, t_opt, opt_info = joint_optimize(
        R_phase1, t_phase2,
        sync_t, ext_acc, ext_gyro, mid_acc, mid_gyro,
        lidar_rel_poses, imu_preint, T_LI,
    )

    # ─── Phase 5: 置信度 ───
    print("\n[Phase 5] 置信度评估...")
    conf = {}
    if opt_info["jacobian"] is not None and opt_info["residuals"] is not None:
        conf = compute_confidence(opt_info["jacobian"], opt_info["residuals"])
        print_confidence(conf, R_opt, t_opt)

    # ─── 输出结果 ───
    q_opt = mat_to_quat(R_opt)
    print("\n" + "=" * 60)
    print("  最终标定结果: T_lidar_imu")
    print("=" * 60)
    print(f"\n  平移 (m):   [{t_opt[0]:.6f}, {t_opt[1]:.6f}, {t_opt[2]:.6f}]")
    print(f"  四元数 (xyzw): [{q_opt[0]:.6f}, {q_opt[1]:.6f}, {q_opt[2]:.6f}, {q_opt[3]:.6f}]")
    print(f"\n  旋转矩阵:")
    for row in R_opt:
        print(f"    [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}]")

    # 输出适合直接粘贴到配置的格式
    print(f"\n  params_sensors.yaml 格式:")
    print(f"  T_lidar_imu: [{t_opt[0]:.5f}, {t_opt[1]:.5f}, {t_opt[2]:.5f}, "
          f"{q_opt[0]:.5f}, {q_opt[1]:.5f}, {q_opt[2]:.5f}, {q_opt[3]:.5f}]")

    # 与初始值比较
    q_init_orig = mat_to_quat(R_init)
    angle_final = np.degrees(np.linalg.norm(
        mat_to_rotvec(R_opt) - mat_to_rotvec(R_init)))
    trans_final = np.linalg.norm(t_opt - t_init) * 1000
    print(f"\n  与初始值比较:")
    print(f"    旋转差: {angle_final:.3f}°")
    print(f"    平移差: {trans_final:.2f} mm")

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.1f}s")

    # ─── 保存结果 ───
    out_path = args.output
    if not out_path:
        from pathlib import Path
        data_path = Path(args.data)
        out_path = str(data_path.with_name(data_path.stem + "_result.npz"))

    save_dict = {
        "T_lidar_imu_t": t_opt,
        "T_lidar_imu_q": q_opt,
        "T_lidar_imu_R": R_opt,
        "T_init_t": t_init,
        "T_init_q": q_init_orig,
    }
    if conf:
        save_dict["information_matrix"] = conf["information_matrix"]
        save_dict["covariance"] = conf["covariance"]
        save_dict["eigenvalues_info"] = conf["eigenvalues_info"]
        save_dict["eigenvectors_info"] = conf["eigenvectors_info"]
        save_dict["std_devs"] = conf["std_devs"]
        save_dict["condition_number"] = np.array(conf["condition_number"])

    np.savez(out_path, **save_dict)
    print(f"\n  结果已保存: {out_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
