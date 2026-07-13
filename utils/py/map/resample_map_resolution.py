#!/usr/bin/env python3
"""
Msgpack 地图分辨率重采样工具

读取一个已有的 msgpack 地图，保持物理尺寸不变，
通过最近邻插值调整到新的分辨率。

用法:
  uv run resample_map_resolution.py <输入.msgpack> <新分辨率> [输出.msgpack]

示例:
  uv run resample_map_resolution.py RMUC2026.msgpack 0.05  # → RMUC2026_0.05.msgpack
  uv run resample_map_resolution.py test.msgpack 0.02 output.msgpack
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "msgpack",
#     "numpy",
# ]
# ///

from __future__ import annotations

import os
import sys
from pathlib import Path

import msgpack
import numpy as np


def nearest_neighbor_resize(
    arr: np.ndarray,
    new_shape: tuple[int, int],
) -> np.ndarray:
    """最近邻插值缩放 2D uint8 数组。"""
    h_old, w_old = arr.shape
    h_new, w_new = new_shape

    # 新图每个像素中心映射回原图的浮点坐标
    # 像素中心对齐：旧/新图的 (0,0) 像素中心均在 world 原点
    xs = (np.arange(w_new, dtype=np.float64) + 0.5) * w_old / w_new - 0.5
    ys = (np.arange(h_new, dtype=np.float64) + 0.5) * h_old / h_new - 0.5

    # 四舍五入取最近邻，并钳位到有效范围
    xi = np.round(xs).astype(np.intp).clip(0, w_old - 1)
    yi = np.round(ys).astype(np.intp).clip(0, h_old - 1)

    return arr[np.ix_(yi, xi)]


def resample_map(
    in_path: str,
    new_resolution: float,
    out_path: str | None = None,
) -> str:
    """读取 msgpack，重采样到 new_resolution，保存并返回输出路径。"""
    # ── 读取 ────────────────────────────────────────────────
    with open(in_path, "rb") as f:
        data = msgpack.unpackb(f.read())

    old_resolution = data["resolution"]
    if abs(old_resolution - new_resolution) < 1e-9:
        print("新分辨率与原始分辨率相同，无需处理。")
        return in_path

    w_old: int = data["width"]
    h_old: int = data["height"]
    terrain_old = np.frombuffer(data["terrain"], dtype=np.uint8).reshape(h_old, w_old)
    direction_old = np.frombuffer(data["direction"], dtype=np.uint8).reshape(h_old, w_old)

    # ── 计算新尺寸（保持物理大小） ──────────────────────────
    # physical_width  = w_old * old_resolution
    #                 = w_new * new_resolution
    scale = old_resolution / new_resolution
    w_new = int(round(w_old * scale))
    h_new = int(round(h_old * scale))
    # 确保至少 1px
    w_new = max(1, w_new)
    h_new = max(1, h_new)

    phys_w_old = w_old * old_resolution
    phys_h_old = h_old * old_resolution
    phys_w_new = w_new * new_resolution
    phys_h_new = h_new * new_resolution

    print(f"  输入: {w_old}×{h_old}  @ {old_resolution:.6f} m/px  ({phys_w_old:.3f}×{phys_h_old:.3f} m)")
    print(f"  输出: {w_new}×{h_new}  @ {new_resolution:.6f} m/px  ({phys_w_new:.3f}×{phys_h_new:.3f} m)")
    print(f"  尺寸变化: {scale:.4f}×")

    # ── 最近邻插值 ──────────────────────────────────────────
    terrain_new = nearest_neighbor_resize(terrain_old, (h_new, w_new))
    direction_new = nearest_neighbor_resize(direction_old, (h_new, w_new))

    # ── 保存 ────────────────────────────────────────────────
    if out_path is None:
        stem = Path(in_path).stem
        out_path = f"{stem}_{new_resolution:.6f}.msgpack".rstrip("0").rstrip(".") + ".msgpack"
        # 上面的 rstripping 是为了避免 "0.050000" 这样的文件名
        # 更简单：直接用字符串格式化
        out_path = f"{stem}_{new_resolution}.msgpack"

    packed = msgpack.packb({
        "width": w_new,
        "height": h_new,
        "resolution": new_resolution,
        "terrain": terrain_new.tobytes(),
        "direction": direction_new.tobytes(),
    })
    with open(out_path, "wb") as f:
        f.write(packed)

    print(f"  已保存: {out_path}")
    return out_path


def main() -> None:
    if len(sys.argv) < 3:
        print("用法: uv run resample_map_resolution.py <输入.msgpack> <新分辨率> [输出.msgpack]")
        print("示例: uv run resample_map_resolution.py RMUC2026.msgpack 0.05")
        sys.exit(1)

    in_path = sys.argv[1]
    if not os.path.isfile(in_path):
        print(f"错误: 文件不存在 — {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        new_resolution = float(sys.argv[2])
    except ValueError:
        print(f"错误: 分辨率必须是数字 — {sys.argv[2]}", file=sys.stderr)
        sys.exit(1)

    if new_resolution <= 0:
        print("错误: 分辨率必须为正数", file=sys.stderr)
        sys.exit(1)

    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    resample_map(in_path, new_resolution, out_path)


if __name__ == "__main__":
    main()
