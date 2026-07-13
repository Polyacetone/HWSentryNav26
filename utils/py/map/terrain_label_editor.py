#!/usr/bin/env python3
"""
Msgpack 地形语义标注工具

两通道 msgpack 地图格式:
  - terrain:  uint8[H, W], 0~6 语义标签
  - direction: uint8[H, W], 0~255 → 0~360°

语义标签:
  0 = 平地 (无方向)
  1 = 障碍物 (无方向)
  2 = 斜坡 (有方向, 沿斜坡向上)
  3 = 一级台阶 (有方向, 沿台阶向上)
  4 = 二级台阶 (有方向, 沿台阶向上)
  5 = 飞坡 (有方向, 飞坡飞出正方向)
  6 = 高台阶 (有方向, 沿台阶向上)

方向标注: 画线 → 左侧垂直向外为方向向量 → 编码为角度 0~255

快捷键:
  0-6       选择语义标签
  O         打开 msgpack
  S         保存
  Ctrl+Z    撤销
"""

from __future__ import annotations

import math
import os
import tkinter as tk
from collections.abc import Iterable
from tkinter import filedialog, messagebox, ttk, simpledialog
from typing import Optional

import msgpack
import numpy as np
from PIL import Image, ImageTk

try:
    import open3d as o3d

    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False

# ── 地形类型定义 ──────────────────────────────────────────────

TERRAIN_FLAT = 0
TERRAIN_OBSTACLE = 1
TERRAIN_SLOPE = 2
TERRAIN_STEP_L1 = 3
TERRAIN_STEP_L2 = 4
TERRAIN_FLY_SLOPE = 5
TERRAIN_STEP_HIGH = 6

TERRAIN_NAMES: dict[int, str] = {
    0: "平地",
    1: "障碍物",
    2: "斜坡",
    3: "一级台阶",
    4: "二级台阶",
    5: "飞坡",
    6: "高台阶",
}

DIRECTIONAL_LABELS = {TERRAIN_SLOPE, TERRAIN_STEP_L1, TERRAIN_STEP_L2, TERRAIN_FLY_SLOPE, TERRAIN_STEP_HIGH}

TERRAIN_COLORS: dict[int, tuple[int, int, int]] = {
    0: (76, 175, 80),
    1: (244, 67, 54),
    2: (255, 152, 0),
    3: (255, 235, 59),
    4: (156, 39, 176),
    5: (0, 188, 212),
    6: (0, 242, 255),
}

BG_COLOR = (30, 30, 30)
ELEVATION_OPACITY = 0.40
ARROW_SPACING_PX = 15        # 采样间距（屏幕像素）
ARROW_LENGTH_PX = 20          # 箭头杆长度（屏幕像素，固定值不随缩放变化）
ARROW_HEAD_ANGLE = math.radians(25)
ARROW_HEAD_LENGTH_RATIO = 0.35
MAX_ARROWS = 50000


def direction_angle_from_line(x1: int, y1: int, x2: int, y2: int) -> int:
    """画线 (x1,y1)→(x2,y2)，返回左侧垂直向外方向的 msgpack 编码值 (0~255)。"""
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0
    nx = -dy / length
    ny = dx / length
    angle = math.atan2(ny, nx)
    if angle < 0:
        angle += 2.0 * math.pi
    val = int(round(angle / (2.0 * math.pi) * 255.0))
    return max(0, min(255, val))


def angle_to_display_radians(dir_val: int) -> float:
    """将 msgpack 方向值转为显示坐标系弧度 (y-up)。"""
    angle_img = dir_val / 255.0 * 2.0 * math.pi
    return -angle_img


def terrain_color_hex(label: int) -> str:
    r, g, b = TERRAIN_COLORS.get(label, (200, 200, 200))
    return f"#{r:02x}{g:02x}{b:02x}"


def terrain_color_light_hex(label: int, factor: float = 0.45) -> str:
    r, g, b = TERRAIN_COLORS.get(label, (200, 200, 200))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── 高程亮度叠加 ──────────────────────────────────────────────


def elevation_to_grayscale(elev: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """高程归一化为灰度图，仅用亮度表示高低，不引入色调。"""
    if not np.any(valid_mask):
        return np.zeros((*elev.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(elev[valid_mask], [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    norm = np.clip((elev - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[~valid_mask] = 0
    return rgb


# ── 主应用 ────────────────────────────────────────────────────


class TerrainLabelEditor:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("地形语义标注工具 — msgpack")
        self.root.geometry("1920x1080")

        # ─ 数据 ─
        self.terrain: Optional[np.ndarray] = None
        self.direction: Optional[np.ndarray] = None
        self.elevation: Optional[np.ndarray] = None
        self.elev_valid: Optional[np.ndarray] = None
        self.map_path: Optional[str] = None
        self.resolution: float = 0.05

        # ─ 状态 ─
        self.mode = "pixel"
        self.current_label = TERRAIN_FLAT
        self.current_dir_value = tk.IntVar(value=64)  # 默认北向
        self.line_width = 3
        self.scale = 1.0
        self.show_elevation = True
        self.show_arrows = True
        self._last_directional_label = TERRAIN_SLOPE
        self._pixel_dirty = False
        self._lazy_refresh_id: Optional[str] = None

        self.history: list[tuple[np.ndarray, np.ndarray]] = []
        self.start_x: Optional[int] = None
        self.start_y: Optional[int] = None
        self.temp_draw_id: Optional[int] = None
        self._temp_preview: list[int] = []
        self._brush_preview_id: Optional[int] = None
        self._hover_xy: Optional[tuple[int, int]] = None
        self._last_paint_xy: Optional[tuple[int, int]] = None

        self._render_cache: Optional[ImageTk.PhotoImage] = None
        self._arrow_items: list[int] = []

        self._build_ui()
        self._update_mode_ui()

    # ── UI 构建 ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="新建", command=self.new_map).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="打开", command=self.open_map).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="保存", command=self.save_map).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="另存为", command=self.save_map_as).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Frame(toolbar, width=6).pack(side=tk.LEFT)

        tk.Button(toolbar, text="撤销", command=self.undo).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Frame(toolbar, width=6).pack(side=tk.LEFT)

        tk.Button(toolbar, text="Z+", width=3, command=self.zoom_in).pack(side=tk.LEFT, padx=1, pady=2)
        tk.Button(toolbar, text="Z-", width=3, command=self.zoom_out).pack(side=tk.LEFT, padx=1, pady=2)
        tk.Frame(toolbar, width=10).pack(side=tk.LEFT)

        self.mode_var = tk.StringVar(value="pixel")
        for text, val in [("像素", "pixel"), ("矩形", "rect"), ("画线", "line")]:
            tk.Radiobutton(toolbar, text=text, variable=self.mode_var,
                           value=val, command=self._on_mode_change).pack(side=tk.LEFT, padx=2)

        tk.Frame(toolbar, width=6).pack(side=tk.LEFT)

        tk.Label(toolbar, text="笔刷大小:").pack(side=tk.LEFT, padx=1)
        self.spin_width = tk.Spinbox(toolbar, from_=1, to=30, width=3,
                                     command=self._update_line_width)
        self.spin_width.pack(side=tk.LEFT, padx=2)
        self.spin_width.delete(0, "end")
        self.spin_width.insert(0, "3")
        self.spin_width.bind("<KeyRelease>", lambda _: self._update_line_width())
        self.spin_width.bind("<FocusOut>", lambda _: self._update_line_width())

        tk.Frame(toolbar, width=10).pack(side=tk.LEFT)

        tk.Label(toolbar, text="标签:").pack(side=tk.LEFT, padx=1)
        self.label_var = tk.StringVar(value=TERRAIN_NAMES[0])
        self.label_combo = ttk.Combobox(toolbar, textvariable=self.label_var,
                                        values=list(TERRAIN_NAMES.values()),
                                        state="readonly", width=10)
        self.label_combo.pack(side=tk.LEFT, padx=2)
        self.label_combo.bind("<<ComboboxSelected>>", self._on_label_change)

        self.color_preview = tk.Canvas(toolbar, width=22, height=22,
                                       highlightthickness=1, highlightbackground="#888")
        self.color_preview.pack(side=tk.LEFT, padx=4)

        # ─ 方向滑块 (仅方向标签可用) ─
        self.dir_slider_label = tk.Label(toolbar, text="方向:", fg="#888")
        self.dir_slider_label.pack(side=tk.LEFT, padx=(10, 0))
        self.dir_slider = tk.Scale(toolbar, from_=0, to=255, orient=tk.HORIZONTAL,
                                   variable=self.current_dir_value, showvalue=False,
                                   length=120, sliderlength=16)
        self.dir_slider.pack(side=tk.LEFT, padx=2)
        self.dir_value_label = tk.Label(toolbar, text="90° (64)", width=10, anchor="w")
        self.dir_value_label.pack(side=tk.LEFT, padx=1)
        self.current_dir_value.trace_add("write", self._on_dir_slider_change)

        tk.Frame(toolbar, width=10).pack(side=tk.LEFT)

        self.elev_var = tk.BooleanVar(value=True)
        tk.Checkbutton(toolbar, text="叠加高程", variable=self.elev_var,
                       command=self._on_toggle_elevation).pack(side=tk.LEFT, padx=2)
        self.arrow_var = tk.BooleanVar(value=True)
        tk.Checkbutton(toolbar, text="方向箭头", variable=self.arrow_var,
                       command=self._on_toggle_arrows).pack(side=tk.LEFT, padx=2)

        tk.Frame(toolbar, width=6).pack(side=tk.LEFT)
        tk.Button(toolbar, text="加载 PCD", command=self.load_pcd).pack(side=tk.LEFT, padx=2, pady=2)

        # ─ 画布 ─
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.canvas = tk.Canvas(canvas_frame, bg="#1a1a1a",
                                yscrollcommand=self.v_scroll.set,
                                xscrollcommand=self.h_scroll.set)
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ─ 状态栏 ─
        status_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_coord = tk.Label(status_bar, text="X:-, Y:-", anchor="w", width=18)
        self.status_coord.pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(status_bar, text="标签: -, 方向: -, 高程: -", anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=15)
        self.status_info = tk.Label(status_bar, text="就绪", anchor="w", fg="#42a5f5")
        self.status_info.pack(side=tk.RIGHT, padx=15)

        # ─ 事件绑定 ─
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        self.root.bind("<Control-z>", lambda _: self.undo())
        self.root.bind("<Key>", self._on_key)

        self._update_color_preview()

    # ── 模式 / 参数更新 ──────────────────────────────────────

    def _on_mode_change(self) -> None:
        self.mode = self.mode_var.get()
        self.spin_width.config(state=tk.NORMAL if self.mode in ("pixel", "line") else tk.DISABLED)
        self._update_brush_preview()
        self._update_mode_ui()

    def _update_mode_ui(self) -> None:
        if self.terrain is None:
            return
        mode_names = {"pixel": "像素", "rect": "矩形", "line": "画线"}
        label_name = TERRAIN_NAMES.get(self.current_label, "?")
        self.status_info.config(text=f"模式: {mode_names.get(self.mode, self.mode)} | 标签: {label_name}")

    def _update_line_width(self) -> None:
        try:
            self.line_width = min(30, max(1, int(self.spin_width.get())))
        except ValueError:
            return
        self._update_brush_preview()

    def _on_label_change(self, _event=None) -> None:
        selected_name = self.label_var.get()
        for label, name in TERRAIN_NAMES.items():
            if name == selected_name:
                self.current_label = label
                break
        if self.current_label in DIRECTIONAL_LABELS:
            self._last_directional_label = self.current_label
        self._update_color_preview()
        self._update_dir_slider_state()
        self._update_brush_preview()
        self._update_mode_ui()

    def _on_dir_slider_change(self, _name=None, _index=None, _mode=None) -> None:
        v = self.current_dir_value.get()
        angle = v / 255.0 * 360.0
        self.dir_value_label.config(text=f"{angle:.0f}° ({v})")

    def _update_dir_slider_state(self) -> None:
        if self.current_label in DIRECTIONAL_LABELS:
            self.dir_slider_label.config(fg="#000")
            self.dir_slider.config(state=tk.NORMAL)
            self.dir_value_label.config(fg="#000")
        else:
            self.dir_slider_label.config(fg="#888")
            self.dir_slider.config(state=tk.DISABLED)
            self.dir_value_label.config(fg="#888")

    def _update_color_preview(self) -> None:
        color = TERRAIN_COLORS.get(self.current_label, (128, 128, 128))
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        self.color_preview.config(bg=hex_color)

    def _on_key(self, event: tk.Event) -> None:
        if event.char in ("0", "1", "2", "3", "4", "5", "6"):
            label = int(event.char)
            if label in TERRAIN_NAMES:
                self.current_label = label
                if label in DIRECTIONAL_LABELS:
                    self._last_directional_label = label
                self.label_var.set(TERRAIN_NAMES[label])
                self._update_color_preview()
                self._update_dir_slider_state()
                self._update_brush_preview()
                self._update_mode_ui()
        elif event.char in ("o", "O"):
            self.open_map()
        elif event.char in ("s", "S"):
            self.save_map()

    # ── 显示控制 ─────────────────────────────────────────────

    def _on_toggle_elevation(self) -> None:
        self.show_elevation = self.elev_var.get()
        self._refresh_display()

    def _on_toggle_arrows(self) -> None:
        self.show_arrows = self.arrow_var.get()
        self._refresh_display()

    # ── 文件操作 ─────────────────────────────────────────────

    def new_map(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("新建地图")
        dialog.geometry("320x200")
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="宽度 (像素):").grid(row=0, column=0, sticky="w", pady=6)
        w_entry = ttk.Entry(frame, width=12)
        w_entry.insert(0, "200")
        w_entry.grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(frame, text="高度 (像素):").grid(row=1, column=0, sticky="w", pady=6)
        h_entry = ttk.Entry(frame, width=12)
        h_entry.insert(0, "200")
        h_entry.grid(row=1, column=1, sticky="w", padx=10)

        ttk.Label(frame, text="分辨率 (m/px):").grid(row=2, column=0, sticky="w", pady=6)
        r_entry = ttk.Entry(frame, width=12)
        r_entry.insert(0, "0.05")
        r_entry.grid(row=2, column=1, sticky="w", padx=10)

        def do_create() -> None:
            try:
                w = int(w_entry.get())
                h = int(h_entry.get())
                res = float(r_entry.get())
                if w <= 0 or h <= 0 or res <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("参数错误", "请检查输入参数")
                return
            self.map_path = None
            self.resolution = res
            self.terrain = np.zeros((h, w), dtype=np.uint8)
            self.direction = np.zeros((h, w), dtype=np.uint8)
            self.elevation = None
            self.elev_valid = None
            self.history.clear()
            self.scale = 1.0
            self._refresh_display()
            self.status_info.config(text=f"新建空白地图: {w}×{h}, {res}m/px")
            dialog.destroy()

        ttk.Button(frame, text="创建", command=do_create).grid(row=3, column=0, columnspan=2, pady=16)

    def open_map(self) -> None:
        path = filedialog.askopenfilename(
            title="打开 msgpack 地图",
            filetypes=[("Msgpack 地图", "*.msgpack"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self._load_msgpack(path)

    def save_map(self) -> None:
        if self.terrain is None:
            return
        if self.map_path:
            self._save_msgpack(self.map_path)
        else:
            self.save_map_as()

    def save_map_as(self) -> None:
        if self.terrain is None:
            return
        path = filedialog.asksaveasfilename(
            title="保存 msgpack 地图",
            defaultextension=".msgpack",
            filetypes=[("Msgpack 地图", "*.msgpack"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self._save_msgpack(path)

    def _load_msgpack(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                data = msgpack.unpackb(f.read())
        except Exception as e:
            messagebox.showerror("打开失败", f"无法读取文件:\n{e}")
            return

        try:
            w = data["width"]
            h = data["height"]
            terrain = np.frombuffer(data["terrain"], dtype=np.uint8).reshape(h, w).copy()
            direction = np.frombuffer(data["direction"], dtype=np.uint8).reshape(h, w).copy()
        except (KeyError, ValueError, TypeError) as e:
            messagebox.showerror("格式错误", f"msgpack 数据字段不完整:\n{e}")
            return

        self.map_path = path
        self.resolution = data.get("resolution", 0.05)
        self.terrain = terrain
        self.direction = direction
        self.elevation = None
        self.elev_valid = None
        self.history.clear()
        self.scale = 1.0
        self._refresh_display()
        self.status_info.config(text=f"已加载: {os.path.basename(path)} ({w}×{h})")

    def _save_msgpack(self, path: str) -> None:
        assert self.terrain is not None
        assert self.direction is not None
        try:
            packed = msgpack.packb({
                "width": self.terrain.shape[1],
                "height": self.terrain.shape[0],
                "resolution": self.resolution,
                "terrain": self.terrain.tobytes(),
                "direction": self.direction.tobytes(),
            })
            with open(path, "wb") as f:
                f.write(packed)
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            return

        self.map_path = path
        self.status_info.config(text=f"已保存: {os.path.basename(path)}")

    # ── PCD 加载 ─────────────────────────────────────────────

    def load_pcd(self) -> None:
        if not _HAS_OPEN3D:
            messagebox.showerror("缺少依赖", "需要安装 open3d:\n  pip install open3d")
            return

        path = filedialog.askopenfilename(
            title="加载 PCD 点云",
            filetypes=[("PCD 文件", "*.pcd"), ("所有文件", "*.*")],
        )
        if not path:
            return

        try:
            pcd = o3d.io.read_point_cloud(path)
            points = np.asarray(pcd.points, dtype=np.float64)
            if points.shape[0] == 0:
                messagebox.showerror("错误", "PCD 点云为空")
                return
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载 PCD:\n{e}")
            return

        if self.terrain is None:
            messagebox.showwarning("提示", "请先新建或打开地图后再加载 PCD")
            return

        grid_x = (points[:, 0] / self.resolution).astype(np.int32)
        grid_y = (points[:, 1] / self.resolution).astype(np.int32)
        h, w = self.terrain.shape

        in_bounds = (grid_x >= 0) & (grid_x < w) & (grid_y >= 0) & (grid_y < h)
        if not np.any(in_bounds):
            messagebox.showwarning("提示", "点云不在当前地图范围内")
            return

        cell_id = grid_y.astype(np.int64) * w + grid_x.astype(np.int64)
        flat_size = h * w

        cnt = np.bincount(cell_id[in_bounds], minlength=flat_size).astype(np.float32)
        sum_z = np.bincount(cell_id[in_bounds], weights=points[in_bounds, 2], minlength=flat_size).astype(np.float32)

        self.elevation = np.zeros((h, w), dtype=np.float32)
        self.elev_valid = cnt.reshape(h, w) > 0
        m = cnt > 0
        self.elevation.flat[m] = sum_z[m] / cnt[m]

        self.status_info.config(
            text=f"已加载 PCD: {os.path.basename(path)}, {points.shape[0]} 点"
        )
        self._refresh_display()

    # ── 显示渲染 ─────────────────────────────────────────────

    def _refresh_display(self) -> None:
        if self.terrain is None:
            self.canvas.delete("all")
            return

        rgb = self._render_base_image()
        img = Image.fromarray(rgb)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        h, w = self.terrain.shape
        sw = max(1, int(w * self.scale))
        sh = max(1, int(h * self.scale))
        img = img.resize((sw, sh), Image.NEAREST)
        self._render_cache = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.temp_draw_id = None
        self._temp_preview.clear()
        self._brush_preview_id = None
        self.canvas.create_image(0, 0, image=self._render_cache, anchor=tk.NW)
        self.canvas.config(scrollregion=(0, 0, sw, sh))

        self._draw_arrows()
        self._update_brush_preview()

    def _render_base_image(self) -> np.ndarray:
        assert self.terrain is not None
        h, w = self.terrain.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for label, color in TERRAIN_COLORS.items():
            mask = self.terrain == label
            rgb[mask] = color

        empty = (self.terrain == 0) & (rgb == 0).all(axis=2)
        rgb[empty] = BG_COLOR

        if self.show_elevation and self.elevation is not None and self.elev_valid is not None:
            elev_gray = elevation_to_grayscale(self.elevation, self.elev_valid)
            rgb = ((1.0 - ELEVATION_OPACITY) * rgb.astype(np.float32)
                   + ELEVATION_OPACITY * elev_gray.astype(np.float32)).astype(np.uint8)

        return rgb

    def _draw_arrows(self) -> None:
        for item_id in self._arrow_items:
            self.canvas.delete(item_id)
        self._arrow_items.clear()

        if not self.show_arrows or self.terrain is None:
            return

        h, w = self.terrain.shape
        arrow_len = ARROW_LENGTH_PX
        spacing = max(4, int(round(ARROW_SPACING_PX / max(self.scale, 0.1))))

        drawn = 0
        for y in range(0, h, spacing):
            for x in range(0, w, spacing):
                if drawn >= MAX_ARROWS:
                    break
                label = int(self.terrain[y, x])
                if label not in DIRECTIONAL_LABELS:
                    continue
                dir_val = int(self.direction[y, x])

                cx = (x + 0.5) * self.scale
                cy = ((h - 1 - y) + 0.5) * self.scale
                angle = angle_to_display_radians(dir_val)
                ex = cx + arrow_len * math.cos(angle)
                ey = cy + arrow_len * math.sin(angle)
                color = terrain_color_light_hex(label)
                lw = 1

                line_id = self.canvas.create_line(
                    cx, cy, ex, ey, fill=color, width=lw,
                )
                self._arrow_items.append(line_id)

                head_len = arrow_len * ARROW_HEAD_LENGTH_RATIO
                a1x = ex - head_len * math.cos(angle - ARROW_HEAD_ANGLE)
                a1y = ey - head_len * math.sin(angle - ARROW_HEAD_ANGLE)
                a2x = ex - head_len * math.cos(angle + ARROW_HEAD_ANGLE)
                a2y = ey - head_len * math.sin(angle + ARROW_HEAD_ANGLE)

                head_id = self.canvas.create_polygon(
                    ex, ey, a1x, a1y, a2x, a2y, fill=color, outline="",
                )
                self._arrow_items.append(head_id)
                drawn += 1
            if drawn >= MAX_ARROWS:
                break

    # ── 缩放 ─────────────────────────────────────────────────

    def zoom_in(self) -> None:
        self.scale = min(self.scale * 1.5, 20.0)
        self._refresh_display()

    def zoom_out(self) -> None:
        self.scale = max(self.scale / 1.5, 0.1)
        self._refresh_display()

    # ── 撤销 ─────────────────────────────────────────────────

    def undo(self) -> None:
        if not self.history:
            return
        self.terrain, self.direction = self.history.pop()
        self._refresh_display()
        self.status_info.config(text="已撤销")

    def _save_state(self) -> None:
        if self.terrain is not None:
            self.history.append((self.terrain.copy(), self.direction.copy()))
            if len(self.history) > 30:
                self.history.pop(0)

    # ── 坐标系转换 ───────────────────────────────────────────

    def _canvas_to_img_xy(self, event, *, clamp: bool = False) -> Optional[tuple[int, int]]:
        if self.terrain is None:
            return None
        h, w = self.terrain.shape
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        x = math.floor(cx / self.scale)
        y_disp = math.floor(cy / self.scale)
        y = (h - 1) - y_disp
        if clamp:
            return min(w - 1, max(0, x)), min(h - 1, max(0, y))
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x, y

    # ── 鼠标事件 ─────────────────────────────────────────────

    def _on_mouse_move(self, event: tk.Event) -> None:
        if self.terrain is None:
            self.status_coord.config(text="X:-, Y:-")
            self.status_label.config(text="标签: -, 方向: -, 高程: -")
            return

        xy = self._canvas_to_img_xy(event)
        if xy is None:
            self._hover_xy = None
            self._clear_brush_preview()
            self.status_coord.config(text="X:-, Y:-")
            self.status_label.config(text="标签: -, 方向: -, 高程: -")
            return

        x, y = xy
        self._hover_xy = xy
        self._update_brush_preview()
        self.status_coord.config(text=f"X:{x}, Y:{y}")

        label = int(self.terrain[y, x])
        dir_val = int(self.direction[y, x])
        angle_deg = dir_val / 255.0 * 360.0
        elev_str = f"{self.elevation[y, x]:.3f}" if (self.elevation is not None
                                                       and self.elev_valid is not None
                                                       and self.elev_valid[y, x]) else "-"
        self.status_label.config(
            text=f"标签: {TERRAIN_NAMES.get(label, '?')} | "
                 f"方向: {angle_deg:.0f}° ({dir_val}) | "
                 f"高程: {elev_str}"
        )

    def _on_mouse_leave(self, _event: tk.Event) -> None:
        self._hover_xy = None
        self._clear_brush_preview()
        self.status_coord.config(text="X:-, Y:-")
        self.status_label.config(text="标签: -, 方向: -, 高程: -")

    def _on_mouse_down(self, event: tk.Event) -> None:
        if self.terrain is None:
            return
        xy = self._canvas_to_img_xy(event)
        if xy is None:
            return

        self._save_state()
        self._pixel_dirty = False
        self._last_paint_xy = None
        x, y = xy
        self._hover_xy = xy
        self.canvas.grab_set()

        if self.mode == "pixel":
            self._paint_brush(x, y)
            self._last_paint_xy = xy
        else:
            self.start_x = x
            self.start_y = y
            sx = x * self.scale
            sy = (self.terrain.shape[0] - 1 - y) * self.scale

            if self.mode == "rect":
                self.temp_draw_id = self.canvas.create_rectangle(
                    sx, sy, sx, sy,
                    outline=terrain_color_hex(self.current_label), width=2, dash=(4, 2),
                )
            elif self.mode == "line":
                self._update_temp_line(x, y)
        self._update_brush_preview()

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self.terrain is None:
            return
        xy = self._canvas_to_img_xy(event, clamp=True)
        if xy is None:
            return
        x, y = xy
        self._hover_xy = xy

        if self.mode == "pixel":
            if self._last_paint_xy is None:
                self._paint_brush(x, y)
            else:
                self._paint_brush_stroke(*self._last_paint_xy, x, y)
            self._last_paint_xy = xy
        elif self.mode == "rect" and self.temp_draw_id is not None:
            self._update_temp_rect(x, y)
        elif self.mode == "line" and self.start_x is not None:
            self._update_temp_line(x, y)
        self._update_brush_preview()

    def _on_mouse_up(self, event: tk.Event) -> None:
        if self.terrain is None:
            return
        xy = self._canvas_to_img_xy(event, clamp=True)
        try:
            if xy is None:
                return
            x, y = xy

            if self.mode == "rect" and self.temp_draw_id is not None:
                self._apply_rect(x, y)
            elif self.mode == "line" and self.start_x is not None:
                self._apply_line(x, y)
            elif self.mode == "pixel" and self._pixel_dirty:
                self._schedule_lazy_refresh()
        finally:
            self._last_paint_xy = None
            if self.canvas.grab_current() == self.canvas:
                self.canvas.grab_release()

    # ── 标注操作 ─────────────────────────────────────────────

    def _paint_brush(self, x: int, y: int) -> None:
        self._paint_pixels(self._brush_footprint(x, y, self.line_width))

    def _paint_brush_stroke(self, x0: int, y0: int, x1: int, y1: int) -> None:
        pixels = {
            pixel
            for x, y in self._rasterize_line(x0, y0, x1, y1)
            for pixel in self._brush_footprint(x, y, self.line_width)
        }
        self._paint_pixels(pixels)

    def _paint_pixels(self, pixels: Iterable[tuple[int, int]]) -> None:
        assert self.terrain is not None and self.direction is not None
        h = self.terrain.shape[0]
        color = terrain_color_hex(self.current_label)
        final_dir = self.current_dir_value.get() if self.current_label in DIRECTIONAL_LABELS else 0
        painted = False
        for px, py in pixels:
            if not (0 <= py < h and 0 <= px < self.terrain.shape[1]):
                continue
            painted = True
            self.terrain[py, px] = self.current_label
            self.direction[py, px] = final_dir
            sx = px * self.scale
            sy = (h - 1 - py) * self.scale
            self.canvas.create_rectangle(
                sx, sy, sx + self.scale, sy + self.scale,
                outline="", fill=color,
            )
        self._pixel_dirty = self._pixel_dirty or painted

    def _clear_brush_preview(self) -> None:
        if self._brush_preview_id is not None:
            self.canvas.delete(self._brush_preview_id)
            self._brush_preview_id = None

    def _update_brush_preview(self) -> None:
        self._clear_brush_preview()
        if self.terrain is None or self._hover_xy is None or self.mode not in ("pixel", "line"):
            return

        x, y = self._hover_xy
        h = self.terrain.shape[0]
        w = self.terrain.shape[1]
        if not (0 <= x < w and 0 <= y < h):
            self._hover_xy = None
            return
        offset = -(self.line_width // 2)
        left = (x + offset) * self.scale
        right = (x + offset + self.line_width) * self.scale
        top = (h - y - offset - self.line_width) * self.scale
        bottom = (h - y - offset) * self.scale
        self._brush_preview_id = self.canvas.create_rectangle(
            left, top, right, bottom,
            outline=terrain_color_light_hex(self.current_label), width=2, dash=(4, 3),
        )

    def _schedule_lazy_refresh(self) -> None:
        if self._lazy_refresh_id is not None:
            self.root.after_cancel(self._lazy_refresh_id)
        self._lazy_refresh_id = self.root.after(200, self._do_lazy_refresh)

    def _do_lazy_refresh(self) -> None:
        self._lazy_refresh_id = None
        self._pixel_dirty = False
        self._refresh_display()

    def _update_temp_rect(self, x: int, y: int) -> None:
        assert self.start_x is not None and self.start_y is not None
        h = self.terrain.shape[0]
        x0, x1 = min(self.start_x, x), max(self.start_x, x)
        y0, y1 = min(self.start_y, y), max(self.start_y, y)
        left = x0 * self.scale
        right = (x1 + 1) * self.scale
        top = (h - 1 - y1) * self.scale
        bottom = (h - 1 - y0 + 1) * self.scale
        self.canvas.coords(self.temp_draw_id, left, top, right, bottom)

    def _apply_rect(self, x: int, y: int) -> None:
        assert self.terrain is not None and self.direction is not None
        assert self.start_x is not None and self.start_y is not None
        x0, x1 = min(self.start_x, x), max(self.start_x, x) + 1
        y0, y1 = min(self.start_y, y), max(self.start_y, y) + 1

        self.terrain[y0:y1, x0:x1] = self.current_label
        if self.current_label in DIRECTIONAL_LABELS:
            self.direction[y0:y1, x0:x1] = self.current_dir_value.get()
        else:
            self.direction[y0:y1, x0:x1] = 0

        self.canvas.delete(self.temp_draw_id)
        self.temp_draw_id = None
        self.start_x = self.start_y = None
        self._refresh_display()

    def _update_temp_line(self, x: int, y: int) -> None:
        assert self.start_x is not None and self.start_y is not None
        for rid in self._temp_preview:
            self.canvas.delete(rid)
        self._temp_preview.clear()

        h = self.terrain.shape[0]
        color = terrain_color_hex(self.current_label)

        pixels = self._line_footprint(
            self.start_x, self.start_y, x, y, self.line_width,
            self.terrain.shape[1], h,
        )
        for px, py in pixels:
            sx = px * self.scale
            sy = (h - 1 - py) * self.scale
            rid = self.canvas.create_rectangle(
                sx, sy, sx + self.scale, sy + self.scale,
                outline="", fill=color,
            )
            self._temp_preview.append(rid)

    def _apply_line(self, x: int, y: int) -> None:
        assert self.terrain is not None and self.direction is not None
        assert self.start_x is not None and self.start_y is not None

        for rid in self._temp_preview:
            self.canvas.delete(rid)
        self._temp_preview.clear()

        dir_val = direction_angle_from_line(self.start_x, self.start_y, x, y)
        label = self.current_label

        final_dir = dir_val if label in DIRECTIONAL_LABELS else 0
        pixels = self._line_footprint(
            self.start_x, self.start_y, x, y, self.line_width,
            self.terrain.shape[1], self.terrain.shape[0],
        )
        for px, py in pixels:
            self.terrain[py, px] = label
            self.direction[py, px] = final_dir

        if self.temp_draw_id is not None:
            self.canvas.delete(self.temp_draw_id)
            self.temp_draw_id = None
        self.start_x = self.start_y = None
        self._refresh_display()

    @staticmethod
    def _brush_footprint(x: int, y: int, width: int) -> list[tuple[int, int]]:
        """返回以 (x, y) 为落点、边长严格为 width 的方形笔刷像素。"""
        offset = -(width // 2)
        return [
            (x + dx, y + dy)
            for dy in range(offset, offset + width)
            for dx in range(offset, offset + width)
        ]

    @classmethod
    def _line_footprint(
        cls,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        width: int,
        map_width: int,
        map_height: int,
    ) -> list[tuple[int, int]]:
        """用方形笔刷沿中心线扫掠，起终点均保留完整笔刷。"""
        if (x1, y1) < (x0, y0):
            x0, y0, x1, y1 = x1, y1, x0, y0

        pixels = {
            pixel
            for x, y in cls._rasterize_line(x0, y0, x1, y1)
            for pixel in cls._brush_footprint(x, y, width)
            if 0 <= pixel[0] < map_width and 0 <= pixel[1] < map_height
        }
        return sorted(pixels, key=lambda pixel: (pixel[1], pixel[0]))

    @staticmethod
    def _rasterize_line(x0: int, y0: int, x1: int, y1: int):
        """Bresenham 直线栅格化。"""
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points


# ── 入口 ──────────────────────────────────────────────────────


def main() -> None:
    root = tk.Tk()
    TerrainLabelEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
