#!/usr/bin/env python3
"""交互式平面点云补充工具.

在已有场景点云的基础上可视化地"绘制"若干矩形平面点云
(墙、地面、坡道等), 可实时调整每个平面的大小、方向(欧拉角)、
位置、点密度和表面抖动, 支持撤销, 最终合并保存为 .pcd.

交互方式:
    - 打开已有点云后, 点击"添加平面"或用快捷键 N 绘制新平面
    - 在右侧面板中调整选中平面的参数, 场景实时更新
    - Ctrl + 点击场景中的平面: 拾取并选中该平面
    - Ctrl+Z: 撤销添加/复制/删除/清空操作
    - Delete: 删除选中平面
    - ←/→: 微调最近拖动过的位置轴
    - Ctrl+S: 保存合并点云

用法:
    ~/.venv/bin/python draw_planes_gui.py [cloud.pcd]

保存结果为纯几何点云(不含颜色), 便于后续建图/仿真使用.
Requires: open3d >= 0.18, numpy.
"""

import argparse
import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# 单平面最大点数, 防止密度/尺寸过大时卡死
MAX_PLANE_POINTS = 2_000_000
MAX_UNDO = 30


def _make_palette(count=500):
    colors = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        hue = (index * 0.618033988749895) % 1.0
        sector = int(hue * 6.0)
        fraction = hue * 6.0 - sector
        color_table = (
            (1.0, fraction, 0.0),
            (1.0 - fraction, 1.0, 0.0),
            (0.0, 1.0, fraction),
            (0.0, 1.0 - fraction, 1.0),
            (fraction, 0.0, 1.0),
            (1.0, 0.0, 1.0 - fraction),
        )
        colors[index] = np.asarray(color_table[sector % 6]) * 0.85 + 0.15
    return colors


PALETTE = _make_palette()
CLOUD_COLOR = np.array((0.55, 0.55, 0.55))


class PlaneParam:
    """单个平面参数: 中心位置 + 欧拉角方向 + 尺寸 + 密度 + 抖动."""

    def __init__(self, center=(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, roll=0.0,
                 width=4.0, length=4.0, density=10.0, jitter=0.005):
        self.center = np.asarray(center, dtype=np.float64).copy()
        self.yaw = float(yaw)          # deg, 绕 Z
        self.pitch = float(pitch)      # deg, 绕 Y
        self.roll = float(roll)        # deg, 绕 X
        self.width = float(width)      # 沿局部 X 的尺寸 (m)
        self.length = float(length)    # 沿局部 Y 的尺寸 (m)
        self.density = float(density)  # 每米点数 (pts/m)
        self.jitter = float(jitter)    # 沿法向的高斯抖动 (m)

    def copy(self):
        return PlaneParam(
            self.center, self.yaw, self.pitch, self.roll,
            self.width, self.length, self.density, self.jitter)


def _basis_from_euler(yaw_deg, pitch_deg, roll_deg):
    """由欧拉角 (ZYX 顺序) 计算平面局部坐标系的三个基向量.

    返回 (u, v, n): u 沿宽度方向, v 沿长度方向, n 为平面法向.
    """
    yaw, pitch, roll = np.radians((yaw_deg, pitch_deg, roll_deg))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    rot = rz @ ry @ rx
    return rot[:, 0], rot[:, 1], rot[:, 2]


def count_plane_points(param):
    """不实际生成点, 快速估算一个平面将产生的点数(含上限)."""
    nx = int(np.floor(param.width * param.density)) + 1
    ny = int(np.floor(param.length * param.density)) + 1
    return min(nx * ny, MAX_PLANE_POINTS)


def generate_plane_points(param, rng):
    """按参数生成矩形平面点云 (N x 3), 抖动沿法向叠加."""
    width = max(param.width, 1e-6)
    length = max(param.length, 1e-6)
    density = max(param.density, 1e-6)
    nx = int(np.floor(width * density)) + 1
    ny = int(np.floor(length * density)) + 1
    total = nx * ny
    if total > MAX_PLANE_POINTS:
        # 超上限时按比例稀释网格, 保持平面铺满
        factor = np.sqrt(MAX_PLANE_POINTS / total)
        nx = max(2, int(nx * factor))
        ny = max(2, int(ny * factor))
        total = nx * ny

    xs = (np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0) / density
    ys = (np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0) / density
    gx, gy = np.meshgrid(xs, ys)
    u, v, n = _basis_from_euler(param.yaw, param.pitch, param.roll)
    points = (param.center
              + np.outer(gx.ravel(), u)
              + np.outer(gy.ravel(), v))
    if param.jitter > 0.0:
        points = points + np.outer(rng.standard_normal(total), n) * param.jitter
    return points


class PlaneDrawerApp:
    """GUI: 在已有点云上交互式绘制平面点云."""

    def __init__(self, cloud_path=None):
        self.app = gui.Application.instance
        self.app.initialize()
        self._configure_chinese_font()

        self.window = self.app.create_window("点云补平面工具", 1440, 900)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.set_background([0.1, 0.1, 0.1, 1.0])
        self._scene.set_on_mouse(self._on_mouse)

        # ---- 状态 (需在 _build_ui 之前初始化) ----
        self.original_pcd = None
        self.planes = []          # list[PlaneParam]
        self.selected = -1        # 当前选中平面索引
        self.history = []         # 结构操作(增/删/复制/清空)的历史
        self._pick_pts = None     # 所有平面点 (N x 3), 用于鼠标拾取
        self._pick_ids = None     # 每个点所属平面索引
        self._last_pos_key = None # 最近拖动的平移轴, 用于方向键微调
        self._updating_ui = False
        self._pos_limits = ((-50.0, 50.0), (-50.0, 50.0), (-50.0, 50.0))
        self._rng = np.random.default_rng()
        self._sliders = {}
        self._slider_labels = {}
        self._formats = {}
        self._limits = {}  # 每个滑块的 (min, max), 用于回读/夹取

        self._panel = gui.Vert(0, gui.Margins(self.window.theme.font_size,
                                              self.window.theme.font_size,
                                              self.window.theme.font_size,
                                              self.window.theme.font_size))
        self._build_ui()
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        self.window.add_child(self._scene)
        self.window.add_child(self._panel)

        self._cloud_mat = rendering.MaterialRecord()
        self._cloud_mat.shader = "defaultUnlit"
        self._cloud_mat.point_size = 2.0
        self._plane_mat = rendering.MaterialRecord()
        self._plane_mat.shader = "defaultUnlit"
        self._plane_mat.point_size = 3.0
        self._frame_mat = rendering.MaterialRecord()
        self._frame_mat.shader = "unlitLine"
        self._frame_mat.line_width = 2.0

        if cloud_path:
            self._load_cloud(cloud_path)

    # ------------------------------------------------------------- UI

    def _configure_chinese_font(self):
        for path in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ):
            if os.path.exists(path):
                font = gui.FontDescription(path)
                font.add_typeface_for_language(path, "zh_all")
                gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)
                return

    def _build_ui(self):
        em = self.window.theme.font_size

        file_section = self._section("文件")
        load_button = gui.Button("加载点云")
        load_button.set_on_clicked(self._on_load)
        save_button = gui.Button("保存合并点云 (Ctrl+S)")
        save_button.set_on_clicked(self._on_save)
        self._lbl_info = gui.Label("就绪 - 请加载点云")
        self._lbl_info.text_color = gui.Color(0.8, 0.8, 0.8)
        file_section.add_child(load_button)
        file_section.add_child(gui.VGrid(1, 0.25 * em))
        file_section.add_child(save_button)
        file_section.add_child(gui.VGrid(1, 0.25 * em))
        file_section.add_child(self._lbl_info)

        param_section = self._section("平面参数")
        self._combo = gui.Combobox()
        self._combo.set_on_selection_changed(self._on_combo_changed)
        param_section.add_child(self._combo)

        self._lbl_selected = gui.Label("未选中平面")
        self._lbl_selected.text_color = gui.Color(1.0, 1.0, 0.5)
        param_section.add_child(gui.VGrid(1, 0.5 * em))
        param_section.add_child(self._lbl_selected)

        self._make_slider(param_section, "位置 X (m)", "cx", -50, 50, 0.0, ".3f")
        self._make_slider(param_section, "位置 Y (m)", "cy", -50, 50, 0.0, ".3f")
        self._make_slider(param_section, "位置 Z (m)", "cz", -50, 50, 0.0, ".3f")
        self._make_slider(param_section, "偏航角 (deg)", "yaw", -180, 180, 0.0, ".2f")
        self._make_slider(param_section, "俯仰角 (deg)", "pitch", -180, 180, 0.0, ".2f")
        self._make_slider(param_section, "横滚角 (deg)", "roll", -180, 180, 0.0, ".2f")
        self._make_slider(param_section, "宽度 (m)", "width", 0.1, 100, 4.0, ".2f")
        self._make_slider(param_section, "长度 (m)", "length", 0.1, 100, 4.0, ".2f")
        self._make_slider(param_section, "点密度 (pts/m)", "density", 1, 200, 10, "d", True)
        self._make_slider(param_section, "表面抖动 (m)", "jitter", 0.0, 0.1, 0.005, ".4f")

        edit_section = self._section("编辑")
        add_button = gui.Button("添加平面 (N)")
        add_button.set_on_clicked(self._on_add)
        duplicate_button = gui.Button("复制选中平面")
        duplicate_button.set_on_clicked(self._on_duplicate)
        delete_button = gui.Button("删除选中平面 (Delete)")
        delete_button.set_on_clicked(self._on_delete)
        clear_button = gui.Button("清空全部平面")
        clear_button.set_on_clicked(self._on_clear)
        undo_button = gui.Button("撤销 (Ctrl+Z)")
        undo_button.set_on_clicked(self._on_undo)
        for button in (add_button, duplicate_button, delete_button, clear_button, undo_button):
            edit_section.add_child(button)
            edit_section.add_child(gui.VGrid(1, 0.25 * em))

        hint_section = self._section("操作提示")
        hint_section.add_child(gui.Label("Ctrl + 点击: 选中场景中的平面"))
        hint_section.add_child(gui.Label("N: 添加平面   Delete: 删除"))
        hint_section.add_child(gui.Label("Ctrl+Z: 撤销   ←/→: 微调位置"))
        hint_section.add_child(gui.Label("俯仰 90° 可使平面竖直(墙)"))

        for section in (file_section, param_section, edit_section, hint_section):
            self._panel.add_child(section)

    def _section(self, title):
        section = gui.CollapsableVert(title, 0.25 * self.window.theme.font_size,
                                      gui.Margins(self.window.theme.font_size, 0, 0, 0))
        section.set_is_open(True)
        return section

    def _make_slider(self, parent, label, key, minimum, maximum, default, fmt, is_int=False):
        container = gui.Vert()
        slider = gui.Slider(gui.Slider.INT if is_int else gui.Slider.DOUBLE)
        slider.set_limits(float(minimum), float(maximum))
        slider.double_value = float(default)
        value_label = gui.Label(self._fmt_value(key, float(default)))

        def on_change(value, k=key):
            value_label.text = self._fmt_value(k, float(value))
            self._on_slider_changed(k, float(value))

        slider.set_on_value_changed(on_change)
        container.add_child(gui.Label(label))
        row = gui.Horiz(5)
        row.add_child(slider)
        row.add_child(value_label)
        container.add_child(row)
        parent.add_child(container)
        self._sliders[key] = slider
        self._slider_labels[key] = value_label
        self._limits[key] = (float(minimum), float(maximum))
        self._formats[key] = fmt
        return slider

    def _fmt_value(self, key, value):
        fmt = self._formats.get(key, ".3f")
        return str(int(round(value))) if fmt == "d" else fmt.format(value)

    def _set_slider_value(self, key, value):
        """程序化设置滑块值并同步数值标签.

        注意: 直接赋值 slider.double_value 不会触发 set_on_value_changed 回调,
        因此标签和业务状态都必须在这里手动更新.
        """
        value = float(np.clip(value, *self._slider_limits(key)))
        self._sliders[key].double_value = value
        self._slider_labels[key].text = self._fmt_value(key, value)
        return value

    def _on_layout(self, context):
        rect = self.window.content_rect
        sidebar = min(rect.width - 10, 22 * context.theme.font_size)
        self._scene.frame = gui.Rect(rect.x, rect.y, rect.width - sidebar, rect.height)
        self._panel.frame = gui.Rect(rect.x + rect.width - sidebar, rect.y, sidebar, rect.height)

    # ----------------------------------------------------- 加载 / 保存

    def _on_load(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, "选择点云文件", self.window.theme)
        dialog.add_filter(".pcd .ply .xyz .pts", "点云文件")
        dialog.set_on_cancel(self.window.close_dialog)
        dialog.set_on_done(self._on_load_done)
        self.window.show_dialog(dialog)

    def _on_load_done(self, path):
        self.window.close_dialog()
        self._load_cloud(path)

    def _load_cloud(self, path):
        try:
            cloud = o3d.io.read_point_cloud(path)
        except Exception as error:
            self._set_info(f"加载失败: {error}")
            return
        if not cloud.has_points():
            self._set_info("空点云")
            return

        self.original_pcd = cloud
        self.planes = []
        self.selected = -1
        self.history.clear()
        self._last_pos_key = None

        # 位置滑块范围跟随点云包围盒
        min_b = cloud.get_min_bound()
        max_b = cloud.get_max_bound()
        pad = (max_b - min_b) * 0.1
        self._pos_limits = tuple(
            (min_b[i] - pad[i], max_b[i] + pad[i]) for i in range(3))
        for key, limits in zip(("cx", "cy", "cz"), self._pos_limits):
            self._sliders[key].set_limits(*limits)

        scene = self._scene.scene
        scene.clear_geometry()
        scene.add_geometry("cloud", self._paint_cloud(), self._cloud_mat)
        bounds = scene.bounding_box
        self._scene.setup_camera(60.0, bounds, bounds.get_center())

        self._rebuild_combobox()
        self._refresh_scene()
        self._set_info(f"已加载 {os.path.basename(path)} ({len(cloud.points)} 点), 按 N 添加平面")

    def _paint_cloud(self):
        """原始点云统一显示为灰色, 与新增平面区分."""
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(self.original_pcd.points))
        cloud.paint_uniform_color(CLOUD_COLOR)
        return cloud

    def _on_save(self):
        if self.original_pcd is None and not self.planes:
            self._set_info("没有可保存的内容")
            return
        dialog = gui.FileDialog(gui.FileDialog.SAVE, "保存合并点云", self.window.theme)
        dialog.add_filter(".pcd", "PCD 文件")
        dialog.set_on_cancel(self.window.close_dialog)
        dialog.set_on_done(self._on_save_done)
        self.window.show_dialog(dialog)

    def _on_save_done(self, path):
        self.window.close_dialog()
        if not path.endswith(".pcd"):
            path += ".pcd"
        arrays = []
        if self.original_pcd is not None:
            arrays.append(np.asarray(self.original_pcd.points))
        for param in self.planes:
            arrays.append(generate_plane_points(param, self._rng))
        merged = np.concatenate(arrays, axis=0)
        # 与仓库其他编辑工具一致: 保存纯几何点云, 剥离颜色
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(merged)
        if o3d.io.write_point_cloud(path, cloud, write_ascii=False):
            self._set_info(f"已保存 {len(merged)} 点至 {os.path.basename(path)}")

    # ------------------------------------------------- 平面结构操作

    def _push_history(self):
        self.history.append([p.copy() for p in self.planes])
        if len(self.history) > MAX_UNDO:
            self.history.pop(0)

    def _default_center(self):
        if self.original_pcd is not None:
            return (self.original_pcd.get_max_bound() + self.original_pcd.get_min_bound()) / 2.0
        return np.zeros(3)

    def _on_add(self):
        self._push_history()
        self.planes.append(PlaneParam(center=self._default_center()))
        self.selected = len(self.planes) - 1
        self._after_structure_change()
        self._set_info(f"已添加平面 #{self.selected}")

    def _on_duplicate(self):
        if self.selected < 0 or not self.planes:
            return
        self._push_history()
        duplicate = self.planes[self.selected].copy()
        duplicate.center = duplicate.center + np.array((0.5, 0.5, 0.0))
        self.planes.append(duplicate)
        self.selected = len(self.planes) - 1
        self._after_structure_change()
        self._set_info(f"已复制为平面 #{self.selected}")

    def _on_delete(self):
        if self.selected < 0 or not self.planes:
            return
        self._push_history()
        del self.planes[self.selected]
        self.selected = min(self.selected, len(self.planes) - 1)
        self._after_structure_change()
        self._set_info("已删除平面")

    def _on_clear(self):
        if not self.planes:
            return
        self._push_history()
        self.planes = []
        self.selected = -1
        self._after_structure_change()
        self._set_info("已清空全部平面")

    def _on_undo(self):
        if not self.history:
            self._set_info("没有可撤销的操作")
            return
        self.planes = self.history.pop()
        self.selected = min(self.selected, len(self.planes) - 1)
        self._after_structure_change()
        self._set_info("已撤销")

    def _after_structure_change(self):
        self._rebuild_combobox()
        self._load_params_to_sliders()
        self._refresh_scene()

    # ------------------------------------------------- 参数编辑

    def _on_slider_changed(self, key, value):
        if self._updating_ui or self.selected < 0 or self.selected >= len(self.planes):
            return
        param = self.planes[self.selected]
        if key == "cx":
            param.center[0] = value
        elif key == "cy":
            param.center[1] = value
        elif key == "cz":
            param.center[2] = value
        elif key == "yaw":
            param.yaw = value
        elif key == "pitch":
            param.pitch = value
        elif key == "roll":
            param.roll = value
        elif key == "width":
            param.width = value
        elif key == "length":
            param.length = value
        elif key == "density":
            param.density = value
        elif key == "jitter":
            param.jitter = value
        if key in ("cx", "cy", "cz"):
            self._last_pos_key = key
        self._refresh_scene()

    def _on_combo_changed(self, text, index):
        if self._updating_ui:
            return
        self.selected = index
        self._load_params_to_sliders()
        self._refresh_scene()

    def _select_plane(self, index):
        if index < 0 or index >= len(self.planes):
            return
        self.selected = index
        self._updating_ui = True
        self._combo.selected_index = index
        self._updating_ui = False
        self._load_params_to_sliders()
        self._refresh_scene()

    def _load_params_to_sliders(self):
        if self.selected < 0 or self.selected >= len(self.planes):
            self._lbl_selected.text = "未选中平面"
            return
        param = self.planes[self.selected]
        self._updating_ui = True
        for key, value in {
            "cx": param.center[0], "cy": param.center[1], "cz": param.center[2],
            "yaw": param.yaw, "pitch": param.pitch, "roll": param.roll,
            "width": param.width, "length": param.length,
            "density": param.density, "jitter": param.jitter,
        }.items():
            self._set_slider_value(key, value)
        self._updating_ui = False
        self._lbl_selected.text = self._selected_label()

    def _slider_limits(self, key):
        if key in ("cx", "cy", "cz"):
            return self._pos_limits[("cx", "cy", "cz").index(key)]
        return self._limits[key]

    def _selected_label(self):
        param = self.planes[self.selected]
        count = count_plane_points(param)
        return (f"平面 #{self.selected} · 中心 "
                f"({param.center[0]:.2f}, {param.center[1]:.2f}, {param.center[2]:.2f}) "
                f"· {count} 点")

    # ------------------------------------------------- 场景刷新

    def _rebuild_combobox(self):
        self._updating_ui = True
        self._combo.clear_items()
        for index, param in enumerate(self.planes):
            self._combo.add_item(
                f"平面 #{index}  {param.width:.2f}x{param.length:.2f} m")
        if self.selected >= 0 and self.planes:
            self._combo.selected_index = min(self.selected, len(self.planes) - 1)
        self._updating_ui = False

    def _refresh_scene(self, reset_camera=False):
        self._update_plane_geometry()
        self._update_frame()
        self._update_info()
        if reset_camera:
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60.0, bounds, bounds.get_center())
        self._scene.force_redraw()

    def _update_plane_geometry(self):
        scene = self._scene.scene
        scene.remove_geometry("planes")
        self._pick_pts = None
        self._pick_ids = None
        if not self.planes:
            return
        arrays, colors, ids = [], [], []
        for index, param in enumerate(self.planes):
            points = generate_plane_points(param, self._rng)
            arrays.append(points)
            colors.append(np.tile(PALETTE[index % len(PALETTE)], (len(points), 1)))
            ids.append(np.full(len(points), index, dtype=np.int32))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.concatenate(arrays, axis=0))
        cloud.colors = o3d.utility.Vector3dVector(np.concatenate(colors, axis=0))
        scene.add_geometry("planes", cloud, self._plane_mat)
        self._pick_pts = np.concatenate(arrays, axis=0)
        self._pick_ids = np.concatenate(ids, axis=0)

    def _update_frame(self):
        scene = self._scene.scene
        scene.remove_geometry("frame")
        if self.selected < 0 or self.selected >= len(self.planes):
            return
        param = self.planes[self.selected]
        u, v, n = _basis_from_euler(param.yaw, param.pitch, param.roll)
        half = np.array((param.width, param.length)) / 2.0
        corners = [
            param.center - half[0] * u - half[1] * v,
            param.center + half[0] * u - half[1] * v,
            param.center + half[0] * u + half[1] * v,
            param.center - half[0] * u + half[1] * v,
        ]
        # 矩形边框 + 局部坐标轴 (x 绿, y 蓝, z 红)
        axis_len = max(0.4, 0.25 * max(param.width, param.length))
        points = corners + [
            param.center, param.center + axis_len * u,
            param.center + axis_len * v,
            param.center + axis_len * n,
        ]
        lines = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (4, 6), (4, 7)]
        line_colors = [
            (1.0, 0.85, 0.2)] * 4 + [(0.2, 1.0, 0.2), (0.2, 0.6, 1.0), (1.0, 0.3, 0.2)]
        frame = o3d.geometry.LineSet()
        frame.points = o3d.utility.Vector3dVector(np.asarray(points))
        frame.lines = o3d.utility.Vector2iVector(np.asarray(lines))
        frame.colors = o3d.utility.Vector3dVector(np.asarray(line_colors))
        scene.add_geometry("frame", frame, self._frame_mat)

    def _update_info(self):
        original_count = len(self.original_pcd.points) if self.original_pcd is not None else 0
        plane_count = sum(count_plane_points(p) for p in self.planes)
        self._set_info(
            f"原始 {original_count} | 平面 {len(self.planes)} | "
            f"合成 {original_count + plane_count} 点")

    def _set_info(self, text):
        self._lbl_info.text = text

    # ------------------------------------------------- 鼠标拾取

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
            self._handle_pick(event.x, event.y)
            return True
        return False

    def _handle_pick(self, window_x, window_y):
        if self._pick_pts is None or len(self._pick_pts) == 0:
            return
        scene_x = window_x - self._scene.frame.x
        scene_y = window_y - self._scene.frame.y
        width, height = self._scene.frame.width, self._scene.frame.height
        if width <= 0 or height <= 0:
            return
        camera = self._scene.scene.camera
        try:
            near = camera.unproject(float(scene_x), float(scene_y), -1.0, float(width), float(height))
            far = camera.unproject(float(scene_x), float(scene_y), 0.0, float(width), float(height))
            edge_far = camera.unproject(float(min(scene_x + 8, width - 1)), float(scene_y),
                                        0.0, float(width), float(height))
        except Exception:
            return
        ray = far - near
        ray_norm = np.linalg.norm(ray)
        edge_ray = edge_far - near
        edge_norm = np.linalg.norm(edge_ray)
        if ray_norm < 1e-12 or edge_norm < 1e-12:
            return
        ray /= ray_norm
        edge_ray /= edge_norm
        angular_tolerance = np.arccos(np.clip(ray @ edge_ray, -1.0, 1.0))
        origin = np.linalg.inv(camera.get_view_matrix())[:3, 3]
        vectors = self._pick_pts - origin
        depths = vectors @ ray
        perpendicular = np.linalg.norm(vectors - np.outer(depths, ray), axis=1)
        candidates = (depths > 0.0) & (perpendicular <= depths * np.tan(angular_tolerance))
        if not np.any(candidates):
            return
        picked = np.flatnonzero(candidates)[np.argmin(depths[candidates])]
        self._select_plane(int(self._pick_ids[picked]))

    # ------------------------------------------------- 键盘快捷键

    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False
        try:
            ctrl = event.is_modifier_down(gui.KeyModifier.CTRL)
        except Exception:
            ctrl = False
        if ctrl and event.key == ord("z"):
            self._on_undo()
            return True
        if ctrl and event.key == ord("s"):
            self._on_save()
            return True
        if event.key == gui.KeyName.N:
            self._on_add()
            return True
        if event.key == gui.KeyName.DELETE:
            self._on_delete()
            return True
        if event.key == gui.KeyName.LEFT:
            return self._nudge_position(-1.0)
        if event.key == gui.KeyName.RIGHT:
            return self._nudge_position(1.0)
        return False

    def _nudge_position(self, direction):
        if self._last_pos_key is None or self.selected < 0 or self.selected >= len(self.planes):
            return False
        key = self._last_pos_key
        axis = ("cx", "cy", "cz").index(key)
        limits = self._pos_limits[axis]
        step = max((limits[1] - limits[0]) * 0.002, 0.005)
        value = self._set_slider_value(key, self._sliders[key].double_value + direction * step)
        self.planes[self.selected].center[axis] = value
        self._refresh_scene()
        return True

    def run(self):
        self.app.run()


def main():
    parser = argparse.ArgumentParser(description="在已有点云上交互式补充平面点云")
    parser.add_argument("cloud", nargs="?", default=None, help="已有点云文件 (.pcd/.ply/.xyz)")
    args = parser.parse_args()
    PlaneDrawerApp(args.cloud).run()


if __name__ == "__main__":
    main()
