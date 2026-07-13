#!/usr/bin/env python3
"""Interactive point-cloud segment remover.

The cloud is partitioned into large planar segments and compact spatial
clusters.  Every extracted segment is independently selectable and removable.
Requires: open3d >= 0.18, numpy.
"""

import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


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
NOISE_COLOR = np.array((0.25, 0.25, 0.25))
HIGHLIGHT_COLOR = (1.0, 0.92, 0.2)
PLANE = 0
CLUSTER = 1


class PointCloudCleaner:
    """Partition a point cloud into planes and clusters for interactive removal."""

    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()
        self._configure_chinese_font()

        self.window = self.app.create_window("点云分块删除", 1400, 900)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.set_background([0.1, 0.1, 0.1, 1.0])
        self._scene.set_on_mouse(self._on_mouse)

        em = self.window.theme.font_size
        self._panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self._build_ui()
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        self.window.add_child(self._scene)
        self.window.add_child(self._panel)

        self.original_pcd = None
        self.working_points = None
        self.segment_labels = None
        self.segment_types = {}
        self.visible_mask = None
        self.selected_segment = -1
        self.undo_stack = []
        self.max_undo = 30

        self._point_material = rendering.MaterialRecord()
        self._point_material.shader = "defaultUnlit"
        self._point_material.point_size = 3.0
        self._highlight_material = rendering.MaterialRecord()
        self._highlight_material.shader = "defaultUnlit"
        self._highlight_material.point_size = 6.0

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
        save_button = gui.Button("保存处理结果")
        save_button.set_on_clicked(self._on_save)
        self._lbl_info = gui.Label("就绪 - 请加载点云")
        self._lbl_info.text_color = gui.Color(0.8, 0.8, 0.8)
        file_section.add_child(load_button)
        file_section.add_child(gui.VGrid(1, 0.25 * em))
        file_section.add_child(save_button)
        file_section.add_child(gui.VGrid(1, 0.5 * em))
        file_section.add_child(self._lbl_info)

        plane_section = self._section("平面提取")
        self._sld_voxel = self._make_slider(
            plane_section, "体素尺寸 (m)", 0.0, 1.0, 0.08, ".3f")
        self._sld_plane_distance = self._make_slider(
            plane_section, "平面距离阈值 (m)", 0.01, 1.0, 0.12, ".3f")
        self._sld_plane_min_points = self._make_slider(
            plane_section, "最小平面点数", 20, 5000, 200, "d", True)
        self._sld_max_planes = self._make_slider(
            plane_section, "最多平面数", 1, 100, 20, "d", True)

        cluster_section = self._section("剩余点聚类")
        self._sld_cluster_eps = self._make_slider(
            cluster_section, "聚类距离 (m)", 0.05, 3.0, 0.30, ".3f")
        self._sld_cluster_min_points = self._make_slider(
            cluster_section, "最小聚块点数", 1, 500, 10, "d", True)

        process_button = gui.Button("重新分割")
        process_button.set_on_clicked(self._on_process)
        cluster_section.add_child(gui.VGrid(1, 0.5 * em))
        cluster_section.add_child(process_button)

        edit_section = self._section("编辑")
        self._lbl_selected = gui.Label("未选择")
        self._lbl_selected.text_color = gui.Color(1.0, 1.0, 0.5)
        delete_button = gui.Button("删除选中块 (Delete)")
        delete_button.set_on_clicked(self._on_delete)
        undo_button = gui.Button("撤销 (Ctrl+Z)")
        undo_button.set_on_clicked(self._on_undo)
        deselect_button = gui.Button("取消选择 (Esc)")
        deselect_button.set_on_clicked(self._on_deselect)
        edit_section.add_child(self._lbl_selected)
        for button in (delete_button, undo_button, deselect_button):
            edit_section.add_child(gui.VGrid(1, 0.25 * em))
            edit_section.add_child(button)

        hint_section = self._section("操作提示")
        hint_section.add_child(gui.Label("Ctrl + 点击: 选择最前方块"))
        hint_section.add_child(gui.Label("Delete: 删除选中块"))
        hint_section.add_child(gui.Label("Ctrl+Z: 撤销删除"))

        for section in (file_section, plane_section, cluster_section, edit_section, hint_section):
            self._panel.add_child(section)

    def _section(self, title):
        section = gui.CollapsableVert(title, 0.25 * self.window.theme.font_size,
                                      gui.Margins(self.window.theme.font_size, 0, 0, 0))
        section.set_is_open(True)
        return section

    @staticmethod
    def _make_slider(parent, label, minimum, maximum, default, fmt, is_int=False):
        container = gui.Vert()
        slider = gui.Slider(gui.Slider.INT if is_int else gui.Slider.DOUBLE)
        slider.set_limits(float(minimum), float(maximum))
        slider.double_value = float(default)
        def format_value(value):
            return format(int(round(value)), fmt) if is_int else format(value, fmt)

        value_label = gui.Label(format_value(default))

        def on_change(value):
            value_label.text = format_value(value)

        slider.set_on_value_changed(on_change)
        container.add_child(gui.Label(label))
        row = gui.Horiz(5)
        row.add_child(slider)
        row.add_child(value_label)
        container.add_child(row)
        parent.add_child(container)
        return slider

    def _on_layout(self, context):
        rect = self.window.content_rect
        sidebar = min(rect.width - 10, 22 * context.theme.font_size)
        self._scene.frame = gui.Rect(rect.x, rect.y, rect.width - sidebar, rect.height)
        self._panel.frame = gui.Rect(rect.x + rect.width - sidebar, rect.y, sidebar, rect.height)

    def _on_load(self):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, "选择点云文件", self.window.theme)
        dialog.add_filter(".pcd .ply .xyz .pts", "点云文件")
        dialog.set_on_cancel(self.window.close_dialog)
        dialog.set_on_done(self._on_load_done)
        self.window.show_dialog(dialog)

    def _on_load_done(self, path):
        self.window.close_dialog()
        try:
            cloud = o3d.io.read_point_cloud(path)
        except Exception as error:
            self._set_info(f"加载失败: {error}")
            return
        if not cloud.has_points():
            self._set_info("空点云")
            return
        self._reset_state()
        self.original_pcd = cloud
        self._run_segmentation()
        self._set_info(f"已加载 {os.path.basename(path)}，正在显示分割结果")

    def _on_save(self):
        if self.working_points is None:
            return
        dialog = gui.FileDialog(gui.FileDialog.SAVE, "保存处理结果", self.window.theme)
        dialog.add_filter(".pcd", "PCD 文件")
        dialog.set_on_cancel(self.window.close_dialog)
        dialog.set_on_done(self._on_save_done)
        self.window.show_dialog(dialog)

    def _on_save_done(self, path):
        self.window.close_dialog()
        if not path.endswith(".pcd"):
            path += ".pcd"
        cloud = self._build_visible_cloud()
        if cloud is not None and o3d.io.write_point_cloud(path, cloud):
            self._set_info(f"已保存至 {os.path.basename(path)}")

    def _on_process(self):
        self._run_segmentation()

    def _run_segmentation(self):
        if self.original_pcd is None:
            return
        voxel_size = self._sld_voxel.double_value
        cloud = (self.original_pcd.voxel_down_sample(voxel_size)
                 if voxel_size > 0.001 else self.original_pcd)
        self.working_points = np.asarray(cloud.points).copy()
        if len(self.working_points) == 0:
            self._set_info("降采样后点云为空")
            return
        self._extract_segments()
        self._update_visualization(reset_camera=True)
        self._update_info()

    def _extract_segments(self):
        """Extract connected RANSAC planes, then cluster every remaining point."""
        points = self.working_points
        count = len(points)
        labels = np.full(count, -1, dtype=np.int32)
        self.segment_types = {}
        remaining = np.arange(count, dtype=np.int32)
        next_label = 0
        distance = self._sld_plane_distance.double_value
        min_plane_points = int(self._sld_plane_min_points.double_value)

        for _ in range(int(self._sld_max_planes.double_value)):
            if len(remaining) < min_plane_points:
                break
            candidate = o3d.geometry.PointCloud()
            candidate.points = o3d.utility.Vector3dVector(points[remaining])
            _, inliers = candidate.segment_plane(distance, 3, 1000)
            if len(inliers) < min_plane_points:
                break
            inlier_indices = remaining[np.asarray(inliers, dtype=np.int32)]
            connected = self._largest_connected_component(points[inlier_indices], distance * 3.0)
            if len(connected) < min_plane_points:
                break
            plane_indices = inlier_indices[connected]
            labels[plane_indices] = next_label
            self.segment_types[next_label] = PLANE
            next_label += 1
            keep = np.ones(len(remaining), dtype=bool)
            keep[np.isin(remaining, plane_indices)] = False
            remaining = remaining[keep]

        if len(remaining):
            remaining_cloud = o3d.geometry.PointCloud()
            remaining_cloud.points = o3d.utility.Vector3dVector(points[remaining])
            cluster_labels = np.asarray(remaining_cloud.cluster_dbscan(
                eps=self._sld_cluster_eps.double_value,
                min_points=int(self._sld_cluster_min_points.double_value),
                print_progress=False), dtype=np.int32)
            for cluster_id in range(int(cluster_labels.max()) + 1 if len(cluster_labels) else 0):
                indices = remaining[cluster_labels == cluster_id]
                labels[indices] = next_label
                self.segment_types[next_label] = CLUSTER
                next_label += 1

        self.segment_labels = labels
        self.visible_mask = np.ones(count, dtype=bool)
        self.selected_segment = -1
        self.undo_stack.clear()
        self._lbl_selected.text = "未选择"

    @staticmethod
    def _largest_connected_component(points, radius):
        """Split RANSAC inliers so disconnected coplanar surfaces stay selectable."""
        if len(points) < 2:
            return np.arange(len(points), dtype=np.int32)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        labels = np.asarray(cloud.cluster_dbscan(eps=max(radius, 0.01), min_points=1,
                                                 print_progress=False), dtype=np.int32)
        return np.flatnonzero(labels == labels[np.argmax(np.bincount(labels))])

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
            self._handle_pick(event.x, event.y)
            return True
        return False

    def _handle_pick(self, window_x, window_y):
        if self.working_points is None or not np.any(self.visible_mask):
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
        active_indices = np.flatnonzero(self.visible_mask)
        vectors = self.working_points[active_indices] - origin
        depths = vectors @ ray
        perpendicular = np.linalg.norm(vectors - np.outer(depths, ray), axis=1)
        candidates = (depths > 0.0) & (perpendicular <= depths * np.tan(angular_tolerance))
        if not np.any(candidates):
            return
        picked = active_indices[np.flatnonzero(candidates)[np.argmin(depths[candidates])]]
        segment_id = int(self.segment_labels[picked])
        if segment_id >= 0:
            self._select_segment(segment_id)

    def _select_segment(self, segment_id):
        self.selected_segment = segment_id
        count = int(np.count_nonzero(self.segment_labels == segment_id))
        kind = "平面" if self.segment_types[segment_id] == PLANE else "聚块"
        self._lbl_selected.text = f"选中{kind} #{segment_id} ({count} 点)"
        self._update_highlight()

    def _on_deselect(self):
        self.selected_segment = -1
        self._lbl_selected.text = "未选择"
        self._update_highlight()

    def _on_delete(self):
        if self.selected_segment < 0:
            return
        deleted = (self.segment_labels == self.selected_segment) & self.visible_mask
        if not np.any(deleted):
            return
        self.undo_stack.append(self.visible_mask.copy())
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.visible_mask[deleted] = False
        self._on_deselect()
        self._update_visualization()
        self._update_info()

    def _on_undo(self):
        if not self.undo_stack:
            return
        self.visible_mask = self.undo_stack.pop()
        self._on_deselect()
        self._update_visualization()
        self._update_info()

    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False
        if event.key == gui.KeyName.DELETE:
            self._on_delete()
            return True
        if event.key == gui.KeyName.ESCAPE:
            self._on_deselect()
            return True
        try:
            ctrl_down = event.is_modifier_down(gui.KeyModifier.CTRL)
        except Exception:
            ctrl_down = False
        if ctrl_down and event.key == ord("z"):
            self._on_undo()
            return True
        return False

    def _update_visualization(self, reset_camera=False):
        scene = self._scene.scene
        scene.remove_geometry("segments")
        cloud = self._build_visible_cloud()
        if cloud is not None:
            scene.add_geometry("segments", cloud, self._point_material)
            if reset_camera:
                bounds = cloud.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60.0, bounds, bounds.get_center())
        self._update_highlight()
        self._scene.force_redraw()

    def _update_highlight(self):
        scene = self._scene.scene
        scene.remove_geometry("highlight")
        if self.selected_segment < 0 or self.visible_mask is None:
            return
        selected = (self.segment_labels == self.selected_segment) & self.visible_mask
        if not np.any(selected):
            return
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.working_points[selected])
        cloud.paint_uniform_color(HIGHLIGHT_COLOR)
        scene.add_geometry("highlight", cloud, self._highlight_material)

    def _build_visible_cloud(self):
        if self.visible_mask is None or not np.any(self.visible_mask):
            return None
        labels = self.segment_labels[self.visible_mask]
        colors = np.empty((len(labels), 3), dtype=np.float64)
        valid = labels >= 0
        colors[~valid] = NOISE_COLOR
        colors[valid] = PALETTE[labels[valid] % len(PALETTE)]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.working_points[self.visible_mask])
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    def _update_info(self):
        if self.working_points is None:
            self._set_info("就绪")
            return
        plane_count = sum(kind == PLANE for kind in self.segment_types.values())
        cluster_count = sum(kind == CLUSTER for kind in self.segment_types.values())
        noise_count = int(np.count_nonzero(self.segment_labels == -1))
        visible_count = int(np.count_nonzero(self.visible_mask))
        deleted_count = len(self.visible_mask) - visible_count
        self._set_info(
            f"原始 {len(self.original_pcd.points)} -> 处理 {len(self.working_points)} | "
            f"平面 {plane_count} | 聚块 {cluster_count} | 噪声 {noise_count} | "
            f"已删 {deleted_count} | 剩余 {visible_count}")

    def _set_info(self, text):
        self._lbl_info.text = text

    def _reset_state(self):
        self.working_points = None
        self.segment_labels = None
        self.segment_types = {}
        self.visible_mask = None
        self.selected_segment = -1
        self.undo_stack.clear()
        self._scene.scene.clear_geometry()
        self._lbl_selected.text = "未选择"

    def run(self):
        self.app.run()


if __name__ == "__main__":
    PointCloudCleaner().run()
