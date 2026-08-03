#!/usr/bin/env python3
"""Interactive ICP fine-registration of two roughly aligned point clouds.

Load a fixed reference cloud (配准模板) and a source cloud (待配准点云), run
point-to-plane / point-to-point ICP with adjustable multi-scale parameters,
then inspect the residual difference in the GUI: either a two-color overlay
or a distance heatmap. The aligned source cloud and the 4x4 transform can be
saved.

Usage:
    ~/.venv/bin/python icp_align_cloud_gui.py [template.pcd] [source.pcd]

Requires: open3d >= 0.18, numpy.
"""

import argparse
import os

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


TEMPLATE_COLOR = (0.35, 0.55, 0.95)   # blue
SOURCE_COLOR = (1.00, 0.45, 0.10)     # orange
IDENTITY = np.eye(4)


def _jet_colors(distances, threshold):
    """Matplotlib-style jet colormap, distances clipped at `threshold`."""
    t = np.clip(np.asarray(distances, dtype=np.float64) / max(threshold, 1e-9), 0.0, 1.0)
    colors = np.empty((len(t), 3), dtype=np.float64)
    colors[:, 0] = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    colors[:, 1] = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    colors[:, 2] = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return colors


class ICPAlignApp:
    """GUI for ICP fine-registration and difference inspection."""

    def __init__(self, template_path=None, source_path=None):
        self.app = gui.Application.instance
        self.app.initialize()
        self._configure_chinese_font()

        self.window = self.app.create_window("ICP 点云精配准", 1440, 900)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.set_background([0.1, 0.1, 0.1, 1.0])

        em = self.window.theme.font_size
        self._panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self._build_ui()
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._panel)

        self.template_pcd = None
        self.source_pcd = None
        self.aligned_source = None
        self.accum_T = IDENTITY.copy()
        self.last_fitness = 0.0
        self.last_rmse = 0.0
        self.has_registered = False
        self._dist_cache = None  # source -> template distances, recomputed after alignment

        self._template_mat = rendering.MaterialRecord()
        self._template_mat.shader = "defaultUnlit"
        self._template_mat.point_size = 2.5
        self._source_mat = rendering.MaterialRecord()
        self._source_mat.shader = "defaultUnlit"
        self._source_mat.point_size = 3.5

        if template_path:
            self._load_cloud(template_path, as_template=True)
        if source_path:
            self._load_cloud(source_path, as_template=False)

    # ------------------------------------------------------------------ UI

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
        load_template = gui.Button("加载配准模板")
        load_template.set_on_clicked(self._on_load_template)
        load_source = gui.Button("加载待配准点云")
        load_source.set_on_clicked(self._on_load_source)
        save_aligned = gui.Button("保存配准后的点云")
        save_aligned.set_on_clicked(self._on_save_aligned)
        save_transform = gui.Button("保存变换矩阵 (.txt)")
        save_transform.set_on_clicked(self._on_save_transform)
        for button in (load_template, load_source, save_aligned, save_transform):
            file_section.add_child(button)
            file_section.add_child(gui.VGrid(1, 0.25 * em))

        icp_section = self._section("ICP 配准参数")
        self._sld_voxel = self._make_slider(
            icp_section, "体素降采样 (m, 0=不降采样)", 0.0, 0.5, 0.05, ".3f")
        self._sld_max_dist = self._make_slider(
            icp_section, "最大对应距离 (m)", 0.01, 2.0, 0.30, ".3f")
        self._sld_iter = self._make_slider(
            icp_section, "每层迭代次数", 5, 100, 40, "d", True)
        self._sld_levels = self._make_slider(
            icp_section, "多尺度层次 (距离逐层减半)", 1, 6, 3, "d", True)
        self._chk_plane = gui.Checkbox("点到面 ICP (需模板法向量)")
        self._chk_plane.checked = True
        icp_section.add_child(gui.VGrid(1, 0.25 * em))
        icp_section.add_child(self._chk_plane)
        run_button = gui.Button("运行 ICP 配准")
        run_button.set_on_clicked(self._on_run_icp)
        reset_button = gui.Button("重置变换")
        reset_button.set_on_clicked(self._on_reset)
        icp_section.add_child(gui.VGrid(1, 0.25 * em))
        icp_section.add_child(run_button)
        icp_section.add_child(gui.VGrid(1, 0.25 * em))
        icp_section.add_child(reset_button)

        display_section = self._section("显示")
        self._chk_show_template = gui.Checkbox("显示配准模板 (蓝)")
        self._chk_show_template.checked = True
        self._chk_show_template.set_on_checked(self._on_display_changed)
        self._chk_show_source = gui.Checkbox("显示待配准点云 (橙)")
        self._chk_show_source.checked = True
        self._chk_show_source.set_on_checked(self._on_display_changed)
        self._chk_diff = gui.Checkbox("差异热力图 (按到模板距离染色)")
        self._chk_diff.checked = False
        self._chk_diff.set_on_checked(self._on_display_changed)
        display_section.add_child(self._chk_show_template)
        display_section.add_child(self._chk_show_source)
        display_section.add_child(gui.VGrid(1, 0.25 * em))
        display_section.add_child(self._chk_diff)
        self._sld_threshold = self._make_slider(
            display_section, "热力图距离上限 (m)", 0.01, 2.0, 0.10, ".3f",
            on_change=self._on_display_changed)
        self._sld_point_size = self._make_slider(
            display_section, "点大小", 1, 12, 3, "d", True, self._on_point_size)

        hint_section = self._section("操作提示")
        hint_section.add_child(gui.Label("1. 依次加载模板与待配准点云"))
        hint_section.add_child(gui.Label("2. 调整参数后点击“运行 ICP 配准”"))
        hint_section.add_child(gui.Label("3. 勾选差异热力图查看残差分布"))
        hint_section.add_child(gui.Label("4. 保存配准后的点云或变换矩阵"))

        for section in (file_section, icp_section, display_section, hint_section):
            self._panel.add_child(section)
        self._panel.add_child(gui.VGrid(1, 0.5 * em))
        self._lbl_state = gui.Label("就绪 - 请加载点云")
        self._lbl_state.text_color = gui.Color(0.8, 0.8, 0.8)
        self._panel.add_child(self._lbl_state)

    def _section(self, title):
        section = gui.CollapsableVert(title, 0.25 * self.window.theme.font_size,
                                      gui.Margins(self.window.theme.font_size, 0, 0, 0))
        section.set_is_open(True)
        return section

    def _make_slider(self, parent, label, minimum, maximum, default, fmt,
                     is_int=False, on_change=None):
        container = gui.Vert()
        slider = gui.Slider(gui.Slider.INT if is_int else gui.Slider.DOUBLE)
        slider.set_limits(float(minimum), float(maximum))
        slider.double_value = float(default)
        value_label = gui.Label(format(int(round(default)), fmt) if is_int
                                else format(default, fmt))

        def handle(value):
            value_label.text = (format(int(round(value)), fmt) if is_int
                                else format(value, fmt))
            if on_change is not None:
                on_change(value)

        slider.set_on_value_changed(handle)
        container.add_child(gui.Label(label))
        row = gui.Horiz(5)
        row.add_child(slider)
        row.add_child(value_label)
        container.add_child(row)
        parent.add_child(container)
        return slider

    def _on_layout(self, context):
        rect = self.window.content_rect
        sidebar = min(rect.width - 10, 24 * context.theme.font_size)
        self._scene.frame = gui.Rect(rect.x, rect.y, rect.width - sidebar, rect.height)
        self._panel.frame = gui.Rect(rect.x + rect.width - sidebar, rect.y, sidebar, rect.height)

    # --------------------------------------------------------------- files

    def _on_load_template(self):
        self._show_open_dialog("选择配准模板点云", self._load_cloud, True)

    def _on_load_source(self):
        self._show_open_dialog("选择待配准点云", self._load_cloud, False)

    def _show_open_dialog(self, title, callback, as_template):
        dialog = gui.FileDialog(gui.FileDialog.OPEN, title, self.window.theme)
        dialog.add_filter(".pcd .ply .xyz .pts", "点云文件")
        dialog.set_on_cancel(self.window.close_dialog)
        dialog.set_on_done(lambda path: (self.window.close_dialog(), callback(path, as_template)))
        self.window.show_dialog(dialog)

    def _load_cloud(self, path, as_template):
        try:
            cloud = o3d.io.read_point_cloud(path)
        except Exception as error:
            self._set_info(f"加载失败: {error}")
            return False
        if not cloud.has_points():
            self._set_info(f"点云为空: {os.path.basename(path)}")
            return False
        if as_template:
            self.template_pcd = cloud
            name = "配准模板"
        else:
            self.source_pcd = cloud
            self.aligned_source = o3d.geometry.PointCloud(cloud)
            self.accum_T = IDENTITY.copy()
            self.has_registered = False
            self._dist_cache = None
            name = "待配准点云"
        self._update_visualization(reset_camera=True)
        self._update_info()
        self._set_info(f"已加载{name}: {os.path.basename(path)}")
        return True

    def _on_save_aligned(self):
        if self.aligned_source is None:
            self._set_info("请先加载待配准点云")
            return
        dialog = gui.FileDialog(gui.FileDialog.SAVE, "保存配准后的点云", self.window.theme)
        dialog.add_filter(".pcd", "PCD 文件")
        dialog.add_filter(".ply", "PLY 文件")
        dialog.set_on_cancel(self.window.close_dialog)
        dialog.set_on_done(self._on_save_aligned_done)
        self.window.show_dialog(dialog)

    def _on_save_aligned_done(self, path):
        self.window.close_dialog()
        if not (path.endswith(".pcd") or path.endswith(".ply")):
            path += ".pcd"
        if o3d.io.write_point_cloud(path, self.aligned_source, write_ascii=False):
            self._set_info(f"已保存配准结果: {os.path.basename(path)}")
        else:
            self._set_info("保存失败")

    def _on_save_transform(self):
        if not self.has_registered:
            self._set_info("尚未执行配准，无变换矩阵")
            return
        dialog = gui.FileDialog(gui.FileDialog.SAVE, "保存变换矩阵", self.window.theme)
        dialog.add_filter(".txt", "文本文件")
        dialog.set_on_cancel(self.window.close_dialog)
        dialog.set_on_done(self._on_save_transform_done)
        self.window.show_dialog(dialog)

    def _on_save_transform_done(self, path):
        self.window.close_dialog()
        if not path.endswith(".txt"):
            path += ".txt"
        try:
            np.savetxt(path, self.accum_T, fmt="%.9f")
            self._set_info(f"已保存变换矩阵: {os.path.basename(path)}")
        except Exception as error:
            self._set_info(f"保存失败: {error}")

    # ----------------------------------------------------------------- ICP

    @staticmethod
    def _downsample(cloud, voxel):
        return cloud.voxel_down_sample(voxel) if voxel > 0.001 else o3d.geometry.PointCloud(cloud)

    def _on_run_icp(self):
        if self.template_pcd is None or self.source_pcd is None:
            self._set_info("请先加载配准模板和待配准点云")
            return
        voxel = self._sld_voxel.double_value
        source_work = self._downsample(self.source_pcd, voxel)
        target_work = self._downsample(self.template_pcd, voxel)
        if len(source_work.points) == 0 or len(target_work.points) == 0:
            self._set_info("降采样后点云为空，请增大体素尺寸")
            return

        use_plane = self._chk_plane.checked
        if use_plane:
            radius = max(voxel * 2.0, 0.05) if voxel > 0.001 else 0.1
            target_work.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
            if not np.all(np.isfinite(np.asarray(target_work.normals))):
                use_plane = False
                self._set_info("法向量估计失败，已回退到点到点 ICP")
        estimation = (o3d.pipelines.registration.TransformationEstimationPointToPlane()
                      if use_plane else
                      o3d.pipelines.registration.TransformationEstimationPointToPoint())

        max_dist = self._sld_max_dist.double_value
        levels = int(self._sld_levels.double_value)
        iterations = int(self._sld_iter.double_value)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=iterations, relative_fitness=1e-6, relative_rmse=1e-6)
        init = self.accum_T.copy()
        result = None
        try:
            for level in range(levels):
                distance = max_dist / (2 ** level)
                result = o3d.pipelines.registration.registration_icp(
                    source_work, target_work, distance, init, estimation, criteria)
                init = result.transformation
        except Exception as error:
            self._set_info(f"ICP 失败: {error}")
            return

        self.accum_T = np.asarray(result.transformation).copy()
        self.last_fitness = float(result.fitness)
        self.last_rmse = float(result.inlier_rmse)
        self.has_registered = True
        self._dist_cache = None
        self.aligned_source = o3d.geometry.PointCloud(self.source_pcd).transform(self.accum_T)
        self._update_visualization()
        self._update_info()
        self._set_info(
            f"ICP 完成: fitness {self.last_fitness:.4f}, RMSE {self.last_rmse:.4f} m, "
            f"{levels} 层, 最终距离 {distance:.4f} m")

    def _on_reset(self):
        if self.source_pcd is None:
            return
        self.accum_T = IDENTITY.copy()
        self.has_registered = False
        self._dist_cache = None
        self.aligned_source = o3d.geometry.PointCloud(self.source_pcd)
        self._update_visualization()
        self._update_info()
        self._set_info("已重置变换")

    # ------------------------------------------------------------- display

    def _on_display_changed(self, _value):
        self._update_visualization()

    def _on_point_size(self, value):
        self._template_mat.point_size = float(value)
        self._source_mat.point_size = min(float(value) + 1.0, 15.0)
        self._update_visualization()

    def _update_visualization(self, reset_camera=False):
        scene = self._scene.scene
        scene.remove_geometry("template")
        scene.remove_geometry("source")
        if self.template_pcd is not None and self._chk_show_template.checked:
            cloud = o3d.geometry.PointCloud(self.template_pcd)
            cloud.paint_uniform_color(TEMPLATE_COLOR)
            scene.add_geometry("template", cloud, self._template_mat)
        if self.aligned_source is not None and self._chk_show_source.checked:
            cloud = o3d.geometry.PointCloud(self.aligned_source)
            if self._chk_diff.checked:
                self._paint_difference(cloud)
            else:
                cloud.paint_uniform_color(SOURCE_COLOR)
            scene.add_geometry("source", cloud, self._source_mat)
        if reset_camera:
            self._fit_camera()
        self._scene.force_redraw()

    def _paint_difference(self, cloud):
        if self.template_pcd is None:
            cloud.paint_uniform_color(SOURCE_COLOR)
            return
        if self._dist_cache is None:
            self._dist_cache = np.asarray(
                cloud.compute_point_cloud_distance(self.template_pcd))
        threshold = self._sld_threshold.double_value
        cloud.colors = o3d.utility.Vector3dVector(_jet_colors(self._dist_cache, threshold))

    def _fit_camera(self):
        clouds = []
        if self.template_pcd is not None and self._chk_show_template.checked:
            clouds.append(self.template_pcd)
        if self.aligned_source is not None and self._chk_show_source.checked:
            clouds.append(self.aligned_source)
        if not clouds:
            return
        lower = clouds[0].get_min_bound().copy()
        upper = clouds[0].get_max_bound().copy()
        for cloud in clouds[1:]:
            lower = np.minimum(lower, cloud.get_min_bound())
            upper = np.maximum(upper, cloud.get_max_bound())
        bounds = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
        self._scene.setup_camera(60.0, bounds, bounds.get_center())

    # --------------------------------------------------------------- info

    def _update_info(self):
        parts = []
        if self.template_pcd is not None:
            parts.append(f"模板 {len(self.template_pcd.points)}")
        if self.source_pcd is not None:
            parts.append(f"待配准 {len(self.source_pcd.points)}")
        if self.has_registered:
            parts.append(f"fitness {self.last_fitness:.4f}")
            parts.append(f"RMSE {self.last_rmse:.4f} m")
        self._lbl_state.text = " | ".join(parts) if parts else "就绪"

    def _set_info(self, text):
        self._lbl_state.text = text

    def run(self):
        self.app.run()


def main():
    parser = argparse.ArgumentParser(description="ICP 点云精配准 GUI")
    parser.add_argument("template", nargs="?", help="配准模板点云 (.pcd/.ply/.xyz/.pts)")
    parser.add_argument("source", nargs="?", help="待配准点云 (.pcd/.ply/.xyz/.pts)")
    args = parser.parse_args()
    app = ICPAlignApp(args.template, args.source)
    app.run()


if __name__ == "__main__":
    main()
