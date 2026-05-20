#!/usr/bin/env python3
"""
点云动态障碍物去除工具

功能:
  - 地面提取 (PatchWork++ 启发: 极坐标网格 + Z-法向约束 RANSAC)
  - 非地面点 DBSCAN 聚类染色
  - Ctrl+点击选择障碍物块
  - 右侧面板删除 / 撤销
  - 可调参数滑块

依赖: open3d >= 0.18, numpy
"""

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os

# ---------------------------------------------------------------------------
# 预设调色板 (HSV 黄金角分布)
# ---------------------------------------------------------------------------
def _make_palette(n=500):
    colors = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        # HSV -> RGB
        hi = int(h * 6)
        f = h * 6 - hi
        p, q, t = 0.0, 1.0 - f, f
        r, g, b = {
            0: (1.0, t, 0.0),
            1: (q, 1.0, 0.0),
            2: (0.0, 1.0, t),
            3: (0.0, q, 1.0),
            4: (t, 0.0, 1.0),
            5: (1.0, 0.0, q),
        }[hi % 6]
        colors.append([r * 0.85 + 0.15, g * 0.85 + 0.15, b * 0.85 + 0.15])
    return colors


_PALETTE = _make_palette(500)
_GROUND_COLOR = (0.65, 0.65, 0.65)
_GROUND_REMAIN_COLOR = (0.45, 0.55, 0.45)
_NOISE_COLOR = (0.25, 0.25, 0.25)
_HIGHLIGHT_COLOR = (1.0, 0.92, 0.2)

# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class PointCloudCleaner:
    """交互式点云动态障碍物去除器."""

    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()

        # ----- 字体 (中文支持) -----
        for fp in [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]:
            if os.path.exists(fp):
                fd = gui.FontDescription(fp)
                fd.add_typeface_for_language(fp, "zh_all")
                gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, fd)
                break

        self.window = self.app.create_window("点云动态障碍物去除", 1400, 900)
        w = self.window
        em = w.theme.font_size

        # ----- 3D 场景 -----
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([0.1, 0.1, 0.1, 1.0])
        self._scene.set_on_mouse(self._on_mouse)

        # ----- 右侧面板 -----
        self._panel = gui.Vert(0, gui.Margins(em, em, em, em))
        self._build_ui()

        # ----- 布局 -----
        w.set_on_layout(self._on_layout)
        w.set_on_key(self._on_key)
        w.add_child(self._scene)
        w.add_child(self._panel)

        # ----- 数据状态 -----
        self.original_pcd = None       # 刚加载的原始点云 (未降采样)
        self.voxel_pcd = None          # 降采样后用于处理
        self.ground_pts = None         # (M,3) 地面点坐标
        self.ground_plane_ids = None   # (M,) 每个地面点所属平面 ID (-1 = 非平面细碎地面)
        self.obstacle_pts = None       # (N,3) 非地面点坐标
        self.obstacle_labels = None    # (N,) DBSCAN 标签
        self.obstacle_mask = None      # (N,) bool — True 表示尚未被删除

        self.selected_cluster = -1

        # 撤销栈: 存 obstacle_mask 副本
        self.undo_stack = []
        self.max_undo = 30

        # ----- 材质 -----
        self._mat = rendering.MaterialRecord()
        self._mat.shader = "defaultUnlit"
        self._mat.point_size = 3.0

        self._mat_small = rendering.MaterialRecord()
        self._mat_small.shader = "defaultUnlit"
        self._mat_small.point_size = 1.5

        self._mat_hl = rendering.MaterialRecord()
        self._mat_hl.shader = "defaultUnlit"
        self._mat_hl.point_size = 6.0

    # ======================================================================
    #  UI 构建
    # ======================================================================

    def _build_ui(self):
        em = self.window.theme.font_size

        # ---- 文件操作 ----
        sec_file = gui.CollapsableVert("文件", 0.25 * em, gui.Margins(em, 0, 0, 0))
        sec_file.set_is_open(True)

        btn_load = gui.Button("加载点云")
        btn_load.set_on_clicked(self._on_load)
        btn_save = gui.Button("保存结果")
        btn_save.set_on_clicked(self._on_save)
        sec_file.add_child(btn_load)
        sec_file.add_child(gui.VGrid(1, 0.25 * em))
        sec_file.add_child(btn_save)

        # ---- 信息 ----
        self._lbl_info = gui.Label("就绪 — 请加载点云")
        self._lbl_info.text_color = gui.Color(0.8, 0.8, 0.8)
        sec_file.add_child(gui.VGrid(1, 0.5 * em))
        sec_file.add_child(self._lbl_info)

        self._panel.add_child(sec_file)

        # ---- 地面提取参数 ----
        sec_ground = gui.CollapsableVert("地面提取", 0.25 * em, gui.Margins(em, 0, 0, 0))
        sec_ground.set_is_open(True)

        self._sld_voxel = self._make_slider(sec_ground, "体素尺寸 (m)", 0.0, 1.0, 0.08, 0.01,
                                             self._on_param_changed, fmt=".3f")
        self._sld_thresh = self._make_slider(sec_ground, "RANSAC 阈值 (m)", 0.01, 1.0, 0.20, 0.01,
                                             self._on_param_changed, fmt=".3f")
        self._sld_angle = self._make_slider(sec_ground, "地面法向夹角 (°)", 1.0, 45.0, 15.0, 0.5,
                                            self._on_param_changed, fmt=".1f")

        btn_process = gui.Button("重新提取 + 聚类")
        btn_process.set_on_clicked(self._on_process)
        sec_ground.add_child(gui.VGrid(1, 0.5 * em))
        sec_ground.add_child(btn_process)

        self._panel.add_child(sec_ground)

        # ---- 聚类参数 ----
        sec_cluster = gui.CollapsableVert("聚类", 0.25 * em, gui.Margins(em, 0, 0, 0))
        sec_cluster.set_is_open(True)

        self._sld_eps = self._make_slider(sec_cluster, "DBSCAN eps (m)", 0.05, 2.0, 0.30, 0.01,
                                          self._on_param_changed, fmt=".3f")
        self._sld_minpts = self._make_slider(sec_cluster, "min points", 1, 200, 10, 1,
                                             self._on_param_changed, fmt="d", is_int=True)

        btn_recluster = gui.Button("重新聚类")
        btn_recluster.set_on_clicked(self._on_recluster)
        sec_cluster.add_child(gui.VGrid(1, 0.5 * em))
        sec_cluster.add_child(btn_recluster)

        self._panel.add_child(sec_cluster)

        # ---- 选择 & 删除 ----
        sec_edit = gui.CollapsableVert("编辑", 0.25 * em, gui.Margins(em, 0, 0, 0))
        sec_edit.set_is_open(True)

        self._lbl_selected = gui.Label("未选择")
        self._lbl_selected.text_color = gui.Color(1, 1, 0.5)
        sec_edit.add_child(self._lbl_selected)

        btn_del = gui.Button("删除选中块 (Delete)")
        btn_del.set_on_clicked(self._on_delete)
        sec_edit.add_child(gui.VGrid(1, 0.25 * em))
        sec_edit.add_child(btn_del)

        btn_undo = gui.Button("撤销 (Ctrl+Z)")
        btn_undo.set_on_clicked(self._on_undo)
        sec_edit.add_child(gui.VGrid(1, 0.25 * em))
        sec_edit.add_child(btn_undo)

        btn_deselect = gui.Button("取消选择 (Esc)")
        btn_deselect.set_on_clicked(self._on_deselect)
        sec_edit.add_child(gui.VGrid(1, 0.25 * em))
        sec_edit.add_child(btn_deselect)

        btn_show_ground = gui.Checkbox("显示地面")
        btn_show_ground.checked = True
        btn_show_ground.set_on_checked(self._on_toggle_ground)
        sec_edit.add_child(gui.VGrid(1, 0.5 * em))
        sec_edit.add_child(btn_show_ground)

        self._panel.add_child(sec_edit)

        # ---- 操作提示 ----
        sec_hint = gui.CollapsableVert("操作提示", 0.25 * em, gui.Margins(em, 0, 0, 0))
        sec_hint.set_is_open(True)
        for t in [
            "Ctrl + 点击 → 选择块",
            "Delete → 删除选中块",
            "Ctrl+Z → 撤销删除",
        ]:
            sec_hint.add_child(gui.Label(t))
        self._panel.add_child(sec_hint)

        # 引用存储
        self._show_ground_check = btn_show_ground

    @staticmethod
    def _make_slider(parent, label, vmin, vmax, vdefault, step, callback, fmt=".3f", is_int=False):
        container = gui.Vert()
        lbl = gui.Label(label)
        sld = gui.Slider(gui.Slider.INT if is_int else gui.Slider.DOUBLE)
        sld.set_limits(float(vmin), float(vmax))
        sld.double_value = float(vdefault)

        val_lbl = gui.Label(format(vdefault, fmt))

        def on_change(val):
            val_lbl.text = format(val, fmt)
            callback()

        sld.set_on_value_changed(on_change)
        container.add_child(lbl)
        row = gui.Horiz(5)
        row.add_child(sld)
        row.add_child(val_lbl)
        container.add_child(row)
        parent.add_child(container)
        return sld

    # ======================================================================
    #  布局
    # ======================================================================

    def _on_layout(self, ctx):
        r = self.window.content_rect
        sidebar = min(r.width - 10, 22 * ctx.theme.font_size)
        self._scene.frame = gui.Rect(r.x, r.y, r.width - sidebar, r.height)
        self._panel.frame = gui.Rect(r.x + r.width - sidebar, r.y, sidebar, r.height)

    # ======================================================================
    #  鼠标 / 键盘
    # ======================================================================

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and \
           event.is_modifier_down(gui.KeyModifier.CTRL):
            self._handle_pick(event.x, event.y)
            return True
        return False

    def _handle_pick(self, win_x, win_y):
        if self.obstacle_pts is None or len(self.obstacle_pts) == 0:
            return

        # 转换到 scene widget 坐标系
        sx = win_x - self._scene.frame.x
        sy = win_y - self._scene.frame.y
        sw = self._scene.frame.width
        sh = self._scene.frame.height
        if sw <= 0 or sh <= 0:
            return

        cam = self._scene.scene.camera
        try:
            p_near = cam.unproject(float(sx), float(sy), -1.0, float(sw), float(sh))
            p_far  = cam.unproject(float(sx), float(sy), 0.0, float(sw), float(sh))
        except Exception:
            return
        if np.any(np.isnan(p_near)) or np.any(np.isnan(p_far)):
            return

        # 射线方向
        ray_dir = p_far - p_near
        nrm = np.linalg.norm(ray_dir)
        if nrm < 1e-12:
            return
        ray_dir /= nrm

        # 相机位置
        view_inv = np.linalg.inv(cam.get_view_matrix())
        ray_origin = view_inv[:3, 3]

        # 在障碍物点中找离射线最近的 (只考虑射线前方)
        vec = self.obstacle_pts - ray_origin
        t = vec @ ray_dir
        valid = t > 0
        if not np.any(valid):
            return

        # perpendicular distance squared
        proj = ray_origin + np.outer(t, ray_dir)
        dist_sq = np.sum((self.obstacle_pts - proj) ** 2, axis=1)
        dist_sq[~valid] = np.inf

        best = np.argmin(dist_sq)
        if dist_sq[best] > 1.0:
            return
        if not self.obstacle_mask[best]:
            return

        cluster_id = self.obstacle_labels[best]
        if cluster_id < 0:
            return  # noise 不选择
        self._select_cluster(int(cluster_id))

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
            ctrl = event.is_modifier_down(gui.KeyModifier.CTRL)
        except Exception:
            ctrl = False
        if ctrl and event.key == ord('z'):
            self._on_undo()
            return True

        return False

    # ======================================================================
    #  文件操作
    # ======================================================================

    def _on_load(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "选择点云文件", self.window.theme)
        dlg.add_filter(".pcd .ply .xyz .pts", "点云文件")
        dlg.set_on_cancel(lambda: self.window.close_dialog())
        dlg.set_on_done(self._on_load_done)
        self.window.show_dialog(dlg)

    def _on_load_done(self, path):
        self.window.close_dialog()
        try:
            pcd = o3d.io.read_point_cloud(path)
        except Exception as e:
            self._set_info(f"加载失败: {e}")
            return
        if not pcd.has_points():
            self._set_info("空点云")
            return

        self._reset_state()
        self.original_pcd = pcd
        self._set_info(f"已加载: {os.path.basename(path)}  ({len(pcd.points)} 点)")

        # 立即跑一次处理管线
        self._run_pipeline()

    def _on_save(self):
        if self.original_pcd is None:
            return
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "保存点云", self.window.theme)
        dlg.add_filter(".pcd", "PCD 文件")
        dlg.set_on_cancel(lambda: self.window.close_dialog())
        dlg.set_on_done(self._on_save_done)
        self.window.show_dialog(dlg)

    def _on_save_done(self, path):
        self.window.close_dialog()
        if not path.endswith(".pcd"):
            path += ".pcd"
        merged = self._build_merged_cloud()
        if merged is not None:
            o3d.io.write_point_cloud(path, merged)
            self._set_info(f"已保存至 {os.path.basename(path)}")

    # ======================================================================
    #  管线
    # ======================================================================

    def _on_process(self):
        self._run_pipeline()

    def _on_recluster(self):
        if self.obstacle_pts is None or len(self.obstacle_pts) == 0:
            return
        self.undo_stack.clear()
        self._cluster_obstacles()
        self._update_visualization()
        self._update_info()

    def _on_param_changed(self):
        pass  # 参数变化由按钮触发重新处理

    def _run_pipeline(self):
        """地面提取 → 聚类 → 可视化."""
        if self.original_pcd is None:
            return

        # 1. 降采样
        vs = self._sld_voxel.double_value
        if vs > 0.001:
            self.voxel_pcd = self.original_pcd.voxel_down_sample(vs)
        else:
            self.voxel_pcd = o3d.geometry.PointCloud(self.original_pcd)

        if not self.voxel_pcd.has_points():
            self._set_info("降采样后点云为空")
            return

        # 2. 地面提取
        self._extract_ground()

        # 3. 聚类
        self._cluster_obstacles()

        # 4. 可视化
        self._update_visualization()
        self._update_info()

    # ------------------------------------------------------------------
    #  地面提取  (PatchWork++ 启发: 全局 RANSAC + 极坐标网格精化)
    # ------------------------------------------------------------------

    def _extract_ground(self):
        points = np.asarray(self.voxel_pcd.points)
        n = len(points)
        threshold = self._sld_thresh.double_value
        max_angle_rad = np.radians(self._sld_angle.double_value)

        ground_mask = np.zeros(n, dtype=bool)
        ground_plane_id = np.full(n, -1, dtype=np.int32)
        remaining = np.arange(n)
        next_plane_id = 0

        # ---- A) 迭代 RANSAC — 每个水平平面独立编号 ----
        temp = o3d.geometry.PointCloud()
        for _ in range(10):
            if len(remaining) < 5:
                break
            temp.points = o3d.utility.Vector3dVector(points[remaining])
            model, inliers = temp.segment_plane(threshold, 3, 500)
            if len(inliers) < 5:
                break
            normal = model[:3] / (np.linalg.norm(model[:3]) + 1e-12)
            angle = np.arccos(np.abs(normal[2]))
            idx = remaining[inliers]
            if angle <= max_angle_rad:
                ground_mask[idx] = True
                ground_plane_id[idx] = next_plane_id
                next_plane_id += 1
                remaining = np.setdiff1d(remaining, idx)
            else:
                remaining = np.setdiff1d(remaining, idx)

        # ---- B) 极坐标网格精化 (捕获起伏 / 小块地面) ----
        nonground_idx = np.where(~ground_mask)[0]
        if len(nonground_idx) > 5:
            ng = points[nonground_idx]
            x, y, z = ng[:, 0], ng[:, 1], ng[:, 2]
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)

            r_max = max(np.max(r), 0.1)
            n_ring = 24
            n_sect = 36

            ri = np.floor(r / r_max * n_ring).astype(np.int32)
            ri = np.clip(ri, 0, n_ring - 1)
            si = np.floor((theta + np.pi) / (2 * np.pi) * n_sect).astype(np.int32)
            si = np.clip(si, 0, n_sect - 1)

            for ring in range(n_ring):
                for sect in range(n_sect):
                    cell = (ri == ring) & (si == sect)
                    cell_idx = nonground_idx[cell]
                    if len(cell_idx) < 4:
                        continue
                    cell_pts = points[cell_idx]
                    cen = cell_pts.mean(axis=0)
                    cent = cell_pts - cen
                    _, _, vh = np.linalg.svd(cent, full_matrices=False)
                    nml = vh[-1] / (np.linalg.norm(vh[-1]) + 1e-12)
                    ang = np.arccos(np.abs(nml[2]))
                    if ang > max_angle_rad or cen[2] > 0.5:
                        continue
                    dist = np.abs(cent @ nml)
                    fidx = cell_idx[dist < threshold * 1.5]
                    ground_mask[fidx] = True
                    ground_plane_id[fidx] = -1  # 精化面不单独编号

        # ---- 存储 ----
        gnd_idx = np.where(ground_mask)[0]
        obs_idx = np.where(~ground_mask)[0]

        self.ground_pts = points[gnd_idx].copy()
        self.ground_plane_ids = ground_plane_id[gnd_idx].copy()

        if len(obs_idx) > 0:
            self.obstacle_pts = points[obs_idx].copy()
            self.obstacle_mask = np.ones(len(obs_idx), dtype=bool)
        else:
            self.obstacle_pts = np.empty((0, 3), dtype=np.float64)
            self.obstacle_mask = np.array([], dtype=bool)
        self.undo_stack.clear()
        self.selected_cluster = -1

    # ------------------------------------------------------------------
    #  聚类  (DBSCAN)
    # ------------------------------------------------------------------

    def _cluster_obstacles(self):
        if self.obstacle_pts is None or len(self.obstacle_pts) < 2:
            self.obstacle_labels = np.array([], dtype=np.int32)
            return
        eps = self._sld_eps.double_value
        min_pts = int(self._sld_minpts.double_value)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.obstacle_pts)
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False),
                          dtype=np.int32)
        self.obstacle_labels = labels
        self.selected_cluster = -1

        n_clusters = int(labels.max()) + 1 if len(labels) > 0 else 0
        n_noise = int((labels == -1).sum()) if len(labels) > 0 else 0
        self._set_info(f"聚类: {n_clusters} 块, {n_noise} 噪声点")

    # ======================================================================
    #  选择 / 删除 / 撤销
    # ======================================================================

    def _select_cluster(self, cid):
        self.selected_cluster = cid
        cnt = int((self.obstacle_labels == cid).sum())
        self._lbl_selected.text = f"选中块 #{cid}  ({cnt} 点)"
        self._update_highlight()

    def _on_deselect(self):
        self.selected_cluster = -1
        self._lbl_selected.text = "未选择"
        self._update_highlight()

    def _on_delete(self):
        if self.selected_cluster < 0 or self.obstacle_labels is None:
            return
        cid = self.selected_cluster

        # 保存撤销状态
        self.undo_stack.append(self.obstacle_mask.copy())
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

        mask = self.obstacle_labels == cid
        self.obstacle_mask[mask] = False
        removed = int(mask.sum())
        self.selected_cluster = -1
        self._lbl_selected.text = "未选择"

        self._update_visualization()
        self._update_info()
        self._set_info(f"已删除 {removed} 点 (块 #{cid})")

    def _on_undo(self):
        if not self.undo_stack:
            return
        self.obstacle_mask = self.undo_stack.pop()
        self.selected_cluster = -1
        self._lbl_selected.text = "未选择"
        self._update_visualization()
        self._update_info()
        self._set_info("已撤销")

    def _on_toggle_ground(self, checked):
        self._update_visualization()

    # ======================================================================
    #  可视化
    # ======================================================================

    def _update_visualization(self):
        sc = self._scene.scene
        # --- 地面 (按 plane_id 染色) ---
        sc.remove_geometry("ground")
        if self.ground_pts is not None and len(self.ground_pts) > 0 and \
           self._show_ground_check.checked:
            gnd = self._build_ground_cloud()
            if gnd is not None:
                sc.add_geometry("ground", gnd, self._mat_small)

        # --- 障碍物 (聚类染色) ---
        sc.remove_geometry("obstacles")
        if self.obstacle_pts is not None and len(self.obstacle_pts) > 0 and \
           self.obstacle_labels is not None:
            cloud = self._build_obstacle_cloud()
            if cloud is not None and cloud.has_points():
                sc.add_geometry("obstacles", cloud, self._mat)

        # --- 高亮 ---
        self._update_highlight()

        sc.remove_geometry("coord_frame")
        if self.voxel_pcd is not None:
            bbox = self.voxel_pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            extent = bbox.get_max_extent()
            size = max(extent * 0.15, 0.3)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            sc.add_geometry("coord_frame", frame, rendering.MaterialRecord())

        self._scene.force_redraw()

    def _update_highlight(self):
        sc = self._scene.scene
        sc.remove_geometry("highlight")
        if self.selected_cluster < 0 or self.obstacle_labels is None:
            return
        mask = (self.obstacle_labels == self.selected_cluster) & self.obstacle_mask
        if not np.any(mask):
            return
        pts = self.obstacle_pts[mask]
        hl = o3d.geometry.PointCloud()
        hl.points = o3d.utility.Vector3dVector(pts)
        hl.paint_uniform_color(_HIGHLIGHT_COLOR)
        sc.add_geometry("highlight", hl, self._mat_hl)

    def _build_obstacle_cloud(self):
        visible = self.obstacle_mask
        if not np.any(visible):
            return None
        pts = self.obstacle_pts[visible]
        labels = self.obstacle_labels[visible]
        colors = np.zeros((len(pts), 3))
        for i, lbl in enumerate(labels):
            if lbl < 0:
                colors[i] = _NOISE_COLOR
            else:
                colors[i] = _PALETTE[lbl % len(_PALETTE)]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    def _build_ground_cloud(self):
        if self.ground_pts is None or len(self.ground_pts) == 0:
            return None
        colors = np.zeros((len(self.ground_pts), 3))
        for i, pid in enumerate(self.ground_plane_ids):
            if pid >= 0:
                colors[i] = _PALETTE[pid % len(_PALETTE)]
            else:
                colors[i] = _GROUND_REMAIN_COLOR
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(self.ground_pts)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    def _build_merged_cloud(self):
        """返回 ground + 剩余 obstacles 合并的点云."""
        parts = []
        gnd = self._build_ground_cloud()
        if gnd is not None:
            parts.append(gnd)
        if self.obstacle_pts is not None and np.any(self.obstacle_mask):
            cloud = self._build_obstacle_cloud()
            if cloud is not None:
                parts.append(cloud)
        if not parts:
            return None
        merged = parts[0]
        for p in parts[1:]:
            merged += p
        return merged

    # ======================================================================
    #  辅助
    # ======================================================================

    def _update_info(self):
        if self.original_pcd is None:
            self._lbl_selected.text = "未选择"
            self._set_info("就绪")
            return

        total_orig = len(self.original_pcd.points)
        total_vox = len(self.voxel_pcd.points) if self.voxel_pcd else 0
        n_ground = len(self.ground_pts) if self.ground_pts is not None else 0
        n_ground_planes = 0
        if self.ground_plane_ids is not None and len(self.ground_plane_ids) > 0:
            n_ground_planes = len(set(self.ground_plane_ids) - {-1})
        n_obs = int(self.obstacle_mask.sum()) if self.obstacle_mask is not None else 0
        n_del = 0
        if self.obstacle_mask is not None and self.obstacle_labels is not None:
            n_del = len(self.obstacle_mask) - int(self.obstacle_mask.sum())

        n_clusters = 0
        if self.obstacle_labels is not None and len(self.obstacle_labels) > 0:
            unique = set(self.obstacle_labels[self.obstacle_mask])
            n_clusters = len(unique - {-1})
        n_noise = 0
        if self.obstacle_labels is not None and self.obstacle_mask is not None:
            n_noise = int(((self.obstacle_labels == -1) & self.obstacle_mask).sum())

        self._set_info(
            f"原始 {total_orig} → 处理 {total_vox}  "
            f"| 地面 {n_ground} ({n_ground_planes}面)  "
            f"| 块 {n_clusters} (+{n_noise}噪声)  "
            f"| 已删 {n_del}  "
            f"| 剩余 {n_obs}"
        )

    def _set_info(self, text):
        self._lbl_info.text = text

    def _reset_state(self):
        self.voxel_pcd = None
        self.ground_pts = None
        self.ground_plane_ids = None
        self.obstacle_pts = None
        self.obstacle_labels = None
        self.obstacle_mask = None
        self.selected_cluster = -1
        self.undo_stack.clear()
        self._scene.scene.clear_geometry()
        self._lbl_selected.text = "未选择"

    # ======================================================================
    #  运行
    # ======================================================================

    def run(self):
        self.app.run()


# =======================================================================

if __name__ == "__main__":
    app = PointCloudCleaner()
    app.run()
