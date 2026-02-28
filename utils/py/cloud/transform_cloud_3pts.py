#!/usr/bin/env python3
"""
点云对齐工具 — 三点法定义坐标系

通过在点云上点击三个点来建立新的坐标系并对齐点云:
  P1 (红): 新原点 (地板角落)
  P2 (绿): X 轴正方向 (原点 → 该点)
  P3 (蓝): Y 轴正方向 (原点 → 该点)

正交化策略:
  1. Z = normalize(X_raw × Y_raw)            — 法线由两向量叉积确定
  2. 将 X_raw / Y_raw 分别投影到 ⊥Z 平面
  3. X_c1 = X_proj,  X_c2 = Y_proj × Z       — 两种 X 估计
  4. X = normalize(X_c1 + X_c2)               — 取平均, 误差均分到 X/Y
  5. Y = Z × X

用法:
  python align_cloud_3point.py [点云文件]
"""

import argparse
import numpy as np
import os

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


# ─── helpers ────────────────────────────────────────────────────────────────

def _fmt_mat4(m):
    """4×4 矩阵 → 格式化字符串"""
    lines = []
    for row in m:
        lines.append("  [" + ", ".join(f"{v:12.6f}" for v in row) + "]")
    return "\n".join(lines)


def _rotation_to_euler_deg(R):
    """旋转矩阵 → XYZ 内旋欧拉角 (度)"""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees(rx), np.degrees(ry), np.degrees(rz)


# ─── App ────────────────────────────────────────────────────────────────────

class PointCloudAlignApp:
    """三点法交互式点云对齐工具"""

    PT_COLORS = [
        [1.0, 0.1, 0.1],  # P1 红
        [0.1, 1.0, 0.1],  # P2 绿
        [0.2, 0.6, 1.0],  # P3 蓝
    ]
    PT_NAMES = ["P1(原点)", "P2(X轴)", "P3(Y轴)"]

    # ── 初始化 ──────────────────────────────────────────────────────────────

    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        font = gui.FontDescription(font_path)
        font.add_typeface_for_language(font_path, "zh_all")
        gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, font)
        self.window = self.app.create_window("点云对齐 — 三点法", 1400, 900)

        # 3‑D 场景
        self.scene_w = gui.SceneWidget()
        self.scene_w.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_w.scene.set_background([0.12, 0.12, 0.12, 1.0])
        self.scene_w.scene.show_axes(False)

        # 状态
        self.original_cloud = None   # 原始点云 (不变)
        self.display_cloud = None    # 当前显示 & 变换用
        self.picked_pts = []         # 最多 3 个 np.ndarray
        self.picking = False
        self.flip_z = False
        self.sphere_r = 0.05
        self.axis_size = 1.0
        self.cum_T = np.eye(4)       # 累计变换矩阵

        # 材质
        self.cloud_mat = rendering.MaterialRecord()
        self.cloud_mat.shader = "defaultUnlit"
        self.cloud_mat.point_size = 2.0

        self.lit_mat = rendering.MaterialRecord()
        self.lit_mat.shader = "defaultLit"

        # UI 面板
        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self._build_panel()

        # 布局
        self.window.add_child(self.scene_w)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._layout)

        # 鼠标
        self.scene_w.set_on_mouse(self._on_mouse)

        # 初始坐标轴
        self._refresh_axis()

    # ── UI 构建 ─────────────────────────────────────────────────────────────

    def _build_panel(self):
        add = self.panel.add_child
        sep = lambda: add(gui.Label(""))

        # ─ 文件 ─
        add(gui.Label("-- 文件 --"))
        hf = gui.Horiz(4)
        b = gui.Button("打开点云"); b.set_on_clicked(self._dlg_open); hf.add_child(b)
        b = gui.Button("保存点云"); b.set_on_clicked(self._dlg_save); hf.add_child(b)
        add(hf)
        self.lbl_file = gui.Label("未加载文件")
        add(self.lbl_file)
        sep()

        # ─ 使用说明 ─
        add(gui.Label("-- 三点对齐 --"))
        for t in [
            "1. 加载点云, 点击 [开始选点]",
            "2. Ctrl+左键 依次选 3 个点:",
            "   P1: 新原点 (红色标记)",
            "   P2: X 轴方向 (绿色标记)",
            "   P3: Y 轴方向 (蓝色标记)",
            "3. 确认预览坐标系后点 [应用对齐]",
        ]:
            add(gui.Label(t))
        sep()

        # ─ 选点控制 ─
        hp = gui.Horiz(4)
        self.btn_pick = gui.Button("开始选点")
        self.btn_pick.set_on_clicked(self._toggle_pick)
        hp.add_child(self.btn_pick)
        b = gui.Button("撤销"); b.set_on_clicked(self._undo_pick); hp.add_child(b)
        b = gui.Button("清除"); b.set_on_clicked(self._clear_picks); hp.add_child(b)
        add(hp)

        # 坐标显示
        self.lbl_pts = []
        for i in range(3):
            lbl = gui.Label(f"  {self.PT_NAMES[i]}: --")
            self.lbl_pts.append(lbl)
            add(lbl)
        sep()

        # ─ 翻转 Z ─
        self.cb_flip = gui.Checkbox("翻转 Z 轴 (若 Z 方向反了)")
        self.cb_flip.checked = False
        self.cb_flip.set_on_checked(self._on_flip_z)
        add(self.cb_flip)
        sep()

        # ─ 应用 / 重置 ─
        ha = gui.Horiz(4)
        b = gui.Button("应用对齐"); b.set_on_clicked(self._apply); ha.add_child(b)
        b = gui.Button("重置原始"); b.set_on_clicked(self._reset_cloud); ha.add_child(b)
        add(ha)
        self.lbl_status = gui.Label("就绪")
        add(self.lbl_status)
        sep()

        # ─ 显示设置 ─
        add(gui.Label("-- 显示设置 --"))
        for label, lo, hi, val, cb in [
            ("点大小",   1.0,  10.0, 2.0,  self._cb_pt_size),
            ("标记半径", 0.005, 1.0, 0.05, self._cb_sphere),
            ("坐标轴",   0.1,  10.0, 1.0,  self._cb_axis),
        ]:
            h = gui.Horiz(4)
            h.add_child(gui.Label(f"{label}:"))
            sld = gui.Slider(gui.Slider.DOUBLE)
            sld.set_limits(lo, hi)
            sld.double_value = val
            sld.set_on_value_changed(cb)
            h.add_child(sld)
            add(h)
            if label == "标记半径":
                self.sld_sphere = sld

    # ── 布局 ────────────────────────────────────────────────────────────────

    def _layout(self, _ctx):
        r = self.window.content_rect
        pw = 360
        self.scene_w.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def _status(self, msg):
        self.lbl_status.text = msg

    # ── 文件 I/O ────────────────────────────────────────────────────────────

    def _dlg_open(self):
        d = gui.FileDialog(gui.FileDialog.OPEN, "选择点云", self.window.theme)
        d.add_filter(".pcd .ply .xyz .pts", "点云文件 (*.pcd *.ply …)")
        d.set_on_cancel(lambda: self.window.close_dialog())
        d.set_on_done(self._do_load)
        self.window.show_dialog(d)

    def _do_load(self, path):
        self.window.close_dialog()
        self._load(path)

    def _load(self, path):
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            self._status("读取失败: 点云为空")
            return
        self.original_cloud = pcd
        self.display_cloud = o3d.geometry.PointCloud(pcd)
        self.cum_T = np.eye(4)
        self._clear_picks()
        self._refresh_cloud()
        self._fit_camera()
        n = len(pcd.points)
        self.lbl_file.text = f"{os.path.basename(path)} ({n} 点)"
        self._status("已加载")
        print(f"[load] {path}  ({n} points)")
        # 自动标记半径
        ext = pcd.get_axis_aligned_bounding_box().get_extent()
        self.sphere_r = float(np.max(ext)) * 0.008
        self.sld_sphere.double_value = self.sphere_r

    def load_file(self, path):
        """命令行直接打开"""
        self._load(path)

    def _dlg_save(self):
        if self.display_cloud is None:
            self._status("无点云可保存"); return
        d = gui.FileDialog(gui.FileDialog.SAVE, "保存对齐后的点云", self.window.theme)
        d.add_filter(".pcd", "PCD 格式 (*.pcd)")
        d.add_filter(".ply", "PLY 格式 (*.ply)")
        d.set_on_cancel(lambda: self.window.close_dialog())
        d.set_on_done(self._do_save)
        self.window.show_dialog(d)

    def _do_save(self, path):
        self.window.close_dialog()
        o3d.io.write_point_cloud(path, self.display_cloud)
        self._status(f"已保存 {os.path.basename(path)}")
        self._print_result(path)

    def _print_result(self, path):
        T = self.cum_T
        R, t = T[:3, :3], T[:3, 3]
        rx, ry, rz = _rotation_to_euler_deg(R)
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  已保存至: {path}")
        print(sep)
        print("累计变换矩阵 (original → aligned):")
        print(_fmt_mat4(T))
        print("\n逆变换矩阵 (aligned → original):")
        print(_fmt_mat4(np.linalg.inv(T)))
        print(f"\n平移 (translation): [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
        print(f"旋转 (XYZ euler deg): [{rx:.4f}, {ry:.4f}, {rz:.4f}]")
        print(sep + "\n")

    # ── 场景刷新 ────────────────────────────────────────────────────────────

    def _refresh_cloud(self):
        s = self.scene_w.scene
        if s.has_geometry("cloud"):
            s.remove_geometry("cloud")
        if self.display_cloud is not None:
            s.add_geometry("cloud", self.display_cloud, self.cloud_mat)

    def _fit_camera(self):
        if self.display_cloud and len(self.display_cloud.points) > 0:
            bb = self.display_cloud.get_axis_aligned_bounding_box()
            self.scene_w.setup_camera(60, bb, bb.get_center())

    def _refresh_axis(self):
        s = self.scene_w.scene
        if s.has_geometry("origin_axis"):
            s.remove_geometry("origin_axis")
        ax = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_size)
        s.add_geometry("origin_axis", ax, rendering.MaterialRecord())

    def _remove_geom(self, name):
        if self.scene_w.scene.has_geometry(name):
            self.scene_w.scene.remove_geometry(name)

    # ── 选点 ────────────────────────────────────────────────────────────────

    def _toggle_pick(self):
        self.picking = not self.picking
        self.btn_pick.text = "停止选点" if self.picking else "开始选点"
        if self.picking:
            self._status(f"Ctrl+左键选点 ({len(self.picked_pts)}/3)")
        else:
            self._status("选点模式已关闭")

    def _undo_pick(self):
        if not self.picked_pts:
            return
        i = len(self.picked_pts) - 1
        self.picked_pts.pop()
        self._remove_geom(f"sphere_{i}")
        self._remove_preview()
        self._refresh_labels()
        self._status(f"已撤销 → {len(self.picked_pts)}/3")

    def _clear_picks(self):
        for i in range(len(self.picked_pts)):
            self._remove_geom(f"sphere_{i}")
        self.picked_pts.clear()
        self._remove_preview()
        self._refresh_labels()

    def _refresh_labels(self):
        for i in range(3):
            if i < len(self.picked_pts):
                p = self.picked_pts[i]
                self.lbl_pts[i].text = (
                    f"  {self.PT_NAMES[i]}: "
                    f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
            else:
                self.lbl_pts[i].text = f"  {self.PT_NAMES[i]}: --"

    def _add_sphere(self, pt, idx):
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=self.sphere_r)
        sp.translate(pt)
        sp.paint_uniform_color(self.PT_COLORS[idx])
        sp.compute_vertex_normals()
        name = f"sphere_{idx}"
        self._remove_geom(name)
        self.scene_w.scene.add_geometry(name, sp, self.lit_mat)

    # ── 鼠标回调 (深度缓冲 + KDTree 拾取) ──────────────────────────────────

    def _on_mouse(self, event):
        if (not self.picking
                or self.display_cloud is None
                or len(self.picked_pts) >= 3):
            return gui.Widget.EventCallbackResult.IGNORED

        if not (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.is_modifier_down(gui.KeyModifier.CTRL)):
            return gui.Widget.EventCallbackResult.IGNORED

        # 屏幕坐标 (相对于 scene widget)
        sx = event.x - self.scene_w.frame.x
        sy = event.y - self.scene_w.frame.y

        def _depth_cb(depth_img):
            arr = np.asarray(depth_img)
            ix, iy = int(round(sx)), int(round(sy))

            # 在 11×11 窗口内搜索最近有效深度 (点云稀疏时容易点到间隙)
            HALF = 5
            y0, y1 = max(0, iy - HALF), min(arr.shape[0], iy + HALF + 1)
            x0, x1 = max(0, ix - HALF), min(arr.shape[1], ix + HALF + 1)
            patch = arr[y0:y1, x0:x1]
            valid = patch < 1.0
            if not np.any(valid):
                return  # 未命中

            ys, xs = np.where(valid)
            ys += y0; xs += x0
            best = np.argmin((ys - iy) ** 2 + (xs - ix) ** 2)
            bx, by = int(xs[best]), int(ys[best])
            depth = float(arr[by, bx])

            # 反投影到世界坐标
            world = self.scene_w.scene.camera.unproject(
                bx, by, depth,
                self.scene_w.frame.width,
                self.scene_w.frame.height)
            world_np = np.asarray(world).flatten()[:3]

            # 吸附到最近的点云点
            tree = o3d.geometry.KDTreeFlann(self.display_cloud)
            k, idxs, _ = tree.search_knn_vector_3d(world_np, 1)
            if k > 0:
                nearest = np.asarray(self.display_cloud.points)[idxs[0]].copy()
            else:
                nearest = world_np.copy()

            gui.Application.instance.post_to_main_thread(
                self.window, lambda p=nearest: self._on_picked(p))

        self.scene_w.scene.scene.render_to_depth_image(_depth_cb)
        return gui.Widget.EventCallbackResult.HANDLED

    def _on_picked(self, pt):
        i = len(self.picked_pts)
        if i >= 3:
            return
        self.picked_pts.append(pt)
        self._add_sphere(pt, i)
        self._refresh_labels()
        n = len(self.picked_pts)
        print(f"[pick] {self.PT_NAMES[i]}: "
              f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
        if n == 3:
            self._show_preview()
            self._status("3 点已选 → 确认预览后 [应用对齐]")
        else:
            self._status(f"Ctrl+左键选点 ({n}/3)")

    # ── 计算变换矩阵 ───────────────────────────────────────────────────────

    def _compute_transform(self):
        """
        三点 → 4×4 齐次变换矩阵

        正交化策略 (误差均分):
          Z = normalize(X_raw × Y_raw)
          X_c1 = project(X_raw, ⊥Z)
          X_c2 = Y_proj × Z            (从 Y 约束得到的 X 估计)
          X = normalize(X_c1 + X_c2)   (角平分, 误差对半)
          Y = Z × X
        """
        p1 = np.asarray(self.picked_pts[0], dtype=np.float64)
        p2 = np.asarray(self.picked_pts[1], dtype=np.float64)
        p3 = np.asarray(self.picked_pts[2], dtype=np.float64)

        origin = p1
        x_raw = p2 - origin
        y_raw = p3 - origin

        lx, ly = np.linalg.norm(x_raw), np.linalg.norm(y_raw)
        if lx < 1e-8 or ly < 1e-8:
            print("[error] 选点距离过近, 无法计算"); return None
        x_raw /= lx
        y_raw /= ly

        # 1) Z 轴 = 两方向叉积
        z_axis = np.cross(x_raw, y_raw)
        zn = np.linalg.norm(z_axis)
        if zn < 1e-8:
            print("[error] 三点近似共线, 无法确定平面"); return None
        z_axis /= zn

        # 可选翻转
        if self.flip_z:
            z_axis = -z_axis

        # 2) 投影到 ⊥Z 平面
        x_proj = x_raw - x_raw.dot(z_axis) * z_axis
        x_proj /= np.linalg.norm(x_proj)
        y_proj = y_raw - y_raw.dot(z_axis) * z_axis
        y_proj /= np.linalg.norm(y_proj)

        # 3) 两种 X 估计
        x_c1 = x_proj                          # 直接从 X 选点
        x_c2 = np.cross(y_proj, z_axis)        # 从 Y 选点推导
        x_c2 /= np.linalg.norm(x_c2)

        # 4) 平均
        x_axis = x_c1 + x_c2
        x_axis /= np.linalg.norm(x_axis)

        # 5) Y = Z × X
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # 偏差报告
        phi = np.degrees(np.arccos(np.clip(
            (p2 - origin).dot(p3 - origin) / (lx * ly), -1, 1)))
        print(f"[info] X-Y 夹角 = {phi:.2f}° (理想 90°, 偏差 {abs(phi - 90):.2f}°)")

        # R 列向量 = 新坐标轴在旧坐标系中的方向
        R = np.column_stack([x_axis, y_axis, z_axis])

        # T_new = R^T (p - origin)
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ origin
        return T

    # ── 预览 ────────────────────────────────────────────────────────────────

    def _show_preview(self):
        T = self._compute_transform()
        if T is None:
            self._status("计算失败, 请重新选点"); return

        R_inv = T[:3, :3].T  # 新坐标轴在旧坐标系里的方向
        origin = np.asarray(self.picked_pts[0])

        # 预览坐标系
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.axis_size * 1.5)
        M = np.eye(4)
        M[:3, :3] = R_inv
        M[:3, 3] = origin
        frame.transform(M)
        self._remove_geom("preview_frame")
        self.scene_w.scene.add_geometry(
            "preview_frame", frame, rendering.MaterialRecord())

        # 连线 origin→P2 / origin→P3
        pts = o3d.utility.Vector3dVector([
            origin,
            np.asarray(self.picked_pts[1]),
            np.asarray(self.picked_pts[2]),
        ])
        ls = o3d.geometry.LineSet()
        ls.points = pts
        ls.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2]])
        ls.colors = o3d.utility.Vector3dVector(
            [self.PT_COLORS[1], self.PT_COLORS[2]])
        lm = rendering.MaterialRecord()
        lm.shader = "unlitLine"
        lm.line_width = 3.0
        self._remove_geom("preview_lines")
        self.scene_w.scene.add_geometry("preview_lines", ls, lm)

    def _remove_preview(self):
        for n in ("preview_frame", "preview_lines"):
            self._remove_geom(n)

    # ── 应用 / 重置 ────────────────────────────────────────────────────────

    def _apply(self):
        if self.display_cloud is None:
            self._status("无点云"); return
        if len(self.picked_pts) < 3:
            self._status(f"需要 3 个点 (当前 {len(self.picked_pts)}/3)"); return

        T = self._compute_transform()
        if T is None:
            self._status("计算失败"); return

        self.display_cloud.transform(T)
        self.cum_T = T @ self.cum_T

        self._clear_picks()
        self._refresh_cloud()
        self._fit_camera()
        self._refresh_axis()
        self._status("对齐完成 ✓")

        print("\n--- 本次变换 ---")
        print(_fmt_mat4(T))
        print("--- 累计变换 ---")
        print(_fmt_mat4(self.cum_T))

    def _reset_cloud(self):
        if self.original_cloud is None:
            return
        self.display_cloud = o3d.geometry.PointCloud(self.original_cloud)
        self.cum_T = np.eye(4)
        self._clear_picks()
        self._refresh_cloud()
        self._fit_camera()
        self._refresh_axis()
        self._status("已重置为原始点云")

    # ── 翻转 Z ─────────────────────────────────────────────────────────────

    def _on_flip_z(self, checked):
        self.flip_z = checked
        if len(self.picked_pts) == 3:
            self._show_preview()

    # ── 显示回调 ────────────────────────────────────────────────────────────

    def _cb_pt_size(self, v):
        self.cloud_mat.point_size = v
        self._refresh_cloud()

    def _cb_sphere(self, v):
        self.sphere_r = v
        for i, pt in enumerate(self.picked_pts):
            self._add_sphere(pt, i)

    def _cb_axis(self, v):
        self.axis_size = v
        self._refresh_axis()
        if len(self.picked_pts) == 3:
            self._show_preview()

    # ── 运行 ────────────────────────────────────────────────────────────────

    def run(self):
        self.app.run()


# ─── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="三点法点云对齐工具 — 交互式选 3 点定义新坐标系")
    parser.add_argument("file", nargs="?", default=None,
                        help="可选: 直接打开的点云文件路径 (*.pcd, *.ply …)")
    args = parser.parse_args()

    viewer = PointCloudAlignApp()
    if args.file and os.path.isfile(args.file):
        viewer.load_file(args.file)
    viewer.run()
