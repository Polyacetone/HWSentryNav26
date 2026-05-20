import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import copy

class PointCloudEditor:
    def __init__(self):
        # 1. 初始化 Application
        self.app = gui.Application.instance
        self.app.initialize()
        
        self.window = self.app.create_window("Open3D Point Cloud Editor", 1280, 800)
        w = self.window
        em = w.theme.font_size
        
        # 2. 3D 场景控件 (主要工作区 - 左侧)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([0.1, 0.1, 0.1, 1.0])
        
        # 3. 右侧控制面板 (Sidebar - 右侧)
        # Margins(left, top, right, bottom)
        self._settings_panel = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        
        # === 文件操作 ===
        file_group = gui.CollapsableVert("File Operations", 0.25 * em, gui.Margins(em, 0, 0, 0))
        load_btn = gui.Button("Load Point Cloud")
        load_btn.set_on_clicked(self._on_load_clicked)
        save_btn = gui.Button("Save Point Cloud")
        save_btn.set_on_clicked(self._on_save_clicked)
        
        file_group.add_child(load_btn)
        file_group.add_child(gui.VGrid(1, 0.25 * em)) 
        file_group.add_child(save_btn)
        
        # === 裁剪/编辑操作 ===
        self._crop_group = gui.CollapsableVert("Crop & Edit", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._crop_group.set_is_open(True)

        self._info_label = gui.Label("Status: Waiting for load...")
        self._info_label.text_color = gui.Color(0.8, 0.8, 0.8)
        
        self._show_box_check = gui.Checkbox("Show Crop Box")
        self._show_box_check.checked = True
        self._show_box_check.set_on_checked(self._on_show_box_changed)
        
        # 新增撤销按钮
        self._undo_btn = gui.Button("Undo (Ctrl+Z)")
        self._undo_btn.enabled = False # 默认禁用
        self._undo_btn.set_on_clicked(self._undo_operation)
        
        self._del_inside_btn = gui.Button("Delete INSIDE Box")
        self._del_inside_btn.set_on_clicked(self._on_delete_inside)
        
        self._keep_inside_btn = gui.Button("Keep INSIDE Box")
        self._keep_inside_btn.set_on_clicked(self._on_keep_inside)

        self._crop_group.add_child(self._info_label)
        self._crop_group.add_child(gui.VGrid(1, 0.5 * em))
        self._crop_group.add_child(self._show_box_check)
        self._crop_group.add_child(self._undo_btn) # 添加到按钮组
        self._crop_group.add_child(self._del_inside_btn)
        self._crop_group.add_child(self._keep_inside_btn)
        self._crop_group.add_child(gui.Label("Adjust Box Bounds:"))
        
        # === 预创建滑块 (修复滑块不显示问题) ===
        self.sliders = {}
        slider_names = [
            ("min_x", "Min X"), ("max_x", "Max X"),
            ("min_y", "Min Y"), ("max_y", "Max Y"),
            ("min_z", "Min Z"), ("max_z", "Max Z")
        ]
        
        for key, label_text in slider_names:
            container = gui.Vert()
            lbl = gui.Label(label_text)
            sld = gui.Slider(gui.Slider.DOUBLE)
            sld.set_limits(0.0, 1.0)
            sld.enabled = False
            
            def on_val_changed(val, k=key):
                self._on_slider_changed(k, val)
                
            sld.set_on_value_changed(on_val_changed)
            container.add_child(lbl)
            container.add_child(sld)
            self._crop_group.add_child(container)
            self.sliders[key] = sld

        # === 轴选择（用于方向键微调） ===
        self._crop_group.add_child(gui.Label("Axis for Arrow Keys:"))
        self._axis_combo = gui.Combobox()
        for axis_item in ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z"]:
            self._axis_combo.add_item(axis_item)
        self._axis_combo.selected_text = "min_x"
        self._axis_combo.set_on_selection_changed(self._on_axis_selected)
        self._crop_group.add_child(self._axis_combo)

        # === 显示设置 ===
        self._display_group = gui.CollapsableVert("Display", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._display_group.set_is_open(True)

        self._color_height_check = gui.Checkbox("Color by Height")
        self._color_height_check.set_on_checked(self._on_color_height_changed)

        size_lbl = gui.Label("Point Size")
        self._size_slider = gui.Slider(gui.Slider.INT)
        self._size_slider.set_limits(1, 20)
        self._size_slider.int_value = 3
        self._size_slider.set_on_value_changed(self._on_size_changed)

        self._display_group.add_child(self._color_height_check)
        self._display_group.add_child(gui.VGrid(1, 0.25 * em))
        self._display_group.add_child(size_lbl)
        self._display_group.add_child(self._size_slider)
        self._display_group.add_child(gui.VGrid(1, 0.25 * em))

        self._highlight_check = gui.Checkbox("Highlight Inside")
        self._highlight_check.set_on_checked(self._on_highlight_changed)
        self._display_group.add_child(self._highlight_check)

        self._settings_panel.add_child(file_group)
        self._settings_panel.add_child(self._crop_group)
        self._settings_panel.add_child(self._display_group)

        # 4. 键盘回调（方向键微调滑块）
        w.set_on_key(self._on_key)

        # 5. 组装窗口布局
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # 内部状态数据
        self.pcd = None
        self.pcd_name = "cloud"
        self.box_name = "bbox"
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3.0
        self.box_bounds = {"min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 1}
        self._base_colors = None   # numpy array, same length as pcd points
        self._height_color_active = False
        self._highlight_name = "cloud_highlight"
        self._highlight_active = False
        self._selected_axis = "min_x"
        self._slider_limits = {}   # 每个滑块的 (lo, hi)，用于方向键步长
        
        # --- 撤销功能相关 ---
        self.history = []
        self.MAX_HISTORY_SIZE = 10 

    def _on_layout(self, layout_context):
        """定义窗口改变大小时的布局逻辑：左3D，右侧边栏"""
        r = self.window.content_rect
        sidebar_width = 20 * layout_context.theme.font_size
        sidebar_width = min(r.width - 10, sidebar_width)
        
        # 3D 场景占据左侧
        self._scene.frame = gui.Rect(r.x, r.y, r.width - sidebar_width, r.height)
        # 设置面板占据右侧
        self._settings_panel.frame = gui.Rect(r.x + r.width - sidebar_width, r.y, sidebar_width, r.height)

    # ---------------- 文件加载与状态更新 ----------------

    def _on_load_clicked(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select Point Cloud", self.window.theme)
        dlg.add_filter(".pcd .ply .xyz", "Point Cloud Files")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_load_dialog_done(self, path):
        self.window.close_dialog()
        try:
            pcd = o3d.io.read_point_cloud(path)
            if not pcd.has_points():
                self._info_label.text = "Error: Empty file."
                return
            
            self.pcd = pcd
            self._base_colors = np.asarray(self.pcd.colors).copy() if self.pcd.has_colors() else None
            
            # 1. 清空历史记录并保存初始状态
            self.history.clear()
            self._save_state() # 保存初始加载状态
            
            # 2. 应用高度染色（如果已激活，直接修改颜色，稍后统一 add_geometry）
            if self._height_color_active:
                z = np.asarray(self.pcd.points)[:, 2]
                self.pcd.colors = o3d.utility.Vector3dVector(
                    self._compute_height_colors(z))
            
            # 3. 更新 3D 场景
            self._scene.scene.clear_geometry()
            self._scene.scene.add_geometry(self.pcd_name, self.pcd, self.mat)
            
            # 4. 设置相机
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())
            
            # 5. 初始化裁剪工具 UI 和边界
            self._setup_crop_ui_and_bounds(reset_bounds_to_pcd=True)
            self._update_info_text()
            
            self._scene.force_redraw()
            
        except Exception as e:
            print(f"Load Error: {e}")
            self._info_label.text = "Load Error!"

    def _update_info_text(self):
        """ 更新点云数量显示 """
        if self.pcd:
            n = len(self.pcd.points)
            self._info_label.text = f"Points: {n}"

    # ---------------- 撤销逻辑 ----------------

    def _save_state(self):
        """保存当前点云状态到历史记录"""
        if self.pcd and self.pcd.has_points():
            # 使用深拷贝确保历史记录独立
            state_pcd = copy.deepcopy(self.pcd)
            state_base = self._base_colors.copy() if self._base_colors is not None else None
            state_bounds = copy.deepcopy(self.box_bounds)
            self.history.append((state_pcd, state_base, state_bounds))
            
            # 限制历史记录大小，移除最旧的状态
            if len(self.history) > self.MAX_HISTORY_SIZE:
                self.history.pop(0) 
                
            self._undo_btn.enabled = True
            
    def _undo_operation(self):
        """撤销上一步操作"""
        if not self.history:
            self._info_label.text = "No history to undo."
            self._undo_btn.enabled = False
            return
            
        # 1. 恢复上一个状态
        previous_pcd, prev_base, prev_bounds = self.history.pop()
        self.pcd = previous_pcd
        self._base_colors = prev_base
        self.box_bounds = copy.deepcopy(prev_bounds)
        
        # 2. 根据当前高度染色开关决定颜色
        if self._height_color_active:
            self._apply_height_coloring()
        else:
            self._restore_base_colors()
        
        # 3. 重新初始化裁剪工具的边界，恢复滑块位置
        self._setup_crop_ui_and_bounds(reset_bounds_to_pcd=False)
        for key, val in self.box_bounds.items():
            self.sliders[key].double_value = val
        
        # 4. 更新信息和按钮状态
        self._update_info_text()
        self._undo_btn.enabled = bool(self.history)
        self._info_label.text = f"Undo successful. Points: {len(self.pcd.points)}"

    # ---------------- 裁剪工具 UI/核心逻辑 ----------------
    
    def _setup_crop_ui_and_bounds(self, reset_bounds_to_pcd=True):
        """设置滑块的范围，并可选地重置滑块的当前值"""
        if not self.pcd: return

        min_b = self.pcd.get_min_bound()
        max_b = self.pcd.get_max_bound()
        
        span = max_b - min_b
        pad = span * 0.1 
        limits_min = min_b - pad
        limits_max = max_b + pad

        configs = [
            ('min_x', 0), ('max_x', 0),
            ('min_y', 1), ('max_y', 1),
            ('min_z', 2), ('max_z', 2)
        ]

        for key, axis_idx in configs:
            slider = self.sliders[key]
            slider.enabled = True
            # 设置滑块允许拖动的范围
            lo, hi = limits_min[axis_idx], limits_max[axis_idx]
            slider.set_limits(lo, hi)
            self._slider_limits[key] = (lo, hi)

            if reset_bounds_to_pcd:
                # 重置选框范围到点云的完整边界
                val = max_b[axis_idx] if 'max' in key else min_b[axis_idx]
                self.box_bounds[key] = val
                slider.double_value = val
            # 否则（如用户拖动滑块时），保留滑块当前值

        self._update_wireframe_box()

    def _on_slider_changed(self, key, val):
        """滑块拖动时的回调"""
        self._selected_axis = key
        self.box_bounds[key] = val
        self._update_wireframe_box()

    def _update_wireframe_box(self):
        """绘制红色的 AABB 线框，并更新高亮叠加层"""
        # 高亮叠加层依赖当前 box，必须最先更新
        self._update_highlight()

        if self._scene.scene.has_geometry(self.box_name):
            self._scene.scene.remove_geometry(self.box_name)
            
        if not self._show_box_check.checked:
            return

        min_pt = [self.box_bounds['min_x'], self.box_bounds['min_y'], self.box_bounds['min_z']]
        max_pt = [self.box_bounds['max_x'], self.box_bounds['max_y'], self.box_bounds['max_z']]
        
        final_min = np.minimum(min_pt, max_pt)
        final_max = np.maximum(min_pt, max_pt)

        bbox = o3d.geometry.AxisAlignedBoundingBox(final_min, final_max)
        bbox.color = [1.0, 0.0, 0.0]
        
        lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 2.0
        
        self._scene.scene.add_geometry(self.box_name, lines, mat)

    def _on_show_box_changed(self, is_checked):
        self._update_wireframe_box()

    # ---------------- Display 辅助 ----------------

    def _reload_geometry(self):
        """用当前 self.pcd / self.mat 刷新场景中的几何体"""
        if self.pcd is None:
            return
        if self._scene.scene.has_geometry(self.pcd_name):
            self._scene.scene.remove_geometry(self.pcd_name)
        self._scene.scene.add_geometry(self.pcd_name, self.pcd, self.mat)
        self._scene.force_redraw()

    @staticmethod
    def _compute_height_colors(z_values):
        """将 Z 值归一化后映射到 jet 风格色图"""
        z_min, z_max = z_values.min(), z_values.max()
        if z_max - z_min < 1e-10:
            return np.full((len(z_values), 3), 0.5)

        z_norm = (z_values - z_min) / (z_max - z_min)
        colors = np.zeros((len(z_values), 3))

        mask1 = z_norm <= 0.25
        t1 = z_norm[mask1] / 0.25
        colors[mask1, 0] = 0
        colors[mask1, 1] = t1
        colors[mask1, 2] = 1

        mask2 = (z_norm > 0.25) & (z_norm <= 0.5)
        t2 = (z_norm[mask2] - 0.25) / 0.25
        colors[mask2, 0] = t2
        colors[mask2, 1] = 1
        colors[mask2, 2] = 1 - t2

        mask3 = (z_norm > 0.5) & (z_norm <= 0.75)
        t3 = (z_norm[mask3] - 0.5) / 0.25
        colors[mask3, 0] = 1
        colors[mask3, 1] = 1 - t3
        colors[mask3, 2] = 0

        mask4 = z_norm > 0.75
        t4 = (z_norm[mask4] - 0.75) / 0.25
        colors[mask4, 0] = 1
        colors[mask4, 1] = 1 - t4
        colors[mask4, 2] = 0

        return colors

    def _apply_height_coloring(self):
        """根据高度计算并设置点云颜色"""
        if self.pcd is None or not self.pcd.has_points():
            return
        z = np.asarray(self.pcd.points)[:, 2]
        colors = self._compute_height_colors(z)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self._reload_geometry()

    def _restore_base_colors(self):
        """恢复为原始颜色（若无则设为灰色）"""
        if self.pcd is None or not self.pcd.has_points():
            return
        if self._base_colors is not None:
            self.pcd.colors = o3d.utility.Vector3dVector(self._base_colors.copy())
        else:
            n = len(self.pcd.points)
            self.pcd.colors = o3d.utility.Vector3dVector(np.full((n, 3), 0.5))
        self._reload_geometry()

    def _on_color_height_changed(self, checked):
        self._height_color_active = checked
        if checked:
            self._apply_height_coloring()
        else:
            self._restore_base_colors()

    def _on_size_changed(self, val):
        self.mat.point_size = float(val)
        self._reload_geometry()

    # ---------------- 高亮叠加层 ----------------

    def _update_highlight(self):
        """将框内点渲染为红色叠加层"""
        # 清除旧高亮
        if self._scene.scene.has_geometry(self._highlight_name):
            self._scene.scene.remove_geometry(self._highlight_name)

        if not self._highlight_active or self.pcd is None or not self.pcd.has_points():
            self._scene.force_redraw()
            return

        min_pt = [self.box_bounds['min_x'], self.box_bounds['min_y'], self.box_bounds['min_z']]
        max_pt = [self.box_bounds['max_x'], self.box_bounds['max_y'], self.box_bounds['max_z']]
        final_min = np.minimum(min_pt, max_pt)
        final_max = np.maximum(min_pt, max_pt)

        bbox = o3d.geometry.AxisAlignedBoundingBox(final_min, final_max)
        indices = bbox.get_point_indices_within_bounding_box(self.pcd.points)

        if len(indices) == 0:
            self._scene.force_redraw()
            return

        inside_pcd = self.pcd.select_by_index(indices)
        red = np.full((len(inside_pcd.points), 3), [1.0, 0.0, 0.0])
        inside_pcd.colors = o3d.utility.Vector3dVector(red)

        hl_mat = rendering.MaterialRecord()
        hl_mat.shader = "defaultUnlit"
        hl_mat.point_size = max(self.mat.point_size * 1.8, 5.0)
        self._scene.scene.add_geometry(self._highlight_name, inside_pcd, hl_mat)
        self._scene.force_redraw()

    def _on_highlight_changed(self, checked):
        self._highlight_active = checked
        self._update_highlight()

    # ---------------- 轴选择 & 方向键微调 ----------------

    def _on_axis_selected(self, text, idx):
        self._selected_axis = text

    def _on_key(self, event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return False
        if self.pcd is None or not self.pcd.has_points():
            return False

        axis = self._selected_axis
        slider = self.sliders.get(axis)
        limits = self._slider_limits.get(axis)
        if slider is None or limits is None:
            return False

        lo, hi = limits
        step = (hi - lo) * 0.005
        if step < 1e-8:
            return False

        if event.key == gui.KeyName.LEFT:
            val = max(lo, slider.double_value - step)
        elif event.key == gui.KeyName.RIGHT:
            val = min(hi, slider.double_value + step)
        else:
            return False

        slider.double_value = val
        self.box_bounds[axis] = val
        self._update_wireframe_box()
        return True

    # ---------------- 裁剪操作核心逻辑 ----------------
    def _crop_operation(self, keep_inside):
        """
        核心裁剪操作。先保存状态，再执行裁剪，并安全更新场景。
        """
        if not self.pcd: return
        
        # 0. 裁剪前保存当前状态（用于撤销）
        self._save_state() 
        
        min_pt = [self.box_bounds['min_x'], self.box_bounds['min_y'], self.box_bounds['min_z']]
        max_pt = [self.box_bounds['max_x'], self.box_bounds['max_y'], self.box_bounds['max_z']]
        final_min = np.minimum(min_pt, max_pt)
        final_max = np.maximum(min_pt, max_pt)
        
        bbox = o3d.geometry.AxisAlignedBoundingBox(final_min, final_max)
        indices = bbox.get_point_indices_within_bounding_box(self.pcd.points)
        
        # 1. 建立布尔掩码，便于同步裁剪 base_colors
        keep_mask = np.zeros(len(self.pcd.points), dtype=bool)
        keep_mask[indices] = True
        if not keep_inside:
            keep_mask = ~keep_mask
        
        # 2. 执行裁剪
        new_pcd = self.pcd.select_by_index(np.where(keep_mask)[0])
        
        # 3. **致命错误修复**：检查点云是否为空，防止渲染空点云导致的程序崩溃
        if not new_pcd.has_points():
            print("Result is empty point cloud. Reverting state.")
            # 恢复刚刚保存的状态（撤销本次操作）
            self.history.pop()
            self._info_label.text = "Error: Result is empty. Operation cancelled."
            self._undo_btn.enabled = bool(self.history)
            return
            
        # 4. 更新引用
        self.pcd = new_pcd
        if self._base_colors is not None:
            self._base_colors = self._base_colors[keep_mask].copy()
        
        # 5. 根据当前高度染色开关重新着色
        if self._height_color_active:
            self._apply_height_coloring()
        else:
            self._restore_base_colors()
        
        self._update_wireframe_box()
        self._update_info_text()

    def _on_delete_inside(self):
        self._crop_operation(keep_inside=False)

    def _on_keep_inside(self):
        self._crop_operation(keep_inside=True)

    # ---------------- 保存 ----------------
    def _on_save_clicked(self):
        if not self.pcd: return
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Save File", self.window.theme)
        dlg.add_filter(".pcd", "PCD File")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._on_save_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog_done(self, path):
        self.window.close_dialog()
        if self.pcd:
            if not path.endswith(".pcd"):
                path += ".pcd"
            # 只保存几何信息，剥离所有颜色（避免写入高度染色数据）
            save_pcd = o3d.geometry.PointCloud()
            save_pcd.points = o3d.utility.Vector3dVector(np.asarray(self.pcd.points))
            o3d.io.write_point_cloud(path, save_pcd, write_ascii=False)
            self._info_label.text = f"Saved to {path}"

    def run(self):
        self.app.run()

if __name__ == "__main__":
    gui_app = PointCloudEditor()
    gui_app.run()