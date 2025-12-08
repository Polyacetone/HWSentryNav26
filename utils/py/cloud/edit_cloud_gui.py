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

        self._settings_panel.add_child(file_group)
        self._settings_panel.add_child(self._crop_group)

        # 4. 组装窗口布局
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
            
            # 1. 清空历史记录并保存初始状态
            self.history.clear()
            self._save_state() # 保存初始加载状态
            
            # 2. 更新 3D 场景
            self._scene.scene.clear_geometry()
            self._scene.scene.add_geometry(self.pcd_name, self.pcd, self.mat)
            
            # 3. 设置相机
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())
            
            # 4. 初始化裁剪工具 UI 和边界
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
            state = copy.deepcopy(self.pcd)
            self.history.append(state)
            
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
        previous_pcd = self.history.pop()
        self.pcd = previous_pcd
        
        # 2. 更新场景
        if self._scene.scene.has_geometry(self.pcd_name):
            self._scene.scene.remove_geometry(self.pcd_name)
            
        self._scene.scene.add_geometry(self.pcd_name, self.pcd, self.mat)
        
        # 3. 重新初始化裁剪工具的边界，因为点云边界可能变了
        self._setup_crop_ui_and_bounds(reset_bounds_to_pcd=True)
        
        # 4. 更新信息和按钮状态
        self._update_info_text()
        self._undo_btn.enabled = bool(self.history)
        self._info_label.text = f"Undo successful. Points: {len(self.pcd.points)}"
        self._scene.force_redraw()

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
            slider.set_limits(limits_min[axis_idx], limits_max[axis_idx])

            if reset_bounds_to_pcd:
                # 重置选框范围到点云的完整边界
                val = max_b[axis_idx] if 'max' in key else min_b[axis_idx]
                self.box_bounds[key] = val
                slider.double_value = val
            # 否则（如用户拖动滑块时），保留滑块当前值

        self._update_wireframe_box()

    def _on_slider_changed(self, key, val):
        """滑块拖动时的回调"""
        self.box_bounds[key] = val
        self._update_wireframe_box()

    def _update_wireframe_box(self):
        """绘制红色的 AABB 线框"""
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
        
        # 1. 执行裁剪
        new_pcd = self.pcd.select_by_index(indices, invert=not keep_inside)
        
        # 2. **致命错误修复**：检查点云是否为空，防止渲染空点云导致的程序崩溃
        if not new_pcd.has_points():
            print("Result is empty point cloud. Reverting state.")
            # 恢复刚刚保存的状态（撤销本次操作）
            self.history.pop()
            self._info_label.text = "Error: Result is empty. Operation cancelled."
            self._undo_btn.enabled = bool(self.history)
            return
            
        # 3. 更新引用
        self.pcd = new_pcd
        
        # 4. 安全更新场景
        if self._scene.scene.has_geometry(self.pcd_name):
            self._scene.scene.remove_geometry(self.pcd_name)
            
        self._scene.scene.add_geometry(self.pcd_name, self.pcd, self.mat)
        
        self._update_info_text()
        self._scene.force_redraw()

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
            o3d.io.write_point_cloud(path, self.pcd)
            self._info_label.text = f"Saved to {path}"

    def run(self):
        self.app.run()

if __name__ == "__main__":
    gui_app = PointCloudEditor()
    gui_app.run()