import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os

class PointCloudApp:
    def __init__(self):
        # 1. 初始化 App 和 窗口
        self.app = gui.Application.instance
        self.app.initialize()
        self.app.set_font(gui.Application.DEFAULT_FONT_ID, gui.FontDescription("UbuntuMono-R"))
        self.window = self.app.create_window("交互式点云变换 (Open3D)", 1024, 768)

        # 2. 创建 3D 场景部件
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([0.1, 0.1, 0.1, 1.0])

        # 3. 初始化数据
        # base_cloud 用于保存原始状态（未变换前），current_cloud 用于显示
        self.base_cloud = self.create_demo_point_cloud()
        self.current_cloud = o3d.geometry.PointCloud(self.base_cloud)
        
        # 材质设置
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3.0
        
        # 添加几何体到场景
        self.widget3d.scene.add_geometry("point_cloud", self.current_cloud, self.mat)
        
        # 添加坐标轴 (可调整大小)
        self.axis_size = 1.0
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_size, origin=[0, 0, 0])
        self.widget3d.scene.add_geometry("axis", self.axis, rendering.MaterialRecord())

        # 设置相机
        self.reset_camera()

        # 4. 创建右侧控制面板
        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        
        # 变换参数变量
        self.trans_x = 0.0
        self.trans_y = 0.0
        self.trans_z = 0.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0

        # --- UI 控件存储 (以便重置时修改显示的数值) ---
        self.inputs = {} 

        # 构建 UI
        self.add_file_ui()
        self.panel.add_child(gui.Label("")) # 空行
        self.add_transform_ui()

        # 5. 布局管理
        self.window.add_child(self.widget3d)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

    def create_demo_point_cloud(self):
        """如果没有加载文件，生成一个圆环作为演示"""
        mesh = o3d.geometry.TriangleMesh.create_torus(torus_radius=1.0, tube_radius=0.5)
        pcd = mesh.sample_points_poisson_disk(5000)
        # 上色
        points = np.asarray(pcd.points)
        colors = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def reset_camera(self):
        bbox = self.current_cloud.get_axis_aligned_bounding_box()
        self.widget3d.setup_camera(60, bbox, bbox.get_center())

    def add_file_ui(self):
        """添加文件操作区"""
        self.panel.add_child(gui.Label("File Operations"))
        
        # 水平布局放置两个按钮
        h_layout = gui.Horiz(10)
        
        btn_open = gui.Button("Open")
        btn_open.set_on_clicked(self.on_open_file)
        h_layout.add_child(btn_open)

        btn_save = gui.Button("Save")
        btn_save.set_on_clicked(self.on_save_file)
        h_layout.add_child(btn_save)
        
        self.panel.add_child(h_layout)

    def add_transform_ui(self):
        """添加变换控制区 (使用 NumberEdit)"""
        
        # 辅助函数：创建带标签的数字输入框
        def add_input_row(label_text, key, callback):
            h = gui.Horiz(5)
            h.add_child(gui.Label(label_text))
            
            # NumberEdit: 支持双精度浮点数，带上下调节功能
            num_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
            num_edit.double_value = 0.0
            num_edit.set_on_value_changed(callback)
            
            # 保存控件引用，方便后续代码修改它的值
            self.inputs[key] = num_edit
            h.add_child(num_edit)
            self.panel.add_child(h)

        self.panel.add_child(gui.Label("Translation"))
        add_input_row("X:", 'tx', self.on_trans_x)
        add_input_row("Y:", 'ty', self.on_trans_y)
        add_input_row("Z:", 'tz', self.on_trans_z)
        
        self.panel.add_child(gui.Label(""))

        self.panel.add_child(gui.Label("Rotation (deg)"))
        add_input_row("Roll  (X):", 'rx', self.on_rot_x)
        add_input_row("Pitch (Y):", 'ry', self.on_rot_y)
        add_input_row("Yaw   (Z):", 'rz', self.on_rot_z)

        self.panel.add_child(gui.Label(""))
        
        # Axis size slider
        self.panel.add_child(gui.Label("Axis"))
        h_axis = gui.Horiz(5)
        h_axis.add_child(gui.Label("Size:"))
        sld_axis = gui.Slider(gui.Slider.DOUBLE)
        sld_axis.set_limits(1.0, 10.0)
        # 使用 double_value 设置初始值（部分 Open3D 版本不支持 set_value）
        sld_axis.double_value = self.axis_size
        sld_axis.set_on_value_changed(self.on_axis_size)
        self.inputs['axis_size'] = sld_axis
        h_axis.add_child(sld_axis)
        self.panel.add_child(h_axis)
        
        btn_reset = gui.Button("Reset Transform")
        btn_reset.set_on_clicked(self.reset_transform)
        self.panel.add_child(btn_reset)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 300 # 面板宽度
        self.widget3d.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

    # --- 文件操作回调 ---
    def on_open_file(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "选择点云文件", self.window.theme)
        dlg.add_filter(".pcd .ply .xyz .pts", "Point Cloud Files")
        dlg.set_on_cancel(lambda: self.window.close_dialog())
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        try:
            print(f"Loading {filename}...")
            new_cloud = o3d.io.read_point_cloud(filename)
            if new_cloud.is_empty():
                print("Failed to load cloud or file is empty.")
                return
            
            # 更新基础数据
            self.base_cloud = new_cloud
            
            # 重置所有变换参数和UI
            self.reset_transform()
            
            # 重新调整相机视角适应新物体
            self.reset_camera()
            print("Loaded successfully.")
        except Exception as e:
            print(f"Error loading file: {e}")

    def on_save_file(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "保存变换后的点云", self.window.theme)
        dlg.add_filter(".pcd", "PCD File")
        dlg.add_filter(".ply", "PLY File")
        dlg.set_on_cancel(lambda: self.window.close_dialog())
        dlg.set_on_done(self._on_save_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog_done(self, filename):
        self.window.close_dialog()
        # 保存当前变换后的点云 (self.current_cloud)
        o3d.io.write_point_cloud(filename, self.current_cloud)
        print(f"Saved to {filename}")

    # --- 变换回调 ---
    def on_trans_x(self, val): self.trans_x = val; self.apply_transform()
    def on_trans_y(self, val): self.trans_y = val; self.apply_transform()
    def on_trans_z(self, val): self.trans_z = val; self.apply_transform()
    def on_rot_x(self, val): self.rot_x = val; self.apply_transform()
    def on_rot_y(self, val): self.rot_y = val; self.apply_transform()
    def on_rot_z(self, val): self.rot_z = val; self.apply_transform()

    def reset_transform(self):
        # 重置逻辑值
        self.trans_x = self.trans_y = self.trans_z = 0.0
        self.rot_x = self.rot_y = self.rot_z = 0.0
        
        # 重置 UI 显示值 (重要)
        self.inputs['tx'].double_value = 0.0
        self.inputs['ty'].double_value = 0.0
        self.inputs['tz'].double_value = 0.0
        self.inputs['rx'].double_value = 0.0
        self.inputs['ry'].double_value = 0.0
        self.inputs['rz'].double_value = 0.0
        
        # 恢复轴大小为默认值并更新 UI
        self.axis_size = 1.0
        if 'axis_size' in self.inputs:
            self.inputs['axis_size'].double_value = self.axis_size
        self.update_axis()
        
        self.apply_transform()

    def apply_transform(self):
        # 总是基于原始数据变换
        transformed_cloud = o3d.geometry.PointCloud(self.base_cloud)

        # 平移
        transformed_cloud.translate((self.trans_x, self.trans_y, self.trans_z))

        # 旋转
        R = transformed_cloud.get_rotation_matrix_from_xyz((
            np.radians(self.rot_x),
            np.radians(self.rot_y),
            np.radians(self.rot_z)
        ))
        transformed_cloud.rotate(R, center=(0, 0, 0))

        # 更新显示数据
        self.current_cloud.points = transformed_cloud.points
        self.current_cloud.colors = transformed_cloud.colors
        
        # 通知场景数据已更新
        self.widget3d.scene.remove_geometry("point_cloud")
        self.widget3d.scene.add_geometry("point_cloud", transformed_cloud, self.mat)

    def on_axis_size(self, val):
        """滑块回调：调整轴的大小并更新场景"""
        self.axis_size = val
        self.update_axis()

    def update_axis(self):
        """更新坐标轴几何并重绘场景"""
        if self.widget3d.scene.has_geometry("axis"):
            self.widget3d.scene.remove_geometry("axis")
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_size, origin=[0, 0, 0])
        self.widget3d.scene.add_geometry("axis", self.axis, rendering.MaterialRecord())
        try:
            self.widget3d.force_redraw()
        except Exception:
            # 某些 Open3D 版本可能没有 force_redraw，忽略即可
            pass

    def run(self):
        self.app.run()

if __name__ == "__main__":
    app = PointCloudApp()
    app.run()