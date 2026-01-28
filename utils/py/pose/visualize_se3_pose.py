"""
SE(3) 变换可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
import sys

# 修复 Arrow3D 兼容性问题
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

def quat_to_rot_matrix(q):
    """将 [x, y, z, w] 四元数转换为 3x3 旋转矩阵"""
    x, y, z, w = q
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("四元数模长为0")
    x, y, z, w = np.array(q) / norm

    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R

def plot_coordinate_frame(ax, origin, R, label="", color_map=None, axis_length=1.0):
    """绘制带箭头的坐标系"""
    if color_map is None:
        color_map = {'x': 'red', 'y': 'green', 'z': 'blue'}

    for i, axis in enumerate(['x', 'y', 'z']):
        direction = R[:, i] * axis_length
        arrow = Arrow3D(
            [origin[0], origin[0] + direction[0]],
            [origin[1], origin[1] + direction[1]],
            [origin[2], origin[2] + direction[2]],
            mutation_scale=20, lw=2.5, arrowstyle="-|>", color=color_map[axis],
            linestyle='solid'
        )
        ax.add_artist(arrow)
        
        # 轴标签
        label_pos = origin + direction * 1.3
        ax.text(*label_pos, f'{label}{axis.upper()}', color=color_map[axis],
                fontsize=11, weight='bold', zorder=100)

def visualize_se3(translation, quaternion):
    t = np.array(translation)
    R = quat_to_rot_matrix(quaternion)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 世界坐标系（深色）
    plot_coordinate_frame(ax, np.zeros(3), np.eye(3), label="W", 
                          color_map={'x': '#8B0000', 'y': '#006400', 'z': '#00008B'},
                          axis_length=1.0)

    # 变换后坐标系（亮色）
    plot_coordinate_frame(ax, t, R, label="B",
                          color_map={'x': 'red', 'y': 'green', 'z': 'blue'},
                          axis_length=1.0)

    # 设置视图范围
    points = np.vstack([np.zeros(3), t, t + R @ np.eye(3)])
    max_range = np.max(np.abs(points - np.mean(points, axis=0))) * 1.5
    center = np.mean(points, axis=0)
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # 标签与标题
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', fontsize=12, labelpad=10)
    ax.set_zlabel('Z', fontsize=12, labelpad=10)
    ax.set_title(f'SE(3) 变换可视化\n平移: {translation}\n四元数: {quaternion}',
                 fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_box_aspect([1, 1, 1])

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#8B0000', lw=2.5, label='World X'),
        Line2D([0], [0], color='#006400', lw=2.5, label='World Y'),
        Line2D([0], [0], color='#00008B', lw=2.5, label='World Z'),
        Line2D([0], [0], color='red', lw=2.5, label='Body X'),
        Line2D([0], [0], color='green', lw=2.5, label='Body Y'),
        Line2D([0], [0], color='blue', lw=2.5, label='Body Z'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    # 设置初始视角
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.show()

def parse_input(prompt, expected_len, value_name, default=None):
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                print(f"  → 使用默认值: {default}")
                return default
            
            values = list(map(float, user_input.replace(',', ' ').split()))
            if len(values) != expected_len:
                print(f"❌ 错误：{value_name}需要{expected_len}个数值，当前输入了{len(values)}个")
                continue
            return values
        except ValueError as e:
            print(f"❌ 输入格式错误：{e}")
        except KeyboardInterrupt:
            print("\n👋 程序已退出")
            sys.exit(0)

def main():
    print("="*60)
    print("SE(3) 变换可视化工具 (ROS 2 四元数顺序: x y z w)")
    print("="*60)
    
    translation = parse_input(
        "\n请输入平移向量 [x y z]（单位：米），例如：0.5 0 1.0\n"
        "（直接回车使用默认 [0, 0, 1]）: ",
        3, "平移", default=[0.0, 0.0, 1.0]
    )
    
    quaternion = parse_input(
        "\n请输入四元数 [x y z w]，例如：0 0 0.7071 0.7071 (绕Z轴旋转90°)\n"
        "（直接回车使用默认 [0, 0, 0, 1]）: ",
        4, "四元数", default=[0.0, 0.0, 0.0, 1.0]
    )
    
    print("\n✅ 变换参数:")
    print(f"   平移: {translation}")
    print(f"   四元数: {quaternion}")
    
    try:
        visualize_se3(translation, quaternion)
    except Exception as e:
        print(f"\n❌ 可视化出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✓ 依赖检查通过 (Matplotlib {matplotlib.__version__})")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请安装: pip install numpy matplotlib")
        sys.exit(1)
    
    main()