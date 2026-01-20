import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def generate_elevation_map(pcd, resolution=0.05, search_radius=0.15, min_points=8):
    """
    改进版：基于2D邻域搜索的局部平面拟合高程图生成
    """
    # 1. 预处理
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    points = np.asarray(pcd.points)
    
    # 2. 建立 2D 空间索引 (只考虑 X 和 Y)
    points_2d = points[:, :2]
    tree_2d = cKDTree(points_2d)
    
    # 计算边界
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    width = int(np.ceil((max_bound[0] - min_bound[0]) / resolution))
    height = int(np.ceil((max_bound[1] - min_bound[1]) / resolution))
    
    elevation_map = np.full((width, height), np.nan)

    # 3. 遍历网格
    for i in range(width):
        for j in range(height):
            # 网格中心的 XY 坐标
            cx = min_bound[0] + (i + 0.5) * resolution
            cy = min_bound[1] + (j + 0.5) * resolution
            
            # 在 2D 平面上搜索半径范围内的点索引
            indices = tree_2d.query_ball_point([cx, cy], r=search_radius)
            
            if len(indices) >= min_points:
                # 获取这些索引对应的 3D 坐标
                neighbor_points = points[indices, :]
                
                # 4. 最小二乘法拟合平面: z = Ax + By + C
                # 矩阵构造
                A_mat = np.c_[neighbor_points[:, 0], neighbor_points[:, 1], np.ones(len(indices))]
                Z_mat = neighbor_points[:, 2]
                
                try:
                    # 使用 rcond 处理奇异矩阵，增加鲁棒性
                    fit, _, _, _ = np.linalg.lstsq(A_mat, Z_mat, rcond=None)
                    # A=fit[0], B=fit[1], C=fit[2]
                    z_val = fit[0] * cx + fit[1] * cy + fit[2]
                    
                    # 额外检查：计算出的 z 值不应偏离邻域均值太远（防止拟合平面过于倾斜）
                    z_mean = np.mean(Z_mat)
                    if abs(z_val - z_mean) < 0.5: # 50cm 阈值，可调
                        elevation_map[i, j] = z_val
                except np.linalg.LinAlgError:
                    continue
                    
    return elevation_map

pcd = o3d.io.read_point_cloud("base_cropped.pcd")
grid = generate_elevation_map(pcd, resolution=0.05, search_radius=0.15)
plt.imshow(grid.T, origin='lower', cmap='viridis')
plt.colorbar(label='Height (m)')
plt.show()