import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy import ndimage
import cv2

def crop_cloud_by_bounds(cloud, min_bound, max_bound):
    points = np.asarray(cloud.points)
    mask = (
        (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
        (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
        (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
    )
    cropped_pcd = cloud.select_by_index(np.where(mask)[0])
    return cropped_pcd

def cloud_to_height_map(cloud, min_bound, max_bound, resolution, fill_radius, hole_area_threshold, bilateral_d, bilateral_sigma_color, bilateral_sigma_space):
    # 1. 提取点云数据
    points = np.asarray(cloud.points)
    
    # 2. 计算边界
    if min_bound is None:
        min_bound = points.min(axis=0)
    if max_bound is None:
        max_bound = points.max(axis=0)
    min_x, min_y = min_bound[0], min_bound[1]
    max_x, max_y = max_bound[0], max_bound[1]

    # 3. 确定图像尺寸
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))

    # 4. 初始化高度图（NaN表示无数据）
    height_map = np.full((height, width), np.nan, dtype=np.float32)

    # 5. 将点投影到栅格并保留最大高度（或可选平均/最小）
    for pt in points:
        x, y, z = pt
        col = int((x - min_x) / resolution)
        row = int((y - min_y) / resolution)
        # 防止边界误差
        if 0 <= row < height and 0 <= col < width:
            if np.isnan(height_map[row, col]) or z > height_map[row, col]:
                height_map[row, col] = z  # 保留最高点（适合地面以上障碍物）

    # 6. 改进的缺失值补充逻辑：区分小空洞和大块缺失
    valid_mask = ~np.isnan(height_map)
    holes_mask = ~valid_mask
    
    # 使用连通域分析识别空洞大小
    labeled_holes, num_features = ndimage.label(holes_mask)
    
    if num_features > 0:
        # 计算每个连通域的像素数量
        component_sizes = ndimage.sum(holes_mask, labeled_holes, range(num_features + 1))
        
        # 找出需要填充的小空洞（面积小于阈值）
        small_holes_mask = np.zeros_like(holes_mask, dtype=bool)
        # 忽略背景(0)，检查1到num_features
        for i in range(1, num_features + 1):
            if component_sizes[i] < hole_area_threshold:
                small_holes_mask[labeled_holes == i] = True
        
        nan_rows, nan_cols = np.where(small_holes_mask)
        
        if len(nan_rows) > 0:
            print(f"正在填充 {len(nan_rows)} 个小空洞像素 (忽略大块缺失)...")
            
            # 准备有效点用于KDTree
            valid_rows, valid_cols = np.where(valid_mask)
            valid_x = valid_cols * resolution + min_x + resolution / 2.0
            valid_y = valid_rows * resolution + min_y + resolution / 2.0
            valid_z = height_map[valid_rows, valid_cols]
            valid_coords = np.column_stack((valid_x, valid_y))
            
            if len(valid_coords) > 0:
                tree = cKDTree(valid_coords)
                
                nan_world_x = nan_cols * resolution + min_x + resolution / 2.0
                nan_world_y = nan_rows * resolution + min_y + resolution / 2.0
                nan_coords = np.column_stack((nan_world_x, nan_world_y))

                # 批量查询k近邻
                distances, indices = tree.query(nan_coords, k=10, distance_upper_bound=fill_radius, workers=-1)

                for i in range(len(nan_rows)):
                    idxs = indices[i]
                    dists = distances[i]
                    valid_neighbors = (idxs != tree.n) & np.isfinite(dists)
                    if np.any(valid_neighbors):
                        neighbor_heights = valid_z[idxs[valid_neighbors]]
                        height_map[nan_rows[i], nan_cols[i]] = np.mean(neighbor_heights)
    
    # 7. 双边滤波去噪 (保持边缘，滤除噪声)
    # 更新有效掩码
    valid_mask = ~np.isnan(height_map)
    if np.any(valid_mask):
        # 填充NaN以便进行滤波（使用均值填充，减少边界影响）
        temp_map = height_map.copy()
        mean_val = np.nanmean(temp_map)
        temp_map[~valid_mask] = mean_val
        
        # 应用双边滤波
        # d: 邻域直径, sigmaColor: 高度差异敏感度, sigmaSpace: 空间距离敏感度
        filtered_map = cv2.bilateralFilter(temp_map, d=bilateral_d, 
                                           sigmaColor=bilateral_sigma_color, 
                                           sigmaSpace=bilateral_sigma_space)
        
        # 仅更新有效区域，保持大块缺失为NaN
        height_map[valid_mask] = filtered_map[valid_mask]
        print("双边滤波完成。")

    return height_map, (min_x, min_y)

def enlarge_height_map(height_map, factor):
    new_size = (int(height_map.shape[1] * factor), int(height_map.shape[0] * factor))
    enlarged_map = cv2.resize(height_map, new_size, interpolation=cv2.INTER_CUBIC)
    return enlarged_map

def save_height_map_matplotlib(height_map, output_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(height_map, cmap='terrain', origin='lower')
    plt.colorbar(label='Height (m)')
    plt.title('Height Map')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"高度图图像已保存至: {output_path}")

def save_height_map_cv(height_map, output_path):
    import cv2
    # 归一化高度图到0-255范围
    valid_mask = ~np.isnan(height_map)
    min_height = np.nanmin(height_map)
    norm_height_map = np.zeros_like(height_map, dtype=np.uint8)
    norm_height_map[valid_mask] = ((height_map[valid_mask] - min_height) * 100).astype(np.uint8)
    print(f"高度归一化: 最小高度={min_height:.3f}米, 像素值单位=1cm")
    
    # 将NaN区域设为白色
    norm_height_map[~valid_mask] = 255
    
    # 保存图像
    cv2.imwrite(output_path, norm_height_map)
    print(f"高度图图像已保存至: {output_path}")


if __name__ == "__main__":
    pcd_path="RMUC2026_aligned.pcd"
    cloud = o3d.io.read_point_cloud(pcd_path)
    cloud = crop_cloud_by_bounds(cloud, min_bound=(0, 0, 0), max_bound=(1e3, 1e3, 1.5))

    height_map, origin = cloud_to_height_map(
        cloud=cloud,
        min_bound=(0, 0),
        max_bound=None,
        resolution=0.2,
        fill_radius=0.25,
        hole_area_threshold=30,
        bilateral_d=3,
        bilateral_sigma_color=0.15,
        bilateral_sigma_space=3
    )

    print(f"地图原点世界坐标: x={origin[0]:.3f}, y={origin[1]:.3f}")
    print(f"地图尺寸: {height_map.shape[1]} x {height_map.shape[0]} 像素")

    save_height_map_cv(height_map, "height_map.png")