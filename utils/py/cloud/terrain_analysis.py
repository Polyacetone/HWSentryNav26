import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt
import numba as nb


@nb.njit(parallel=True, fastmath=True, cache=False)
def _classify_normals_numba(normals, ground_thr, slope_thr, label_ground, label_slope, label_obstacle):
    n = normals.shape[0]
    labels = np.empty(n, dtype=np.int32)
    candidate = np.empty(n, dtype=np.uint8)

    for i in nb.prange(n):
        nz = normals[i, 2]
        if nz < 0.0:
            nz = -nz
        if nz > 1.0:
            nz = 1.0
        theta = np.arccos(nz)

        if theta <= ground_thr:
            labels[i] = label_ground
            candidate[i] = 0
        elif theta <= slope_thr:
            labels[i] = label_slope
            candidate[i] = 0
        else:
            labels[i] = label_obstacle
            candidate[i] = 1

    return labels, candidate


@nb.njit(cache=False)
def _build_grid_csr_numba(px, py, min_x, min_y, cell_size, grid_w, grid_h):
    n = px.shape[0]
    n_cells = grid_w * grid_h
    counts = np.zeros(n_cells, dtype=np.int32)

    for i in range(n):
        cx = int((px[i] - min_x) / cell_size)
        cy = int((py[i] - min_y) / cell_size)
        if cx < 0:
            cx = 0
        elif cx >= grid_w:
            cx = grid_w - 1
        if cy < 0:
            cy = 0
        elif cy >= grid_h:
            cy = grid_h - 1
        cell = cy * grid_w + cx
        counts[cell] += 1

    offsets = np.empty(n_cells + 1, dtype=np.int32)
    offsets[0] = 0
    for c in range(n_cells):
        offsets[c + 1] = offsets[c] + counts[c]

    cursor = offsets[:-1].copy()
    indices = np.empty(n, dtype=np.int32)
    for i in range(n):
        cx = int((px[i] - min_x) / cell_size)
        cy = int((py[i] - min_y) / cell_size)
        if cx < 0:
            cx = 0
        elif cx >= grid_w:
            cx = grid_w - 1
        if cy < 0:
            cy = 0
        elif cy >= grid_h:
            cy = grid_h - 1
        cell = cy * grid_w + cx

        pos = cursor[cell]
        indices[pos] = i
        cursor[cell] = pos + 1

    return offsets, indices


@nb.njit(parallel=True, fastmath=True, cache=False)
def _analyze_candidates_numba(
    points,
    candidate_indices,
    cell_offsets,
    cell_indices,
    min_x,
    min_y,
    cell_size,
    grid_w,
    grid_h,
    neighbor_r2,
    min_pts,
    step_min_h,
    step_max_h,
    label_step,
    label_obstacle,
    refined_labels_out,
    step_vectors_out,
):
    n_candidates = candidate_indices.shape[0]
    radius = np.sqrt(neighbor_r2)
    r_cells = int(np.ceil(radius / cell_size))
    if r_cells < 1:
        r_cells = 1

    for k in nb.prange(n_candidates):
        idx = candidate_indices[k]
        qx = points[idx, 0]
        qy = points[idx, 1]

        cx = int((qx - min_x) / cell_size)
        cy = int((qy - min_y) / cell_size)
        if cx < 0:
            cx = 0
        elif cx >= grid_w:
            cx = grid_w - 1
        if cy < 0:
            cy = 0
        elif cy >= grid_h:
            cy = grid_h - 1

        n_nb = 0
        z_min = 1e9
        z_max = -1e9

        x0 = cx - r_cells
        x1 = cx + r_cells
        y0 = cy - r_cells
        y1 = cy + r_cells
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 >= grid_w:
            x1 = grid_w - 1
        if y1 >= grid_h:
            y1 = grid_h - 1

        for yy in range(y0, y1 + 1):
            base = yy * grid_w
            for xx in range(x0, x1 + 1):
                cell = base + xx
                start = cell_offsets[cell]
                end = cell_offsets[cell + 1]
                for ppos in range(start, end):
                    j = cell_indices[ppos]
                    dx = points[j, 0] - qx
                    dy = points[j, 1] - qy
                    d2 = dx * dx + dy * dy
                    if d2 <= neighbor_r2:
                        n_nb += 1
                        z = points[j, 2]
                        if z < z_min:
                            z_min = z
                        if z > z_max:
                            z_max = z

        if n_nb < min_pts * 2:
            refined_labels_out[idx] = label_obstacle
            continue

        if (z_max - z_min) < step_min_h:
            refined_labels_out[idx] = label_obstacle
            continue

        m_low = z_min
        m_high = z_max

        for _ in range(4):
            t = 0.5 * (m_low + m_high)
            sum_low = 0.0
            sum_high = 0.0
            cnt_low = 0
            cnt_high = 0

            for yy in range(y0, y1 + 1):
                base = yy * grid_w
                for xx in range(x0, x1 + 1):
                    cell = base + xx
                    start = cell_offsets[cell]
                    end = cell_offsets[cell + 1]
                    for ppos in range(start, end):
                        j = cell_indices[ppos]
                        dx = points[j, 0] - qx
                        dy = points[j, 1] - qy
                        d2 = dx * dx + dy * dy
                        if d2 <= neighbor_r2:
                            z = points[j, 2]
                            if z < t:
                                sum_low += z
                                cnt_low += 1
                            else:
                                sum_high += z
                                cnt_high += 1

            if cnt_low == 0 or cnt_high == 0:
                break
            m_low = sum_low / cnt_low
            m_high = sum_high / cnt_high

        if m_high < m_low:
            tmp = m_low
            m_low = m_high
            m_high = tmp

        height_diff = m_high - m_low
        t = 0.5 * (m_low + m_high)

        sum_low_x = 0.0
        sum_low_y = 0.0
        sum_high_x = 0.0
        sum_high_y = 0.0
        cnt_low = 0
        cnt_high = 0

        for yy in range(y0, y1 + 1):
            base = yy * grid_w
            for xx in range(x0, x1 + 1):
                cell = base + xx
                start = cell_offsets[cell]
                end = cell_offsets[cell + 1]
                for ppos in range(start, end):
                    j = cell_indices[ppos]
                    dx = points[j, 0] - qx
                    dy = points[j, 1] - qy
                    d2 = dx * dx + dy * dy
                    if d2 <= neighbor_r2:
                        z = points[j, 2]
                        if z < t:
                            sum_low_x += points[j, 0]
                            sum_low_y += points[j, 1]
                            cnt_low += 1
                        else:
                            sum_high_x += points[j, 0]
                            sum_high_y += points[j, 1]
                            cnt_high += 1

        if (
            height_diff >= step_min_h
            and height_diff <= step_max_h
            and cnt_low >= min_pts
            and cnt_high >= min_pts
        ):
            refined_labels_out[idx] = label_step
            cx_low = sum_low_x / cnt_low
            cy_low = sum_low_y / cnt_low
            cx_high = sum_high_x / cnt_high
            cy_high = sum_high_y / cnt_high

            vx = cx_high - cx_low
            vy = cy_high - cy_low
            nrm = np.sqrt(vx * vx + vy * vy)
            if nrm > 1e-6:
                step_vectors_out[idx, 0] = vx / nrm
                step_vectors_out[idx, 1] = vy / nrm
            else:
                step_vectors_out[idx, 0] = 0.0
                step_vectors_out[idx, 1] = 0.0
        else:
            refined_labels_out[idx] = label_obstacle

class TerrainAnalyzer:
    def __init__(self, 
                 ground_angle_threshold=5.0,       # 地面最大倾角（度）
                 slope_angle_threshold=30.0,       # 斜坡最大倾角（度）
                 step_min_height=0.05,             # 台阶最小高度（米）
                 step_max_height=0.3,             # 台阶最大高度（米）
                 neighbor_radius=0.3,              # 邻域分析半径（米）
                 min_points_per_cluster=10,        # 每个聚类最少点数
                 voxel_size=0.05,                   # 体素下采样大小
                 map_resolution=0.1):             # 导航地图分辨率（米/像素）
        
        self.ground_angle_threshold = np.radians(ground_angle_threshold)
        self.slope_angle_threshold = np.radians(slope_angle_threshold)
        self.step_min_height = step_min_height
        self.step_max_height = step_max_height
        self.neighbor_radius = neighbor_radius
        self.min_points_per_cluster = min_points_per_cluster
        self.voxel_size = voxel_size
        self.map_resolution = map_resolution
        
        # 语义标签定义
        self.LABEL_GROUND = 0
        self.LABEL_SLOPE = 1
        self.LABEL_STEP = 2
        self.LABEL_OBSTACLE = 3
        
        self.label_colors = {
            self.LABEL_GROUND: [0, 255, 0],      # 绿色
            self.LABEL_SLOPE: [0, 128, 0],       # 深绿色
            self.LABEL_STEP: [255, 255, 0],      # 黄色
            self.LABEL_OBSTACLE: [255, 0, 0]     # 红色
        }
    
    def preprocess_pointcloud(self, pcd):
        """点云预处理：去噪 + 下采样"""
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return pcd
    
    def estimate_normals(self, pcd, k_neighbors=15):
        """估计法向量"""
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )
        normals = np.asarray(pcd.normals)
        normals[normals[:, 2] < 0] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd
    
    def classify_by_normals(self, pcd):
        """基于法向量倾角进行初步分类（Numba 加速）"""
        normals = np.asarray(pcd.normals, dtype=np.float32)
        labels, candidate_mask_u8 = _classify_normals_numba(
            normals,
            float(self.ground_angle_threshold),
            float(self.slope_angle_threshold),
            self.LABEL_GROUND,
            self.LABEL_SLOPE,
            self.LABEL_OBSTACLE,
        )
        return labels, candidate_mask_u8.astype(bool)
    
    def analyze_candidate_region(self, points, candidate_indices):
        """候选区域邻域分析 + 台阶向量（Numba 并行加速）。"""
        n_points = points.shape[0]
        refined_labels = np.full(n_points, -1, dtype=np.int32)
        step_vectors = np.zeros((n_points, 2), dtype=np.float32)

        if candidate_indices.size == 0:
            return refined_labels, step_vectors

        points_f32 = np.asarray(points, dtype=np.float32)
        candidate_indices_i64 = np.asarray(candidate_indices, dtype=np.int64)

        # 用 2D 栅格哈希替代 KDTree：构建 cell -> point indices 的 CSR 结构
        min_xy = np.min(points_f32[:, :2], axis=0)
        max_xy = np.max(points_f32[:, :2], axis=0)

        cell_size = float(self.neighbor_radius)
        grid_w = int(np.floor((max_xy[0] - min_xy[0]) / cell_size)) + 1
        grid_h = int(np.floor((max_xy[1] - min_xy[1]) / cell_size)) + 1

        cell_offsets, cell_indices = _build_grid_csr_numba(
            points_f32[:, 0],
            points_f32[:, 1],
            float(min_xy[0]),
            float(min_xy[1]),
            float(cell_size),
            grid_w,
            grid_h,
        )

        neighbor_r2 = float(self.neighbor_radius * self.neighbor_radius)
        _analyze_candidates_numba(
            points_f32,
            candidate_indices_i64,
            cell_offsets,
            cell_indices,
            float(min_xy[0]),
            float(min_xy[1]),
            float(cell_size),
            grid_w,
            grid_h,
            neighbor_r2,
            int(self.min_points_per_cluster),
            float(self.step_min_height),
            float(self.step_max_height),
            int(self.LABEL_STEP),
            int(self.LABEL_OBSTACLE),
            refined_labels,
            step_vectors,
        )

        return refined_labels, step_vectors

    def generate_navigation_map(self, points, labels, step_vectors):
        """
        生成符合要求的三通道 BGR 导航地图。
        R: 障碍物 (255)，包含闭合填充
        G: 台阶 Y 方向分量 (映射到 1-255)
        B: 台阶 X 方向分量 (映射到 1-255)
        """
        # 1. 计算地图边界和尺寸
        min_xy = np.min(points[:, :2], axis=0)
        max_xy = np.max(points[:, :2], axis=0)

        width = int(np.ceil((max_xy[0] - min_xy[0]) / self.map_resolution)) + 1
        height = int(np.ceil((max_xy[1] - min_xy[1]) / self.map_resolution)) + 1
        
        print(f"地图尺寸: {width} x {height}")
        
        # 初始化图像 (H, W, 3) - OpenCV 格式
        nav_map = np.zeros((height, width, 3), dtype=np.uint8)
        # 2. 栅格化点云数据（用 bincount 聚合，避免 Python 循环）
        x_idx = ((points[:, 0] - min_xy[0]) / self.map_resolution).astype(np.int32)
        y_idx = ((points[:, 1] - min_xy[1]) / self.map_resolution).astype(np.int32)
        in_bounds = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)

        flat_size = height * width
        cell_id = (y_idx.astype(np.int64) * width + x_idx.astype(np.int64))

        # 障碍物掩码（写 255 是幂等的，重复写没影响）
        obstacle_flat = np.zeros(flat_size, dtype=np.uint8)
        obstacle_sel = in_bounds & (labels == self.LABEL_OBSTACLE)
        obstacle_flat[cell_id[obstacle_sel]] = 255
        obstacle_mask = obstacle_flat.reshape((height, width))
        
        # 3. 处理障碍物通道 (R 通道) - 包含闭合区域填充
        # 先做一次形态学闭操作，把密集的障碍物点连接成线/块
        # kernel_size = 3
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        # obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        
        # === 闭合区域填充逻辑 ===
        # 复制一份作为 mask
        # 增加 2 像素的 padding，这是 floodFill 的要求
        # h, w = obstacle_mask.shape
        # mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # 从 (0,0) 开始泛洪填充背景。假设 (0,0) 是安全的外部区域。
        # 如果地图边缘全是障碍物，这里可能需要调整起始点。通常假设这一角是空的。
        # im_floodfill = obstacle_mask.copy()
        # cv2.floodFill(im_floodfill, mask, (0,0), 255)
        
        # 让我们理清 floodFill 逻辑：
        # 1. 原图: 障碍物=255, 背景=0
        # 2. FloodFill (从0,0, 填成255): 外部背景=255, 障碍物=255, 内部空洞=0 (因为水进不去)
        # 3. Invert FloodFill: 外部背景=0, 障碍物=0, 内部空洞=255
        # 4. Final Result = Original | Inverted FloodFill
        # final_obstacles = obstacle_mask | cv2.bitwise_not(im_floodfill)
        
        nav_map[:, :, 2] = obstacle_mask
        
        # 4. 处理台阶向量通道 (B, G 通道)
        step_vecs = np.asarray(step_vectors, dtype=np.float32)
        step_nonzero = (np.abs(step_vecs[:, 0]) + np.abs(step_vecs[:, 1])) > 0
        step_sel = in_bounds & (labels == self.LABEL_STEP) & step_nonzero

        if np.any(step_sel):
            ids = cell_id[step_sel]
            cnt = np.bincount(ids, minlength=flat_size).astype(np.float32)
            sum_x = np.bincount(ids, weights=step_vecs[step_sel, 0], minlength=flat_size).astype(np.float32)
            sum_y = np.bincount(ids, weights=step_vecs[step_sel, 1], minlength=flat_size).astype(np.float32)

            avg_x = np.zeros(flat_size, dtype=np.float32)
            avg_y = np.zeros(flat_size, dtype=np.float32)
            m = cnt > 0
            avg_x[m] = sum_x[m] / cnt[m]
            avg_y[m] = sum_y[m] / cnt[m]

            # 归一化
            norms = np.sqrt(avg_x * avg_x + avg_y * avg_y)
            nm = norms > 1e-6
            avg_x[nm] /= norms[nm]
            avg_y[nm] /= norms[nm]

            avg_x = avg_x.reshape((height, width))
            avg_y = avg_y.reshape((height, width))
            mask_step = (cnt.reshape((height, width)) > 0)

            b_channel = np.zeros((height, width), dtype=np.uint8)
            g_channel = np.zeros((height, width), dtype=np.uint8)

            map_x = 128 + avg_x * 127
            b_channel[mask_step] = np.clip(map_x[mask_step], 1, 255).astype(np.uint8)

            map_y = 128 + avg_y * 127
            g_channel[mask_step] = np.clip(map_y[mask_step], 1, 255).astype(np.uint8)

            nav_map[:, :, 0] = b_channel
            nav_map[:, :, 1] = g_channel

        # 5. 障碍物不是台阶
        red_mask = (nav_map[:, :, 2] == 255)
        nav_map[red_mask, 0] = 0
        nav_map[red_mask, 1] = 0
        
        # 翻转 Y 轴以符合图像坐标系 (如果需要存成图片看的话，通常希望 Origin 在左下角，但图片存储是从左上角开始)
        # 这里的 nav_map[0,0] 对应的是 min_xy。如果直接 imshow，(0,0)在左上。
        # 为了让图片看起来像地图（y轴向上），我们需要上下翻转。
        # nav_map = cv2.flip(nav_map, 0)
        
        return nav_map, min_xy, self.map_resolution

    def analyze_terrain(self, pcd_path, output_image_path="nav_map.png"):
        print("1. 加载点云...")
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        print("2. 预处理 & 法向量估计...")
        pcd = self.preprocess_pointcloud(pcd)
        pcd = self.estimate_normals(pcd)
        
        print("3. 初步分类...")
        labels, candidate_mask = self.classify_by_normals(pcd)
        
        print("4. 局部特征分析 & 向量计算...")
        points = np.asarray(pcd.points)
        candidate_indices = np.where(candidate_mask)[0]
        # 注意：这里我们修改了 analyze_candidate_region 的返回值，增加了 vectors
        refined_labels, step_vectors = self.analyze_candidate_region(points, candidate_indices)
        
        # 合并标签
        final_labels = labels.copy()
        final_step_vectors = np.zeros((len(points), 2))
        
        # 将 refined 的结果填回去
        for i, idx in enumerate(candidate_indices):
            if refined_labels[idx] != -1:
                final_labels[idx] = refined_labels[idx]
                if refined_labels[idx] == self.LABEL_STEP:
                    final_step_vectors[idx] = step_vectors[idx]
        
        print("5. 生成 BGR 导航地图...")
        nav_map, origin, res = self.generate_navigation_map(points, final_labels, final_step_vectors)
        
        print(f"6. 保存地图至 {output_image_path}")
        cv2.imwrite(output_image_path, nav_map)
        
        # 可视化一下生成的地图
        plt.figure(figsize=(10, 8))
        # 转换 BGR 到 RGB 方便 matplotlib 显示
        plt.imshow(cv2.cvtColor(nav_map, cv2.COLOR_BGR2RGB))
        plt.title(f"Navigation Map (Res: {res}m)\nR=Obstacle, G=Dy, B=Dx")
        plt.axis('off')
        plt.show()
        
        return nav_map

if __name__ == "__main__":
    # 使用示例
    analyzer = TerrainAnalyzer()
    cloud_path = input("请输入点云文件路径（.pcd 格式）：")
    nav_map = analyzer.analyze_terrain(cloud_path, output_image_path=os.path.splitext(cloud_path)[0] + ".png")
    print("处理完成。")