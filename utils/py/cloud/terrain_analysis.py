import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import warnings

warnings.filterwarnings('ignore')

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
        """基于法向量倾角进行初步分类"""
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        cos_theta = np.abs(normals[:, 2])
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        labels = np.full(len(points), self.LABEL_OBSTACLE, dtype=int)
        
        ground_mask = theta <= self.ground_angle_threshold
        labels[ground_mask] = self.LABEL_GROUND
        
        slope_mask = (theta > self.ground_angle_threshold) & \
                     (theta <= self.slope_angle_threshold)
        labels[slope_mask] = self.LABEL_SLOPE
        
        candidate_mask = theta > self.slope_angle_threshold
        labels[candidate_mask] = self.LABEL_OBSTACLE 
        
        return labels, candidate_mask
    
    def analyze_candidate_region(self, points, candidate_indices):
        """
        对候选区域进行邻域分层分析，并计算台阶向量。
        Returns:
            refined_labels: 修正后的标签
            step_vectors: 对应的台阶方向向量 (N, 2)，非台阶点为(0,0)
        """
        refined_labels = np.full(len(points), -1, dtype=int)
        step_vectors = np.zeros((len(points), 2), dtype=float) # 存储 dx, dy

        if len(candidate_indices) == 0:
            return refined_labels, step_vectors
        
        tree = cKDTree(points[:, :2])
        
        num_completed = 0
        total_candidates = len(candidate_indices)
        
        for idx in candidate_indices:
            num_completed += 1
            if num_completed % 500 == 0:
                print(f"分析进度: {round(num_completed / total_candidates * 100, 1)}% ({num_completed}/{total_candidates})")
            
            # 获取邻域点
            query_point = points[idx, :2]
            indices = tree.query_ball_point(query_point, self.neighbor_radius)
            
            if len(indices) < self.min_points_per_cluster * 2:
                refined_labels[idx] = self.LABEL_OBSTACLE
                continue
            
            neighbor_z = points[indices, 2]
            neighbor_xy = points[indices, :2]
            
            try:
                # K-Means 聚类高度 (Z轴)
                kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
                clusters = kmeans.fit_predict(neighbor_z.reshape(-1, 1))
                centers = kmeans.cluster_centers_.flatten()
                
                # 确定哪个簇更高
                if centers[0] > centers[1]:
                    high_idx = 0
                    low_idx = 1
                else:
                    high_idx = 1
                    low_idx = 0
                
                height_diff = abs(centers[0] - centers[1])
                
                cluster_high_count = np.sum(clusters == high_idx)
                cluster_low_count = np.sum(clusters == low_idx)
                
                # 判断是否为台阶
                if (self.step_min_height <= height_diff <= self.step_max_height and
                    cluster_high_count >= self.min_points_per_cluster and
                    cluster_low_count >= self.min_points_per_cluster):
                    
                    refined_labels[idx] = self.LABEL_STEP
                    
                    # === 新增：计算台阶向量 ===
                    # 获取高处点集和低处点集的重心 (XY平面)
                    high_points_xy = neighbor_xy[clusters == high_idx]
                    low_points_xy = neighbor_xy[clusters == low_idx]
                    
                    center_high_xy = np.mean(high_points_xy, axis=0)
                    center_low_xy = np.mean(low_points_xy, axis=0)
                    
                    # 向量方向：从低处指向高处 (Ascending direction)
                    vec = center_high_xy - center_low_xy
                    norm = np.linalg.norm(vec)
                    
                    if norm > 1e-6:
                        step_vectors[idx] = vec / norm # 归一化
                    else:
                        step_vectors[idx] = [0, 0]
                        
                else:
                    refined_labels[idx] = self.LABEL_OBSTACLE
                    
            except Exception:
                refined_labels[idx] = self.LABEL_OBSTACLE
        
        return refined_labels, step_vectors

    def generate_navigation_map(self, points, labels, step_vectors):
        """
        生成符合要求的三通道 BGR 导航地图。
        R: 障碍物 (255)，包含闭合填充
        G: 台阶 Y 方向分量 (映射到 1-255)
        B: 台阶 X 方向分量 (映射到 1-255)
        """
        # 1. 计算地图边界和尺寸
        min_xy = [0, 0]
        max_xy = np.max(points[:, :2], axis=0)
        
        width = int(np.ceil((max_xy[0] - min_xy[0]) / self.map_resolution))
        height = int(np.ceil((max_xy[1] - min_xy[1]) / self.map_resolution))
        
        print(f"地图尺寸: {width} x {height}")
        
        # 初始化图像 (H, W, 3) - OpenCV 格式
        # grid_accumulators 用于累积同一个栅格内的向量，取平均值
        nav_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 临时存储用于平均向量的数据
        vec_accum = np.zeros((height, width, 2), dtype=float)
        vec_count = np.zeros((height, width), dtype=float)
        
        # 障碍物临时掩码
        obstacle_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 2. 栅格化点云数据
        for i, point in enumerate(points):
            # 坐标转换：World -> Image Pixel
            # img_x 对应 B 通道 (X向量), img_y 对应 G 通道 (Y向量)
            # 注意：图像坐标系通常 y 轴向下，地图通常 y 轴向上。
            # 这里我们做一个简单的平移缩放，保持 map_y 与 world_y 方向一致（可视化的习惯），
            # 或者将其翻转。为了路径规划方便，通常保持物理坐标系方向一致，即 origin='lower'。
            # 但 OpenCV 图像索引是 (row, col)，即 (y, x)。
            
            x_idx = int((point[0] - min_xy[0]) / self.map_resolution)
            y_idx = int((point[1] - min_xy[1]) / self.map_resolution)
            
            if x_idx < 0 or x_idx >= width or y_idx < 0 or y_idx >= height:
                continue
            
            label = labels[i]
            
            if label == self.LABEL_OBSTACLE:
                obstacle_mask[y_idx, x_idx] = 255
            
            elif label == self.LABEL_STEP:
                vec = step_vectors[i]
                if abs(vec[0]) > 0 or abs(vec[1]) > 0:
                    vec_accum[y_idx, x_idx] += vec
                    vec_count[y_idx, x_idx] += 1
        
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
        # 计算平均向量
        mask_step = (vec_count > 0)
        avg_vecs = np.zeros_like(vec_accum)
        avg_vecs[mask_step] = vec_accum[mask_step] / vec_count[mask_step][..., None]
        
        # 再次归一化平均向量 (防止平均后长度变短)
        norms = np.linalg.norm(avg_vecs, axis=2)
        norm_mask = norms > 1e-6
        avg_vecs[norm_mask] /= norms[norm_mask][..., None]
        
        # 映射到 [1, 255], 中心 128
        # Formula: value = 128 + vec * 127
        # B channel -> X component (vec[:,:,0])
        # G channel -> Y component (vec[:,:,1])
        
        # 初始化为0
        b_channel = np.zeros((height, width), dtype=np.uint8)
        g_channel = np.zeros((height, width), dtype=np.uint8)
        
        # 只在有台阶的地方赋值
        # X component
        map_x = 128 + avg_vecs[:, :, 0] * 127
        b_channel[mask_step] = np.clip(map_x[mask_step], 1, 255).astype(np.uint8)
        
        # Y component
        map_y = 128 + avg_vecs[:, :, 1] * 127
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
    nav_map = analyzer.analyze_terrain(cloud_path, output_image_path="nav_map.png") 
    print("处理完成。")