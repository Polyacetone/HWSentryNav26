# terrain_analysis.py
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import argparse
import warnings
warnings.filterwarnings('ignore')

class TerrainAnalyzer:
    def __init__(self, 
                 ground_angle_threshold=5.0,      # 地面最大倾角（度）
                 slope_angle_threshold=20.0,       # 斜坡最大倾角（度）
                 step_min_height=0.05,             # 台阶最小高度（米）
                 step_max_height=0.25,             # 台阶最大高度（米）
                 neighbor_radius=0.5,              # 邻域分析半径（米）
                 min_points_per_cluster=10,        # 每个聚类最少点数
                 voxel_size=0.05):                 # 体素下采样大小
        
        self.ground_angle_threshold = np.radians(ground_angle_threshold)
        self.slope_angle_threshold = np.radians(slope_angle_threshold)
        self.step_min_height = step_min_height
        self.step_max_height = step_max_height
        self.neighbor_radius = neighbor_radius
        self.min_points_per_cluster = min_points_per_cluster
        self.voxel_size = voxel_size
        
        # 语义标签定义
        self.LABEL_GROUND = 0
        self.LABEL_SLOPE = 1
        self.LABEL_STEP = 2
        self.LABEL_OBSTACLE = 3
        self.label_colors = {
            self.LABEL_GROUND: [0, 255, 0],      # 绿色
            self.LABEL_SLOPE: [0, 255, 0],       # 绿色
            self.LABEL_STEP: [255, 255, 0],      # 黄色
            self.LABEL_OBSTACLE: [255, 0, 0]     # 红色
        }
    
    def preprocess_pointcloud(self, pcd):
        """点云预处理：去噪 + 下采样"""
        # 统计离群点移除
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 体素下采样
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        return pcd
    
    def estimate_normals(self, pcd, k_neighbors=15):
        """估计法向量"""
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )
        # 确保法向量朝上（nz > 0）
        normals = np.asarray(pcd.normals)
        normals[normals[:, 2] < 0] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)
        return pcd
    
    def classify_by_normals(self, pcd):
        """基于法向量倾角进行初步分类"""
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # 计算法向量与Z轴的夹角
        cos_theta = np.abs(normals[:, 2])  # |nz|
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        labels = np.full(len(points), self.LABEL_OBSTACLE, dtype=int)
        
        # 地面：theta <= ground_angle_threshold
        ground_mask = theta <= self.ground_angle_threshold
        labels[ground_mask] = self.LABEL_GROUND
        
        # 斜坡：ground_angle_threshold < theta <= slope_angle_threshold
        slope_mask = (theta > self.ground_angle_threshold) & \
                     (theta <= self.slope_angle_threshold)
        labels[slope_mask] = self.LABEL_SLOPE
        
        # 候选障碍物/台阶：theta > slope_angle_threshold
        candidate_mask = theta > self.slope_angle_threshold
        labels[candidate_mask] = self.LABEL_OBSTACLE  # 临时标记为障碍物
        
        return labels, candidate_mask
    
    def analyze_candidate_region(self, points, candidate_indices):
        """对候选区域进行邻域分层分析"""
        if len(candidate_indices) == 0:
            return np.array([])
        
        # 构建KD树用于快速邻域搜索
        tree = cKDTree(points[:, :2])  # 只在XY平面搜索
        
        refined_labels = np.full(len(points), -1, dtype=int)

        num_completed = 0
        total_candidates = len(candidate_indices)
        
        for idx in candidate_indices:
            num_completed += 1
            if num_completed % 100 == 0 or num_completed == total_candidates:
                print(f"分析候选点：{round(num_completed / total_candidates * 100, 1)}% ({num_completed}/{total_candidates})")
            
            # 获取邻域点
            query_point = points[idx, :2]
            indices = tree.query_ball_point(query_point, self.neighbor_radius)
            
            if len(indices) < self.min_points_per_cluster * 2:
                refined_labels[idx] = self.LABEL_OBSTACLE
                continue
            
            # 提取邻域点的高度
            neighbor_z = points[indices, 2]
            
            # 使用K-means进行高度聚类（K=2）
            try:
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                clusters = kmeans.fit_predict(neighbor_z.reshape(-1, 1))
                
                # 获取两个簇的中心高度
                centers = kmeans.cluster_centers_.flatten()
                height_diff = abs(centers[0] - centers[1])
                
                # 检查每个簇的点数
                cluster_0_count = np.sum(clusters == 0)
                cluster_1_count = np.sum(clusters == 1)
                
                # 判断是否为台阶
                if (self.step_min_height <= height_diff <= self.step_max_height and
                    cluster_0_count >= self.min_points_per_cluster and
                    cluster_1_count >= self.min_points_per_cluster):
                    refined_labels[idx] = self.LABEL_STEP
                else:
                    refined_labels[idx] = self.LABEL_OBSTACLE
                    
            except Exception as e:
                refined_labels[idx] = self.LABEL_OBSTACLE
        
        return refined_labels
    
    def create_elevation_map(self, points, labels, resolution=0.05):
        """创建高程地图和语义地图"""
        # 计算边界
        min_xy = np.min(points[:, :2], axis=0)
        max_xy = np.max(points[:, :2], axis=0)
        
        # 创建网格
        x_bins = int((max_xy[0] - min_xy[0]) / resolution) + 1
        y_bins = int((max_xy[1] - min_xy[1]) / resolution) + 1
        
        elevation_map = np.full((y_bins, x_bins), np.nan)
        semantic_map = np.full((y_bins, x_bins), -1, dtype=int)
        count_map = np.zeros((y_bins, x_bins), dtype=int)
        
        # 填充地图
        for point, label in zip(points, labels):
            x_idx = int((point[0] - min_xy[0]) / resolution)
            y_idx = int((point[1] - min_xy[1]) / resolution)
            
            if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                if np.isnan(elevation_map[y_idx, x_idx]):
                    elevation_map[y_idx, x_idx] = point[2]
                    semantic_map[y_idx, x_idx] = label
                else:
                    # 取最高点（适用于台阶上表面）
                    if point[2] > elevation_map[y_idx, x_idx]:
                        elevation_map[y_idx, x_idx] = point[2]
                        semantic_map[y_idx, x_idx] = label
                count_map[y_idx, x_idx] += 1
        
        return elevation_map, semantic_map, min_xy, resolution
    
    def visualize_results(self, pcd, final_labels, elevation_map, semantic_map, origin):
        """可视化结果"""
        # 1. 3D点云可视化
        colors = np.array([self.label_colors[label] for label in final_labels]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        print("显示3D点云（按Q退出）...")
        o3d.visualization.draw_geometries([pcd])
        
        # 2. 2D地图可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 高程地图
        im1 = axes[0].imshow(elevation_map, cmap='terrain', origin='lower')
        axes[0].set_title('Elevation Map')
        plt.colorbar(im1, ax=axes[0])
        
        # 语义地图
        semantic_colors = np.zeros((*semantic_map.shape, 3))
        for label, color in self.label_colors.items():
            mask = semantic_map == label
            semantic_colors[mask] = np.array(color) / 255.0
        
        axes[1].imshow(semantic_colors, origin='lower')
        axes[1].set_title('Semantic Map')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=np.array(color)/255.0, label=label_name)
            for label_name, color in zip(['Ground', 'Slope', 'Step', 'Obstacle'], 
                                        [self.label_colors[self.LABEL_GROUND],
                                         self.label_colors[self.LABEL_SLOPE],
                                         self.label_colors[self.LABEL_STEP],
                                         self.label_colors[self.LABEL_OBSTACLE]])
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_terrain(self, pcd_path):
        """主分析流程"""
        print("加载点云...")
        pcd = o3d.io.read_point_cloud(pcd_path)
        if len(pcd.points) == 0:
            raise ValueError("点云为空！")
        
        print("预处理点云...")
        pcd = self.preprocess_pointcloud(pcd)
        
        print("估计法向量...")
        pcd = self.estimate_normals(pcd)
        
        print("基于法向量初步分类...")
        labels, candidate_mask = self.classify_by_normals(pcd)
        
        print("分析候选区域...")
        points = np.asarray(pcd.points)
        candidate_indices = np.where(candidate_mask)[0]
        refined_labels = self.analyze_candidate_region(points, candidate_indices)
        
        # 合并标签
        final_labels = labels.copy()
        for i, refined_label in enumerate(refined_labels):
            if refined_label != -1:
                final_labels[i] = refined_label
        
        print("生成高程地图...")
        elevation_map, semantic_map, origin, resolution = self.create_elevation_map(
            points, final_labels
        )
        
        print("可视化结果...")
        self.visualize_results(pcd, final_labels, elevation_map, semantic_map, origin)
        
        return final_labels, elevation_map, semantic_map


if __name__ == "__main__":
    analyzer = TerrainAnalyzer()
    final_labels, elevation_map, semantic_map = analyzer.analyze_terrain("part.pcd")