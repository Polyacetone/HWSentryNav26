import open3d as o3d
import numpy as np
import os

def align_pointcloud_by_pca(pcd, enforce_up=True):
    # 1. 提取点坐标
    points = np.asarray(pcd.points)

    # 2. 计算质心
    centroid = np.mean(points, axis=0)

    # 3. 去中心化
    centered_points = points - centroid

    # 4. 计算协方差矩阵
    cov_matrix = np.cov(centered_points, rowvar=False)

    # 5. 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # eigh 返回升序，我们按降序排列（最大方差在前）
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # 每列为一个主方向

    # eigenvectors[:,0] -> 最长方向，[:,1] -> 次长，[:,2] -> 最短

    # 6. 确定哪个主方向最接近竖直（世界 z 轴）
    world_up = np.array([0.0, 0.0, 1.0])
    dots = np.abs(eigenvectors.T @ world_up)
    vertical_idx = np.argmax(dots)  # 最接近竖直的主轴索引

    # 7. 设置局部 z 轴
    local_z = eigenvectors[:, vertical_idx]
    if enforce_up and (local_z @ world_up) < 0:
        local_z = -local_z

    # 8. 剩余两个轴作为水平方向
    all_idx = [0, 1, 2]
    horiz_idx = [i for i in all_idx if i != vertical_idx]

    # 提取两个水平主方向
    h0 = eigenvectors[:, horiz_idx[0]]
    h1 = eigenvectors[:, horiz_idx[1]]

    # 9. 投影到 xy 平面（消除 z 分量，确保水平）
    def project_to_xy(v):
        v_xy = v.copy()
        v_xy[2] = 0.0
        norm = np.linalg.norm(v_xy)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0])
        return v_xy / norm

    h0_xy = project_to_xy(h0)
    h1_xy = project_to_xy(h1)

    # 10. 构建水平基并保证为右手系（避免镜像）
    # 按特征值选择主要水平方向作为 local_x 的候选（长边为 x）
    if eigenvalues[horiz_idx[0]] >= eigenvalues[horiz_idx[1]]:
        primary_h = h0_xy
    else:
        primary_h = h1_xy

    # 如果投影太小，使用默认方向
    if np.linalg.norm(primary_h) < 1e-8:
        primary_h = np.array([1.0, 0.0, 0.0])

    local_x = primary_h / np.linalg.norm(primary_h)

    # local_y = cross(local_z, local_x) 确保右手系（x × y = z）
    local_y = np.cross(local_z, local_x)
    norm_y = np.linalg.norm(local_y)
    if norm_y < 1e-8:
        # 退化情况，尝试使用世界 x 轴生成一个正交方向
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, local_z)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        local_y = np.cross(local_z, tmp)
        norm_y = np.linalg.norm(local_y)
        if norm_y < 1e-8:
            # 最后兜底
            local_y = np.array([0.0, 1.0, 0.0])
            norm_y = 1.0
    local_y = local_y / norm_y

    # 重新正交化 local_x，防止数值误差，确保正交且右手系
    local_x = np.cross(local_y, local_z)
    local_x = local_x / np.linalg.norm(local_x)

    # 12. 构建旋转矩阵（列：x, y, z）并确保行列式为 +1（无镜像）
    R_new = np.column_stack((local_x, local_y, local_z))
    if np.linalg.det(R_new) < 0:
        # 翻转 y 轴以修正为右手系
        local_y = -local_y
        R_new = np.column_stack((local_x, local_y, local_z))

    # 13. 构建变换矩阵：先旋转（使主轴对齐坐标系），再平移（质心到原点）
    # 注意：点云原始坐标 = centroid + R_new @ p_local
    # 所以 p_local = R_new.T @ (p_world - centroid)
    T = np.eye(4)
    T[:3, :3] = R_new.T
    T[:3, 3] = -R_new.T @ centroid

    return T, R_new, eigenvalues

pcd_path = input("请输入点云文件路径（.pcd 格式）：")
save_path = os.path.splitext(pcd_path)[0] + "_aligned.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)

# 可选：先去离群点（推荐）
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# PCA 对齐
T_pca, R_pca, eigs = align_pointcloud_by_pca(pcd_clean, enforce_up=True)

# 应用变换
pcd_aligned = pcd_clean.transform(T_pca)

print(T_pca)

# 保存
o3d.io.write_point_cloud(save_path, pcd_aligned)