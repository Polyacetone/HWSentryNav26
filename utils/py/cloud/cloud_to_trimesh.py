import open3d as o3d
import numpy as np
import os

def poisson_reconstruct(pcd, depth=9, scale=1.1, quantile=0.01):
    """
    使用泊松方法对点云进行三角网格重建
    
    参数:
        pcd (open3d.geometry.PointCloud): 输入点云（需含法向）
        depth (int): 树深度，越大越精细（默认 9）
        scale (float): 点云包围盒缩放比例（>1.0，建议 1.1~1.2）
        quantile (float): 用于剔除低密度顶点的比例（如 0.01 表示移除最稀疏的 1%）
    
    返回:
        mesh (open3d.geometry.TriangleMesh): 重建后的网格
    """
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=scale, linear_fit=False
    )

    # 将密度转为 numpy 数组
    densities = np.asarray(densities)
    
    # 移除低密度区域（通常由噪声或孤立点引起）
    if quantile > 0:
        vertices_to_remove = densities < np.quantile(densities, quantile)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(f"Removed {vertices_to_remove.sum()} low-density vertices (quantile={quantile})")

    # 计算法线以便正确着色和光照
    mesh.compute_vertex_normals()
    return mesh

def main():
    config = {
        "input": "RMUL2026.pcd", # 输入点云文件路径
        "output": "RMUL2026.ply", # 输出网格文件路径（可选）
        "voxel_size": 0.1, # 下采样体素大小（米）
        "depth": 12, # 泊松重建树深度
        "scale": 1.1, # 泊松重建缩放比例
        "quantile": 0.01, # 低密度顶点剔除比例
        "no_filter": False # 是否跳过去噪步骤
    }

    # 加载点云
    if not os.path.exists(config["input"]):
        raise FileNotFoundError(f"Input file not found: {config["input"]}")
    pcd = o3d.io.read_point_cloud(config["input"])
    print(f"Loaded {len(pcd.points)} points from {config["input"]}")

    if len(pcd.points) == 0:
        raise ValueError("Point cloud is empty!")

    # 预处理：下采样
    print(f"Downsampling with voxel size = {config["voxel_size"]} m")
    pcd = pcd.voxel_down_sample(voxel_size=config["voxel_size"])

    # 去噪（可选）
    if not config["no_filter"]:
        print("Removing statistical outliers...")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 估计法向量（泊松重建必需）
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config["voxel_size"] * 5, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=100)

    # 泊松重建
    mesh = poisson_reconstruct(
        pcd, 
        depth=config["depth"], 
        scale=config["scale"], 
        quantile=config["quantile"]
    )

    # 可视化
    print("Visualizing reconstructed mesh...")
    o3d.visualization.draw_geometries([mesh], window_name="Poisson Reconstruction")

    # 保存（可选）
    if config["output"]:
        success = o3d.io.write_triangle_mesh(config["output"], mesh)
        if success:
            print(f"Mesh saved to {config["output"]}")
        else:
            print(f"Failed to save mesh to {config["output"]}")

if __name__ == "__main__":
    main()