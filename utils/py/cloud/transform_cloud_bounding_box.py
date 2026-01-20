import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("part.pcd")
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

obb = pcd_clean.get_oriented_bounding_box()

center = np.array(obb.center)        # (3,)
R = np.array(obb.R)                  # (3,3)，列向量为OBB的局部坐标轴（对应x,y,z方向）
extent = np.array(obb.extent)        # (3,)，沿局部坐标轴的长宽高

if (R[2][2] < 0): R = (-np.identity(3)) @ R # 如果z轴向下就转正

T = np.eye(4)
T[:3, :3] = R.T
T[:3, 3] = -R.T @ (center - R @ (extent / 2))

print(T)

pcd_aligned = pcd_clean.transform(T)
o3d.io.write_point_cloud("part_aligned.pcd", pcd_aligned)