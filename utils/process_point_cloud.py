import open3d as o3d
import copy

cloud = o3d.io.read_point_cloud("RMUC.pcd")
moved = copy.deepcopy(cloud).translate((0.7, 3.3, 0.6))
rotation = moved.get_rotation_matrix_from_xyz((0, 0, -0.018))
rotated = copy.deepcopy(moved).rotate(rotation, (0, 0, 0))
# cloud.scale(1000, (0, 0, 0))
o3d.io.write_point_cloud("RMUC.pcd", cloud)