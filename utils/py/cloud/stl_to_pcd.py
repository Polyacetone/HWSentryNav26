import open3d as o3d
import os
import sys
import numpy as np

def stl_to_pcd(stl_path, pcd_path, num_points=100000, visualize=False, scale=1.0):
    """
    Converts an STL mesh file to a PCD point cloud file by sampling points.

    Args:
        stl_path (str): Path to the input STL file.
        pcd_path (str): Path to the output PCD file.
        num_points (int): Number of points to sample from the mesh.
        visualize (bool): Whether to visualize the result after conversion.
        scale (float): Scale factor applied to the sampled points.
    """
    # 1. Check if input file exists
    if not os.path.exists(stl_path):
        print(f"Error: Input file '{stl_path}' not found.")
        sys.exit(1)

    print(f"Loading mesh from: {stl_path}")
    
    # 2. Read the STL file
    try:
        mesh = o3d.io.read_triangle_mesh(stl_path)
        # Check if the mesh is empty
        if len(mesh.triangles) == 0:
            print("Error: The loaded mesh has no triangles. Please check the file format.")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading STL file: {e}")
        sys.exit(1)

    # 3. Sample points from the mesh
    # Poisson disk sampling is generally preferred for uniform distribution,
    # but uniform_sample_points is faster. Here we use Poisson for better quality.
    print(f"Sampling {num_points} points from mesh...")
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    
    # Alternative: Uniform sampling (faster but less uniform)
    # pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    # 4. Apply the scale factor
    if scale != 1.0:
        print(f"Applying scale factor {scale} to point cloud...")
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * scale)

    # 5. Save the PCD file
    print(f"Saving point cloud to: {pcd_path}")
    try:
        o3d.io.write_point_cloud(pcd_path, pcd)
        print("Conversion successful.")
    except Exception as e:
        print(f"Error writing PCD file: {e}")
        sys.exit(1)

    # 6. Optional Visualization
    if visualize:
        print("Visualizing the generated point cloud...")
        # Add a coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(pcd.get_max_bound() - pcd.get_min_bound()) * 0.5, origin=[0, 0, 0]
        )
        o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="STL to PCD Result")

def main():
    input_filename = input("请输入 STL 文件路径: ")
    output_filename = os.path.splitext(os.path.basename(input_filename))[0] + ".pcd"
    num_points = int(input("请输入采样点数量 (默认 100000): ") or "100000")
    visualize_input = input("是否可视化结果？(y/n, 默认 n): ") or "n"
    visualize = visualize_input.lower() == 'y'
    scale_input = input("请输入缩放因子 (默认 1.0): ")
    scale = float(scale_input) if scale_input else 1.0

    stl_to_pcd(input_filename, output_filename, num_points, visualize, scale)

if __name__ == "__main__":
    main()