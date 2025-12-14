#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/mls.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam_points/factors/integrated_gicp_factor.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/ann/kdtree.hpp>

#include <fstream>
#include <regex>
#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <unordered_map>

using gtsam::symbol_shorthand::X;

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const {
        return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
    }
};

struct {
    int num_threads;
    int max_iterations;

    int k_neighbors;
    double loop_dist_thres;
    double gicp_max_correspondence_distance;

    bool enable_raycasting_filter;
    double voxel_resolution;
    int pass_through_threshold;

    bool enable_downsample;
    double downsample_leaf_size;

    bool enable_statistical_outlier_removal;
    int mean_k;
    double std_dev_mul_thresh;

    bool enable_mls_smoothing;
    double search_radius;
    int polynomial_order;
} node_params;

rclcpp::Logger get_logger() {
    return rclcpp::get_logger("map_optimizer");
}

gtsam::Pose3 parse_pose(const std::string& line) {
    std::regex re(R"(se3\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+)\))");
    std::smatch match;
    if (std::regex_search(line, match, re)) {
        double x = std::stod(match[1]);
        double y = std::stod(match[2]);
        double z = std::stod(match[3]);
        double qx = std::stod(match[4]);
        double qy = std::stod(match[5]);
        double qz = std::stod(match[6]);
        double qw = std::stod(match[7]);
        return gtsam::Pose3(gtsam::Rot3(qw, qx, qy, qz), gtsam::Point3(x, y, z));
    }
    throw std::runtime_error("Failed to parse pose: " + line);
}

std::vector<int> find_neighbors(
    const Eigen::Vector4d* points,
    const int num_points,
    const int k
) {
    gtsam_points::KdTree tree(points, num_points);
    std::vector<int> neighbors(num_points * k);
    const auto perpoint_task = [&](int i) {
        std::vector<size_t> k_indices(k);
        std::vector<double> k_sq_dists(k);
        tree.knn_search(points[i].data(), k, k_indices.data(), k_sq_dists.data());
        std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + i * k);
    };
    #pragma omp parallel for num_threads(node_params.num_threads) schedule(guided, 8)
    for (int i = 0; i < num_points; i++) {
        perpoint_task(i);
    }
    return neighbors;
}

Eigen::Matrix4d regularize(const Eigen::Matrix4d& cov, Eigen::Vector3d* eigenvalues, Eigen::Matrix3d* eigenvectors) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
    eig.computeDirect(cov.block<3, 3>(0, 0));
    if (eigenvalues) *eigenvalues = eig.eigenvalues();
    if (eigenvectors) *eigenvectors = eig.eigenvectors();
    Eigen::Vector3d values(1e-3, 1.0, 1.0);
    Eigen::Matrix4d c = Eigen::Matrix4d::Zero();
    c.block<3, 3>(0, 0) = eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().transpose();
    return c;
}

void estimate_cov(
    const std::vector<Eigen::Vector4d>& points,
    const std::vector<int>& neighbors,
    const int k_neighbors,
    std::vector<Eigen::Vector4d>& normals,
    std::vector<Eigen::Matrix4d>& covs
) {
    if (points.empty()) {
        return;
    }

    const int k_correspondences = neighbors.size() / points.size();
    assert(k_correspondences * points.size() == neighbors.size());
    assert(k_neighbors <= k_correspondences);

    // Precompute pt * pt.transpose()
    std::vector<Eigen::Matrix4d> pt_cross(points.size());
    #pragma omp parallel for num_threads(node_params.num_threads) schedule(guided, 64)
    for (int i = 0; i < points.size(); i++) {
        pt_cross[i] = points[i] * points[i].transpose();
    }

    normals.resize(points.size());
    covs.resize(points.size());

    const auto calc_cov = [&](int i) {
        Eigen::Vector4d sum_points = Eigen::Vector4d::Zero();
        Eigen::Matrix4d sum_cross = Eigen::Matrix4d::Zero();

        const int begin = k_correspondences * i;
        for (int j = 0; j < k_neighbors; j++) {
            const int index = neighbors[begin + j];
            sum_points += points[index];
            sum_cross += pt_cross[index];
        }

        const Eigen::Vector4d mean = sum_points / k_neighbors;
        const Eigen::Matrix4d cov = (sum_cross - mean * sum_points.transpose()) / k_neighbors;

        Eigen::Matrix3d eigenvectors;
        covs[i] = regularize(cov, nullptr, &eigenvectors);
        covs[i](3, 3) = 0.0;

        normals[i] << eigenvectors.col(0), 0.0;
        if (points[i].dot(normals[i]) > 0.0) {
            normals[i] = -normals[i];
        }
    };

    // Calculate covariances
    #pragma omp parallel for num_threads(node_params.num_threads) schedule(guided, 8)
    for (int i = 0; i < points.size(); i++) {
        calc_cov(i);
    }
}

gtsam_points::PointCloud::Ptr pcl_to_gtsam_points(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud) {
    std::vector<Eigen::Vector4d> points;
    points.reserve(pcl_cloud->size());
    for (const auto& pt : *pcl_cloud) {
        points.emplace_back(pt.x, pt.y, pt.z, 1.0);
    }
    auto frame = std::make_shared<gtsam_points::PointCloudCPU>(points);
    std::vector<int> neighbors = find_neighbors(frame->points, frame->size(), node_params.k_neighbors);
    std::vector<Eigen::Vector4d> normals;
    std::vector<Eigen::Matrix4d> covs;
    estimate_cov(points, neighbors, node_params.k_neighbors, normals, covs);
    frame->add_covs(covs);
    frame->add_normals(normals);
    return frame;
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("map_optimizer");
    node_params = {
        (int)node->declare_parameter<int>("num_threads"),
        (int)node->declare_parameter<int>("max_iterations"),
        (int)node->declare_parameter<int>("k_neighbors"),
        node->declare_parameter<double>("loop_dist_thres"),
        node->declare_parameter<double>("gicp_max_correspondence_distance"),
        node->declare_parameter<bool>("enable_raycasting_filter"),
        node->declare_parameter<double>("voxel_resolution"),
        (int)node->declare_parameter<int>("pass_through_threshold"),
        node->declare_parameter<bool>("enable_downsample"),
        node->declare_parameter<double>("downsample_leaf_size"),
        node->declare_parameter<bool>("enable_statistical_outlier_removal"),
        (int)node->declare_parameter<int>("mean_k"),
        node->declare_parameter<double>("std_dev_mul_thresh"),
        node->declare_parameter<bool>("enable_mls_smoothing"),
        node->declare_parameter<double>("search_radius"),
        (int)node->declare_parameter<int>("polynomial_order")
    };
    std::string data_path = node->declare_parameter<std::string>("data_path");
    std::vector<gtsam::Pose3> initial_poses;
    std::vector<gtsam_points::PointCloud::Ptr> frames;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pcl_frames;

    // Load poses
    std::string poses_path = data_path + "/poses.txt";
    std::ifstream poses_file(poses_path);
    if (!poses_file.is_open()) {
        RCLCPP_ERROR(get_logger(), "Failed to open %s", poses_path.c_str());
        return 1;
    }
    std::string line;
    while (std::getline(poses_file, line)) {
        if (line.empty()) continue;
        try {
            initial_poses.push_back(parse_pose(line));
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "Skipping invalid line: %s", line.c_str());
        }
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu poses", initial_poses.size());

    // Load frames
    for (size_t i = 0; i < initial_poses.size(); i++) {
        std::string pcd_path = data_path + "/frame_" + std::to_string(i) + ".pcd";
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        if (pcl::io::loadPCDFile(pcd_path, *cloud) == -1) {
            RCLCPP_ERROR(get_logger(), "Failed to load %s", pcd_path.c_str());
            return 1;
        }
        pcl_frames.push_back(cloud);
        frames.push_back(pcl_to_gtsam_points(cloud));
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu frames", frames.size());

    if (initial_poses.size() != frames.size()) {
        RCLCPP_ERROR(get_logger(), "Number of poses and frames do not match");
        return 1;
    }

    // Build graph
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;

    auto prior_noise = gtsam::noiseModel::Isotropic::Precision(6, 1e6);

    for (size_t i = 0; i < initial_poses.size(); i++) {
        values.insert(X(i), initial_poses[i]);

        if (i == 0) {
            graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), initial_poses[0], prior_noise));
        } else {
            // GICP factor (sequential)
            auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(
                X(i-1), X(i), frames[i-1], frames[i]
            );
            factor->set_num_threads(node_params.num_threads);
            factor->set_max_correspondence_distance(node_params.gicp_max_correspondence_distance);
            graph.add(factor);
        }
    }

    // Loop closure (simple distance based)
    int loop_count = 0;
    for (size_t i = 0; i < initial_poses.size(); i++) {
        for (size_t j = i + 2; j < initial_poses.size(); j++) {
            double dist = (initial_poses[i].translation() - initial_poses[j].translation()).norm();
            if (dist < node_params.loop_dist_thres) {
                auto factor = gtsam::make_shared<gtsam_points::IntegratedGICPFactor>(
                    X(i), X(j), frames[i], frames[j]
                );
                factor->set_num_threads(node_params.num_threads);
                factor->set_max_correspondence_distance(node_params.gicp_max_correspondence_distance);
                graph.add(factor);
                loop_count++;
            }
        }
    }
    RCLCPP_INFO(get_logger(), "Added %d loop closure factors", loop_count);

    // Optimize
    RCLCPP_INFO(get_logger(), "Optimizing...");
    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(node_params.max_iterations);
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
    gtsam::Values result = optimizer.optimize();
    RCLCPP_INFO(get_logger(), "Optimization done");

    // Merge
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for (size_t i = 0; i < initial_poses.size(); i++) {
        gtsam::Pose3 pose = result.at<gtsam::Pose3>(X(i));
        Eigen::Matrix4f T = pose.matrix().cast<float>();
        pcl::PointCloud<pcl::PointXYZ> transformed;
        pcl::transformPointCloud(*pcl_frames[i], transformed, T);
        *merged += transformed;
    }

    // Dynamic Object Removal
    if (node_params.enable_raycasting_filter) {
        RCLCPP_INFO(get_logger(), "Removing dynamic objects...");
        double voxel_res = node_params.voxel_resolution;
        std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_to_index;
        std::vector<VoxelKey> index_to_voxel;

        // Build voxel map from merged cloud
        for (const auto& pt : *merged) {
            VoxelKey k = {
                (int)std::floor(pt.x / voxel_res),
                (int)std::floor(pt.y / voxel_res),
                (int)std::floor(pt.z / voxel_res)
            };
            if (voxel_to_index.find(k) == voxel_to_index.end()) {
                voxel_to_index[k] = index_to_voxel.size();
                index_to_voxel.push_back(k);
            }
        }

        std::vector<std::atomic<int>> pass_through_counts(index_to_voxel.size());

        // Ray casting
        #pragma omp parallel for num_threads(node_params.num_threads) schedule(guided)
        for (size_t i = 0; i < initial_poses.size(); i++) {
            gtsam::Pose3 pose = result.at<gtsam::Pose3>(X(i));
            Eigen::Vector3d origin = pose.translation();
            Eigen::Matrix4d T = pose.matrix();

            for (const auto& pt_local : *pcl_frames[i]) {
                Eigen::Vector4d pt_global_h = T * Eigen::Vector4d(pt_local.x, pt_local.y, pt_local.z, 1.0);
                Eigen::Vector3d pt_global = pt_global_h.head<3>();

                Eigen::Vector3d direction = pt_global - origin;
                double distance = direction.norm();
                direction.normalize();

                VoxelKey target_k = {
                    (int)std::floor(pt_global.x() / voxel_res),
                    (int)std::floor(pt_global.y() / voxel_res),
                    (int)std::floor(pt_global.z() / voxel_res)
                };
                
                int target_idx = -1;
                auto it_target = voxel_to_index.find(target_k);
                if (it_target != voxel_to_index.end()) {
                    target_idx = it_target->second;
                }

                int last_idx = -1;
                for (double d = 0.0; d < distance - voxel_res; d += voxel_res / 2.0) {
                    Eigen::Vector3d current_pos = origin + direction * d;
                    VoxelKey k = {
                        (int)std::floor(current_pos.x() / voxel_res),
                        (int)std::floor(current_pos.y() / voxel_res),
                        (int)std::floor(current_pos.z() / voxel_res)
                    };

                    auto it = voxel_to_index.find(k);
                    if (it != voxel_to_index.end()) {
                        int idx = it->second;
                        if (idx != target_idx && idx != last_idx) {
                            pass_through_counts[idx].fetch_add(1);
                            last_idx = idx;
                        }
                    }
                }
            }
        }

        // Filter
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        for (const auto& pt : *merged) {
            VoxelKey k = {
                (int)std::floor(pt.x / voxel_res),
                (int)std::floor(pt.y / voxel_res),
                (int)std::floor(pt.z / voxel_res)
            };
            if (pass_through_counts[voxel_to_index[k]] <= node_params.pass_through_threshold) {
                filtered->push_back(pt);
            }
        }
        RCLCPP_INFO(get_logger(), "Dynamic removal done. Kept %zu / %zu points", filtered->size(), merged->size());
        *merged = *filtered;
    }

    // Downsample
    if (node_params.enable_downsample) {
        RCLCPP_INFO(get_logger(), "Downsampling...");
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(merged);
        sor.setLeafSize(
            node_params.downsample_leaf_size,
            node_params.downsample_leaf_size,
            node_params.downsample_leaf_size
        );
        sor.filter(*downsampled);
        RCLCPP_INFO(get_logger(), "Downsampling done. Kept %zu / %zu points", downsampled->size(), merged->size());
        *merged = *downsampled;
    }

    // Remove statistical outliers
    if (node_params.enable_statistical_outlier_removal) {
        RCLCPP_INFO(get_logger(), "Removing statistical outliers...");
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(merged);
        sor.setMeanK(node_params.mean_k);
        sor.setStddevMulThresh(node_params.std_dev_mul_thresh);
        sor.filter(*filtered);
        RCLCPP_INFO(get_logger(), "Statistical outlier removal done. Kept %zu / %zu points", filtered->size(), merged->size());
        *merged = *filtered;
    }

    // MLS smoothing
    if (node_params.enable_mls_smoothing) {
        RCLCPP_INFO(get_logger(), "Applying MLS smoothing...");
        pcl::PointCloud<pcl::PointXYZ>::Ptr smoothed = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
        pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
        mls.setInputCloud(merged);
        mls.setSearchMethod(tree);
        mls.setSearchRadius(node_params.search_radius);
        mls.setPolynomialOrder(node_params.polynomial_order);
        mls.process(*smoothed);
        RCLCPP_INFO(get_logger(), "MLS smoothing done.");
        *merged = *smoothed;
    }

    std::string pcd_output_path = data_path + "/optimized_map.pcd";
    pcl::io::savePCDFileBinary(pcd_output_path, *merged);
    RCLCPP_INFO(get_logger(), "Saved optimized map to %s", pcd_output_path.c_str());
    
    // Save optimized poses
    std::string pose_output_path = data_path + "/optimized_poses.txt";
    std::ofstream out_poses(pose_output_path);
    for(size_t i = 0; i < initial_poses.size(); i++) {
        gtsam::Pose3 pose = result.at<gtsam::Pose3>(X(i));
        auto q = pose.rotation().toQuaternion();
        auto t = pose.translation();
        // se3(x,y,z,qx,qy,qz,qw)
        out_poses << "se3(" << t.x() << "," << t.y() << "," << t.z() << "," 
            << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << ")" << std::endl;
    }
    RCLCPP_INFO(get_logger(), "Saved optimized poses to %s", pose_output_path.c_str());
}