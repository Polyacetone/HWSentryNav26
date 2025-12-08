#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

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

using gtsam::symbol_shorthand::X;

struct {
    int num_threads;
    int max_iterations;
    int k_neighbors;
    double loop_dist_thres;
    double gicp_max_correspondence_distance;
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

void run(const std::string& data_path) {
    std::vector<gtsam::Pose3> initial_poses;
    std::vector<gtsam_points::PointCloud::Ptr> frames;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> pcl_frames;

    // Load poses
    std::string poses_path = data_path + "/poses.txt";
    std::ifstream poses_file(poses_path);
    if (!poses_file.is_open()) {
        RCLCPP_ERROR(get_logger(), "Failed to open %s", poses_path.c_str());
        return;
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
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile(pcd_path, *cloud) == -1) {
            RCLCPP_ERROR(get_logger(), "Failed to load %s", pcd_path.c_str());
            return;
        }
        pcl_frames.push_back(cloud);
        frames.push_back(pcl_to_gtsam_points(cloud));
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu frames", frames.size());

    if (initial_poses.empty()) return;

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

    RCLCPP_INFO(get_logger(), "Optimizing...");
    gtsam::LevenbergMarquardtParams params;
    params.setMaxIterations(node_params.max_iterations);
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
    gtsam::Values result = optimizer.optimize();
    RCLCPP_INFO(get_logger(), "Optimization done");

    // Merge and save
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < initial_poses.size(); i++) {
        gtsam::Pose3 pose = result.at<gtsam::Pose3>(X(i));
        Eigen::Matrix4f T = pose.matrix().cast<float>();
        pcl::PointCloud<pcl::PointXYZ> transformed;
        pcl::transformPointCloud(*pcl_frames[i], transformed, T);
        *merged += transformed;
    }

    std::string output_path = data_path + "/optimized_map.pcd";
    pcl::io::savePCDFileBinary(output_path, *merged);
    RCLCPP_INFO(get_logger(), "Saved optimized map to %s", output_path.c_str());
    
    // Save optimized poses
    std::ofstream out_poses(data_path + "/optimized_poses.txt");
    for(size_t i = 0; i < initial_poses.size(); i++) {
        gtsam::Pose3 pose = result.at<gtsam::Pose3>(X(i));
        auto q = pose.rotation().toQuaternion();
        auto t = pose.translation();
        // se3(x,y,z,qw,qx,qy,qz)
        out_poses << "se3(" << t.x() << "," << t.y() << "," << t.z() << "," 
            << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << ")" << std::endl;
    }
    RCLCPP_INFO(get_logger(), "Saved optimized poses");
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("map_optimizer");
    node_params = {
        (int)node->declare_parameter<int>("num_threads"),
        (int)node->declare_parameter<int>("max_iterations"),
        (int)node->declare_parameter<int>("k_neighbors"),
        node->declare_parameter<double>("loop_dist_thres"),
        node->declare_parameter<double>("gicp_max_correspondence_distance")
    };
    std::string data_path = node->declare_parameter<std::string>("data_path");
    run(data_path);
}