#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
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

#include <map_optimizer/voxel_key.hpp>

#include <regex>

#include <atomic>
#include <fstream>
#include <limits>
#include <unordered_map>

#include <omp.h>

#if defined(MAP_OPTIMIZER_USE_CUDA)
#include <map_optimizer/raycasting_cuda.hpp>
#endif

using gtsam::symbol_shorthand::X;

// Amanatides & Woo style voxel traversal (3D DDA).
// Traverses voxels from origin toward end_pt, excluding the end voxel.
// For each traversed voxel that exists in voxel_to_index, increments the per-thread counter map.
inline void traverse_voxels_dda(
    const Eigen::Vector3d& origin,
    const Eigen::Vector3d& end_pt,
    const double voxel_res,
    const double inv_voxel_res,
    const std::unordered_map<VoxelKey, int, VoxelKeyHash>& voxel_to_index,
    std::unordered_map<int, int>& local_counts
) {
    Eigen::Vector3d delta = end_pt - origin;
    const double distance = delta.norm();
    if (distance <= 1e-9) {
        return;
    }
    if (distance < voxel_res * 0.5) {
        return;
    }

    const Eigen::Vector3d dir = delta / distance;

    VoxelKey v = voxel_key_from_xyz(origin.x(), origin.y(), origin.z(), inv_voxel_res);
    const VoxelKey vend = voxel_key_from_xyz(end_pt.x(), end_pt.y(), end_pt.z(), inv_voxel_res);
    if (v == vend) {
        return;
    }

    const int step_x = (dir.x() > 0.0) ? 1 : ((dir.x() < 0.0) ? -1 : 0);
    const int step_y = (dir.y() > 0.0) ? 1 : ((dir.y() < 0.0) ? -1 : 0);
    const int step_z = (dir.z() > 0.0) ? 1 : ((dir.z() < 0.0) ? -1 : 0);

    const double inf = std::numeric_limits<double>::infinity();

    auto next_boundary = [&](int voxel_coord, int step) {
        // boundary in world coordinates
        return (step > 0) ? (double)(voxel_coord + 1) * voxel_res : (double)voxel_coord * voxel_res;
    };

    double t_max_x = inf;
    double t_max_y = inf;
    double t_max_z = inf;
    double t_delta_x = inf;
    double t_delta_y = inf;
    double t_delta_z = inf;

    if (step_x != 0) {
        const double bx = next_boundary(v.x, step_x);
        t_max_x = (bx - origin.x()) / dir.x();
        t_delta_x = voxel_res / std::abs(dir.x());
    }
    if (step_y != 0) {
        const double by = next_boundary(v.y, step_y);
        t_max_y = (by - origin.y()) / dir.y();
        t_delta_y = voxel_res / std::abs(dir.y());
    }
    if (step_z != 0) {
        const double bz = next_boundary(v.z, step_z);
        t_max_z = (bz - origin.z()) / dir.z();
        t_delta_z = voxel_res / std::abs(dir.z());
    }

    // Step into the next voxel first, so we don't count the origin cell.
    // Also mimic the previous sampling behavior: only count voxels before (distance - voxel_res).
    while (!(v == vend)) {
        double t_next = 0.0;
        if (t_max_x < t_max_y) {
            if (t_max_x < t_max_z) {
                t_next = t_max_x;
                v.x += step_x;
                t_max_x += t_delta_x;
            } else {
                t_next = t_max_z;
                v.z += step_z;
                t_max_z += t_delta_z;
            }
        } else {
            if (t_max_y < t_max_z) {
                t_next = t_max_y;
                v.y += step_y;
                t_max_y += t_delta_y;
            } else {
                t_next = t_max_z;
                v.z += step_z;
                t_max_z += t_delta_z;
            }
        }

        if (t_next > distance - voxel_res) {
            break;
        }

        if (v == vend) {
            break;
        }

        auto it = voxel_to_index.find(v);
        if (it != voxel_to_index.end()) {
            local_counts[it->second] += 1;
        }
    }
}

struct {
    int num_threads;
    int max_iterations;

    bool enable_factor_graph_optimization;
    int k_neighbors;
    double loop_dist_thres;
    double gicp_max_correspondence_distance;

    bool enable_raycasting_filter;
    double voxel_resolution;
    int pass_through_threshold;

    bool enable_downsample;
    double downsample_leaf_size;

    bool enable_radius_outlier_removal;
    double radius_search;
    int min_neighbors_in_radius;

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

gtsam_points::PointCloudCPU::Ptr pcl_to_gtsam_points(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud) {
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
        node->declare_parameter<bool>("enable_factor_graph_optimization"),
        (int)node->declare_parameter<int>("k_neighbors"),
        node->declare_parameter<double>("loop_dist_thres"),
        node->declare_parameter<double>("gicp_max_correspondence_distance"),
        node->declare_parameter<bool>("enable_raycasting_filter"),
        node->declare_parameter<double>("voxel_resolution"),
        (int)node->declare_parameter<int>("pass_through_threshold"),
        node->declare_parameter<bool>("enable_downsample"),
        node->declare_parameter<double>("downsample_leaf_size"),
        node->declare_parameter<bool>("enable_radius_outlier_removal"),
        node->declare_parameter<double>("radius_search"),
        (int)node->declare_parameter<int>("min_neighbors_in_radius"),
        node->declare_parameter<bool>("enable_statistical_outlier_removal"),
        (int)node->declare_parameter<int>("mean_k"),
        node->declare_parameter<double>("std_dev_mul_thresh"),
        node->declare_parameter<bool>("enable_mls_smoothing"),
        node->declare_parameter<double>("search_radius"),
        (int)node->declare_parameter<int>("polynomial_order")
    };
    std::string data_path = node->declare_parameter<std::string>("data_path");
    bool use_cuda_raycasting = node->declare_parameter<bool>("use_cuda_raycasting", true);
    std::vector<gtsam::Pose3> poses;
    std::vector<gtsam_points::PointCloudCPU::Ptr> frames;
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
            poses.push_back(parse_pose(line));
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "Skipping invalid line: %s", line.c_str());
        }
    }
    RCLCPP_INFO(get_logger(), "Loaded %zu poses", poses.size());

    // Load frames
    for (size_t i = 0; i < poses.size(); i++) {
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

    if (poses.size() != frames.size()) {
        RCLCPP_ERROR(get_logger(), "Number of poses and frames do not match");
        return 1;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr merged = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    if (node_params.enable_factor_graph_optimization) {
        // Build graph
        gtsam::NonlinearFactorGraph graph;
        gtsam::Values values;

        auto prior_noise = gtsam::noiseModel::Isotropic::Precision(6, 1e6);

        for (size_t i = 0; i < poses.size(); i++) {
            values.insert(X(i), poses[i]);

            if (i == 0) {
                graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), poses[0], prior_noise));
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
        for (size_t i = 0; i < poses.size(); i++) {
            for (size_t j = i + 2; j < poses.size(); j++) {
                double dist = (poses[i].translation() - poses[j].translation()).norm();
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

        for (size_t i = 0; i < poses.size(); i++) {
            poses[i] = result.at<gtsam::Pose3>(X(i));
        }
    }

    // Merge point clouds
    for (size_t i = 0; i < poses.size(); i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        Eigen::Matrix4f T = poses[i].matrix().cast<float>();
        pcl::transformPointCloud(*pcl_frames[i], *transformed, T);
        *merged += *transformed;
    }

    // Dynamic Object Removal
    if (node_params.enable_raycasting_filter) {
        RCLCPP_INFO(get_logger(), "Removing dynamic objects...");
        double voxel_res = node_params.voxel_resolution;
        const double inv_voxel_res = 1.0 / voxel_res;
        std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_to_index;
        std::vector<VoxelKey> keys_by_index;

        // Heuristic reserve to reduce rehashing (merged points >> unique voxels).
        voxel_to_index.reserve(std::max<size_t>(1024, merged->size() / 8));

        // Build voxel map from merged cloud (unique voxels only)
        keys_by_index.reserve(std::max<size_t>(1024, merged->size() / 16));
        for (const auto& pt : *merged) {
            VoxelKey k = voxel_key_from_xyz(pt.x, pt.y, pt.z, inv_voxel_res);
            const int new_index = (int)voxel_to_index.size();
            auto [it, inserted] = voxel_to_index.emplace(k, new_index);
            if (inserted) {
                keys_by_index.push_back(k);
            }
        }

        std::vector<int> pass_through_counts(keys_by_index.size(), 0);

        bool used_gpu = false;
#if defined(MAP_OPTIMIZER_USE_CUDA)
        if (use_cuda_raycasting) {
            RCLCPP_INFO(get_logger(), "Raycasting (CUDA) ...");

            std::vector<Float3> frame_origins;
            std::vector<float> frame_rotations_rowmajor;
            std::vector<Float3> points_local;
            std::vector<int> frame_offsets;

            frame_origins.resize(poses.size());
            frame_rotations_rowmajor.resize(poses.size() * 9);
            frame_offsets.resize(poses.size() + 1);
            frame_offsets[0] = 0;

            size_t total_points = 0;
            for (size_t i = 0; i < pcl_frames.size(); i++) {
                total_points += pcl_frames[i]->size();
                frame_offsets[i + 1] = (int)total_points;
            }
            points_local.reserve(total_points);

            for (size_t i = 0; i < poses.size(); i++) {
                const gtsam::Pose3 pose = poses[i];
                const Eigen::Matrix3d R = pose.rotation().matrix();
                const auto t = pose.translation();
                frame_origins[i] = Float3{(float)t.x(), (float)t.y(), (float)t.z()};

                // Row-major
                float* R9 = frame_rotations_rowmajor.data() + i * 9;
                R9[0] = (float)R(0, 0);
                R9[1] = (float)R(0, 1);
                R9[2] = (float)R(0, 2);
                R9[3] = (float)R(1, 0);
                R9[4] = (float)R(1, 1);
                R9[5] = (float)R(1, 2);
                R9[6] = (float)R(2, 0);
                R9[7] = (float)R(2, 1);
                R9[8] = (float)R(2, 2);

                for (const auto& pt_local : *pcl_frames[i]) {
                    points_local.push_back(Float3{pt_local.x, pt_local.y, pt_local.z});
                }
            }

            std::string cuda_error;
            std::vector<int> cuda_counts;
            if (raycasting_cuda_compute_counts(keys_by_index,
                                               (float)voxel_res,
                                               frame_origins,
                                               frame_rotations_rowmajor,
                                               points_local,
                                               frame_offsets,
                                               cuda_counts,
                                               &cuda_error)) {
                pass_through_counts = std::move(cuda_counts);
                used_gpu = true;
                RCLCPP_INFO(get_logger(), "Raycasting (CUDA) done.");
            } else {
                RCLCPP_WARN(get_logger(), "CUDA raycasting failed (%s). Falling back to CPU.", cuda_error.c_str());
            }
        }
#else
        if (use_cuda_raycasting) {
            RCLCPP_WARN(get_logger(), "use_cuda_raycasting=true but CUDA support was not compiled. Falling back to CPU.");
        }
#endif

        if (!used_gpu) {
            const int num_threads = std::max(1, node_params.num_threads);
            std::vector<std::unordered_map<int, int>> thread_local_counts(num_threads);
            for (auto& m : thread_local_counts) {
                m.reserve(16384);
            }

            std::atomic<int> processed_frames = 0;

            // Ray casting (voxel DDA). Use thread-local accumulation to avoid atomic contention.
            #pragma omp parallel for num_threads(num_threads) schedule(guided)
            for (size_t i = 0; i < poses.size(); i++) {
                const gtsam::Pose3 pose = poses[i];
                const Eigen::Matrix3d R = pose.rotation().matrix();
                const Eigen::Vector3d t(pose.translation().x(), pose.translation().y(), pose.translation().z());
                const Eigen::Vector3d origin = t;

                auto& local_counts = thread_local_counts[omp_get_thread_num()];
                for (const auto& pt_local : *pcl_frames[i]) {
                    const Eigen::Vector3d pl(pt_local.x, pt_local.y, pt_local.z);
                    const Eigen::Vector3d pt_global = R * pl + t;
                    traverse_voxels_dda(origin, pt_global, voxel_res, inv_voxel_res, voxel_to_index, local_counts);
                }

                processed_frames.fetch_add(1);
                if (omp_get_thread_num() == 0) {
                    RCLCPP_INFO(
                        get_logger(), "Raycasting progress: %.1f%% (%d / %zu)",
                        100.0 * processed_frames.load() / poses.size(), processed_frames.load(), poses.size()
                    );
                }
            }

            // Merge thread-local counts
            for (const auto& local : thread_local_counts) {
                for (const auto& kv : local) {
                    pass_through_counts[kv.first] += kv.second;
                }
            }
        }

        // Filter
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        for (const auto& pt : *merged) {
            VoxelKey k = voxel_key_from_xyz(pt.x, pt.y, pt.z, inv_voxel_res);
            auto it = voxel_to_index.find(k);
            if (it == voxel_to_index.end()) {
                // Shouldn't happen since voxel map is built from merged, but keep safe.
                filtered->push_back(pt);
                continue;
            }
            if (pass_through_counts[it->second] <= node_params.pass_through_threshold) {
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

    // Remove radius outliers
    if (node_params.enable_radius_outlier_removal) {
        RCLCPP_INFO(get_logger(), "Removing radius outliers...");
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
        ror.setInputCloud(merged);
        ror.setRadiusSearch(node_params.radius_search);
        ror.setMinNeighborsInRadius(node_params.min_neighbors_in_radius);
        ror.filter(*filtered);
        RCLCPP_INFO(get_logger(), "Radius outlier removal done. Kept %zu / %zu points", filtered->size(), merged->size());
        *merged = *filtered;
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
    for(size_t i = 0; i < poses.size(); i++) {
        const gtsam::Pose3 pose = poses[i];
        auto q = pose.rotation().toQuaternion();
        auto t = pose.translation();
        // se3(x,y,z,qx,qy,qz,qw)
        out_poses << "se3(" << t.x() << "," << t.y() << "," << t.z() << "," 
            << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << ")" << std::endl;
    }
    RCLCPP_INFO(get_logger(), "Saved optimized poses to %s", pose_output_path.c_str());
}