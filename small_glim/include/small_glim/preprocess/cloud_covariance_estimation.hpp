#pragma once

#include <vector>
#include <Eigen/Dense>

namespace small_glim {

enum class RegularizationMethod { NONE, PLANE, NORMALIZED_MIN_EIG, FROBENIUS };

class CloudCovarianceEstimation {
public:
    explicit CloudCovarianceEstimation(const int num_threads = 1);

    void estimate(
        const std::vector<Eigen::Vector4d>& points,
        const std::vector<size_t>& neighbors,
        std::vector<Eigen::Vector4d>& normals,
        std::vector<Eigen::Matrix4d>& covs
    ) const;

    void estimate(
        const std::vector<Eigen::Vector4d>& points,
        const std::vector<size_t>& neighbors,
        const size_t k_neighbors,
        std::vector<Eigen::Vector4d>& normals,
        std::vector<Eigen::Matrix4d>& covs
    ) const;

    std::vector<Eigen::Matrix4d> estimate(
        const std::vector<Eigen::Vector4d>& points,
        const std::vector<size_t>& neighbors,
        const size_t k_neighbors
    ) const;

    std::vector<Eigen::Matrix4d> estimate(
        const std::vector<Eigen::Vector4d>& points,
        const std::vector<size_t>& neighbors
    ) const;

    Eigen::Matrix4d regularize(
        const Eigen::Matrix4d& cov,
        Eigen::Vector3d* eigenvalues = nullptr,
        Eigen::Matrix3d* eigenvectors = nullptr
    ) const;

private:
    void compute_covariances(
        const std::vector<Eigen::Vector4d>& points,
        const std::vector<size_t>& neighbors,
        const size_t k_neighbors,
        const size_t k_correspondences,
        std::vector<Eigen::Vector4d>* normals,
        std::vector<Eigen::Matrix4d>& covs
    ) const;

    const RegularizationMethod regularization_method;
    const int num_threads;
};

}