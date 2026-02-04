#pragma once

#include <memory>
#include <small_glim/common/raw_points.hpp>

namespace small_glim {

/**
* @brief Preprocessed point cloud
*/
struct PreprocessedFrame {
public:
    using Ptr = std::shared_ptr<PreprocessedFrame>;
    using ConstPtr = std::shared_ptr<const PreprocessedFrame>;
    /**
    * @brief Number of points
    * @return Number of points
    */
    size_t size() const {
        return points.size();
    }

public:
    double stamp; // Timestamp at the beginning of the scan
    double scan_end_time; // Timestamp at the end of the scan

    std::vector<double> times; // Point timestamps w.r.t. the first pt
    std::vector<double> intensities; // Point intensities
    std::vector<Eigen::Vector4d> points; // Points (homogeneous coordinates)

    size_t k_neighbors; // Number of neighbors of each point
    std::vector<size_t> neighbors; // k-nearest neighbors of each point
};

}