#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace small_glim {

/**
* @brief Raw point cloud frame
*/
class RawPoints {
public:
    using Ptr = std::shared_ptr<RawPoints>;
    using ConstPtr = std::shared_ptr<const RawPoints>;

    RawPoints(
        const sensor_msgs::msg::PointCloud2& points_msg,
        const std::string& intensity_channel,
        const std::string& ring_channel
    );

    /// Number of points
    int size() const {
        return points.size();
    }

public:
    double stamp; ///< Timestamp of the first point
    std::vector<double> times; ///< Per-point timestamps relative to the first point
    std::vector<double> intensities; ///< Point intensities
    std::vector<Eigen::Vector4d> points; ///< Point coordinates
    std::vector<Eigen::Vector4d> colors; ///< Point colors
    std::vector<uint32_t> rings; ///< Ring numbers of scanned points
};

}