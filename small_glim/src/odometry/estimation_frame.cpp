#include <small_glim/odometry/estimation_frame.hpp>
#include <small_glim/common/logger.hpp>

namespace small_glim {

EstimationFrame::Ptr EstimationFrame::clone() const {
    EstimationFrame::Ptr cloned = std::make_shared<EstimationFrame>();
    *cloned = *this;
    return cloned;
}

EstimationFrame::Ptr EstimationFrame::clone_wo_points() const {
    EstimationFrame::Ptr cloned = std::make_shared<EstimationFrame>();
    *cloned = *this;
    cloned->raw_frame.reset();
    cloned->frame.reset();
    cloned->voxelmaps.clear();
    return cloned;
}

const Eigen::Isometry3d EstimationFrame::T_world_frame() const {
    switch (frame_type) {
        case FrameType::WORLD: return Eigen::Isometry3d::Identity();
        case FrameType::LIDAR: return T_world_lidar;
        case FrameType::IMU: return T_world_imu;
    }
    return Eigen::Isometry3d::Identity();
}

void EstimationFrame::set_T_world_frame(FrameType frame_type, const Eigen::Isometry3d& T) {
    switch (frame_type) {
        default: {
            logger::fatal("estimation_frame", "frame_type must be either of LIDAR or IMU");
            std::exit(EXIT_FAILURE);
            break;
        }
        case FrameType::LIDAR: {
            T_world_lidar = T;
            T_world_imu = T_world_lidar * T_lidar_imu;
            break;
        }
        case FrameType::IMU: {
            T_world_imu = T;
            T_world_lidar = T_world_imu * T_lidar_imu.inverse();
            break;
        }
    }
}

}