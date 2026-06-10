#include <small_glim/odometry/async_odometry_estimation.hpp>
#include <small_glim/common/logger.hpp>
#include <cmath>

namespace small_glim {

AsyncOdometryEstimation::AsyncOdometryEstimation(const Config::Ptr config):
    odometry_estimation(std::make_shared<OdometryEstimationCPU>(config)) {
    max_scan_duration = config->param<double>("sensors.max_scan_duration");
    max_imu_wait_time_gap = config->param<double>("odometry_estimation.max_imu_wait_time_gap");
    max_imu_wait_wall_time = config->param<double>("odometry_estimation.max_imu_wait_wall_time");
    max_internal_frame_queue = config->param<int>("odometry_estimation.max_internal_frame_queue");
    kill_switch = false;
    end_of_sequence = false;
    internal_frame_queue_size = 0;
    thread = std::thread([this] { run(); });
}

AsyncOdometryEstimation::~AsyncOdometryEstimation() {
    kill_switch = true;
    join();
}

void AsyncOdometryEstimation::insert_imu(
    const double stamp,
    const Eigen::Vector3d& linear_acc,
    const Eigen::Vector3d& angular_vel
) {
    Eigen::Matrix<double, 7, 1> imu_data;
    imu_data << stamp, linear_acc, angular_vel;
    input_imu_queue.push_back(imu_data);
}

void AsyncOdometryEstimation::insert_frame(const PreprocessedFrame::Ptr frame) {
    input_frame_queue.push_back(frame);
}

void AsyncOdometryEstimation::join() {
    end_of_sequence = true;
    if (thread.joinable()) {
        thread.join();
    }
}

size_t AsyncOdometryEstimation::workload() const {
    return input_frame_queue.size() + internal_frame_queue_size;
}

void AsyncOdometryEstimation::get_results(
    std::vector<EstimationFrame::ConstPtr>& estimation_results,
    std::vector<EstimationFrame::ConstPtr>& target_ivox_frames,
    std::vector<EstimationFrame::ConstPtr>& marginalized_frames
) {
    estimation_results = output_estimation_results.get_all_and_clear();
    target_ivox_frames = output_target_ivox_frames.get_all_and_clear();
    marginalized_frames = output_marginalized_frames.get_all_and_clear();
}

void AsyncOdometryEstimation::run() {
    double last_imu_time = 0;
    std::deque<PreprocessedFrame::Ptr> raw_frames;
    std::deque<std::chrono::steady_clock::time_point> raw_frame_arrival_times;

    while (!kill_switch) {
        auto imu_frames = input_imu_queue.get_all_and_clear();
        auto new_raw_frames = input_frame_queue.get_all_and_clear();
        raw_frames.insert(raw_frames.end(), new_raw_frames.begin(), new_raw_frames.end());
        for (size_t i = 0; i < new_raw_frames.size(); i++) {
            raw_frame_arrival_times.push_back(std::chrono::steady_clock::now());
        }
        internal_frame_queue_size = raw_frames.size();

        while (raw_frames.size() > static_cast<size_t>(max_internal_frame_queue)) {
            logger::warn(
                "odom_estimation",
                "drop LiDAR frame because odometry queue is too large (|frames|={}, max={})",
                raw_frames.size(),
                max_internal_frame_queue
            );
            raw_frames.pop_front();
            raw_frame_arrival_times.pop_front();
        }

        if (imu_frames.empty() && raw_frames.empty()) {
            if (end_of_sequence) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        for (const auto& imu: imu_frames) {
            const double stamp = imu[0];
            const Eigen::Vector3d linear_acc = imu.block<3, 1>(1, 0);
            const Eigen::Vector3d angular_vel = imu.block<3, 1>(4, 0);
            odometry_estimation->insert_imu(stamp, linear_acc, angular_vel);
            last_imu_time = stamp;
        }

        while (!raw_frames.empty()) {
            const double scan_duration = raw_frames.front()->scan_end_time - raw_frames.front()->stamp;
            if (!std::isfinite(raw_frames.front()->stamp)
                || !std::isfinite(raw_frames.front()->scan_end_time)
                || scan_duration < 0.0
                || scan_duration > max_scan_duration) {
                logger::warn(
                    "odom_estimation",
                    "drop LiDAR frame with invalid scan timing stamp={:.6f} scan_end_time={:.6f}",
                    raw_frames.front()->stamp,
                    raw_frames.front()->scan_end_time
                );
                raw_frames.pop_front();
                raw_frame_arrival_times.pop_front();
                internal_frame_queue_size = raw_frames.size();
                continue;
            }

            if (!end_of_sequence && raw_frames.front()->scan_end_time > last_imu_time) {
                const double imu_time_gap = raw_frames.front()->scan_end_time - last_imu_time;
                const double wait_wall_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - raw_frame_arrival_times.front()).count();
                if (imu_time_gap > max_imu_wait_time_gap || wait_wall_time > max_imu_wait_wall_time) {
                    logger::warn(
                        "odom_estimation",
                        "drop LiDAR frame while waiting for IMU data (scan_end_time={:.6f}, last_imu_time={:.6f}, gap={:.6f}, wait={:.3f}, |frames|={})",
                        raw_frames.front()->scan_end_time,
                        last_imu_time,
                        imu_time_gap,
                        wait_wall_time,
                        raw_frames.size()
                    );
                    raw_frames.pop_front();
                    raw_frame_arrival_times.pop_front();
                    internal_frame_queue_size = raw_frames.size();
                    continue;
                }
                logger::debug(
                    "odom_estimation",
                    "waiting for IMU data (scan_end_time={:.6f}, last_imu_time={:.6f} |frames|={})",
                    raw_frames.front()->scan_end_time,
                    last_imu_time,
                    raw_frames.size()
                );
                std::this_thread::sleep_for(std::chrono::milliseconds(10));

                if (raw_frames.size() > 10) {
                    logger::warn(
                        "odom_estimation",
                        "waiting for IMU data (scan_end_time={:.6f}, last_imu_time={:.6f} |frames|={})",
                        raw_frames.front()->scan_end_time,
                        last_imu_time,
                        raw_frames.size()
                    );
                }

                break;
            }

            const auto& frame = raw_frames.front();
            std::vector<EstimationFrame::ConstPtr> marginalized;
            auto estimation_frame = odometry_estimation->insert_frame(frame, marginalized);
            auto target_ivox_frame = odometry_estimation->get_target_ivox_frame();

            if (estimation_frame) output_estimation_results.push_back(estimation_frame);
            if (target_ivox_frame) output_target_ivox_frames.push_back(target_ivox_frame);
            output_marginalized_frames.insert(marginalized);
            raw_frames.pop_front();
            raw_frame_arrival_times.pop_front();
            internal_frame_queue_size = raw_frames.size();
        }
    }

    auto marginalized = odometry_estimation->get_remaining_frames();
    output_marginalized_frames.insert(marginalized);
}

}
