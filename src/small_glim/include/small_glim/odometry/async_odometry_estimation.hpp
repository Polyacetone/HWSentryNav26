#pragma once

#include <thread>
#include <atomic>
#include <small_glim/odometry/odometry_estimation.hpp>
#include <small_glim/common/concurrent_vector.hpp>

namespace small_glim {

/**
* @brief Odometry estimation executor to wrap and asynchronously run OdometryEstimationCPU
* @note  All the exposed public methods are thread-safe
*/
class AsyncOdometryEstimation {
public:
    /**
    * @brief Construct a new Async Odometry Estimation object
    * @param odometry_estimation  Odometry estimation to be wrapped
    */
    explicit AsyncOdometryEstimation(const std::shared_ptr<OdometryEstimationCPU>& odometry_estimation);

    /**
    * @brief Destroy the Async Odometry Estimation object
    */
    ~AsyncOdometryEstimation();

    /**
    * @brief Insert an IMU data into the odometry estimation
    * @param stamp         Timestamp
    * @param linear_acc    Linear acceleration
    * @param angular_vel   Angular velocity
    */
    void insert_imu(const double stamp, const Eigen::Vector3d& linear_acc, const Eigen::Vector3d& angular_vel);

    /**
    * @brief Insert a preprocessed point cloud into odometry estimation
    * @param frame  Preprocessed point cloud
    */
    void insert_frame(const PreprocessedFrame::Ptr& frame);

    /**
    * @brief Wait for the odometry estimation thread
    */
    void join();

    /**
    * @brief   Get the size of the input queue
    */
    int workload() const;

    /**
    * @brief Get the estimation results
    * @param estimation_results    Estimation results
    * @param marginalized_frames   Marginalized frames
    */
    void get_results(std::vector<EstimationFrame::ConstPtr>& estimation_results, std::vector<EstimationFrame::ConstPtr>& marginalized_frames);

private:
    void run();

private:
    std::atomic_bool kill_switch;      // Flag to stop the thread immediately (Hard kill switch)
    std::atomic_bool end_of_sequence;  // Flag to stop the thread when the input queues become empty (Soft kill switch)
    std::thread thread;

    ConcurrentVector<Eigen::Matrix<double, 7, 1>> input_imu_queue;
    ConcurrentVector<PreprocessedFrame::Ptr> input_frame_queue;

    // Output queues
    ConcurrentVector<EstimationFrame::ConstPtr> output_estimation_results;
    ConcurrentVector<EstimationFrame::ConstPtr> output_marginalized_frames;

    std::atomic_int internal_frame_queue_size;
    std::shared_ptr<OdometryEstimationCPU> odometry_estimation;
};

}