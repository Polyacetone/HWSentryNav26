#pragma once

#include <atomic>
#include <mutex>
#include <thread>
#include <small_glim/mapping/global_mapping.hpp>
#include <small_glim/common/concurrent_vector.hpp>

namespace small_glim {

/**
* @brief Global mapping executor to wrap and asynchronously run a global mapping object
* @note  All the exposed public methods except for save() are thread-safe
*/
class AsyncGlobalMapping {
public:
    /**
    * @brief Construct a new Async Global Mapping object
    * @param global_mapping         Global mapping object
    * @param optimization_interval  Optimizer is updated every this interval even if no additional values and factors are given
    */
    explicit AsyncGlobalMapping(
        const std::shared_ptr<GlobalMapping>& global_mapping,
        const int optimization_interval_sec = 5
    );

    /**
    * @brief Destroy the Async Global Mapping object
    */
    ~AsyncGlobalMapping();

    /**
    * @brief Insert an IMU frame
    * @param stamp         Timestamp
    * @param linear_acc    Linear acceleration
    * @param angular_vel   Angular velocity
    */
    void insert_imu(
        const double stamp,
        const Eigen::Vector3d& linear_acc,
        const Eigen::Vector3d& angular_vel
    );

    /**
    * @brief Insert a SubMap
    * @param submap  SubMap
    */
    void insert_submap(const SubMap::Ptr& submap);

    /**
    * @brief Wait for the global mapping thread
    */
    void join();

    /**
    * @brief Number of data in the input queue (for load control)
    * @return Input queue size
    */
    int workload() const;

    /**
    * @brief Save the mapping result
    * @note  This method may not be thread-safe and is expected to be called after join()
    * @param path    Save path
    */
    void save(const std::string& path);

    std::vector<Eigen::Vector4d> export_points();

    std::shared_ptr<GlobalMapping> get_global_mapping() {
        std::lock_guard<std::mutex> lock(global_mapping_mutex);
        return global_mapping;
    }

    void request_optimize() { request_to_optimize = true; }
    void request_recover() { request_to_recover = true; }
    void request_find_overlapping_submaps(double min_overlap) { request_to_find_overlapping_submaps.store(min_overlap); }

private:
    void run();

private:
    std::atomic_bool kill_switch; ///< Flag to stop the thread immediately (Hard kill switch)
    std::atomic_bool end_of_sequence; ///< Flag to stop the thread when the input queues become empty (Soft kill switch)
    std::thread thread;

    ConcurrentVector<Eigen::Matrix<double, 7, 1>> input_imu_queue {DataStorePolicy::UPTO(100)};
    ConcurrentVector<SubMap::Ptr> input_submap_queue {DataStorePolicy::UPTO(100)};

    int optimization_interval;
    std::atomic_bool request_to_optimize;
    std::atomic_bool request_to_recover;
    std::atomic<double> request_to_find_overlapping_submaps;

    std::mutex global_mapping_mutex;
    std::shared_ptr<GlobalMapping> global_mapping;
};

}