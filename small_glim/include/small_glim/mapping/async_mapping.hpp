#pragma once

#include <thread>
#include <atomic>
#include <memory>
#include <small_glim/mapping/mapping.hpp>
#include <small_glim/common/concurrent_vector.hpp>

namespace small_glim {

class AsyncMapping {
public:
    using Ptr = std::shared_ptr<AsyncMapping>;

    explicit AsyncMapping(const Config::Ptr config);
    ~AsyncMapping();

    void insert_frame(const EstimationFrame::ConstPtr frame);
    void save(const std::string& path);
    void save_raw_frames(const std::string& dir);
    void join();

private:
    void run();

    std::shared_ptr<Mapping> mapping;
    std::atomic<bool> kill_switch;
    std::atomic<bool> end_of_sequence;
    std::thread thread;
    ConcurrentVector<EstimationFrame::ConstPtr> input_frame_queue {DataStorePolicy::UPTO(100)};
};

}
