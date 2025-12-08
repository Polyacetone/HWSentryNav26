#include <small_glim/mapping/async_mapping.hpp>

namespace small_glim {

AsyncMapping::AsyncMapping(const Config::Ptr config): mapping(std::make_shared<Mapping>(config)) {
    kill_switch = false;
    end_of_sequence = false;
    thread = std::thread([this] { run(); });
}

AsyncMapping::~AsyncMapping() {
    kill_switch = true;
    join();
}

void AsyncMapping::insert_frame(const EstimationFrame::ConstPtr frame) {
    input_frame_queue.push_back(frame);
}

void AsyncMapping::join() {
    end_of_sequence = true;
    if (thread.joinable()) {
        thread.join();
    }
}

void AsyncMapping::save(const std::string& path) {
    mapping->save(path);
}

void AsyncMapping::save_raw_frames(const std::string& dir) {
    mapping->save_raw_frames(dir);
}

void AsyncMapping::run() {
    while (!kill_switch) {
        auto frames = input_frame_queue.get_all_and_clear();
        
        if (frames.empty()) {
            if (end_of_sequence) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        for (const auto& frame : frames) {
            mapping->insert_frame(frame);
        }
    }
}

}