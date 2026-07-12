#pragma once

#include <condition_variable>
#include <atomic>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <rclcpp/logging.hpp>

#include <nav_executor/path_executor/solver/global_search.hpp>
#include <nav_executor/path_executor/solver/seed_selector.hpp>

namespace nav_executor {

struct GlobalSearchWorkerParams {
    bool enable = true;
    int candidate_count = 2;
    int min_period_ms = 200;
    int max_seed_age_ticks = 8;
    SeedSelectorParams selector;
};

struct GlobalSearchInput {
    SplinePath path;
    CostMap cost_map;
    CostMap masked_global_map;
    std::vector<CostMap> per_step_cost_maps;
    DirectionMap direction_map;
    StateVec x0;
    TrajectorySeed warm_seed;
    TrajectorySeed longitudinal_seed;
    CapabilityProfile blended_profile;
    std::optional<ActiveStepMode> active_step_mode;
    double prediction_dt = MPC_DT;
    double schedule_rho = 0.0;
    double remaining_energy = 0.0;
    double rfr_pwr_limit = 0.0;
    double current_path_u = 0.0;
    uint64_t sequence = 0;
    uint64_t generation = 0;
};

struct GlobalSearchOutput {
    std::optional<TrajectorySeed> injected_seed;
    std::vector<std::vector<Eigen::Vector2d>> debug_paths;
    uint64_t origin_seq = 0;
    uint64_t generation = 0;
};

class GlobalSearchWorker {
public:
    GlobalSearchWorker(const MPCParams& mpc_params, GlobalSearchWorkerParams params, rclcpp::Logger logger);
    ~GlobalSearchWorker();

    GlobalSearchWorker(const GlobalSearchWorker&) = delete;
    GlobalSearchWorker& operator=(const GlobalSearchWorker&) = delete;

    void submit(GlobalSearchInput input);
    [[nodiscard]] std::optional<GlobalSearchOutput> take_latest(uint64_t current_sequence);
    void clear();

private:
    void run(std::stop_token stop_token);
    [[nodiscard]] std::optional<GlobalSearchOutput> process(GlobalSearchInput input);

    MPCParams mpc_params_;
    GlobalSearchWorkerParams params_;
    GlobalSearch search_;
    SeedSelector selector_;
    rclcpp::Logger logger_;
    std::mutex input_mutex_;
    std::condition_variable_any input_ready_;
    std::optional<GlobalSearchInput> input_;
    std::mutex output_mutex_;
    std::optional<GlobalSearchOutput> output_;
    std::jthread thread_;
    std::atomic<uint64_t> generation_ {0};
};

} // namespace nav_executor
