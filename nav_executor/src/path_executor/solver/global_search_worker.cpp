#include <nav_executor/path_executor/solver/global_search_worker.hpp>

#include <chrono>

namespace nav_executor {

GlobalSearchWorker::GlobalSearchWorker(const MPCParams& mpc_params, GlobalSearchWorkerParams params, rclcpp::Logger logger)
    : mpc_params_(mpc_params), params_(params), search_(mpc_params.follow.global_search), selector_(params.selector),
      logger_(std::move(logger)) {
    if (params_.enable) thread_ = std::jthread([this](std::stop_token token) { run(token); });
}

GlobalSearchWorker::~GlobalSearchWorker() {
    if (thread_.joinable()) {
        thread_.request_stop();
        input_ready_.notify_all();
    }
}

void GlobalSearchWorker::submit(GlobalSearchInput input) {
    if (!params_.enable) return;
    input.generation = generation_.load(std::memory_order_acquire);
    {
        std::scoped_lock lock(input_mutex_);
        input_.reset();
        input_.emplace(std::move(input));
    }
    input_ready_.notify_one();
}

std::optional<GlobalSearchOutput> GlobalSearchWorker::take_latest(const uint64_t current_sequence) {
    std::scoped_lock lock(output_mutex_);
    if (!output_) return std::nullopt;
    auto result = std::move(output_);
    output_.reset();
    if (result->generation != generation_.load(std::memory_order_acquire)) return std::nullopt;
    if (current_sequence < result->origin_seq
        || current_sequence - result->origin_seq > static_cast<uint64_t>(std::max(params_.max_seed_age_ticks, 0))) {
        return std::nullopt;
    }
    return result;
}

void GlobalSearchWorker::clear() {
    generation_.fetch_add(1, std::memory_order_acq_rel);
    std::scoped_lock input_lock(input_mutex_);
    std::scoped_lock output_lock(output_mutex_);
    input_.reset();
    output_.reset();
}

void GlobalSearchWorker::run(const std::stop_token stop_token) {
    while (!stop_token.stop_requested()) {
        std::optional<GlobalSearchInput> input;
        {
            std::unique_lock lock(input_mutex_);
            input_ready_.wait(lock, stop_token, [this] { return input_.has_value(); });
            if (stop_token.stop_requested()) return;
            input.emplace(std::move(*input_));
            input_.reset();
        }
        const auto started = std::chrono::steady_clock::now();
        auto result = process(std::move(*input));
        if (result) {
            std::scoped_lock lock(output_mutex_);
            output_ = std::move(result);
        }
        const auto period = std::chrono::milliseconds(std::max(params_.min_period_ms, 0));
        std::this_thread::sleep_until(started + period);
    }
}

std::optional<GlobalSearchOutput> GlobalSearchWorker::process(GlobalSearchInput input) {
    std::vector<CostMapGridView> step_grids;
    if (input.per_step_cost_maps.empty()) {
        step_grids.emplace_back(input.cost_map);
    } else {
        step_grids.reserve(input.per_step_cost_maps.size());
        for (const auto& map : input.per_step_cost_maps) step_grids.emplace_back(map);
    }
    const CostMapGridView masked_grid(input.masked_global_map);
    const FollowProblem problem(
        input.path, mpc_params_, step_grids, make_grid_info(input.cost_map), masked_grid,
        input.prediction_dt, input.schedule_rho,
        input.remaining_energy, input.rfr_pwr_limit, input.blended_profile, input.active_step_mode,
        input.current_path_u
    );
    const auto solve_start = std::chrono::steady_clock::now();
    auto search_result = search_.search(
        problem, input.x0, input.warm_seed, input.longitudinal_seed, params_.candidate_count
    );
    const double solve_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - solve_start
    ).count();
    const double threshold_ms = 0.6 * static_cast<double>(std::max(params_.min_period_ms, 0));
    if (threshold_ms > 0.0 && solve_ms > threshold_ms) {
        RCLCPP_WARN(logger_, "GlobalSearch solve time %.2f ms > %.2f ms (0.6 * min_period_ms=%d)", solve_ms, threshold_ms, params_.min_period_ms);
    }
    auto selected = selector_.select(problem, input.x0, input.warm_seed, search_result.candidates);
    if (input.generation != generation_.load(std::memory_order_acquire)) return std::nullopt;
    if (selected) selected->origin_seq = input.sequence;
    return GlobalSearchOutput {
        .injected_seed = std::move(selected),
        .debug_paths = std::move(search_result.debug_paths),
        .origin_seq = input.sequence,
        .generation = input.generation,
    };
}

} // namespace nav_executor
