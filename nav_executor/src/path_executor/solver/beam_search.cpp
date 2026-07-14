#include <nav_executor/path_executor/solver/beam_search.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nav_executor/path_executor/solver/mpc_utils.hpp>

namespace nav_executor {
namespace {

struct SearchStateKey {
    std::array<int, 9> values {};

    bool operator==(const SearchStateKey&) const = default;
};

struct SearchStateKeyHash {
    size_t operator()(const SearchStateKey& key) const {
        size_t seed = 0;
        for (const int value : key.values) {
            const size_t component = std::hash<int> {}(value);
            seed ^= component + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

struct SearchNode {
    StateVec state = StateVec::Zero();
    double cost = 0.0;
    int parent_index = -1;
    std::array<ControlVec, MAX_SEARCH_MACRO_STEPS> edge_controls {};
    int edge_control_count = 0;
};

int quantize(const double value, const double resolution) {
    return static_cast<int>(std::lround(value / resolution));
}

SearchStateKey state_key(
    const StateVec& state,
    const SplinePath& path,
    const GlobalSearchBeamParams& params
) {
    const auto reference = path.eval(SplinePath::clamp_u_extrapolated(state(ix::PATH_U)));
    const double dx = state(ix::X) - reference.p.x();
    const double dy = state(ix::Y) - reference.p.y();
    const double longitudinal_error = dx * reference.cos_r + dy * reference.sin_r;
    const double lateral_error = -dx * reference.sin_r + dy * reference.cos_r;
    const double heading_error = wrap_pi(state(ix::THETA) - reference.thetar);
    return SearchStateKey {{
        quantize(state(ix::PATH_U), params.progress_bin),
        quantize(longitudinal_error, params.longitudinal_bin),
        quantize(lateral_error, params.lateral_bin),
        quantize(heading_error, params.heading_bin),
        quantize(state(ix::XH), params.hidden_state_bin),
        quantize(state(ix::V), params.velocity_bin),
        quantize(state(ix::W), params.omega_bin),
        quantize(state(ix::DV), params.velocity_bin),
        quantize(state(ix::DW), params.omega_bin),
    }};
}

double symmetric_sample(const double magnitude, const int sample_count, const int index) {
    if (sample_count <= 1) return 0.0;
    return -magnitude + 2.0 * magnitude * static_cast<double>(index) / static_cast<double>(sample_count - 1);
}

} // namespace

BeamSearch::BeamSearch(GlobalSearchBeamParams params): params_(params) {
    const auto valid_bin = [](const double value) { return std::isfinite(value) && value > 0.0; };
    if (params_.macro_steps < 2 || params_.macro_steps > MAX_SEARCH_MACRO_STEPS
        || params_.macro_steps % 2 != 0) {
        throw std::invalid_argument("global_search.beam.macro_steps must be an even number in [2, 8]");
    }
    if (params_.beam_width < 1 || params_.beam_width > 4096) {
        throw std::invalid_argument("global_search.beam.width must be in [1, 4096]");
    }
    if (params_.velocity_acceleration_samples < 1 || params_.velocity_acceleration_samples > 9
        || params_.omega_acceleration_samples < 1 || params_.omega_acceleration_samples > 15) {
        throw std::invalid_argument("global_search beam acceleration sample count is out of range");
    }
    if (params_.per_state_limit < 1 || params_.per_state_limit > params_.beam_width) {
        throw std::invalid_argument("global_search.beam.per_state_limit is out of range");
    }
    if (params_.exact_candidate_count < 1 || params_.exact_candidate_count > params_.beam_width) {
        throw std::invalid_argument("global_search.beam.exact_candidate_count is out of range");
    }
    if (!valid_bin(params_.progress_bin) || !valid_bin(params_.longitudinal_bin)
        || !valid_bin(params_.lateral_bin) || !valid_bin(params_.heading_bin)
        || !valid_bin(params_.hidden_state_bin) || !valid_bin(params_.velocity_bin)
        || !valid_bin(params_.omega_bin)) {
        throw std::invalid_argument("global_search beam state bin resolutions must be finite and positive");
    }
}

std::vector<TrajectorySeed> BeamSearch::search(
    const FollowProblem& problem,
    const StateVec& x0
) const {
    std::vector<TrajectorySeed> result;
    const int macro_steps = params_.macro_steps;
    const int beam_width = params_.beam_width;
    const int velocity_sample_count = params_.velocity_acceleration_samples;
    const int omega_sample_count = params_.omega_acceleration_samples;
    const int per_state_limit = params_.per_state_limit;
    const int output_count = params_.exact_candidate_count;
    if (!x0.allFinite()) return result;

    const auto& path = problem.reference_path();
    const auto& limits = problem.capability_profile().motion_constraints;
    const int layer_count = (MPC_HORIZON + macro_steps - 1) / macro_steps;
    CoarseSearchModel coarse_model(problem);
    std::vector<std::vector<SearchNode>> layers;
    layers.reserve(static_cast<size_t>(layer_count + 1));
    layers.push_back({SearchNode {.state = x0}});

    for (int layer_index = 0; layer_index < layer_count; ++layer_index) {
        const int fine_step = layer_index * macro_steps;
        const int step_count = std::min(macro_steps, MPC_HORIZON - fine_step);
        const auto& frontier = layers.back();
        std::vector<SearchNode> expanded;
        expanded.reserve(
            frontier.size() * static_cast<size_t>(velocity_sample_count) * static_cast<size_t>(omega_sample_count)
        );

        for (size_t parent_index = 0; parent_index < frontier.size(); ++parent_index) {
            const auto& parent = frontier[parent_index];
            for (int velocity_index = 0; velocity_index < velocity_sample_count; ++velocity_index) {
                const double acceleration = symmetric_sample(limits.acc_max, velocity_sample_count, velocity_index);
                for (int omega_index = 0; omega_index < omega_sample_count; ++omega_index) {
                    const double angular_acceleration = symmetric_sample(
                        limits.alpha_max, omega_sample_count, omega_index
                    );
                    const auto transition = coarse_model.transition(
                        parent.state, acceleration, angular_acceleration, fine_step, step_count
                    );
                    if (!transition.valid) continue;
                    expanded.push_back(SearchNode {
                        .state = transition.state,
                        .cost = parent.cost + transition.running_cost,
                        .parent_index = static_cast<int>(parent_index),
                        .edge_controls = transition.controls,
                        .edge_control_count = transition.control_count,
                    });
                }
            }
        }
        if (expanded.empty()) return result;

        std::sort(expanded.begin(), expanded.end(), [](const SearchNode& lhs, const SearchNode& rhs) {
            return lhs.cost < rhs.cost;
        });
        std::unordered_map<SearchStateKey, int, SearchStateKeyHash> state_counts;
        state_counts.reserve(static_cast<size_t>(beam_width));
        std::vector<SearchNode> next_frontier;
        next_frontier.reserve(static_cast<size_t>(beam_width));
        for (auto& node : expanded) {
            const SearchStateKey key = state_key(node.state, path, params_);
            int& count = state_counts[key];
            if (count >= per_state_limit) continue;
            ++count;
            next_frontier.push_back(std::move(node));
            if (static_cast<int>(next_frontier.size()) >= beam_width) break;
        }
        if (next_frontier.empty()) return result;
        layers.push_back(std::move(next_frontier));
    }

    const auto& final_layer = layers.back();
    std::vector<size_t> final_indices(final_layer.size());
    for (size_t i = 0; i < final_indices.size(); ++i) final_indices[i] = i;
    std::sort(final_indices.begin(), final_indices.end(), [&](const size_t lhs, const size_t rhs) {
        const double lhs_cost = final_layer[lhs].cost + problem.terminal_cost(final_layer[lhs].state);
        const double rhs_cost = final_layer[rhs].cost + problem.terminal_cost(final_layer[rhs].state);
        return lhs_cost < rhs_cost;
    });

    std::vector<size_t> selected_indices;
    selected_indices.reserve(static_cast<size_t>(output_count));
    std::unordered_set<SearchStateKey, SearchStateKeyHash> selected_states;
    const int diverse_target = (output_count + 1) / 2;
    for (const size_t index : final_indices) {
        const SearchStateKey key = state_key(final_layer[index].state, path, params_);
        if (!selected_states.insert(key).second) continue;
        selected_indices.push_back(index);
        if (static_cast<int>(selected_indices.size()) >= diverse_target) break;
    }
    for (const size_t index : final_indices) {
        if (std::find(selected_indices.begin(), selected_indices.end(), index) != selected_indices.end()) continue;
        selected_indices.push_back(index);
        if (static_cast<int>(selected_indices.size()) >= output_count) break;
    }

    result.reserve(selected_indices.size());
    for (const size_t final_index : selected_indices) {
        TrajectorySeed candidate;
        candidate.source = SeedSource::GLOBAL;
        size_t node_index = final_index;
        for (size_t layer_index = layers.size() - 1; layer_index > 0; --layer_index) {
            const auto& node = layers[layer_index][node_index];
            const int control_start = static_cast<int>(layer_index - 1) * macro_steps;
            for (int i = 0; i < node.edge_control_count; ++i) {
                candidate.controls[static_cast<size_t>(control_start + i)]
                    = node.edge_controls[static_cast<size_t>(i)];
            }
            node_index = static_cast<size_t>(node.parent_index);
        }
        result.push_back(std::move(candidate));
    }
    return result;
}

} // namespace nav_executor
