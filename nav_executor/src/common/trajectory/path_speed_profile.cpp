#include <nav_executor/common/trajectory/path_speed_profile.hpp>

#include <algorithm>
#include <cmath>

namespace nav_executor {

PathSpeedProfile::PathSpeedProfile(std::vector<SpeedProfileState> states)
    : states_(std::move(states)) {}

SpeedProfileState PathSpeedProfile::eval_arc_length(const double arc_length) const {
    if (states_.empty()) return {};
    if (states_.size() == 1 || arc_length <= states_.front().arc_length) {
        return states_.front();
    }
    if (arc_length >= states_.back().arc_length) return states_.back();

    const auto upper_it = std::lower_bound(
        states_.begin(), states_.end(), arc_length,
        [](const SpeedProfileState& state, const double value) {
            return state.arc_length < value;
        }
    );
    const SpeedProfileState& upper = *upper_it;
    if (std::abs(arc_length - upper.arc_length) <= 1e-12) return upper;
    const SpeedProfileState& lower = *(upper_it - 1);
    const double span = upper.arc_length - lower.arc_length;
    const double fraction = span > 0.0
        ? (arc_length - lower.arc_length) / span
        : 0.0;
    const double velocity_squared = std::lerp(
        lower.velocity * lower.velocity,
        upper.velocity * upper.velocity,
        fraction
    );

    SpeedProfileState result;
    result.arc_length = arc_length;
    result.velocity = std::sqrt(std::max(velocity_squared, 0.0));
    const double distance = arc_length - lower.arc_length;
    const double velocity_sum = lower.velocity + result.velocity;
    result.time = lower.time + (velocity_sum > 1e-12
        ? 2.0 * distance / velocity_sum
        : 0.0);
    return result;
}

double PathSpeedProfile::total_time() const {
    return states_.empty() ? 0.0 : states_.back().time;
}

double PathSpeedProfile::total_arc_length() const {
    return states_.empty() ? 0.0 : states_.back().arc_length;
}

} // namespace nav_executor
