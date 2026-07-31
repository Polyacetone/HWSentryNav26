#pragma once

#include <vector>

namespace nav_executor {

struct SpeedProfileState {
    double arc_length = 0.0;
    double time = 0.0;
    double velocity = 0.0;
};

// 固定空间路径上的不可变速度参数化。区间内 v² 对弧长线性，因此切向加速度恒定。
class PathSpeedProfile {
public:
    PathSpeedProfile() = default;
    explicit PathSpeedProfile(std::vector<SpeedProfileState> states);

    [[nodiscard]] bool empty() const { return states_.size() < 2; }
    [[nodiscard]] SpeedProfileState eval_arc_length(double arc_length) const;
    [[nodiscard]] double total_time() const;
    [[nodiscard]] double total_arc_length() const;
    [[nodiscard]] const std::vector<SpeedProfileState>& states() const { return states_; }

private:
    std::vector<SpeedProfileState> states_;
};

} // namespace nav_executor
