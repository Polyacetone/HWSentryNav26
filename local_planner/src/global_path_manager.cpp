#include <local_planner/global_path_manager.hpp>

namespace local_planner {

void GlobalPathManager::set_path(const std::vector<Eigen::Vector2d>& control_points, bool fixed) {
    if (control_points.size() < 3) {
        // 不合法的路径：清除 spline，但空路径不是 fixed 则取消 fixed 目标
        global_path_ = std::nullopt;
        if (!fixed) {
            fixed_goal_ = false;
        }
        return;
    }

    global_path_ = SplineD(control_points);
    global_path_->setExtrapolate(true);
    path_updated_ = true;

    fixed_goal_ = fixed;
    if (fixed) {
        fixed_goal_pos_ = global_path_->evaluate(1.0);
    }
}

void GlobalPathManager::consume(bool keep_fixed) {
    global_path_ = std::nullopt;
    path_updated_ = false;
    if (!keep_fixed) {
        fixed_goal_ = false;
    }
}

} // namespace local_planner
