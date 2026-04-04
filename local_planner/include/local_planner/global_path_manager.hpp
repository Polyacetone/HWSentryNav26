#pragma once

#include <optional>
#include <vector>

#include <Eigen/Core>
#include <local_planner/utils.hpp>

namespace local_planner {

/// 管理当前全局路径、fixed 目标、路径消费语义。
/// 对应 §5.3.1 GlobalPathManager。
class GlobalPathManager {
public:
    /// 收到新的全局路径控制点
    void set_path(const std::vector<Eigen::Vector2d>& control_points, bool fixed);

    /// 消费当前路径（由 FSM consume_global_path 触发）
    /// 当 keep_fixed=true 时不取消 fixed_goal_ 标志
    void consume(bool keep_fixed);

    /// 清空 fixed 目标标记
    void clear_fixed() { fixed_goal_ = false; }

    // ─── 查询 ───
    [[nodiscard]] bool has_path() const { return global_path_.has_value(); }
    [[nodiscard]] bool path_updated() const { return path_updated_; }
    [[nodiscard]] bool fixed_goal() const { return fixed_goal_; }
    [[nodiscard]] const Eigen::Vector2d& fixed_goal_pos() const { return fixed_goal_pos_; }
    [[nodiscard]] const std::optional<SplineD>& global_path() const { return global_path_; }

    /// 在一次 update 周期结束后调用，重置 path_updated 标记
    void reset_update_flag() { path_updated_ = false; }

private:
    std::optional<SplineD> global_path_;
    bool path_updated_ = false;
    bool fixed_goal_ = false;
    Eigen::Vector2d fixed_goal_pos_ = Eigen::Vector2d::Zero();
};

} // namespace local_planner