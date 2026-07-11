#pragma once

/// @file motion_model.hpp
/// @brief 降阶运动学模型（Q6=A）：一阶滞后速度响应 + 独轮车积分，
///        以及从 LPV 标称模型推导等效速度时间常数 τ_v。

#include <vector>

#include <nav_executor/path_executor/solver/mpc_types.hpp>
#include <nav_executor/path_executor/solver/search/search_types.hpp>

namespace nav_executor::search {

/// 搜索用的降阶运动学模型。
///
/// 速度：一阶滞后 v_{k+1} = v_k + (v_cmd - v_k) * (1 - exp(-dt/τ_v))
/// 位姿：中点法独轮车积分，降低粗离散误差。
class MotionModel {
public:
    MotionModel(
        double dt,
        double tau_v,
        const CapabilityProfile& profile,
        const std::vector<double>& v_primitive_fracs,
        const std::vector<double>& omega_primitive_fracs
    );

    /// 从 state 出发施加基元一步，返回下一状态。
    [[nodiscard]] SearchState step(const SearchState& state, const MotionPrimitive& prim) const;

    /// 剪除违反侧向加速度 / 角加速度约束的基元后的可用基元集。
    [[nodiscard]] const std::vector<MotionPrimitive>& primitives() const { return primitives_; }

    [[nodiscard]] double dt() const { return dt_; }
    [[nodiscard]] double v_max() const { return profile_.command_bounds.vel_max; }

    /// 由 LPV 标称离散模型的速度通道稳态增益反推等效一阶时间常数。
    /// v_{k+1} ≈ ad11 * v_k + bd1 * v_cmd（稳态增益 bd1/(1-ad11) ≈ 1），
    /// 对应连续一阶系统的离散极点 ad11 = exp(-MPC_DT/τ_v)。
    [[nodiscard]] static double derive_tau_v(const LPVDiscreteModel& nominal_model);

private:
    double dt_;
    double alpha_v_; // 1 - exp(-dt/τ_v)，速度趋近系数
    CapabilityProfile profile_;
    std::vector<MotionPrimitive> primitives_;
};

} // namespace nav_executor::search
