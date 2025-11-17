#pragma once

#include <Eigen/Dense>

namespace utils {
template<typename T>
class EMAFilter {
public:
    explicit EMAFilter(const double filter_ratio): filter_ratio_(filter_ratio) { reset(); }
    void initialize(const T& val) { value_ = val; }
    void force_change_value(const T& val) { value_ = val; }
    T value() const { return value_; };
    void reset() {
        if constexpr (std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>) { // 向量
            value_ = T::Zero();
        } else if constexpr (std::is_base_of_v<Eigen::QuaternionBase<std::decay_t<T>>, std::decay_t<T>>) { // 旋转
            value_ = T::Identity();
        } else if constexpr (std::is_same_v<Eigen::Isometry3d, std::decay_t<T>>) { // 3D位姿
            value_ = T::Identity();
        } else {
            static_assert(false, "unsupported type");
        }
    }
    void update(const T& val) {
        if constexpr (std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>) { // 向量
            value_ = filter_ratio_ * value_ + (1 - filter_ratio_) * val;
        } else if constexpr (std::is_base_of_v<Eigen::QuaternionBase<std::decay_t<T>>, std::decay_t<T>>) { // 旋转
            value_ = value_.slerp(1 - filter_ratio_, val);
        } else if constexpr (std::is_same_v<Eigen::Isometry3d, std::decay_t<T>>) { // 3D位姿
            value_.translation() = filter_ratio_ * Eigen::Vector3d(value_.translation()) + (1 - filter_ratio_) * Eigen::Vector3d(val.translation());
            value_.linear() = Eigen::Quaterniond(value_.linear()).slerp(1 - filter_ratio_, Eigen::Quaterniond(val.linear())).toRotationMatrix();
        } else {
            static_assert(false, "unsupported type");
        }
    }
    
private:
    T value_;
    const double filter_ratio_ = 0; // 介于0-1之间，越大越稳定
};
}