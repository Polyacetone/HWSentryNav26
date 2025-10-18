#pragma once

#include <Eigen/Dense>

namespace utils {
template<unsigned D>
class EMAFilter {
public:
    explicit EMAFilter(const double filter_ratio): filter_ratio_(filter_ratio) { reset(); }
    void reset() { initialize(Eigen::Vector<double, D>::Zero()); }
    void initialize(const Eigen::Vector<double, D>& val) { value_ = val; }
    void update(const Eigen::Vector<double, D>& val) { value_ = filter_ratio_ * value_ + (1 - filter_ratio_) * val; }
    void force_change_value(const Eigen::Vector<double, D>& val) { value_ = val; }
    Eigen::Vector<double, D> value() { return value_; };
    
private:
    Eigen::Vector<double, D> value_;
    const double filter_ratio_ = 0; // 介于0-1之间，越大越稳定
};
}