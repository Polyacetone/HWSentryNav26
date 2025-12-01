#include <small_glim/common/config.hpp>
#include <Eigen/Dense>

namespace small_glim {

Config::Config(rclcpp::Node* const node): node_(node) {}

template<> bool Config::param<bool>(const std::string& name) {
    if (!config_map_.contains(name)) {
        config_map_[name] = std::make_any<bool>(node_->declare_parameter<bool>(name));
    }
    return std::any_cast<bool>(config_map_[name]);
}

template<> int Config::param<int>(const std::string& name) {
    if (!config_map_.contains(name)) {
        config_map_[name] = std::make_any<int>(node_->declare_parameter<int>(name));
    }
    return std::any_cast<int>(config_map_[name]);
}

template<> float Config::param<float>(const std::string& name) {
    if (!config_map_.contains(name)) {
        config_map_[name] = std::make_any<float>(node_->declare_parameter<float>(name));
    }
    return std::any_cast<float>(config_map_[name]);
}

template<> double Config::param<double>(const std::string& name) {
    if (!config_map_.contains(name)) {
        config_map_[name] = std::make_any<double>(node_->declare_parameter<double>(name));
    }
    return std::any_cast<double>(config_map_[name]);
}

template<> std::string Config::param<std::string>(const std::string& name) {
    if (!config_map_.contains(name)) {
        config_map_[name] = std::make_any<std::string>(node_->declare_parameter<std::string>(name));
    }
    return std::any_cast<std::string>(config_map_[name]);
}

template<> std::vector<bool> Config::param<std::vector<bool>>(const std::string& name) {
    if (!config_map_.contains(name)) {
        config_map_[name] = std::make_any<std::vector<bool>>(node_->declare_parameter<std::vector<bool>>(name));
    }
    return std::any_cast<std::vector<bool>>(config_map_[name]);
}

template<> std::vector<int> Config::param<std::vector<int>>(const std::string& name) {
    if (!config_map_.contains(name)) {
        const std::vector<int64_t> i64vec = node_->declare_parameter<std::vector<int64_t>>(name);
        std::vector<int> i32vec(i64vec.size());
        std::for_each(i64vec.begin(), i64vec.end(), [&](const int64_t x) { i32vec.push_back(static_cast<int>(x)); });
        config_map_[name] = std::make_any<std::vector<int>>(i32vec);
    }
    return std::any_cast<std::vector<int>>(config_map_[name]);
}

template<> std::vector<double> Config::param<std::vector<double>>(const std::string& name) {
    if (!config_map_.contains(name)) {
        config_map_[name] = std::make_any<std::vector<double>>(node_->declare_parameter<std::vector<double>>(name));
    }
    return std::any_cast<std::vector<double>>(config_map_[name]);
}

template<> std::vector<std::string> Config::param<std::vector<std::string>>(const std::string& name) {
    if (!config_map_.contains(name)) {
        config_map_[name] = std::make_any<std::vector<std::string>>(node_->declare_parameter<std::vector<std::string>>(name));
    }
    return std::any_cast<std::vector<std::string>>(config_map_[name]);
}

template<> Eigen::Vector2d Config::param<Eigen::Vector2d>(const std::string& name) {
    if (!config_map_.contains(name)) {
        const std::vector<double> vec = node_->declare_parameter<std::vector<double>>(name);
        if (vec.size() != 2) throw std::invalid_argument("config " + name + " requires 2 arguments");
        config_map_[name] = std::make_any<Eigen::Vector2d>(vec[0], vec[1]);
    }
    return std::any_cast<Eigen::Vector2d>(config_map_[name]);
}

template<> Eigen::Vector3d Config::param<Eigen::Vector3d>(const std::string& name) {
    if (!config_map_.contains(name)) {
        const std::vector<double> vec = node_->declare_parameter<std::vector<double>>(name);
        if (vec.size() != 3) throw std::invalid_argument("config " + name + " requires 3 arguments");
        config_map_[name] = std::make_any<Eigen::Vector3d>(vec[0], vec[1], vec[2]);
    }
    return std::any_cast<Eigen::Vector3d>(config_map_[name]);
}

template<> Eigen::Vector4d Config::param<Eigen::Vector4d>(const std::string& name) {
    if (!config_map_.contains(name)) {
        const std::vector<double> vec = node_->declare_parameter<std::vector<double>>(name);
        if (vec.size() != 4) throw std::invalid_argument("config " + name + " requires 4 arguments");
        config_map_[name] = std::make_any<Eigen::Vector4d>(vec[0], vec[1], vec[2], vec[3]);
    }
    return std::any_cast<Eigen::Vector4d>(config_map_[name]);
}

template<> Eigen::Quaterniond Config::param<Eigen::Quaterniond>(const std::string& name) {
    if (!config_map_.contains(name)) {
        const std::vector<double> vec = node_->declare_parameter<std::vector<double>>(name);
        if (vec.size() != 4) throw std::invalid_argument("config " + name + " requires 4 arguments");
        config_map_[name] = std::make_any<Eigen::Quaterniond>(vec[3], vec[0], vec[1], vec[2]); // xyzw -> wxyz
    }
    return std::any_cast<Eigen::Quaterniond>(config_map_[name]);
}

template<> Eigen::Isometry3d Config::param<Eigen::Isometry3d>(const std::string& name) {
    if (!config_map_.contains(name)) {
        const std::vector<double> vec = node_->declare_parameter<std::vector<double>>(name);
        if (vec.size() != 7) throw std::invalid_argument("config " + name + " requires 7 arguments");
        const Eigen::Vector3d translation(vec[0], vec[1], vec[2]);
        const Eigen::Quaterniond rotation(vec[6], vec[3], vec[4], vec[5]); // xyzw -> wxyz
        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        transform.translate(translation);
        transform.rotate(rotation);
        config_map_[name] = std::make_any<Eigen::Isometry3d>(transform);
    }
    return std::any_cast<Eigen::Isometry3d>(config_map_[name]);
}

}