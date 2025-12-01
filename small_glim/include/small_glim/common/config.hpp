#pragma once

#include <any>
#include <rclcpp/node.hpp>

namespace small_glim {

class Config {
public:
    using Ptr = std::shared_ptr<Config>;
    explicit Config(rclcpp::Node* const node);
    template<typename T> T param(const std::string& name);

private:
    rclcpp::Node* const node_;
    std::unordered_map<std::string, std::any> config_map_;
};

}